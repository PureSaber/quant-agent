import copy
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from quant_lab.trials import TrialRegistry

from quant_agent.research_workflow.contracts import file_hash
from quant_agent.research_workflow.federation import federate, record_review, review_state


@pytest.fixture
def manifest(tmp_path):
    source = tmp_path / "evidence.json"
    source.write_text('{"result": 42}', encoding="utf-8")
    return {
        "schema": "quant.research-federation-input/v1",
        "research_id": "project",
        "question": "Can this hypothesis be supported?",
        "context": {"project_revision": "v1"},
        "attempts": [
            {
                "id": "a",
                "pathway": "notebook",
                "status": "succeeded",
                "files": [{"path": str(source), "sha256": file_hash(source)}],
            }
        ],
    }


def request(snapshot, state=None, **changes):
    return {
        "expected_snapshot": snapshot["snapshot_sha256"],
        "expected_head": state["head_sha256"] if state else None,
        "to_state": "candidate",
        "verdict": "supports",
        "attempt_ids": ["a"],
        "actor": "researcher",
        "reason": "Synthetic fixture; tests mechanics only",
        "recorded_at": "2026-01-01T00:00:00Z",
        **changes,
    }


def test_read_only_stable_snapshot_and_stale_review(manifest, tmp_path):
    directory = tmp_path / "reviews"
    view = federate(manifest)
    assert view == federate(manifest)
    assert review_state(directory, view)["state"] is None
    assert not directory.exists()
    state = record_review(directory, view, request(view))
    assert state["state"] == "candidate" and not state["execution_authorized"]
    assert not state["reviews"][0]["needs_review"]
    changed = copy.deepcopy(manifest)
    changed["context"]["project_revision"] = "v2"
    assert review_state(directory, federate(changed))["reviews"][0]["needs_review"]
    (tmp_path / "evidence.json").write_text('{"result": 99}')
    stale = review_state(directory, federate(manifest))["reviews"][0]
    assert stale["needs_review"] and stale["reason"].startswith("Synthetic")
    assert stale["integrity"] == "source_changed"
    with pytest.raises(ValueError, match="changed after snapshot"):
        record_review(directory, view, request(view, state, to_state="review"))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"attempt_ids": ["absent"]}, "unlinked"),
        ({"expected_snapshot": "old"}, "research changed"),
        ({"verdict": "profitable"}, "unsupported"),
        ({"to_state": "observe"}, "invalid research state"),
        ({"actor": ""}, "actor"),
        ({"verdict": "execution_failed"}, "technical failure"),
    ],
)
def test_invalid_reviews_do_not_create_journal(manifest, tmp_path, change, match):
    snapshot = federate(manifest)
    with pytest.raises(ValueError, match=match):
        record_review(tmp_path / "reviews", snapshot, request(snapshot, **change))
    assert not list((tmp_path / "reviews").glob("reviews/*.json"))


def test_failure_not_negative_evidence_and_no_unscoped_aggregation(manifest, tmp_path):
    manifest["attempts"][0]["status"] = "failed"
    view = federate(manifest)
    with pytest.raises(ValueError, match="completed verified"):
        record_review(tmp_path / "review", view, request(view, verdict="rejects"))
    state = record_review(tmp_path / "review", view, request(view, verdict="execution_failed"))
    assert state["reviews"][0]["verdict"] == "execution_failed"
    manifest["attempts"][0]["status"] = "completed"
    manifest["attempts"].append({**manifest["attempts"][0], "id": "b"})
    view = federate(manifest)
    with pytest.raises(ValueError, match="comparison scopes"):
        record_review(tmp_path / "review", view, request(view, state, attempt_ids=["a", "b"]))


def test_real_lab_preserves_planned_candidates_failed_retry_and_readonly(manifest, tmp_path):
    database = tmp_path / "lab.sqlite"
    registry = TrialRegistry(database)
    registry.register(
        "study",
        {
            "hypothesis": "fixture",
            "parameters": [{"x": 1}, {"x": 2}],
            "code_identity": "abc",
            "selection_rule": "prespecified",
        },
    )
    failed = registry.start("study", {"x": 1})
    registry.finish(failed, "failed", {"error": "synthetic technical failure"})
    running = registry.start("study", {"x": 1})
    before = file_hash(database)
    manifest["lab_bindings"] = [{"database": str(database), "study_ids": ["study"]}]
    view = federate(manifest)
    assert file_hash(database) == before
    assert len(view["registration"][0]["definition"]["definition"]["parameters"]) == 2
    statuses = {
        row["native_attempt_id"]: row["status"]
        for row in view["attempts"]
        if row["pathway"] == "lab"
    }
    assert statuses[failed] == "failed" and statuses[running] == "running"
    assert all(row["verified"] for row in view["attempts"])


@pytest.mark.parametrize("initialize", [False, True])
def test_two_cli_processes_reject_stale_writer(manifest, tmp_path, initialize):
    view = federate(manifest)
    directory = tmp_path / "review"
    state = record_review(directory, view, request(view)) if initialize else None
    payload = {
        **manifest,
        "review_request": request(
            view,
            state,
            to_state="review" if initialize else "candidate",
            recorded_at="2026-01-02T00:00:00Z",
        ),
    }
    argv = [
        sys.executable,
        "-m",
        "quant_agent.research_workflow.cli",
        "federate",
        "--manifest",
        "-",
        "--review-directory",
        str(directory),
        "--review",
        "-",
    ]
    processes = [
        subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        for _ in range(2)
    ]
    for process in processes:
        process.stdin.write(json.dumps(payload))
        process.stdin.close()
    outputs = []
    for process in processes:
        outputs.append(json.loads(process.stdout.read()))
        process.wait(timeout=30)
    assert sorted(process.returncode for process in processes) == [0, 2]
    assert any("stale" in item.get("reason", "") for item in outputs)
    assert len(review_state(directory, view)["reviews"]) == (2 if initialize else 1)


def test_tampered_snapshot_and_duplicate_identity_rejected(manifest, tmp_path):
    view = federate(manifest)
    view["question"] = "tampered"
    with pytest.raises(ValueError, match="digest"):
        record_review(tmp_path / "review", view, request(view))
    manifest["attempts"].append(manifest["attempts"][0])
    with pytest.raises(ValueError, match="duplicate"):
        federate(manifest)


def test_missing_lab_is_diagnostic_not_empty_success(manifest, tmp_path):
    manifest["lab_bindings"] = [
        {"database": str(tmp_path / "absent.sqlite"), "study_ids": ["study"]}
    ]
    view = federate(manifest)
    assert view["missing"] and "Lab binding unavailable" in view["missing"][0]
    assert not (tmp_path / "absent.sqlite").exists()


def test_review_bytes_tamper_and_missing_evidence(manifest, tmp_path):
    view = federate(manifest)
    directory = tmp_path / "review"
    record_review(directory, view, request(view))
    path = next((directory / "reviews").glob("*.json"))
    path.write_text("{}")
    assert review_state(directory, view)["reviews"][0]["integrity"] == "invalid"
    (tmp_path / "evidence.json").unlink()
    current = federate(manifest)
    assert not current["attempts"][0]["verified"]
    with pytest.raises(ValueError, match="completed verified"):
        record_review(tmp_path / "other", current, request(current))


@pytest.mark.parametrize("change", ["delete", "unreadable", "symlink"])
def test_snapshot_recheck_drops_old_hash_on_file_failure(manifest, tmp_path, monkeypatch, change):
    from quant_agent.research_workflow import federation

    view = federate(manifest)
    evidence = tmp_path / "evidence.json"
    if change == "delete":
        evidence.unlink()
    elif change == "unreadable":

        def denied(_):
            raise PermissionError("fixture access denied")

        monkeypatch.setattr(federation, "file_hash", denied)
    else:
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == evidence or original(path))
    with pytest.raises(ValueError, match="sources changed after snapshot"):
        record_review(tmp_path / "review", view, request(view))
    assert not (tmp_path / "review").exists()


def test_registered_attempt_append_invalidates_existing_snapshot(manifest, tmp_path):
    database = tmp_path / "lab.sqlite"
    registry = TrialRegistry(database)
    registry.register(
        "study",
        {
            "hypothesis": "fixture",
            "parameters": [{"x": 1}],
            "code_identity": "abc",
            "selection_rule": "prespecified",
        },
    )
    manifest["lab_bindings"] = [{"database": str(database), "study_ids": ["study"]}]
    view = federate(manifest)
    registry.start("study", {"x": 1})
    with pytest.raises(ValueError, match="inventory changed"):
        record_review(tmp_path / "review", view, request(view))


def test_completed_lab_artifact_is_rechecked_before_review(manifest, tmp_path):
    database = tmp_path / "lab.sqlite"
    registry = TrialRegistry(database)
    registry.register(
        "study",
        {
            "hypothesis": "fixture",
            "parameters": [{"x": 1}],
            "code_identity": "abc",
            "selection_rule": "prespecified",
        },
    )
    attempt = registry.start("study", {"x": 1})
    folder = tmp_path / "attempts" / attempt
    folder.mkdir(parents=True)
    artifact = folder / "metrics.csv"
    artifact.write_text("value\n42\n")
    result = folder / "result.json"
    result.write_text(
        json.dumps(
            {
                "study_id": "study",
                "attempt_id": attempt,
                "status": "completed",
                "artifacts": {"metrics.csv": file_hash(artifact)},
            }
        )
    )
    registry.finish(
        attempt,
        "completed",
        {"result": str(result.relative_to(tmp_path)), "sha256": file_hash(result)},
    )
    manifest["lab_bindings"] = [{"database": str(database), "study_ids": ["study"]}]
    before = file_hash(database)
    view = federate(manifest)
    row = next(item for item in view["attempts"] if item["pathway"] == "lab")
    assert row["verified"] and len(row["files"]) == 2
    artifact.write_text("value\n99\n")
    with pytest.raises(ValueError, match="sources changed"):
        record_review(tmp_path / "review", view, request(view, attempt_ids=[row["id"]]))
    assert file_hash(database) == before


def test_corrupt_linked_lab_preserves_historical_review_and_rejects_write(manifest, tmp_path):
    database = tmp_path / "lab.sqlite"
    registry = TrialRegistry(database)
    registry.register(
        "study",
        {
            "hypothesis": "fixture",
            "parameters": [{"x": 1}],
            "code_identity": "abc",
            "selection_rule": "prespecified",
        },
    )
    manifest["lab_bindings"] = [{"database": str(database), "study_ids": ["study"]}]
    view = federate(manifest)
    directory = tmp_path / "review"
    state = record_review(directory, view, request(view))
    database.write_bytes(b"not a SQLite database")
    current = federate(manifest)
    assert "Lab binding unavailable" in current["missing"][0]
    history = review_state(directory, current)
    review = history["reviews"][0]
    assert history["head_sha256"] == state["head_sha256"]
    assert review["reason"] == request(view)["reason"]
    assert review["needs_review"] and review["integrity"] == "source_changed"
    assert "database" in review["reason_for_review"]
    with pytest.raises(sqlite3.Error):
        record_review(directory, view, request(view, state, to_state="review"))
    assert review_state(directory, current)["head_sha256"] == state["head_sha256"]
