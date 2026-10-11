import copy
import sqlite3

import pytest

from quant_agent.research_history import research_history
from quant_agent.research_workflow.cli import main
from quant_agent.research_workflow.contracts import file_hash, write_json
from quant_agent.research_workflow.evidence import cite_json, evidence_graph
from quant_agent.research_workflow.lifecycle import ResearchLifecycle


@pytest.fixture
def evidence(tmp_path):
    artifact = tmp_path / "attempts/a/metrics.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("value\n1\n")
    study = tmp_path / "study.json"
    write_json(
        study,
        {
            "study_id": "study",
            "recipe": {"hypothesis": "momentum has positive value"},
            "results": [
                {
                    "attempt_id": "a",
                    "status": "completed",
                    "metrics": {"mse": 0.1},
                    "artifacts": {"metrics.csv": file_hash(artifact)},
                }
            ],
            "attempts": [
                {"attempt_id": "b", "status": "failed", "payload": {"error": "source unavailable"}}
            ],
        },
    )
    history = research_history("momentum", study_paths=[study])
    citation = cite_json(study, "/results/0/metrics", "evidence", "test evidence")
    assertion = {
        "id": "first",
        "study_id": "study",
        "attempt_id": "a",
        "claim_key": "positive-momentum",
        "verdict": "supports",
        "reason": "explicit test interpretation",
        "citation_ids": history["matches"][0]["completed_results"][0]["citation_ids"],
        "artifact_root": str(artifact.parent),
        "scope": {
            "dataset_sha256": "a" * 64,
            "interval": "2020",
            "population": "fixture",
            "target": "return",
            "cost_policy": "net fixture",
            "sample_kind": "synthetic",
        },
    }
    return study, artifact, history, citation, assertion


def test_lifecycle_evidence_and_optimistic_concurrency(evidence, tmp_path):
    _, _, _, citation, _ = evidence
    service = ResearchLifecycle(tmp_path / "lifecycle.sqlite", create=True)
    state = service.state("strategy")
    assert state["state"] is None
    states = ["candidate", "review", "observe", "paused", "review", "candidate", "retired"]
    for index, target in enumerate(states):
        state = service.transition(
            "strategy",
            target,
            actor="researcher",
            reason="evidence reviewed",
            evidence=[citation],
            recorded_at=f"2020-01-{index + 1:02d}T00:00:00Z",
            expected_head=state["head_sha256"],
        )
        assert state["execution_authorized"] is False
        assert state["state"] == target
    with pytest.raises(ValueError, match="stale"):
        service.transition(
            "strategy",
            "candidate",
            actor="researcher",
            reason="retry",
            evidence=[citation],
            recorded_at="2020-02-01T00:00:00Z",
            expected_head=None,
        )
    with pytest.raises(ValueError, match="invalid research"):
        service.transition(
            "strategy",
            "candidate",
            actor="researcher",
            reason="retry",
            evidence=[citation],
            recorded_at="2020-02-01T00:00:00Z",
            expected_head=state["head_sha256"],
        )
    with sqlite3.connect(service.path) as db:
        db.execute("UPDATE workflow_events SET sha256='tamper' WHERE sequence=2")
    with pytest.raises(ValueError, match="corrupt"):
        service.state("strategy")


def test_lifecycle_does_not_write_lab_or_grant_execution(evidence, tmp_path):
    study, _, _, citation, _ = evidence
    other = tmp_path / "lab.sqlite"
    with sqlite3.connect(other) as db:
        db.execute("CREATE TABLE studies (id TEXT)")
    before = file_hash(other)
    with pytest.raises(ValueError, match="independent"):
        ResearchLifecycle(other, create=True)
    assert before == file_hash(other)
    with pytest.raises(FileNotFoundError):
        ResearchLifecycle(tmp_path / "missing.sqlite")
    service = ResearchLifecycle(tmp_path / "life.sqlite", create=True)
    kwargs = {
        "actor": "reviewer",
        "reason": "research only",
        "evidence": [citation],
        "recorded_at": "2020-01-01T00:00:00Z",
        "expected_head": None,
    }
    with pytest.raises(ValueError, match="invalid research"):
        service.transition("s", "observe", **kwargs)
    with pytest.raises(ValueError, match="actor"):
        service.transition("s", "candidate", **{**kwargs, "actor": ""})
    with pytest.raises(ValueError, match="citation"):
        service.transition("s", "candidate", **{**kwargs, "evidence": []})
    state = service.transition("s", "candidate", **kwargs)
    with pytest.raises(ValueError, match="increase"):
        service.transition("s", "review", **{**kwargs, "expected_head": state["head_sha256"]})
    study.write_text("{}")
    assert service.state("s")["evidence_verification"]["invalid"] == 1
    with pytest.raises(ValueError, match="citation"):
        service.transition("s", "review", **{**kwargs, "expected_head": state["head_sha256"]})


def test_conflicts_are_scoped_and_failed_runs_do_not_reject_hypotheses(evidence):
    _, _, history, _, assertion = evidence
    negative = {**assertion, "id": "negative", "verdict": "rejects"}
    result = evidence_graph(history, [assertion, negative])
    assert len(result["conflicts"]) == 1
    assert len(result["edges"]) == 6
    assert any(n["kind"] == "technical_failure" for n in result["nodes"])
    different = copy.deepcopy(negative)
    different["scope"]["cost_policy"] = "different-cost"
    result = evidence_graph(history, [assertion, different])
    assert not result["conflicts"] and len(result["incomparable"]) == 1
    failed = {
        **assertion,
        "id": "failed",
        "attempt_id": "b",
        "verdict": "execution_failed",
        "citation_ids": history["matches"][0]["failures"][0]["citation_ids"],
    }
    assert evidence_graph(history, [failed])["nodes"][-1]["verified"]
    with pytest.raises(ValueError, match="compatible"):
        evidence_graph(history, [{**failed, "verdict": "rejects"}])


def test_evidence_rechecks_artifacts_and_citations_after_retrieval(evidence):
    study, artifact, history, _, assertion = evidence
    artifact.write_text("value\n999\n")
    result = evidence_graph(
        history, [assertion, {**assertion, "id": "negative", "verdict": "rejects"}]
    )
    assert not result["conflicts"]
    assert not result["nodes"][-1]["verified"]
    study.write_text("{}")
    assert evidence_graph(history, [assertion])["citation_verification"]["invalid"] > 0


@pytest.mark.parametrize("case", ["citation", "scope", "verdict", "duplicate", "claim", "schema"])
def test_graph_rejects_unsupported_links(evidence, case):
    _, _, history, _, assertion = evidence
    assertions = [copy.deepcopy(assertion)]
    if case == "citation":
        assertions[0]["citation_ids"] = ["unrelated"]
    elif case == "scope":
        del assertions[0]["scope"]["target"]
    elif case == "verdict":
        assertions[0]["verdict"] = "profit-guaranteed"
    elif case == "duplicate":
        assertions.append(assertions[0])
    elif case == "claim":
        assertions[0]["claim_key"] = ""
    else:
        history["schema_version"] = "unsupported"
    with pytest.raises(ValueError):
        evidence_graph(history, assertions)


def test_cli_retrieval_and_lifecycle(evidence, tmp_path):
    study, _, _, citation, assertion = evidence
    db = tmp_path / "life.sqlite"
    assert main(["lifecycle", "init", "--database", str(db), "--strategy", "s"]) == 0
    event = {
        "to_state": "candidate",
        "actor": "reviewer",
        "reason": "test",
        "evidence": [citation],
        "recorded_at": "2020-01-01T00:00:00Z",
        "expected_head": None,
    }
    write_json(tmp_path / "event.json", event)
    assert (
        main(
            [
                "lifecycle",
                "transition",
                "--database",
                str(db),
                "--strategy",
                "s",
                "--event",
                str(tmp_path / "event.json"),
            ]
        )
        == 0
    )
    assert main(["lifecycle", "transition", "--database", str(db), "--strategy", "s"]) == 2
    write_json(tmp_path / "assertions.json", [assertion])
    assert (
        main(
            [
                "graph",
                "momentum",
                "--assertions",
                str(tmp_path / "assertions.json"),
                "--study-path",
                str(study),
                "--output",
                str(tmp_path / "graph.json"),
            ]
        )
        == 0
    )
