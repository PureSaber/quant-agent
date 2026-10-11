"""Read-only federation and explicit reviews; never grants execution authority.

Producers supply complete inventories, not search results. Hash checks establish byte
identity only. Research conclusions remain explicit reviewer interpretations.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

from quant_agent.research_history import verify_citations

from .contracts import digest, file_hash, identifier, read_json, timestamp, write_json
from .evidence import cite_json
from .lifecycle import TRANSITIONS, ResearchLifecycle

SCHEMA = "quant.research-federation/v1"
VERDICTS = {"supports", "rejects", "inconclusive", "execution_failed"}
COMPLETE = {"completed", "succeeded"}
FAILURE = {"failed", "blocked", "interrupted", "cancelled", "timed_out"}


def check_files(references):
    """Only caller-declared local files are read; missing/changed files stay visible."""
    checks = []
    for ref in references:
        if not isinstance(ref, dict) or not re.fullmatch(r"[a-f0-9]{64}", ref.get("sha256", "")):
            raise ValueError("file references require an expected SHA256")
        path = Path(ref["path"])
        if not path.is_absolute():
            raise ValueError("file references must be absolute")
        row = {k: v for k, v in ref.items() if k not in {"verified", "actual_sha256", "reason"}}
        row["verified"] = False
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("missing file or symbolic link")
            row["actual_sha256"] = file_hash(path)
            row["verified"] = row["actual_sha256"] == ref["sha256"]
            if not row["verified"]:
                row["reason"] = "source bytes changed"
        except (OSError, ValueError) as exc:
            row["reason"] = str(exc)
        checks.append(row)
    return checks


def federate(manifest):
    """Build a deterministic snapshot; linked databases are opened read-only."""
    if manifest.get("schema") != "quant.research-federation-input/v1":
        raise ValueError("unsupported federation input schema")
    research_id = identifier(manifest["research_id"])
    if not isinstance(manifest.get("question"), str) or not manifest["question"].strip():
        raise ValueError("research question required")
    context = manifest["context"]
    if not isinstance(context, dict) or not context:
        raise ValueError("explicit current context required")
    external, registration, missing = _lab_sources(manifest.get("lab_bindings", []))
    attempts, seen = [], set()
    for item in [*manifest["attempts"], *external]:
        aid = identifier(item["id"])
        if aid in seen:
            raise ValueError("duplicate federated attempt identity")
        seen.add(aid)
        if item["pathway"] not in {"notebook", "native", "lab", "agent_ml"}:
            raise ValueError("unsupported research pathway")
        checks = check_files(item.get("files", []))
        citations = verify_citations(item.get("citations", []))
        issues = list(item.get("issues", []))
        if not checks and not citations["checked"]:
            issues.append("no declared evidence files")
        if any(not row["verified"] for row in checks):
            issues.append("evidence files missing or changed")
        if citations["invalid"]:
            issues.append("registered event evidence missing or changed")
        attempts.append({**item, "files": checks, "verified": not issues, "issues": issues})
    source_checks = check_files(manifest.get("current_sources", []))
    missing.extend(manifest.get("missing", []))
    if any(not row["verified"] for row in source_checks):
        missing.append("current registered input or source changed")
    result = {
        "schema": SCHEMA,
        "research_id": research_id,
        "question": manifest["question"],
        "context": context,
        "context_sha256": digest(context),
        "current_sources": source_checks,
        "registration": [*manifest.get("registration", []), *registration],
        "attempts": sorted(attempts, key=lambda row: row["id"]),
        "missing": missing,
        "execution_authorized": False,
        "scope": "declared linked sources only; byte integrity is not statistical validity",
    }
    result["snapshot_sha256"] = digest(result)
    return result


def _fresh(snapshot):
    if snapshot.get("schema") != SCHEMA or snapshot.get("execution_authorized") is not False:
        raise ValueError("unsupported federation snapshot")
    if digest({k: v for k, v in snapshot.items() if k != "snapshot_sha256"}) != snapshot.get(
        "snapshot_sha256"
    ):
        raise ValueError("snapshot digest mismatch")
    references = snapshot["current_sources"] + [
        ref for attempt in snapshot["attempts"] for ref in attempt["files"]
    ]
    checks = check_files(references)
    if any(
        (row["verified"], row.get("actual_sha256")) != (old["verified"], old.get("actual_sha256"))
        for row, old in zip(checks, references)
    ):
        raise ValueError("sources changed after snapshot; refresh before reviewing")
    if verify_citations([c for row in snapshot["attempts"] for c in row.get("citations", [])])[
        "invalid"
    ]:
        raise ValueError("registered evidence changed after snapshot")
    for registration in snapshot["registration"]:
        if registration.get("database") and registration.get("study_id"):
            fresh = lab_inventory(registration["database"], [registration["study_id"]])[0]
            if fresh != {k: v for k, v in registration.items() if k != "database"}:
                raise ValueError("Lab attempt inventory changed after snapshot")


def review_state(directory, snapshot):
    """Read existing lifecycle events and recheck their evidence against today's view."""
    directory = Path(directory)
    database = directory / "lifecycle.sqlite"
    if not database.exists():
        return {"state": None, "head_sha256": None, "reviews": [], "execution_authorized": False}
    state = ResearchLifecycle(database).state(snapshot["research_id"])
    reviews = []
    for event in state["events"]:
        citation = event["evidence"][0]
        try:
            path = Path(citation["source"]).resolve()
            if not path.is_relative_to(directory.resolve() / "reviews"):
                raise ValueError("review citation is outside the dedicated review directory")
            review = read_json(path)
            if file_hash(path) != citation["source_sha256"]:
                raise ValueError("review bytes changed")
            original = review["snapshot"]
            stale = original["snapshot_sha256"] != snapshot["snapshot_sha256"]
            row = {**review, "needs_review": stale, "integrity": "verified"}
            try:
                _fresh(original)
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                row.update(
                    needs_review=True, integrity="source_changed", reason_for_review=str(exc)
                )
            reviews.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reviews.append({"needs_review": True, "integrity": "invalid", "reason": str(exc)})
    return {**state, "reviews": reviews}


def record_review(directory, snapshot, request):
    """Persist one explicit interpretation via the existing CAS lifecycle journal.

    A stale writer may leave an unreferenced immutable review file. Only reviews cited
    by committed lifecycle events are authoritative; they are never overwritten.
    """
    _fresh(snapshot)
    if request["expected_snapshot"] != snapshot["snapshot_sha256"]:
        raise ValueError("research changed; refresh the evidence view")
    verdict = request["verdict"]
    if verdict not in VERDICTS:
        raise ValueError("unsupported review verdict")
    for field in ("actor", "reason"):
        if not isinstance(request[field], str) or not request[field].strip():
            raise ValueError("actor and explicit reason required")
        if len(request[field]) > 8000:
            raise ValueError("review text exceeds limit")
    requested = request["attempt_ids"]
    if not isinstance(requested, list) or len(requested) != len(set(requested)):
        raise ValueError("distinct evidence attempt IDs required")
    by_id = {row["id"]: row for row in snapshot["attempts"]}
    if not set(requested) <= set(by_id):
        raise ValueError("review cites an unlinked attempt")
    selected = [by_id[aid] for aid in requested]
    if verdict in {"supports", "rejects"} and (
        not selected
        or any(row["status"] not in COMPLETE or not row["verified"] for row in selected)
    ):
        raise ValueError("a hypothesis verdict requires completed verified evidence")
    if verdict == "execution_failed" and (
        not selected or any(row["status"] not in FAILURE or not row["verified"] for row in selected)
    ):
        raise ValueError("execution_failed requires technical failure evidence")
    if len(selected) > 1 and (
        any(not row.get("comparison_scope") for row in selected)
        or len({digest(row["comparison_scope"]) for row in selected}) != 1
    ):
        raise ValueError("multiple evidence attempts require identical declared comparison scopes")
    target = request["to_state"]
    if target == "observe" and (
        snapshot["missing"]
        or not selected
        or not all(row["verified"] and row["status"] in COMPLETE for row in selected)
    ):
        raise ValueError("observation requires complete, currently verified evidence")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "lifecycle.sqlite"
    if not database.exists():
        temporary = directory / f".initialize-{uuid.uuid4().hex}.sqlite"
        try:
            ResearchLifecycle(temporary, create=True)
            try:
                os.link(temporary, database)
            except FileExistsError:
                pass  # Another process published its fully initialized database.
        finally:
            temporary.unlink(missing_ok=True)
    service = ResearchLifecycle(database)
    state = service.state(snapshot["research_id"])
    if request["expected_head"] != state["head_sha256"]:
        raise ValueError("stale expected event head; refresh the evidence view")
    if target not in TRANSITIONS[state["state"]]:
        raise ValueError("invalid research state transition")
    when = timestamp(request["recorded_at"])
    if state["events"] and when <= timestamp(state["events"][-1]["recorded_at"]):
        raise ValueError("event timestamps must increase")
    review = {
        "schema": "quant.research-review/v1",
        "research_id": snapshot["research_id"],
        "actor": request["actor"].strip(),
        "reason": request["reason"].strip(),
        "recorded_at": when.isoformat(),
        "verdict": verdict,
        "attempt_ids": requested,
        "snapshot": snapshot,
        "to_state": target,
        "execution_authorized": False,
    }
    path = directory / "reviews" / f"{digest(review)}.json"
    if path.exists():
        if read_json(path) != review:
            raise ValueError("review content-address collision")
    else:
        temporary = path.parent / f".pending-{uuid.uuid4().hex}.json"
        try:
            write_json(temporary, review)
            try:
                os.link(temporary, path)
            except FileExistsError:
                if read_json(path) != review:
                    raise ValueError("review content-address collision")
        finally:
            temporary.unlink(missing_ok=True)
    _fresh(snapshot)
    service.transition(
        snapshot["research_id"],
        target,
        actor=review["actor"],
        reason=review["reason"],
        recorded_at=when.isoformat(),
        expected_head=request["expected_head"],
        evidence=[cite_json(path, "", "review", "explicit research interpretation")],
    )
    return review_state(directory, snapshot)


def lab_inventory(database, study_ids):
    """Exact registered studies, all event rows and planned candidates, without search limits."""
    from .lab_adapter import ReadOnlyTrialRegistry

    registry = ReadOnlyTrialRegistry(database)
    studies = []
    for study_id in study_ids:
        identifier(study_id)
        definition = registry.definition(study_id)
        with registry.connect() as db:
            rows = db.execute(
                "SELECT sequence,attempt_id,status,recorded_at,payload FROM trial_events "
                "WHERE study_id=? ORDER BY sequence",
                (study_id,),
            ).fetchall()
        studies.append(
            {
                "study_id": study_id,
                "definition": definition,
                "events": [
                    dict(
                        zip(
                            ("sequence", "attempt_id", "status", "recorded_at", "payload"),
                            (*row[:4], json.loads(row[4])),
                        )
                    )
                    for row in rows
                ],
            }
        )
    return studies


def _lab_sources(bindings):
    from quant_agent.research_history import _make_sqlite_citation, _sqlite_row

    attempts, registration, missing = [], [], []
    for binding in bindings:
        try:
            database = Path(binding["database"]).resolve()
            studies = lab_inventory(database, binding["study_ids"])
            registration.extend({**study, "database": str(database)} for study in studies)
            for study in studies:
                latest = {}
                for event in study["events"]:
                    latest[event["attempt_id"]] = event
                for aid, event in latest.items():
                    key = {"sequence": event["sequence"]}
                    row = _sqlite_row(database, "trial_events", key)
                    if row["payload"] != event["payload"] or row["status"] != event["status"]:
                        raise ValueError("Lab changed during collection; refresh")
                    identity = "lab-" + digest([str(database), study["study_id"], aid])[:40]
                    citation = _make_sqlite_citation(
                        identity,
                        database,
                        "trial_events",
                        key,
                        "",
                        row,
                        row,
                        claim="registered trial event; not a statistical conclusion",
                    )
                    files, issues = [], []
                    payload = event["payload"]
                    if event["status"] == "completed":
                        if payload.get("result") and payload.get("sha256"):
                            path = (database.parent / payload["result"]).resolve()
                            if not path.is_relative_to(database.parent):
                                raise ValueError("Lab result escapes its registered root")
                            files.append({"path": str(path), "sha256": payload["sha256"]})
                            if path.is_file() and file_hash(path) == payload["sha256"]:
                                result = read_json(path)
                                for name, expected in result.get("artifacts", {}).items():
                                    artifact = (path.parent / name).resolve()
                                    if not artifact.is_relative_to(path.parent):
                                        raise ValueError("Lab artifact escapes attempt root")
                                    files.append({"path": str(artifact), "sha256": expected})
                                if not result.get("artifacts"):
                                    issues.append("Lab result has no artifact inventory")
                        else:
                            issues.append("completed Lab attempt has no sealed result reference")
                    attempts.append(
                        {
                            "id": identity,
                            "pathway": "lab",
                            "study_id": study["study_id"],
                            "native_attempt_id": aid,
                            "status": event["status"],
                            "files": files,
                            "citations": [citation],
                            "issues": issues,
                            "message": payload.get("error", ""),
                            "comparison_scope": None,
                        }
                    )
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            missing.append(f"Lab binding unavailable: {exc}")
    return attempts, registration, missing
