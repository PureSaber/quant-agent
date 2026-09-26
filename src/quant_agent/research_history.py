"""Read-only, citation-backed retrieval over registered research history."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", value.lower()))


def _score(query: str, text: str) -> float:
    query_tokens = _tokens(query)
    text_tokens = _tokens(text)
    if not query_tokens:
        return 1.0
    return len(query_tokens & text_tokens) / len(query_tokens | text_tokens) if text_tokens else 0.0


def _json_pointer(value: Any, pointer: str) -> Any:
    current = value
    if pointer == "":
        return current
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with /")
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise KeyError(token)
    return current


def _sqlite_row(database: Path, table: str, key: Mapping[str, Any]) -> dict:
    allowed = {
        "studies": ("study_id", "definition", "sha256", "registered_at"),
        "trial_events": (
            "sequence",
            "study_id",
            "attempt_id",
            "status",
            "recorded_at",
            "payload",
        ),
    }
    if table not in allowed or not key or any(column not in allowed[table] for column in key):
        raise ValueError("Unsupported SQLite citation target")
    where = " AND ".join(f"{column}=?" for column in key)
    columns = allowed[table]
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            f"SELECT {','.join(columns)} FROM {table} WHERE {where}",
            tuple(key.values()),
        ).fetchone()
    if row is None:
        raise KeyError(f"Missing SQLite citation row: {table} {dict(key)}")
    document = dict(zip(columns, row, strict=True))
    for field in ("definition", "payload"):
        if field in document:
            document[field] = json.loads(document[field])
    return document


def _make_json_citation(
    citation_id: str, source: Path, pointer: str, value: Any, *, claim: str
) -> dict:
    return {
        "citation_id": citation_id,
        "claim": claim,
        "source_type": "json",
        "source": str(source.resolve()),
        "locator": pointer,
        "source_sha256": _digest_file(source),
        "claim_sha256": _digest_value(value),
    }


def _make_sqlite_citation(
    citation_id: str,
    source: Path,
    table: str,
    key: dict[str, Any],
    pointer: str,
    value: Any,
    row: dict,
    *,
    claim: str,
) -> dict:
    return {
        "citation_id": citation_id,
        "claim": claim,
        "source_type": "sqlite",
        "source": str(source.resolve()),
        "table": table,
        "key": key,
        "locator": pointer,
        "source_sha256": _digest_value(row),
        "claim_sha256": _digest_value(value),
    }


def verify_citations(citations: list[dict]) -> dict:
    """Re-read every cited source and verify both source and selected claim digests."""

    results = []
    for citation in citations:
        try:
            source = Path(citation["source"])
            if citation["source_type"] == "json":
                if _digest_file(source) != citation["source_sha256"]:
                    raise ValueError("source digest changed")
                document = json.loads(source.read_text(encoding="utf-8"))
            elif citation["source_type"] == "sqlite":
                document = _sqlite_row(source, citation["table"], citation["key"])
                if _digest_value(document) != citation["source_sha256"]:
                    raise ValueError("source row digest changed")
            else:
                raise ValueError("unsupported source type")
            selected = _json_pointer(document, citation["locator"])
            if _digest_value(selected) != citation["claim_sha256"]:
                raise ValueError("claim digest changed")
            results.append({"citation_id": citation["citation_id"], "valid": True})
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
            sqlite3.Error,
        ) as exc:
            results.append(
                {"citation_id": citation.get("citation_id"), "valid": False, "reason": str(exc)}
            )
    valid = sum(result["valid"] for result in results)
    return {
        "checked": len(results),
        "valid": valid,
        "invalid": len(results) - valid,
        "accuracy": valid / len(results) if results else None,
        "results": results,
    }


def _registered_result(database: Path, row: dict) -> dict:
    """Resolve the result reference emitted by execute_study, using its registered hash."""
    payload = row["payload"]
    if row["status"] != "completed" or "result" not in payload:
        return row
    try:
        path = (database.parent / payload["result"]).resolve()
        if database.parent.resolve() not in path.parents:
            raise ValueError("registered result path escapes the study directory")
        if _digest_file(path) != payload.get("sha256"):
            raise ValueError("registered result digest mismatch")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or any(
            document.get(key) != row[key] for key in ("study_id", "attempt_id", "status")
        ):
            raise ValueError("registered result identity mismatch")
        return {
            **row,
            "payload": document,
            "_source": {"type": "json", "path": path, "document": document},
            "_json_pointer": "",
            "_attempt_dir": path.parent,
        }
    except (OSError, ValueError, TypeError) as exc:
        # Preserve the terminal event without inventing metrics or trusting a summary copy.
        return {**row, "_result_error": str(exc)}


def _load_sqlite(database: Path | None) -> list[dict]:
    if database is None:
        return []
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "studies" not in tables:
            raise ValueError("SQLite history has no studies table")
        studies = connection.execute(
            "SELECT study_id,definition,sha256,registered_at FROM studies"
        ).fetchall()
        events = (
            connection.execute(
                "SELECT sequence,study_id,attempt_id,status,recorded_at,payload "
                "FROM trial_events ORDER BY sequence"
            ).fetchall()
            if "trial_events" in tables
            else []
        )
    events_by_study: dict[str, list[dict]] = {}
    for sequence, study_id, attempt_id, status, recorded_at, payload_text in events:
        row = {
            "sequence": sequence,
            "study_id": study_id,
            "attempt_id": attempt_id,
            "status": status,
            "recorded_at": recorded_at,
            "payload": json.loads(payload_text),
        }
        if status in {"completed", "failed", "interrupted", "skipped"}:
            events_by_study.setdefault(study_id, []).append(_registered_result(database, row))
    records = []
    for study_id, definition_text, digest, registered_at in studies:
        definition = json.loads(definition_text)
        row = {
            "study_id": study_id,
            "definition": definition,
            "sha256": digest,
            "registered_at": registered_at,
        }
        source = {"type": "sqlite", "path": database.resolve(), "row": row}
        records.append(
            {
                "study_id": study_id,
                "hypothesis": str(definition.get("hypothesis", "")),
                "definition": definition,
                "attempts": [
                    {**attempt, "_source": attempt.get("_source", source)}
                    for attempt in events_by_study.get(study_id, [])
                ],
                "source": source,
            }
        )
    return records


def _load_study_files(studies_root: Path | None, study_paths: Sequence[Path] = ()) -> list[dict]:
    if studies_root is None and not study_paths:
        return []
    records = []
    paths = set(studies_root.glob("**/study.json")) if studies_root is not None else set()
    for candidate in study_paths:
        paths.update(candidate.glob("**/study.json") if candidate.is_dir() else [candidate])
    for path in sorted(paths):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict) or not document.get("study_id"):
            continue
        recipe = document.get("recipe") if isinstance(document.get("recipe"), dict) else {}
        definition = (
            document.get("definition") if isinstance(document.get("definition"), dict) else recipe
        )
        source = {"type": "json", "path": path.resolve(), "document": document}
        attempts = []
        result_rows = document.get("results") if isinstance(document.get("results"), list) else []
        for index, result in enumerate(result_rows):
            if not isinstance(result, dict):
                continue
            attempts.append(
                {
                    **result,
                    "_json_pointer": f"/results/{index}",
                    "_attempt_dir": path.parent / "attempts" / str(result.get("attempt_id", "")),
                    "_source": source,
                }
            )
        for index, event in enumerate(document.get("attempts", [])):
            if not isinstance(event, dict) or event.get("status") not in {
                "failed",
                "interrupted",
                "skipped",
            }:
                continue
            attempts.append(
                {
                    **event,
                    "_json_pointer": f"/attempts/{index}/payload",
                    "_json_status_pointer": f"/attempts/{index}/status",
                    "_source": source,
                }
            )
        records.append(
            {
                "study_id": str(document["study_id"]),
                "hypothesis": str(recipe.get("hypothesis") or definition.get("hypothesis") or ""),
                "definition": definition,
                "attempts": attempts,
                "source": source,
            }
        )
    return records


def _search_text(record: dict) -> str:
    parts = [record["study_id"], record["hypothesis"], _canonical(record["definition"])]
    for attempt in record["attempts"]:
        parts.extend(
            [
                str(attempt.get("status", "")),
                str(attempt.get("error", "")),
                _canonical(attempt.get("candidate", {})),
                _canonical(attempt.get("payload", {})),
            ]
        )
    return " ".join(parts)


def _find_attempt_dir(
    root: Path | None, attempt_id: str, study_paths: Sequence[Path] = ()
) -> Path | None:
    if not attempt_id:
        return None
    roots = ([root] if root is not None else []) + [
        path if path.is_dir() else path.parent for path in study_paths
    ]
    matches = [match for base in roots for match in base.glob(f"**/attempts/{attempt_id}")]
    return matches[0] if matches else None


def validate_artifacts(artifacts: Mapping[str, str] | None, attempt_dir: Path | None) -> dict:
    """Verify declared artifact SHA-256 values without following paths outside an attempt."""

    if not artifacts:
        return {"checked": 0, "verified": 0, "invalid": 0, "status": "not-declared", "items": []}
    items = []
    for relative, expected in sorted(artifacts.items()):
        item = {"path": relative, "expected_sha256": expected}
        try:
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise TypeError("artifact path and digest must be strings")
            if attempt_dir is None:
                raise FileNotFoundError("attempt directory is unavailable")
            base = attempt_dir.resolve()
            path = (base / relative).resolve()
            try:
                path.relative_to(base)
            except ValueError as exc:
                raise ValueError("artifact path escapes the attempt directory") from exc
            if not path.is_file():
                raise FileNotFoundError("artifact is missing")
            actual = _digest_file(path)
            item["actual_sha256"] = actual
            if not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
                raise ValueError("artifact digest mismatch")
            item["status"] = "verified"
        except (OSError, TypeError, ValueError) as exc:
            item["status"] = "invalid"
            item["reason"] = str(exc)
        items.append(item)
    verified = sum(item["status"] == "verified" for item in items)
    return {
        "checked": len(items),
        "verified": verified,
        "invalid": len(items) - verified,
        "status": "verified" if verified == len(items) else "invalid",
        "items": items,
    }


def _terminal_payload(attempt: dict) -> dict:
    payload = attempt.get("payload")
    return payload if isinstance(payload, dict) else attempt


def _citation_for_study(record: dict, citation_id: str) -> dict:
    source = record["source"]
    if source["type"] == "json":
        document = source["document"]
        pointer = "/recipe/hypothesis" if "recipe" in document else "/definition/hypothesis"
        return _make_json_citation(
            citation_id,
            source["path"],
            pointer,
            record["hypothesis"],
            claim="study hypothesis",
        )
    return _make_sqlite_citation(
        citation_id,
        source["path"],
        "studies",
        {"study_id": record["study_id"]},
        "/definition/hypothesis",
        record["hypothesis"],
        source["row"],
        claim="study hypothesis",
    )


def _citation_for_attempt(
    record: dict, attempt: dict, citation_id: str, pointer: str, value: Any, claim: str
) -> dict:
    source = attempt.get("_source", record["source"])
    if source["type"] == "json":
        target = (
            attempt.get("_json_status_pointer", attempt["_json_pointer"] + "/status")
            if pointer == "$status"
            else attempt["_json_pointer"] + pointer
        )
        return _make_json_citation(
            citation_id,
            source["path"],
            target,
            value,
            claim=claim,
        )
    row = {
        key: attempt[key]
        for key in ("sequence", "study_id", "attempt_id", "status", "recorded_at", "payload")
    }
    return _make_sqlite_citation(
        citation_id,
        source["path"],
        "trial_events",
        {"sequence": attempt["sequence"]},
        "/status" if pointer == "$status" else "/payload" + pointer,
        value,
        row,
        claim=claim,
    )


def _locked_fields(recipe: dict) -> dict:
    fields = {}
    for name in ("inputs", "interval", "costs", "risk", "holdout"):
        if name in recipe:
            fields[name] = {"sha256": _digest_value(recipe[name]), "value": recipe[name]}
    return fields


def minimum_comparison(recipe: dict | None, matches: list[dict]) -> dict | None:
    """Describe the smallest paired comparison while preserving governed fields."""

    if recipe is None:
        return None
    control = None
    for match in matches:
        for result in match["completed_results"]:
            artifacts = result["artifact_validation"]
            if artifacts["checked"] and artifacts["invalid"] == 0:
                control = {
                    "study_id": match["study_id"],
                    "attempt_id": result["attempt_id"],
                    "candidate": result.get("candidate"),
                    "metrics": result.get("metrics"),
                    "citation_ids": result["citation_ids"],
                    "historical_metrics_scope": "context-only-not-a-paired-comparison",
                    "replay_required": True,
                }
                break
        if control:
            break
    return {
        "kind": "paired-minimal-control",
        "control": control,
        "candidate": {
            "factors": recipe.get("factors"),
            "factor_expressions": recipe.get("factor_expressions", {}),
            "strategy": recipe.get("strategy"),
        },
        "locked_fields": _locked_fields(recipe),
        "required_evidence": [
            "same data identity, interval, costs and risk limits",
            "paired out-of-sample or preregistered replay",
            "artifact digests verified before using reported metrics",
        ],
        "limitations": [
            "historical metrics are not directly comparable unless the control is replayed on the locked specification",
            "the recommendation is descriptive and does not claim out-of-sample incremental value",
        ],
        "executed": False,
    }


def research_history(
    query: str,
    *,
    database: str | Path | None = None,
    studies_root: str | Path | None = None,
    study_paths: Sequence[str | Path] | str | Path = (),
    recipe: dict | None = None,
    limit: int = 5,
) -> dict:
    """Retrieve studies, terminal results and failures with re-verifiable citations."""

    if type(limit) is not int or limit < 1 or limit > 50:
        raise ValueError("limit must be an integer in [1, 50]")
    database = Path(database) if database is not None else None
    studies_root = Path(studies_root) if studies_root is not None else None
    if isinstance(study_paths, (str, Path)):
        study_paths = [Path(study_paths)]
    else:
        study_paths = [Path(path) for path in study_paths]
    loaded = _load_sqlite(database) + _load_study_files(studies_root, study_paths)
    merged: dict[str, dict] = {}
    for record in loaded:
        existing = merged.get(record["study_id"])
        if existing is None:
            merged[record["study_id"]] = record
            continue
        existing["attempts"].extend(record["attempts"])
        if existing["source"]["type"] != "sqlite" and record["source"]["type"] == "sqlite":
            existing.update(
                {
                    "hypothesis": record["hypothesis"],
                    "definition": record["definition"],
                    "source": record["source"],
                }
            )
    records = list(merged.values())
    candidates = sorted(
        (
            (score, record)
            for record in records
            if (score := _score(query, _search_text(record))) > 0
        ),
        key=lambda item: (
            -item[0],
            item[1]["study_id"],
            0 if item[1]["source"]["type"] == "sqlite" else 1,
            len(str(item[1]["source"]["path"])),
            str(item[1]["source"]["path"]),
        ),
    )
    ranked = []
    seen_studies: set[str] = set()
    for item in candidates:
        study_id = item[1]["study_id"]
        if study_id in seen_studies:
            continue
        seen_studies.add(study_id)
        ranked.append(item)
        if len(ranked) == limit:
            break
    citations: list[dict] = []
    matches = []
    seen_attempts: set[tuple[str, str, str]] = set()
    for score, record in ranked:
        study_citation_id = f"citation-{len(citations) + 1}"
        citations.append(_citation_for_study(record, study_citation_id))
        completed, failures = [], []
        for attempt in record["attempts"]:
            payload = _terminal_payload(attempt)
            status = str(attempt.get("status") or payload.get("status") or "")
            attempt_id = str(attempt.get("attempt_id") or payload.get("attempt_id") or "")
            dedupe = (record["study_id"], attempt_id, status)
            if dedupe in seen_attempts:
                continue
            seen_attempts.add(dedupe)
            if status == "completed":
                has_metrics = isinstance(payload.get("metrics"), dict) and not attempt.get(
                    "_result_error"
                )
                metrics = payload["metrics"] if has_metrics else {}
                citation_id = f"citation-{len(citations) + 1}"
                citations.append(
                    _citation_for_attempt(
                        record,
                        attempt,
                        citation_id,
                        "/metrics" if has_metrics else "$status",
                        metrics if has_metrics else status,
                        "completed metrics"
                        if has_metrics
                        else "completed status; metrics unavailable",
                    )
                )
                attempt_dir = attempt.get("_attempt_dir") or _find_attempt_dir(
                    studies_root, attempt_id, study_paths
                )
                completed.append(
                    {
                        "attempt_id": attempt_id,
                        "candidate": payload.get("candidate"),
                        "metrics": metrics,
                        "result_error": attempt.get("_result_error"),
                        "artifact_validation": validate_artifacts(
                            payload.get("artifacts"), attempt_dir
                        ),
                        "citation_ids": [citation_id],
                    }
                )
            elif status in {"failed", "interrupted", "skipped"}:
                error = payload.get("error") or payload.get("reason") or ""
                citation_id = f"citation-{len(citations) + 1}"
                if "error" in payload:
                    pointer, cited_value = "/error", payload["error"]
                elif "reason" in payload:
                    pointer, cited_value = "/reason", payload["reason"]
                else:
                    pointer, cited_value = "$status", status
                citations.append(
                    _citation_for_attempt(
                        record, attempt, citation_id, pointer, cited_value, "terminal failure"
                    )
                )
                failures.append(
                    {
                        "attempt_id": attempt_id,
                        "status": status,
                        "error_type": payload.get("error_type"),
                        "error": error,
                        "candidate": payload.get("candidate"),
                        "citation_ids": [citation_id],
                    }
                )
        matches.append(
            {
                "study_id": record["study_id"],
                "hypothesis": record["hypothesis"],
                "lexical_similarity": score,
                "study_citation_ids": [study_citation_id],
                "completed_results": completed,
                "failures": failures,
            }
        )
    verification = verify_citations(citations)
    missing_data = []
    if database is None:
        missing_data.append("SQLite registry was not provided")
    if studies_root is None and not study_paths:
        missing_data.append("study artifact root was not provided")
    if not matches:
        missing_data.append("no lexically related study was found")
    if matches and not any(match["completed_results"] for match in matches):
        missing_data.append("no completed result was found for the matched studies")
    if any(
        result["artifact_validation"]["status"] != "verified"
        for match in matches
        for result in match["completed_results"]
    ):
        missing_data.append("one or more completed results lack fully verified artifacts")
    if matches and not any(match["failures"] for match in matches):
        missing_data.append("no terminal failure evidence was found for the matched studies")
    comparison = minimum_comparison(recipe, matches)
    if comparison is not None and comparison["control"] is None:
        missing_data.append("no completed result with verified artifacts is available as a control")
    return {
        "schema_version": "quant.research-history/v1",
        "query": query,
        "matches": matches,
        "citations": citations,
        "citation_verification": verification,
        "minimum_comparison": comparison,
        "missing_data": missing_data,
        "boundaries": {
            "read_only": True,
            "recipe_fields_modified": [],
            "automatic_run": False,
            "online_model_called": False,
        },
    }


def research_advice(
    query: str,
    study_paths: Sequence[str | Path] | str | Path = (),
    *,
    database: str | Path | None = None,
    studies_root: str | Path | None = None,
    recipe: dict | None = None,
    limit: int = 5,
) -> dict:
    """UI-oriented alias accepting explicit study files or directory roots."""

    return research_history(
        query,
        database=database,
        studies_root=studies_root,
        study_paths=study_paths,
        recipe=recipe,
        limit=limit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--studies-root", type=Path)
    parser.add_argument("--study-path", type=Path, action="append", default=[])
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    recipe = None
    if args.recipe:
        import yaml

        recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    result = research_history(
        args.query,
        database=args.database,
        studies_root=args.studies_root,
        study_paths=args.study_path,
        recipe=recipe,
        limit=args.limit,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
