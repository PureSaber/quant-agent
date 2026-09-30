"""Read-only review of registered investment goals; no goal rewriting or promotion."""

import argparse
import json
import sqlite3
from pathlib import Path

from quant_lab.objectives import evaluate_objective
from quant_lab.research import digest, verify_result


def review_objective(registry: Path, study_id: str):
    with sqlite3.connect(registry.resolve().as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT definition,sha256 FROM studies WHERE study_id=?", (study_id,)
        ).fetchone()
        events = db.execute(
            "SELECT attempt_id,status,payload FROM trial_events WHERE study_id=? ORDER BY sequence",
            (study_id,),
        ).fetchall()
    if row is None or digest(json.loads(row[0])) != row[1]:
        raise ValueError("missing or corrupted study definition")
    spec = json.loads(row[0])
    if "objective" not in spec:
        return {
            "study_id": study_id,
            "status": "not_preregistered",
            "policy": "do not retrofit an objective after seeing returns",
        }
    starts, latest = {}, {}
    for attempt, status, payload in events:
        payload = json.loads(payload)
        if status == "running":
            starts[attempt] = payload
        latest[attempt] = (status, payload)
    reviews = []
    for attempt, (status, payload) in latest.items():
        item = {
            "attempt_id": attempt,
            "attempt_status": status,
            "citation": {
                "registry": str(registry.resolve()),
                "study_id": study_id,
                "definition_sha256": row[1],
                "attempt_id": attempt,
            },
        }
        if status == "completed":
            root = Path(starts[attempt]["context"]["artifact_root"]).resolve()
            path = (root / payload["result"]).resolve()
            if root not in path.parents:
                raise ValueError("result escapes registered artifact root")
            result = verify_result(path, payload["sha256"])
            if result.get("attempt_id") != attempt or result.get("study_id") != study_id:
                raise ValueError("result identity differs from registry")
            evidence = result.get("investment_evidence")
            item["evaluation"] = (
                evaluate_objective(spec["objective"], evidence)
                if evidence is not None
                else {"status": "insufficient_evidence"}
            )
            item["citation"]["result_sha256"] = payload["sha256"]
        else:
            item["evaluation"] = {
                "status": "insufficient_evidence",
                "reason": f"attempt {status}; retained, not discarded",
            }
        reviews.append(item)
    return {
        "study_id": study_id,
        "objective": spec["objective"],
        "reviews": reviews,
        "policy": "read-only research acceptance; no objective changes, risk relaxation or trading authorization",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    args = parser.parse_args()
    print(json.dumps(review_objective(args.registry, args.study_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
