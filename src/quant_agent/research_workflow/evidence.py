"""Explicit hypothesis/evidence/conclusion links on the existing verified search API."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

from quant_agent.research_history import (
    _json_pointer,
    _make_json_citation,
    research_history,
    validate_artifacts,
    verify_citations,
)

from .contracts import SAMPLE_KINDS, digest, identifier, read_json


def cite_json(path, pointer, citation_id, claim):
    path = Path(path)
    value = _json_pointer(read_json(path), pointer)
    return _make_json_citation(identifier(citation_id), path, pointer, value, claim=claim)


def evidence_graph(history, assertions):
    """A negative experiment and a technical execution failure have distinct meanings.

    Assertions are explicit human/reviewer interpretations, never generated from lexical
    similarity. Exact scope equality is required for a logical conflict finding.
    """
    if history.get("schema_version") != "quant.research-history/v1":
        raise ValueError("existing research_history result required")
    citations = {c["citation_id"]: c for c in history["citations"]}
    if len(citations) != len(history["citations"]):
        raise ValueError("duplicate citation IDs")
    nodes, edges, contexts = [], [], {}
    for match in history["matches"]:
        study = match["study_id"]
        hypothesis_id = f"hypothesis:{study}"
        nodes.append(
            {
                "id": hypothesis_id,
                "kind": "hypothesis",
                "text": match["hypothesis"],
                "citations": match["study_citation_ids"],
            }
        )
        for kind, results in (
            ("completed", match["completed_results"]),
            ("technical_failure", match["failures"]),
        ):
            for row in results:
                key = (study, row["attempt_id"])
                node_id = f"evidence:{study}:{row['attempt_id']}:{kind}"
                context = {
                    "kind": kind,
                    "row": row,
                    "node_id": node_id,
                    "hypothesis_id": hypothesis_id,
                }
                contexts.setdefault(key, []).append(context)
                nodes.append(
                    {
                        "id": node_id,
                        "kind": kind,
                        "attempt_id": row["attempt_id"],
                        "citations": row["citation_ids"],
                        "details": row,
                    }
                )
                edges.append(
                    {
                        "from": node_id,
                        "to": hypothesis_id,
                        "relation": "tests" if kind == "completed" else "failed_to_test",
                    }
                )
    conclusions, seen = [], set()
    for assertion in assertions:
        aid = identifier(assertion["id"])
        if aid in seen:
            raise ValueError("duplicate assertion ID")
        seen.add(aid)
        verdict = assertion["verdict"]
        if verdict not in {"supports", "rejects", "inconclusive", "execution_failed"}:
            raise ValueError("unknown conclusion verdict")
        key = (assertion["study_id"], assertion["attempt_id"])
        choices = contexts.get(key, [])
        context = next(
            (
                c
                for c in choices
                if (c["kind"] == "technical_failure") == (verdict == "execution_failed")
            ),
            None,
        )
        if context is None:
            raise ValueError(
                "conclusion has no compatible retrieved evidence; failures cannot reject hypotheses"
            )
        scope = assertion["scope"]
        if (
            set(scope)
            != {"dataset_sha256", "interval", "population", "target", "cost_policy", "sample_kind"}
            or any(v is None or v == "" for v in scope.values())
            or scope["sample_kind"] not in SAMPLE_KINDS
        ):
            raise ValueError("complete explicit comparison scope required")
        if not assertion.get("claim_key") or not assertion.get("reason"):
            raise ValueError("claim_key and conclusion rationale required")
        ids = assertion["citation_ids"]
        if (
            not ids
            or not set(ids) <= set(context["row"]["citation_ids"])
            or any(c not in citations for c in ids)
        ):
            raise ValueError("conclusions must cite their own retrieved attempt")
        check = verify_citations([citations[c] for c in ids])
        artifact = context["row"].get("artifact_validation", {})
        if context["kind"] == "completed":
            # Re-read bytes now: retrieval-time verification is not durable evidence.
            expected = {item["path"]: item["expected_sha256"] for item in artifact.get("items", [])}
            fresh = validate_artifacts(
                expected,
                Path(assertion["artifact_root"]) if assertion.get("artifact_root") else None,
            )
            verified_artifacts = fresh["status"] == "verified" and fresh["checked"] > 0
        else:
            verified_artifacts = True
        valid = not check["invalid"] and verified_artifacts
        node = {
            **assertion,
            "id": f"conclusion:{aid}",
            "kind": "conclusion",
            "scope_sha256": digest(scope),
            "verified": valid,
            "verification": check,
            "artifacts_verified": verified_artifacts,
            "interpretation": "explicit reviewer assertion, citation integrity does not prove the inference",
        }
        conclusions.append(node)
        nodes.append(node)
        edges.extend(
            [
                {"from": context["node_id"], "to": node["id"], "relation": "cited_by"},
                {"from": node["id"], "to": context["hypothesis_id"], "relation": verdict},
            ]
        )
    conflicts, incomparable = [], []
    for a, b in combinations(conclusions, 2):
        if a["claim_key"] != b["claim_key"] or {a["verdict"], b["verdict"]} != {
            "supports",
            "rejects",
        }:
            continue
        pair = {"left": a["id"], "right": b["id"], "claim_key": a["claim_key"]}
        if a["scope_sha256"] != b["scope_sha256"]:
            incomparable.append(
                {**pair, "reason": "different declared scopes; not a logical contradiction"}
            )
        elif a["verified"] and b["verified"]:
            conflicts.append(
                {
                    **pair,
                    "reason": "opposite verified-source assertions on identical declared scope; human review required",
                }
            )
    return {
        "schema": "quant.research-evidence-graph/v1",
        "nodes": nodes,
        "edges": edges,
        "conflicts": conflicts,
        "incomparable": incomparable,
        "citation_verification": verify_citations(list(citations.values())),
        "missing_data": history["missing_data"],
        "search_scope": "retrieved matches only; not a complete research universe",
        "execution_authorized": False,
    }


def search_evidence_graph(query, assertions, **kwargs):
    return evidence_graph(research_history(query, **kwargs), assertions)
