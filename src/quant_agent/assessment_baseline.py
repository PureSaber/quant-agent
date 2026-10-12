"""Small label-blind baseline for the supported evidence schemas.

This is a deterministic research assistant, not a simulated LLM. It computes
from evidence values and never reads evaluator files or dispatches on case IDs.
"""

from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from quant_agent.assistant_assessment import identity, public_packet


def respond(bundle):
    bundle = public_packet(bundle)
    responses = []
    for case in bundle["cases"]:
        sources = {source["type"]: source for source in case["evidence"]}
        claims, failures = [], []
        decision, caveats = "report_only", ["synthetic_fixture"]

        def claim(key, value, refs, claims=claims):
            claims.append(
                {
                    "key": key,
                    "value": str(value) if isinstance(value, Decimal) else value,
                    "evidence": refs,
                }
            )

        def ref(kind, pointer="", sources=sources):
            return sources[kind]["source_id"] + "#" + pointer

        if "ledger" in sources:
            ledger = sources["ledger"]["content"]
            net = Decimal(ledger["gross_pnl"]) - Decimal(ledger["fees"])
            value = net / Decimal(ledger["starting_equity"])
            if (
                "summary" in sources
                and Decimal(sources["summary"]["content"]["net_return"]) != value
            ):
                decision = "abstain"
                caveats.append("conflict_unresolved")
                claim("evidence_conflict", True, [ref("ledger"), ref("summary", "/net_return")])
            else:
                claim("net_pnl", net, [ref("ledger", "/gross_pnl"), ref("ledger", "/fees")])
                claim(
                    "net_return",
                    value,
                    [
                        ref("ledger", "/starting_equity"),
                        ref("ledger", "/gross_pnl"),
                        ref("ledger", "/fees"),
                    ],
                )
        elif "cashflow" in sources:
            flow = sources["cashflow"]["content"]
            start, before, amount, end = (
                Decimal(flow[k]) for k in ("start", "before_flow", "inflow", "end")
            )
            twr = (before / start) * (end / (before + amount)) - 1
            claim("time_weighted_return", twr, [ref("cashflow")])
            caveats.append("external_flow_excluded")
        elif "comparison" in sources:
            runs = sources["comparison"]["content"]
            required = {"currency", "period", "cost_basis", "return_unit", "sample_kind"}
            complete = all(
                set(run["basis"]) == required and all(run["basis"].values()) for run in runs
            )
            comparable = complete and all(run["basis"] == runs[0]["basis"] for run in runs)
            claim(
                "comparable",
                bool(comparable),
                [ref("comparison", f"/{i}/basis") for i in range(len(runs))],
            )
            if not comparable:
                decision = "needs_replay"
                caveats.append("incomparable_basis" if complete else "unknown_basis")
        elif "history" in sources:
            attempts = sources["history"]["content"]
            completed = [a for a in attempts if a["status"] == "completed"]
            failures = [
                a["attempt_id"]
                for a in attempts
                if a["status"] in {"failed", "skipped", "interrupted"}
            ]
            claim("completed_count", len(completed), [ref("history")])
            claim("unsuccessful_count", len(failures), [ref("history")])
            if failures:
                caveats.append("failed_attempts_present")
            if not completed:
                decision = "abstain"
                caveats.append("no_completed_result")
        elif "inference" in sources:
            evidence = sources["inference"]["content"]
            sufficient = evidence["sample_size"] >= evidence["prespecified_minimum"]
            claim(
                "evidence_sufficient",
                sufficient,
                [ref("inference", "/sample_size"), ref("inference", "/prespecified_minimum")],
            )
            if not sufficient:
                decision = "abstain"
                caveats.append("short_sample")
        elif "timing" in sources:
            evidence = sources["timing"]["content"]
            available, decision_time = (
                datetime.fromisoformat(evidence[k].replace("Z", "+00:00"))
                for k in ("available_at", "decision_at")
            )
            valid = available <= decision_time
            claim(
                "timing_valid",
                valid,
                [ref("timing", "/available_at"), ref("timing", "/decision_at")],
            )
            if not valid:
                decision = "abstain"
                caveats.append("lookahead")
        elif "multiplicity" in sources:
            evidence = sources["multiplicity"]["content"]
            threshold = Decimal(evidence["family_alpha"]) / Decimal(evidence["tests"])
            claim(
                "adjusted_threshold",
                threshold,
                [ref("multiplicity", "/family_alpha"), ref("multiplicity", "/tests")],
            )
            claim("reject_null", Decimal(evidence["raw_p"]) <= threshold, [ref("multiplicity")])
            caveats.append("multiple_testing")
        elif "validation_scope" in sources:
            evidence = sources["validation_scope"]["content"]
            live = evidence["evidence_kind"] == "live_external_audit"
            claim("live_validated", live, [ref("validation_scope", "/evidence_kind")])
            if not live:
                decision = "abstain"
                caveats.append("insufficient_external_validation")
        else:
            decision = "abstain"
            caveats.append("insufficient_evidence")
            claim(
                "evidence_sufficient",
                False,
                [source["source_id"] + "#" for source in case["evidence"]],
            )
        responses.append(
            {
                "case_id": case["case_id"],
                "decision": decision,
                "claims": claims,
                "acknowledged_failures": failures,
                "caveats": caveats,
            }
        )
    return {
        "schema": "quant.assistant-submission/v1",
        "public_sha256": identity(bundle),
        "candidate": {
            "id": "deterministic-evidence-reasoner/v1",
            "kind": "offline_deterministic",
            "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "responses": responses,
    }
