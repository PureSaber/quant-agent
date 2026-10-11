"""Unified validation-only family report using existing Lab/Factors statistics."""

from __future__ import annotations

import hashlib
import io
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from quant_factors.validation import benjamini_hochberg
from quant_lab.family_evidence import evaluate_family
from quant_lab.selection import audit_family, matrix

from .contracts import SAMPLE_KINDS, digest, read_json, relative_file, timestamp


def performance(values, periods_per_year):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all() or (x < -1).any():
        raise ValueError("finite simple returns >= -1 required")
    wealth = np.r_[1.0, np.cumprod(1 + x)]
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {
        "observations": len(x),
        "mean": float(x.mean()),
        "compound_return": float(wealth[-1] - 1),
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "sharpe_annualized": float(x.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else None,
    }


def robustness_report(returns, benchmark, plan):
    if plan.get("schema") != "quant.robustness-plan/v1" or plan.get("partition") != "validation":
        raise ValueError("a validation-only robustness plan is required")
    basis = plan["basis"]
    if (
        type(plan["cscv_blocks"]) is not int
        or plan["cscv_blocks"] < 2
        or plan["cscv_blocks"] % 2
        or not plan["block_lengths"]
        or any(type(n) is not int or n < 1 for n in plan["block_lengths"])
        or len(set(plan["block_lengths"])) != len(plan["block_lengths"])
        or type(plan["repetitions"]) is not int
        or plan["repetitions"] < 100
        or type(plan["seed"]) is not int
        or plan["seed"] < 0
        or not 0 < plan["alpha"] < 1
    ):
        raise ValueError("invalid frozen statistical settings")
    if basis["sample_kind"] not in SAMPLE_KINDS:
        raise ValueError("unknown sample kind")
    if set(plan["parameters"]) != set(plan["planned"]):
        raise ValueError("every attempted candidate needs its frozen parameters")
    if any(not isinstance(v, dict) for v in plan["parameters"].values()):
        raise ValueError("parameters must be objects")
    holdout = timestamp(plan["holdout_start"])
    if (
        not isinstance(returns.index, pd.DatetimeIndex)
        or returns.index.tz is None
        or returns.index.hasnans
    ):
        raise ValueError("timezone-aware observation dates required")
    if len(returns) and returns.index.max() >= holdout:
        raise ValueError("holdout rows cannot enter family robustness or parameter sensitivity")
    audit = audit_family(
        returns,
        planned=plan["planned"],
        statuses=plan["statuses"],
        family_id=plan["family_id"],
        basis=basis,
    )
    if (returns.astype(float) < -1).any().any():
        raise ValueError("simple returns below -1")
    if (
        not isinstance(benchmark, pd.Series)
        or not benchmark.index.equals(returns.index)
        or not np.isfinite(benchmark).all()
        or (benchmark < -1).any()
    ):
        raise ValueError(
            "explicit finite matched benchmark required; no date intersection or zero assumption"
        )
    intervals, names, previous = [], set(), None
    for interval in plan["intervals"]:
        start, end = timestamp(interval["start"]), timestamp(interval["end"])
        name = interval["id"]
        if (
            name in names
            or not start < end <= holdout
            or (previous is not None and start < previous)
        ):
            raise ValueError("distinct disjoint ordered pre-holdout intervals required")
        names.add(name)
        previous = end
        intervals.append((name, (returns.index >= start) & (returns.index < end)))
    if not intervals or not np.all(np.sum([mask for _, mask in intervals], axis=0) == 1):
        raise ValueError(
            "preregistered intervals must partition every validation observation exactly once"
        )
    blocks, lengths = plan["cscv_blocks"], plan["block_lengths"]
    result = {
        "schema": "quant.robustness/v1",
        "plan_sha256": digest(plan),
        "plan": plan,
        "audit": audit,
        "intervals": [],
        "parameter_sensitivity": [],
        "execution_authorized": False,
        "selection": "none; descriptive sensitivity cannot retune the consumed holdout",
        "registration_assurance": "frozen input plan, external preregistration chronology must be independently verified",
    }

    def evaluate(data, subset_audit, bm):
        report = evaluate_family(
            data,
            subset_audit,
            benchmark=bm,
            blocks=blocks,
            block_lengths=lengths,
            repetitions=plan["repetitions"],
            seed=plan["seed"],
        )
        for method in ("dsr", "cscv", "bootstrap", "spa_mcs"):
            report["methods"].setdefault(
                method, {"available": False, "reason": "incomplete family"}
            )
        return report

    result["full_validation"] = evaluate(returns, audit, benchmark)
    for name, mask in intervals:
        data = returns.loc[mask]
        interval_audit = audit_family(
            data,
            planned=plan["planned"],
            statuses=plan["statuses"],
            family_id=plan["family_id"],
            basis=basis,
        )
        result["intervals"].append(
            {
                "id": name,
                "evaluation": evaluate(data, interval_audit, benchmark.loc[mask]),
                "performance": {
                    c: performance(data[c], basis["periods_per_year"])
                    for c in data
                    if interval_audit["available"]
                },
            }
        )
    if audit["available"]:
        data = matrix(returns)
        for left, right in combinations(sorted(plan["planned"]), 2):
            a, b = plan["parameters"][left], plan["parameters"][right]
            changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            if len(changed) == 1:
                delta = data[right] - data[left]
                result["parameter_sensitivity"].append(
                    {
                        "left": left,
                        "right": right,
                        "changed_parameter": changed[0],
                        "left_value": a.get(changed[0]),
                        "right_value": b.get(changed[0]),
                        "paired_mean_delta": float(delta.mean()),
                        "paired_observations": len(delta),
                        "positive_delta_fraction": float((delta > 0).mean()),
                    }
                )
    # Correct across candidates, intervals AND block choices, never pick the lowest p.
    p_rows, incomplete = [], []
    all_reports = [("full_validation", result["full_validation"])] + [
        (item["id"], item["evaluation"]) for item in result["intervals"]
    ]
    for scope, report in all_reports:
        method = report["methods"]["bootstrap"]
        if not method["available"]:
            incomplete.append(scope)
            continue
        for sensitivity in method["evidence"]["sensitivity"]:
            for candidate, evidence in sensitivity["candidates"].items():
                p_rows.append(
                    {
                        "scope": scope,
                        "block_length": sensitivity["block_length"],
                        "candidate": candidate,
                        "p_value": evidence["centered_two_sided_p"],
                    }
                )
    if incomplete or not p_rows:
        result["multiple_testing"] = {
            "available": False,
            "reason": "complete family of bootstrap tests unavailable",
            "missing_scopes": incomplete,
        }
    else:
        adjusted = benjamini_hochberg([p["p_value"] for p in p_rows], alpha=plan["alpha"])
        for row, correction in zip(p_rows, adjusted.to_dict("records")):
            row["bh_adjusted_p"] = correction["adjusted_p_value"]
            row["bonferroni_p"] = min(1.0, row["p_value"] * len(p_rows))
        result["multiple_testing"] = {
            "available": True,
            "family_size": len(p_rows),
            "tests": p_rows,
            "alpha": plan["alpha"],
            "bh_assumption": "independence or positive dependence; not guaranteed for these overlapping tests",
            "bonferroni_scope": "FWER bound under arbitrary dependence, conditional on valid marginal bootstrap p-values",
        }
    return result


def load_robustness(bundle_path):
    bundle_path = Path(bundle_path)
    bundle = read_json(bundle_path)
    plan = bundle["plan"]
    path = relative_file(bundle_path.parent, bundle["returns_file"])
    data_bytes = path.read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != bundle["returns_sha256"]:
        raise ValueError("return artifact digest mismatch")
    frame = pd.read_csv(io.BytesIO(data_bytes))
    frame.index = pd.DatetimeIndex([timestamp(value) for value in frame.pop("date")])
    benchmark = frame.pop("benchmark")
    return robustness_report(frame, benchmark, plan)


def registered_robustness(database, family_id, plan, benchmark):
    """Require the exact existing registry family, including every failed/retried trial."""
    from .lab_adapter import registered_family

    values, audit = registered_family(database, family_id)
    if (
        plan["family_id"] != family_id
        or plan["basis"] != audit["basis"]
        or set(plan["planned"]) != set(audit["planned"])
        or plan["statuses"] != audit["statuses"]
    ):
        raise ValueError("robustness plan must match the complete read-only Lab inventory")
    for study in audit["inventory"]["studies"]:
        for event in study.get("events", []):
            if (
                event["status"] == "running"
                and plan["parameters"].get(event["attempt_id"]) != event["payload"]["parameters"]
            ):
                raise ValueError("parameter sensitivity must use registered attempt parameters")
    # No data can mean every attempt failed; still deliver the full registry audit.
    if not audit["available"]:
        return {
            "schema": "quant.robustness/v1",
            "plan_sha256": digest(plan),
            "audit": audit,
            "full_validation": {
                "methods": {
                    name: {"available": False, "reason": "incomplete registered family"}
                    for name in ("dsr", "cscv", "bootstrap", "spa_mcs")
                }
            },
            "multiple_testing": {"available": False, "reason": "incomplete registered family"},
            "execution_authorized": False,
        }
    result = robustness_report(values, benchmark, plan)
    result["registry_audit"] = audit
    return result
