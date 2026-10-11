"""Reproducible local examples. All outputs stay under a new --output directory.

python examples/research_workflow.py --output ../evidence/synthetic
python examples/research_workflow.py --output ../evidence/macro --macrodata-csv PATH
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_agent.research_history import research_history
from quant_agent.research_workflow.contracts import file_hash, verify_archive, write_json
from quant_agent.research_workflow.evidence import cite_json, evidence_graph
from quant_agent.research_workflow.lifecycle import ResearchLifecycle
from quant_agent.research_workflow.ml import run_study
from quant_agent.research_workflow.robustness import robustness_report


def prepare_synthetic(root):
    dates = pd.date_range("2020-01-01", periods=240, tz="UTC")
    rng = np.random.default_rng(20261011)
    x, z, noise = rng.normal(size=(3, 240))
    frame = pd.DataFrame(
        {
            "row_id": [f"synthetic-{i}" for i in range(240)],
            "asset": "fixture",
            "decision_at": dates,
            "target_start": dates,
            "target_end": dates + pd.Timedelta(hours=12),
            "target_known_at": dates + pd.Timedelta(hours=18),
            "target": 0.01 * x + 0.002 * noise,
            "x": x,
            "z": z,
            "x__known_at": dates,
            "z__known_at": dates,
        }
    )
    plan = base_plan(
        "synthetic",
        dates[0],
        [(dates[80], dates[120]), (dates[120], dates[160])],
        (dates[160], dates[-1] + pd.Timedelta(days=1)),
        "2021-01-01T00:00:00Z",
    )
    return (
        frame,
        plan,
        {"source": "numpy fixed seed 20261011; synthetic predictive regression fixture"},
    )


def base_plan(name, start, folds, holdout, as_of):
    return {
        "schema": "quant.ml-study/v1",
        "study_id": name,
        "family_id": f"{name}-family",
        "hypothesis": "lagged observables improve regression over the training mean",
        "selection_metric": "mse",
        "train_start": str(start),
        "folds": [{"start": str(a), "end": str(b)} for a, b in folds],
        "holdout": {"start": str(holdout[0]), "end": str(holdout[1])},
        "evaluation_as_of": as_of,
        "embargo_seconds": 0,
        "models": [
            {"id": "mean", "kind": "mean"},
            {"id": "ridge0", "kind": "ridge", "alpha": 0},
            {"id": "ridge10", "kind": "ridge", "alpha": 10},
            {"id": "knn5", "kind": "knn", "neighbors": 5},
        ],
    }


def prepare_macro(root, source):
    """Historical-vintage macro observations, not historical point-in-time releases.

    The two-quarter feature lag and +100d availability are an explicit replay
    convention. This is never relabeled historical_pit or forward evidence.
    """
    raw = pd.read_csv(source)
    periods = pd.PeriodIndex.from_fields(
        year=raw.year.astype(int), quarter=raw.quarter.astype(int), freq="Q"
    )
    dates = periods.to_timestamp(how="end").normalize().tz_localize("UTC")
    frame = pd.DataFrame(
        {
            "row_id": [f"macro-{i}" for i in range(len(raw))],
            "asset": "US-macro",
            "decision_at": dates,
            "target_start": dates,
            "target_end": pd.Series(dates).shift(-1),
            "target_known_at": pd.Series(dates).shift(-1) + pd.Timedelta(days=100),
            "target": raw.unemp.shift(-1) - raw.unemp,
            "x": raw.unemp.shift(2),
            "z": raw.tbilrate.shift(2),
            "x__known_at": pd.Series(dates).shift(2) + pd.Timedelta(days=100),
            "z__known_at": pd.Series(dates).shift(2) + pd.Timedelta(days=100),
        }
    )
    # Explicit unavoidable construction boundary: two lag rows and last unlabeled row.
    frame = frame.iloc[2:-1].copy()
    plan = base_plan(
        "macro-retrospective",
        "1959-01-01T00:00:00Z",
        [
            ("1994-01-01T00:00:00Z", "2000-01-01T00:00:00Z"),
            ("2000-01-01T00:00:00Z", "2005-01-01T00:00:00Z"),
        ],
        ("2005-01-01T00:00:00Z", "2010-01-01T00:00:00Z"),
        "2011-01-01T00:00:00Z",
    )
    provenance = {
        "source": "statsmodels macrodata, FRED and US Bureau of Labor Statistics, downloaded by original compiler on 2009-12-15",
        "documentation": "https://www.statsmodels.org/stable/datasets/generated/macrodata.html",
        "raw_sha256": file_hash(source),
        "availability_policy": "assumed quarter-end +100d with two-quarter feature lags; not verified original release timestamps",
        "revision_policy": "retrospective historical snapshot; original vintages unavailable",
        "target": "next-quarter unemployment rate change in percentage points",
        "removed_boundary_rows": {"leading_lag": 2, "trailing_unlabeled": 1},
        "not_investable": True,
    }
    return frame, plan, provenance


def synthetic_robustness(root):
    dates = pd.date_range("2020-01-01", periods=128, tz="UTC")
    rng = np.random.default_rng(701)
    values = pd.DataFrame(
        rng.normal(0.0002, 0.009, size=(128, 3)), index=dates, columns=["p1", "p2", "p3"]
    )
    benchmark = pd.Series(0.00002, index=dates)
    plan = {
        "schema": "quant.robustness-plan/v1",
        "family_id": "synthetic-return-family",
        "partition": "validation",
        "holdout_start": "2021-01-01T00:00:00Z",
        "planned": list(values),
        "statuses": {c: "completed" for c in values},
        "parameters": {c: {"lookback": i} for c, i in zip(values, [10, 20, 40])},
        "basis": {
            "currency": "USD",
            "benchmark_id": "synthetic-cash-2e-5",
            "cost_policy": "synthetic net returns; no real cost calibration",
            "periods_per_year": 252,
            "sample_kind": "synthetic",
        },
        "intervals": [
            {"id": "early", "start": dates[0].isoformat(), "end": dates[64].isoformat()},
            {"id": "late", "start": dates[64].isoformat(), "end": "2021-01-01T00:00:00Z"},
        ],
        "cscv_blocks": 4,
        "block_lengths": [2, 4],
        "repetitions": 200,
        "seed": 701,
        "alpha": 0.05,
    }
    write_json(root / "robustness-plan.json", plan)
    values.assign(benchmark=benchmark).rename_axis("date").to_csv(root / "returns.csv")
    write_json(
        root / "robustness-bundle.json",
        {
            "plan": plan,
            "returns_file": "returns.csv",
            "returns_sha256": file_hash(root / "returns.csv"),
        },
    )
    result = robustness_report(values, benchmark, plan)
    write_json(root / "robustness.json", result)
    return result


def demonstrate_governance(root, result, plan):
    # Compatibility export is a study-file view; no Lab DB registration or freeze writes.
    study = root / "studies/demo/study.json"
    artifact_dir = study.parent / "attempts/model"
    artifact_dir.mkdir(parents=True)
    write_json(artifact_dir / "evidence.json", result)
    write_json(
        study,
        {
            "study_id": "ml-demo",
            "recipe": {"hypothesis": plan["hypothesis"]},
            "results": [
                {
                    "attempt_id": "model",
                    "status": "completed",
                    "metrics": result["holdout"]["metrics"],
                    "artifacts": {"evidence.json": file_hash(artifact_dir / "evidence.json")},
                }
            ],
            "attempts": [
                {
                    "attempt_id": "missing-feed",
                    "status": "failed",
                    "payload": {
                        "error": "demonstration of a source-unavailable failure, not a negative investment conclusion"
                    },
                }
            ],
        },
    )
    history = research_history("lagged", study_paths=[study])
    write_json(root / "history.json", history)
    cited = history["matches"][0]["completed_results"][0]["citation_ids"]
    scope = {
        "dataset_sha256": result["dataset_sha256"],
        "interval": plan["holdout"],
        "population": plan["study_id"],
        "target": "regression",
        "cost_policy": "not a trading return",
        "sample_kind": result["sample_kind"],
    }
    assertion = {
        "id": "assessment",
        "study_id": "ml-demo",
        "attempt_id": "model",
        "claim_key": "predictive-improvement",
        "verdict": "inconclusive",
        "reason": "a single consumed holdout does not establish investment profitability",
        "citation_ids": cited,
        "artifact_root": str(artifact_dir.resolve()),
        "scope": scope,
    }
    failure = {
        **assertion,
        "id": "technical-failure",
        "attempt_id": "missing-feed",
        "verdict": "execution_failed",
        "citation_ids": history["matches"][0]["failures"][0]["citation_ids"],
        "reason": "no hypothesis conclusion from missing input",
    }
    write_json(root / "assertions.json", [assertion, failure])
    write_json(root / "graph.json", evidence_graph(history, [assertion, failure]))
    citation = cite_json(
        root / "graph.json", "/nodes", "graph-evidence", "explicit research conclusion graph"
    )
    service = ResearchLifecycle(root / "lifecycle.sqlite", create=True)
    state = service.state("demo")
    for day, target in enumerate(["candidate", "review", "observe", "paused", "retired"], 1):
        state = service.transition(
            "demo",
            target,
            actor="offline-example",
            reason="demonstration of research-only governance; no execution authority",
            evidence=[citation],
            recorded_at=f"2026-01-{day:02d}T00:00:00Z",
            expected_head=state["head_sha256"],
        )
    write_json(root / "lifecycle-state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--macrodata-csv", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    frame, plan, provenance = (
        prepare_macro(root, args.macrodata_csv) if args.macrodata_csv else prepare_synthetic(root)
    )
    frame.to_csv(root / "features.csv", index=False)
    dataset = {
        "schema": "quant.ml-dataset/v1",
        "data_file": "features.csv",
        "data_sha256": file_hash(root / "features.csv"),
        "features": ["x", "z"],
        "target_kind": "regression",
        "sample_kind": "retrospective" if args.macrodata_csv else "synthetic",
        "provenance": provenance,
    }
    write_json(root / "dataset.json", dataset)
    write_json(root / "plan.json", plan)
    result = run_study(root / "plan.json", root / "dataset.json", root / "runs")
    if result["status"] != "completed":
        raise RuntimeError(result)
    verify_archive(root / "runs" / plan["study_id"])
    demonstrate_governance(root, result, plan)
    robust = synthetic_robustness(root)
    write_json(
        root / "summary.json",
        {
            "ml_status": result["status"],
            "sample_kind": result["sample_kind"],
            "input_rows": len(frame),
            "decision_start": str(frame.decision_at.min()),
            "decision_end": str(frame.decision_at.max()),
            "selection": result["selection"],
            "holdout": result["holdout"]["metrics"],
            "robustness_sample_kind": "synthetic",
            "statistics_available": {
                k: v["available"] for k, v in robust["full_validation"]["methods"].items()
            },
            "live_orders": 0,
            "historical_pit_certified": False,
            "forward_validated": False,
        },
    )
    print(root / "summary.json")


if __name__ == "__main__":
    main()
