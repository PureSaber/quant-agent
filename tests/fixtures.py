from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from quant_lab.contracts_v2 import ARTIFACT_SCHEMAS_V2, RESEARCH_PROFILE, write_standard_run_v2


def write_multifactor_fixture(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ic = pd.DataFrame(
        [
            {"factor": "pe_ratio", "mean_ic": 0.04, "ic_ir": 0.5, "ic_positive_ratio": 0.55},
            {
                "factor": "forecast_score",
                "mean_ic": float("nan"),
                "ic_ir": float("nan"),
                "ic_positive_ratio": float("nan"),
            },
            {
                "factor": "northbound_chg_5d",
                "mean_ic": 0.01,
                "ic_ir": 0.1,
                "ic_positive_ratio": 0.51,
            },
        ]
    )
    ic.to_csv(run_dir / "ic_summary.csv", index=False)
    stats = pd.DataFrame(
        [{"portfolio": "long_short", "sharpe": 0.8, "max_drawdown": -0.15, "ann_return": 0.12}]
    )
    stats.to_csv(run_dir / "backtest_stats.csv", index=False)
    decay = pd.DataFrame(
        [
            {"factor": "pe_ratio", "horizon_days": 1, "mean_ic": 0.05, "ir": 0.4},
            {"factor": "pe_ratio", "horizon_days": 20, "mean_ic": 0.03, "ir": 0.3},
        ]
    )
    decay.to_csv(run_dir / "ic_decay.csv", index=False)

    config = {
        "factors": ["pe_ratio", "forecast_score", "northbound_chg_5d"],
        "filters": {"pit_fundamentals": True, "use_historical_universe": True},
        "data": {},
    }
    (run_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _v2_sample_value(column: str):
    if column == "event_time":
        return pd.Timestamp("2025-01-02T03:04:05Z")
    if column.endswith("_units"):
        return 10_000
    if column.endswith("_scale"):
        return 2
    if column in {"gross_return", "net_return", "value"}:
        return 1.0
    if column in {"currency", "base_currency"}:
        return "CNY"
    return f"sample-{column}"


def write_v2_fixture(
    run_dir: Path,
    *,
    project: str = "a-share-multifactor",
    metrics: dict | None = None,
    config: dict | None = None,
    strategy_ids: tuple[str, ...] = ("Q5",),
) -> None:
    artifact_names = ("returns", "positions", "portfolio_snapshots", "exposures")
    frames = {
        name: pd.DataFrame(
            [{column: _v2_sample_value(column) for column in ARTIFACT_SCHEMAS_V2[name]}],
            columns=ARTIFACT_SCHEMAS_V2[name],
        )
        for name in artifact_names
    }
    lineage = {name: ["dataset:prices"] for name in ("metrics", *artifact_names)}
    lineage["config"] = []
    write_standard_run_v2(
        run_dir,
        project=project,
        run_id="r2",
        strategy_ids=strategy_ids,
        profile=RESEARCH_PROFILE,
        frames=frames,
        metrics=metrics or {"total_return": 0.1},
        config=config or {"base_currency": "CNY"},
        code_version="a" * 40,
        internal_dependencies={"quant-lab": "v0.3.1"},
        random_seed=7,
        dataset_snapshots={"prices": "sha256-demo-v2"},
        instrument_master_version="instruments-v1",
        execution_model_version="not-applicable",
        base_currency="CNY",
        lineage=lineage,
        created_at="2025-01-02T00:00:00+00:00",
    )
