from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def write_multifactor_fixture(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ic = pd.DataFrame(
        [
            {"factor": "pe_ratio", "mean_ic": 0.04, "ic_ir": 0.5, "ic_positive_ratio": 0.55},
            {"factor": "forecast_score", "mean_ic": float("nan"), "ic_ir": float("nan"), "ic_positive_ratio": float("nan")},
            {"factor": "northbound_chg_5d", "mean_ic": 0.01, "ic_ir": 0.1, "ic_positive_ratio": 0.51},
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
