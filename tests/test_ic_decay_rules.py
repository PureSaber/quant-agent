from pathlib import Path

from quant_agent.adapters.base import RunContext
from quant_agent.rules.engine import check_ic_decay


def test_ic_decay_fast_warn() -> None:
    ctx = RunContext(
        project="a-share-multifactor",
        run_dir=Path("."),
        ic_decay=[
            {"factor": "momentum_20d", "horizon_days": 1, "mean_ic": 0.08},
            {"factor": "momentum_20d", "horizon_days": 20, "mean_ic": 0.01},
        ],
    )
    thresholds = {
        "ic_decay_short_horizon": 1,
        "ic_decay_long_horizon": 20,
        "ic_decay_ratio_warn": 0.5,
        "min_ic_abs": 0.02,
    }
    findings = check_ic_decay(ctx, thresholds)
    assert any(f["code"] == "ic_decay_fast" for f in findings)
