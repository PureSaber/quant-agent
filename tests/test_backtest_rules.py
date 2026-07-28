from pathlib import Path

from quant_agent.adapters.base import RunContext
from quant_agent.rules.engine import check_backtest_stats


def test_backtest_stats_low_sharpe() -> None:
    ctx = RunContext(
        project="a-share-multifactor",
        run_dir=Path("."),
        backtest_stats=[
            {"portfolio": "long_short", "sharpe": -0.5, "max_drawdown": -0.2},
        ],
    )
    findings = check_backtest_stats(ctx, {"min_sharpe": 0.0, "max_drawdown_limit": -0.35})
    codes = {f["code"] for f in findings}
    assert "sharpe_low" in codes


def test_backtest_stats_high_drawdown() -> None:
    ctx = RunContext(
        project="a-share-multifactor",
        run_dir=Path("."),
        backtest_stats=[
            {"portfolio": "long_short", "sharpe": 1.0, "max_drawdown": -0.5},
        ],
    )
    findings = check_backtest_stats(ctx, {"min_sharpe": 0.0, "max_drawdown_limit": -0.35})
    codes = {f["code"] for f in findings}
    assert "drawdown_high" in codes
