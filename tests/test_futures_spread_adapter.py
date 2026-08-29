from pathlib import Path

import pytest

from quant_agent.adapters.futures_spread import FuturesSpreadAdapter
from tests.fixtures import write_v2_fixture


def test_futures_spread_adapter_detect_and_load(tmp_path: Path) -> None:
    perf = tmp_path / "performance"
    perf.mkdir()
    (perf / "summary.csv").write_text(
        "total_return,calmar,sharpe,max_drawdown\n0.12,1.1,1.0,-0.1\n",
        encoding="utf-8",
    )

    adapter = FuturesSpreadAdapter()
    assert adapter.detect(tmp_path)
    ctx = adapter.load(tmp_path)
    assert ctx.project == "quant-futures-spread"
    assert ctx.backtest_stats


def test_futures_spread_v2_consumes_only_validated_metrics(tmp_path: Path) -> None:
    private = tmp_path / "performance"
    private.mkdir()
    (private / "summary.csv").write_text(
        "total_return,calmar,sharpe,max_drawdown\n999,999,999,0\n",
        encoding="utf-8",
    )
    write_v2_fixture(
        tmp_path,
        project="quant-futures-spread",
        metrics={"total_return": 0.12, "calmar": 1.1, "sharpe": 1.0, "max_drawdown": -0.1},
        config={"base_currency": "CNY", "strategies": ["spread-v2"]},
        strategy_ids=("spread-v2",),
    )

    adapter = FuturesSpreadAdapter()
    assert adapter.detect(tmp_path) is True
    context = adapter.load(tmp_path)
    assert context.backtest_stats == [
        {"total_return": 0.12, "calmar": 1.1, "max_drawdown": -0.1, "sharpe": 1.0}
    ]
    assert context.factor_list == ["spread-v2"]

    override = tmp_path / "override.yaml"
    override.write_text("strategies: [bypass]", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot override"):
        adapter.load(tmp_path, override)
