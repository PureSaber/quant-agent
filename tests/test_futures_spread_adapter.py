from pathlib import Path

from quant_agent.adapters.futures_spread import FuturesSpreadAdapter


def test_futures_spread_adapter_detect_and_load(tmp_path: Path) -> None:
    perf = tmp_path / "performance"
    perf.mkdir()
    (perf / "summary.csv").write_text("total_return,calmar\n0.12,1.1\n", encoding="utf-8")

    adapter = FuturesSpreadAdapter()
    assert adapter.detect(tmp_path)
    ctx = adapter.load(tmp_path)
    assert ctx.project == "quant-futures-spread"
    assert ctx.backtest_stats
