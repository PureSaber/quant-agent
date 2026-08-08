from pathlib import Path

from quant_agent.adapters.base import RunContext
from quant_agent.adapters.futures_spread import FuturesSpreadAdapter
from quant_agent.rules.engine import run_all_rules


def test_spread_skips_ic_missing_error(tmp_path: Path) -> None:
    ctx = RunContext(
        project="quant-futures-spread",
        run_dir=tmp_path,
        ic_summary=[],
        backtest_stats=[{"metric": "total_return", "value": 0.1}, {"metric": "calmar", "value": 1.2}],
    )
    findings = run_all_rules(ctx, {"thresholds": {}, "rules": {}})
    codes = {f["code"] for f in findings}
    assert "ic_missing" not in codes


def test_spread_flags_missing_performance(tmp_path: Path) -> None:
    ctx = RunContext(project="quant-futures-spread", run_dir=tmp_path, ic_summary=[], backtest_stats=[])
    findings = run_all_rules(ctx, {"thresholds": {}, "rules": {}})
    assert any(f["code"] == "spread_perf_missing" for f in findings)


def test_spread_review_passes_without_ic_summary(tmp_path: Path) -> None:
    perf = tmp_path / "performance"
    perf.mkdir()
    (perf / "summary.csv").write_text(
        "total_return,sharpe,max_drawdown,calmar\n0.12,1.1,-0.15,1.1\n",
        encoding="utf-8",
    )

    adapter = FuturesSpreadAdapter()
    ctx = adapter.load(tmp_path)
    findings = run_all_rules(ctx, {"thresholds": {}, "rules": {}})
    severities = {f["severity"] for f in findings}
    assert "error" not in severities
    assert "ic_missing" not in {f["code"] for f in findings}


def test_spread_review_flags_high_drawdown(tmp_path: Path) -> None:
    perf = tmp_path / "performance"
    perf.mkdir()
    (perf / "summary.csv").write_text(
        "sharpe,max_drawdown\n0.5,-0.5\n",
        encoding="utf-8",
    )

    adapter = FuturesSpreadAdapter()
    ctx = adapter.load(tmp_path)
    findings = run_all_rules(
        ctx,
        {"thresholds": {"max_drawdown_limit": -0.35}, "rules": {}},
    )
    codes = {f["code"] for f in findings}
    assert "drawdown_high" in codes
