from pathlib import Path

from quant_agent.adapters.futures_spread import FuturesSpreadAdapter
from quant_agent.rules.engine import run_all_rules


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
