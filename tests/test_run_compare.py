from pathlib import Path

import pandas as pd

from quant_agent.rules.run_compare import compare_with_previous


def test_compare_ic_delta(tmp_path: Path) -> None:
    prev = tmp_path / "20250101_120000"
    curr = tmp_path / "latest"
    prev.mkdir()
    curr.mkdir()

    pd.DataFrame([{"factor": "pe_ratio", "mean_ic": 0.05}]).to_csv(
        prev / "ic_summary.csv", index=False
    )
    pd.DataFrame([{"factor": "pe_ratio", "mean_ic": 0.01}]).to_csv(
        curr / "ic_summary.csv", index=False
    )
    pd.DataFrame([{"portfolio": "long_short", "sharpe": 1.0}]).to_csv(
        prev / "backtest_stats.csv", index=False
    )
    pd.DataFrame([{"portfolio": "long_short", "sharpe": 0.2}]).to_csv(
        curr / "backtest_stats.csv", index=False
    )

    findings = compare_with_previous(curr, {"ic_delta_warn": 0.02, "sharpe_delta_warn": 0.3})
    codes = {f["code"] for f in findings}
    assert "ic_delta" in codes
    assert "sharpe_delta" in codes
