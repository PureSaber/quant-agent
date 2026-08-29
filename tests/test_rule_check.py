from pathlib import Path

from quant_agent.adapters.base import RunContext
from quant_agent.rules.engine import run_all_rules
from tests.fixtures import write_multifactor_fixture


def test_rule_check_flags_nan_alt_factor(tmp_path: Path) -> None:
    write_multifactor_fixture(tmp_path)
    ctx = RunContext(
        project="a-share-multifactor",
        run_dir=tmp_path,
        ic_summary=[
            {"factor": "pe_ratio", "mean_ic": 0.04, "ic_positive_ratio": 0.55},
            {
                "factor": "forecast_score",
                "mean_ic": float("nan"),
                "ic_positive_ratio": float("nan"),
            },
        ],
        factor_list=["pe_ratio", "forecast_score"],
        config_snapshot={
            "filters": {"pit_fundamentals": True, "use_historical_universe": True},
            "data": {},
        },
    )
    findings = run_all_rules(ctx, {"thresholds": {}, "rules": {}})
    codes = {f["code"] for f in findings}
    assert "ic_nan" in codes
    assert "alt_data_nan" in codes
