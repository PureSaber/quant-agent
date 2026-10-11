import copy
import math
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from quant_agent.research_workflow.contracts import file_hash, write_json
from quant_agent.research_workflow.robustness import load_robustness, performance, robustness_report


@pytest.fixture
def family():
    dates = pd.date_range("2020-01-01", periods=64, tz="UTC")
    rng = np.random.default_rng(18)
    data = pd.DataFrame(rng.normal(0.001, 0.01, (64, 3)), index=dates, columns=["a", "b", "c"])
    benchmark = pd.Series(0.0001, index=dates)
    plan = {
        "schema": "quant.robustness-plan/v1",
        "family_id": "fixed-family",
        "partition": "validation",
        "holdout_start": "2020-04-01T00:00:00Z",
        "basis": {
            "currency": "USD",
            "benchmark_id": "explicit-fixture",
            "cost_policy": "net synthetic returns with no modeled cost",
            "periods_per_year": 252,
            "sample_kind": "synthetic",
        },
        "planned": ["a", "b", "c"],
        "statuses": {k: "completed" for k in data},
        "parameters": {"a": {"alpha": 1}, "b": {"alpha": 10}, "c": {"alpha": 100}},
        "intervals": [
            {"id": "first", "start": dates[0].isoformat(), "end": dates[32].isoformat()},
            {"id": "second", "start": dates[32].isoformat(), "end": "2020-04-01T00:00:00Z"},
        ],
        "cscv_blocks": 4,
        "block_lengths": [2, 4],
        "repetitions": 100,
        "seed": 42,
        "alpha": 0.05,
    }
    return data, benchmark, plan


def test_independent_financial_formula_comparison(family):
    data, benchmark, plan = family
    result = robustness_report(data, benchmark, plan)
    methods = result["full_validation"]["methods"]
    dsr = methods["dsr"]["evidence"]
    x = (data.a - benchmark).to_numpy()
    mean = sum(x) / len(x)
    sample_sd = math.sqrt(sum((v - mean) ** 2 for v in x) / (len(x) - 1))
    sr = mean / sample_sd
    population_sd = math.sqrt(sum((v - mean) ** 2 for v in x) / len(x))
    skew = sum(((v - mean) / population_sd) ** 3 for v in x) / len(x)
    kurt = sum(((v - mean) / population_sd) ** 4 for v in x) / len(x)
    psr = NormalDist().cdf(
        sr * math.sqrt(len(x) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) * sr**2 / 4)
    )
    actual = dsr["candidates"][0]
    assert actual["sharpe_period"] == pytest.approx(sr)
    assert actual["psr_zero"] == pytest.approx(psr)
    trial_srs = []
    for name in data:
        series = (data[name] - benchmark).tolist()
        avg = sum(series) / len(series)
        trial_srs.append(avg / math.sqrt(sum((v - avg) ** 2 for v in series) / (len(series) - 1)))
    avg_sr = sum(trial_srs) / 3
    dispersion = math.sqrt(sum((s - avg_sr) ** 2 for s in trial_srs) / 2)
    gamma = 0.5772156649015329
    hurdle = dispersion * (
        (1 - gamma) * NormalDist().inv_cdf(1 - 1 / 3)
        + gamma * NormalDist().inv_cdf(1 - 1 / (3 * math.e))
    )
    expected_dsr = NormalDist().cdf(
        (sr - hurdle) * math.sqrt(len(x) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) * sr**2 / 4)
    )
    dsr_three = next(row for row in actual["sensitivity"] if row["independent_trials"] == 3)
    assert dsr_three["hurdle_period"] == pytest.approx(hurdle)
    assert dsr_three["dsr"] == pytest.approx(expected_dsr)
    # Independent CSCV enumeration, scalar sample moments and direct rank count.
    excess = (data.sub(benchmark, axis=0)).to_numpy()
    blocks = [list(range(i, i + 16)) for i in range(0, 64, 16)]
    weights = []

    def scores(indices):
        values = excess[indices]
        return [
            sum(col)
            / len(col)
            / math.sqrt(sum((v - sum(col) / len(col)) ** 2 for v in col) / (len(col) - 1))
            for col in values.T
        ]

    for subset in combinations(range(4), 2):
        train = scores([row for block in subset for row in blocks[block]])
        test = scores([row for block in range(4) if block not in subset for row in blocks[block]])
        winner = train.index(max(train))
        rank = 1 + sum(v < test[winner] for v in test)
        weights.append(1 if rank < 2 else 0.5 if rank == 2 else 0)
    assert methods["cscv"]["evidence"]["pbo"] == sum(weights) / len(weights)
    assert result["parameter_sensitivity"][0]["paired_mean_delta"] == pytest.approx(
        sum(data.b - data.a) / 64
    )
    summary = performance([0.1, -0.2, 0.25], 252)
    assert summary["compound_return"] == pytest.approx(0.1)
    assert summary["max_drawdown"] == pytest.approx(-0.2)
    assert performance([-1, 0.2], 252)["compound_return"] == -1


def test_all_statistics_and_corrections_or_explicit_optional_dependency_gap(family):
    data, benchmark, plan = family
    result = robustness_report(data, benchmark, plan)
    methods = result["full_validation"]["methods"]
    assert set(methods) == {"dsr", "cscv", "bootstrap", "spa_mcs"}
    if not methods["bootstrap"]["available"]:
        assert "arch" in methods["bootstrap"]["reason"]
        assert not result["multiple_testing"]["available"]
        return
    assert methods["spa_mcs"]["available"]
    correction = result["multiple_testing"]
    assert correction["family_size"] == 18
    values = [row["p_value"] for row in correction["tests"]]
    for row in correction["tests"]:
        assert row["bonferroni_p"] == min(1, row["p_value"] * 18)
        eligible = [(p * 18 / (i + 1)) for i, p in enumerate(sorted(values)) if p >= row["p_value"]]
        assert row["bh_adjusted_p"] == pytest.approx(min(1, min(eligible)))
    assert len(result["intervals"]) == 2
    assert result["execution_authorized"] is False


def test_failed_missing_and_unregistered_trials(family):
    data, benchmark, plan = family
    plan["statuses"]["c"] = "failed"
    result = robustness_report(data.drop(columns="c"), benchmark, plan)
    assert not result["audit"]["available"]
    assert result["audit"]["statuses"]["c"] == "failed"
    assert not result["multiple_testing"]["available"]
    assert not result["parameter_sensitivity"]
    with pytest.raises(ValueError, match="unregistered"):
        robustness_report(data.assign(unregistered=data.a), benchmark, plan)


@pytest.mark.parametrize(
    "case",
    ["holdout", "benchmark", "gap", "overlap", "naive", "parameters", "sample_kind", "partition"],
)
def test_invalid_comparison_is_never_silently_repaired(family, case):
    data, benchmark, source = family
    plan = copy.deepcopy(source)
    if case == "holdout":
        plan["holdout_start"] = data.index[-1].isoformat()
    elif case == "benchmark":
        benchmark = benchmark.iloc[1:]
    elif case == "gap":
        plan["intervals"][1]["start"] = data.index[33].isoformat()
    elif case == "overlap":
        plan["intervals"][1]["start"] = data.index[31].isoformat()
    elif case == "naive":
        data.index = data.index.tz_localize(None)
    elif case == "parameters":
        del plan["parameters"]["a"]
    elif case == "sample_kind":
        plan["basis"]["sample_kind"] = "verified-profit"
    else:
        plan["partition"] = "holdout"
    with pytest.raises(ValueError):
        robustness_report(data, benchmark, plan)


def test_bundle_hash_and_missing_dates(family, tmp_path):
    data, benchmark, plan = family
    data.assign(benchmark=benchmark).rename_axis("date").to_csv(tmp_path / "returns.csv")
    write_json(
        tmp_path / "bundle.json",
        {
            "plan": plan,
            "returns_file": "returns.csv",
            "returns_sha256": file_hash(tmp_path / "returns.csv"),
        },
    )
    assert load_robustness(tmp_path / "bundle.json")["audit"]["available"]
    (tmp_path / "returns.csv").write_text("bad")
    with pytest.raises(ValueError, match="digest"):
        load_robustness(tmp_path / "bundle.json")
    data.loc[data.index[0], "b"] = np.nan
    assert not robustness_report(data, benchmark, plan)["audit"]["available"]
