from __future__ import annotations

import json

import pytest
from quant_lab.contracts import write_standard_run

from quant_agent.adapters.base import detect_project
from quant_agent.adapters.multifactor import MultifactorAdapter
from quant_agent.adapters.standard import flat_metric_rows, metric_rows, string_list
from tests.fixtures import write_multifactor_fixture, write_v2_fixture


def test_adapter_validates_and_exposes_standard_contract(tmp_path) -> None:
    write_multifactor_fixture(tmp_path)
    write_standard_run(
        tmp_path,
        project="a-share-multifactor",
        run_id="r1",
        strategy="Q5",
        frames={},
        metrics={},
        config={},
        code_version="abc",
        dataset_snapshots={"prices": "sha256-demo"},
    )
    ctx = MultifactorAdapter().load(tmp_path)
    contract = ctx.run_meta["standard_contract"]
    assert contract["validated"] is True
    assert contract["dataset_snapshots"]["prices"] == "sha256-demo"
    assert contract["profile"] == "legacy-v1"


def test_adapter_prefers_and_exposes_valid_standard_v2_contract(tmp_path) -> None:
    write_multifactor_fixture(tmp_path)
    write_v2_fixture(
        tmp_path,
        metrics={
            "ic_summary": [
                {
                    "factor": "validated_factor",
                    "mean_ic": 0.06,
                    "ic_ir": 0.7,
                    "ic_positive_ratio": 0.6,
                }
            ],
            "backtest_stats": [{"portfolio": "long_short", "sharpe": 1.2, "max_drawdown": -0.1}],
            "ic_decay": [{"factor": "validated_factor", "horizon_days": 1, "mean_ic": 0.06}],
        },
        config={"base_currency": "CNY", "factors": ["validated_factor"]},
    )

    context = MultifactorAdapter().load(tmp_path)
    contract = context.run_meta["standard_contract"]

    assert contract == {
        "schema_version": "2.0.0",
        "project": "a-share-multifactor",
        "run_id": "r2",
        "code_version": "a" * 40,
        "dataset_snapshots": {"prices": "sha256-demo-v2"},
        "profile": "research",
        "validated": True,
    }
    assert context.factor_list == ["validated_factor"]
    assert context.ic_summary[0]["mean_ic"] == 0.06
    assert context.config_snapshot == {
        "base_currency": "CNY",
        "factors": ["validated_factor"],
    }


def test_v2_corruption_fails_without_falling_back_to_valid_v1(tmp_path) -> None:
    write_multifactor_fixture(tmp_path)
    write_standard_run(
        tmp_path,
        project="a-share-multifactor",
        run_id="r1",
        strategy="Q5",
        frames={},
        metrics={},
        config={},
        code_version="abc",
    )
    write_v2_fixture(tmp_path)
    metrics_path = tmp_path / "standard" / "v2" / "metrics.json"
    metrics_path.write_text('{"total_return": 999}', encoding="utf-8")

    with pytest.raises(ValueError, match="mutated"):
        MultifactorAdapter().load(tmp_path)
    with pytest.raises(ValueError, match="mutated"):
        detect_project(tmp_path)


def test_v2_rejects_external_config_and_project_mismatch(tmp_path) -> None:
    write_v2_fixture(tmp_path)
    override = tmp_path / "override.yaml"
    override.write_text("factors: [bypass]", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot override"):
        MultifactorAdapter().load(tmp_path, override)

    wrong_project = tmp_path / "wrong-project"
    write_v2_fixture(wrong_project, project="quant-futures-spread")
    with pytest.raises(ValueError, match="does not match"):
        MultifactorAdapter().load(wrong_project)


def test_v2_detects_from_validated_manifest_without_private_files(tmp_path) -> None:
    write_v2_fixture(tmp_path)
    assert detect_project(tmp_path) == "a-share-multifactor"
    assert MultifactorAdapter().detect(tmp_path) is True


def test_v2_metric_projection_validates_shapes() -> None:
    assert metric_rows({}, "ic_summary") == []
    assert metric_rows({"ic_summary": [{"factor": "value"}]}, "ic_summary") == [{"factor": "value"}]
    with pytest.raises(ValueError, match="list of objects"):
        metric_rows({"ic_summary": {}}, "ic_summary")
    with pytest.raises(ValueError, match="list of objects"):
        metric_rows({"ic_summary": ["bad"]}, "ic_summary")

    assert flat_metric_rows({"sharpe": 1.0, "ignored": 2}, ("sharpe",)) == [{"sharpe": 1.0}]
    assert flat_metric_rows({}, ("sharpe",)) == []
    assert string_list(None, "config.factors") == []
    assert string_list(["Q5"], "config.factors") == ["Q5"]
    for malformed in ("Q5", [""], [1]):
        with pytest.raises(ValueError, match="non-empty strings"):
            string_list(malformed, "config.factors")


def test_v2_rejects_non_object_json_even_after_mocked_validation(tmp_path, monkeypatch) -> None:
    write_v2_fixture(tmp_path)
    from quant_lab import load_and_validate_standard_run as real_validator

    from quant_agent.adapters import standard

    manifest = real_validator(tmp_path)
    metrics_path = tmp_path / "standard" / "v2" / "metrics.json"
    metrics_path.write_text(json.dumps(["not-an-object"]), encoding="utf-8")
    monkeypatch.setattr("quant_lab.load_and_validate_standard_run", lambda _path: manifest)
    with pytest.raises(TypeError, match="JSON object"):
        standard.load_standard_artifacts(tmp_path)
