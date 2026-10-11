import copy

import numpy as np
import pandas as pd
import pytest

from quant_agent.research_workflow.cli import main
from quant_agent.research_workflow.contracts import (
    file_hash,
    load_dataset,
    read_json,
    verify_archive,
    write_json,
)
from quant_agent.research_workflow.ml import run_study, validate_plan
from quant_agent.research_workflow.models import fit_model, predict_model


@pytest.fixture
def inputs(tmp_path):
    dates = pd.date_range("2020-01-01", periods=100, tz="UTC")
    x = np.arange(100, dtype=float)
    frame = pd.DataFrame(
        {
            "row_id": [f"row-{i}" for i in range(100)],
            "asset": "test",
            "decision_at": dates,
            "target_start": dates,
            "target_end": dates + pd.Timedelta(hours=12),
            "target_known_at": dates + pd.Timedelta(days=2),
            "target": 0.3 * x + np.sin(x),
            "x": x,
            "x__known_at": dates,
            "z": np.cos(x),
            "z__known_at": dates,
        }
    )
    frame.loc[2, "x"] = np.nan
    frame.to_csv(tmp_path / "data.csv", index=False)
    data = {
        "schema": "quant.ml-dataset/v1",
        "data_file": "data.csv",
        "data_sha256": file_hash(tmp_path / "data.csv"),
        "features": ["x", "z"],
        "target_kind": "regression",
        "sample_kind": "synthetic",
        "provenance": {"source": "deterministic fixture"},
    }
    plan = {
        "schema": "quant.ml-study/v1",
        "study_id": "example",
        "family_id": "family",
        "hypothesis": "linear baseline predicts target",
        "selection_metric": "mse",
        "train_start": dates[0].isoformat(),
        "folds": [
            {"start": dates[30].isoformat(), "end": dates[50].isoformat()},
            {"start": dates[50].isoformat(), "end": dates[70].isoformat()},
        ],
        "holdout": {
            "start": dates[70].isoformat(),
            "end": (dates[-1] + pd.Timedelta(days=1)).isoformat(),
        },
        "evaluation_as_of": (dates[-1] + pd.Timedelta(days=3)).isoformat(),
        "embargo_seconds": 3600,
        "models": [
            {"id": "mean", "kind": "mean"},
            {"id": "ridge0", "kind": "ridge", "alpha": 0},
            {"id": "ridge10", "kind": "ridge", "alpha": 10},
            {"id": "knn", "kind": "knn", "neighbors": 3},
        ],
    }
    write_json(tmp_path / "dataset.json", data)
    write_json(tmp_path / "plan.json", plan)
    return tmp_path, frame, data, plan


def test_independent_ridge_normal_equations_and_nearest_neighbors():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 3.0, 6.0, 9.0])
    model = fit_model(x, y, {"kind": "ridge", "alpha": 2})
    z = (x[:, 0] - 2.5) / np.sqrt(1.25)
    coefficient = sum(z * (y - 5)) / (sum(z**2) + 2)
    np.testing.assert_allclose(model["coefficients"], [coefficient])
    np.testing.assert_allclose(
        predict_model(model, [[2.5], [5]]), [5, 5 + (2.5 / np.sqrt(1.25)) * coefficient]
    )
    knn = fit_model(x, y, {"kind": "knn", "neighbors": 2})
    np.testing.assert_allclose(predict_model(knn, [[1.5], [3.5]]), [2.5, 7.5])


def test_ml_archive_temporal_purge_and_single_use(inputs):
    root, _, _, plan = inputs
    result = run_study(root / "plan.json", root / "dataset.json", root / "runs")
    assert result["status"] == "completed"
    assert len(result["attempts"]) == 8
    assert result["holdout"]["status"] == "consumed"
    output = root / "runs/example"
    verify_archive(output)
    prediction = pd.read_csv(output / "predictions.csv")
    assert set(prediction.query("partition == 'holdout'").candidate) == {
        result["selection"]["selected"]
    }
    for attempt in result["attempts"]:
        split = attempt["split"]
        assert split["purged_training_rows"]
        assert pd.Timestamp(split["max_training_label_known_at"]) < pd.Timestamp(split["fit_at"])
        assert not set(split["training_rows"]) & set(split["evaluation_rows"])
    assert main(["verify", str(output)]) == 0
    with pytest.raises(FileExistsError):
        run_study(root / "plan.json", root / "dataset.json", root / "runs")
    (output / "predictions.csv").write_text("tamper")
    with pytest.raises(ValueError, match="digest"):
        verify_archive(output)
    (output / "unexpected.txt").write_text("extra")
    assert main(["verify", str(output)]) == 2
    assert validate_plan(plan)


def test_holdout_values_never_change_selection_or_preprocessing(inputs):
    root, frame, data, _ = inputs
    first = run_study(root / "plan.json", root / "dataset.json", root / "runs1")
    frame.loc[70:, ["x", "z", "target"]] = 1e8
    frame.to_csv(root / "mutated.csv", index=False)
    write_json(
        root / "mutated.json",
        {**data, "data_file": "mutated.csv", "data_sha256": file_hash(root / "mutated.csv")},
    )
    second = run_study(root / "plan.json", root / "mutated.json", root / "runs2")
    assert first["selection"] == second["selection"]
    for a, b in zip(
        sorted((root / "runs1/example/models").glob("*")),
        sorted((root / "runs2/example/models").glob("*")),
    ):
        assert read_json(a)["model"] == read_json(b)["model"]


def test_all_attempts_retained_and_failed_family_never_opens_holdout(inputs):
    root, _, _, plan = inputs
    plan["models"].append({"id": "bad", "kind": "knn", "neighbors": 10000})
    write_json(root / "bad-plan.json", plan)
    result = run_study(root / "bad-plan.json", root / "dataset.json", root / "runs")
    assert result["status"] == "incomplete_family"
    assert len(result["attempts"]) == 10
    assert sum(a["status"] == "failed" for a in result["attempts"]) == 2
    assert not (root / "runs/example/holdout-consumed.json").exists()
    assert result["selection"]["available"] is False
    verify_archive(root / "runs/example")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("future_feature", "after decision"),
        ("duplicate", "row IDs"),
        ("naive", "timezone"),
        ("future_label", "target_start"),
        ("infinite", "infinite features"),
        ("missing", "missing dataset columns"),
    ],
)
def test_data_contract_rejects_leakage_and_corruption(inputs, mutation, match):
    root, frame, data, _ = inputs
    if mutation == "future_feature":
        frame.loc[0, "x__known_at"] = frame.loc[1, "decision_at"]
    elif mutation == "duplicate":
        frame.loc[1, "row_id"] = frame.loc[0, "row_id"]
    elif mutation == "naive":
        frame["decision_at"] = frame.decision_at.dt.tz_localize(None)
    elif mutation == "future_label":
        frame["target_start"] = frame.target_end
    elif mutation == "infinite":
        frame.loc[0, "x"] = np.inf
    else:
        frame = frame.drop(columns="x")
    frame.to_csv(root / "bad.csv", index=False)
    write_json(
        root / "bad.json",
        {**data, "data_file": "bad.csv", "data_sha256": file_hash(root / "bad.csv")},
    )
    with pytest.raises(ValueError, match=match):
        load_dataset(root / "bad.json")


def test_invalid_input_is_archived_and_plans_cannot_overlap(inputs):
    root, _, data, plan = inputs
    write_json(root / "bad.json", {**data, "data_sha256": "0" * 64})
    result = run_study(root / "plan.json", root / "bad.json", root / "runs")
    assert result["status"] == "failed"
    verify_archive(root / "runs/example")
    bad = copy.deepcopy(plan)
    bad["folds"][1]["start"] = bad["folds"][0]["start"]
    with pytest.raises(ValueError, match="disjoint"):
        validate_plan(bad)
    bad = copy.deepcopy(plan)
    bad["study_id"] = "../escape"
    with pytest.raises(ValueError, match="identifier"):
        validate_plan(bad)
    bad["study_id"] = "CON"
    with pytest.raises(ValueError, match="portable"):
        validate_plan(bad)
    bad = copy.deepcopy(plan)
    bad["models"].append({"id": "MEAN", "kind": "mean"})
    with pytest.raises(ValueError, match="unique"):
        validate_plan(bad)
