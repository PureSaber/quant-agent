import sqlite3

import numpy as np
import pandas as pd
import pytest
from quant_lab.trials import TrialRegistry

from quant_agent.research_workflow.contracts import file_hash, write_json
from quant_agent.research_workflow.lab_adapter import ReadOnlyTrialRegistry, registered_family
from quant_agent.research_workflow.robustness import registered_robustness


def test_lab_family_readonly_inventory_keeps_failure_and_unattempted(tmp_path):
    path = tmp_path / "lab.db"
    writer = TrialRegistry(path)
    basis = {
        "currency": "USD",
        "benchmark_id": "cash",
        "cost_policy": "net",
        "periods_per_year": 252,
        "sample_kind": "synthetic",
    }
    writer.register_family(
        "family", {"hypothesis": "test", "study_ids": ["study", "missing"], "basis": basis}
    )
    writer.register(
        "study",
        {
            "family_id": "family",
            "hypothesis": "test",
            "parameters": [{"alpha": 1}, {"alpha": 2}],
            "code_identity": "frozen",
            "selection_rule": "validation only",
        },
    )
    attempt = writer.start("study", {"alpha": 1})
    writer.finish(attempt, "failed", {"error": "data missing"})
    before = file_hash(path)
    matrix, audit = registered_family(path, "family")
    assert isinstance(matrix, pd.DataFrame)
    assert not audit["available"]
    assert audit["statuses"][attempt] == "failed"
    assert len(audit["planned"]) == 3
    assert before == file_hash(path)
    reader = ReadOnlyTrialRegistry(path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.start("study", {"alpha": 2})
    assert before == file_hash(path)
    with pytest.raises(FileNotFoundError):
        ReadOnlyTrialRegistry(tmp_path / "missing.db")


def test_invalid_family_digest_is_not_trusted(tmp_path):
    path = tmp_path / "lab.db"
    writer = TrialRegistry(path)
    with writer.connect() as db:
        db.execute("INSERT INTO research_families VALUES ('family','{}','bad','2020-01-01')")
    with pytest.raises(ValueError, match="digest"):
        registered_family(path, "family")


def test_complete_registry_integrates_without_writes_and_retains_failed_retry(tmp_path):
    path = tmp_path / "lab.db"
    writer = TrialRegistry(path)
    dates = pd.date_range("2020-01-01", periods=32, tz="UTC")
    basis = {
        "currency": "USD",
        "benchmark_id": "explicit-cash",
        "cost_policy": "synthetic-net",
        "periods_per_year": 252,
        "sample_kind": "synthetic",
    }
    writer.register_family("family", {"hypothesis": "test", "study_ids": ["study"], "basis": basis})
    writer.register(
        "study",
        {
            "family_id": "family",
            "hypothesis": "test",
            "parameters": [{"alpha": 1}, {"alpha": 2}],
            "code_identity": "frozen",
            "selection_rule": "validation only",
        },
    )
    parameters = {}
    rng = np.random.default_rng(27)
    for alpha in [1, 2]:
        attempt = writer.start("study", {"alpha": alpha}, context={"artifact_root": str(tmp_path)})
        folder = tmp_path / "attempts" / attempt
        folder.mkdir(parents=True)
        pd.Series(rng.normal(0.001, 0.01, 32), index=dates).to_csv(
            folder / "returns.csv", header=["net_return"]
        )
        result = {
            "study_id": "study",
            "attempt_id": attempt,
            "status": "completed",
            "measurement_basis": basis,
            "comparison": {
                "start": "2020-01-01",
                "end": "2020-02-01",
                "currency": "USD",
                "costs": "synthetic-net",
            },
            "artifacts": {"returns.csv": file_hash(folder / "returns.csv")},
        }
        write_json(folder / "result.json", result)
        writer.finish(
            attempt,
            "completed",
            {
                "result": str((folder / "result.json").relative_to(tmp_path)),
                "sha256": file_hash(folder / "result.json"),
            },
        )
        parameters[attempt] = {"alpha": alpha}
    before = file_hash(path)
    plan = {
        "schema": "quant.robustness-plan/v1",
        "family_id": "family",
        "partition": "validation",
        "basis": basis,
        "planned": list(parameters),
        "statuses": {key: "completed" for key in parameters},
        "parameters": parameters,
        "holdout_start": "2020-02-02T00:00:00Z",
        "intervals": [
            {"id": "all", "start": "2020-01-01T00:00:00Z", "end": "2020-02-02T00:00:00Z"}
        ],
        "cscv_blocks": 4,
        "block_lengths": [2],
        "repetitions": 100,
        "seed": 1,
        "alpha": 0.05,
    }
    result = registered_robustness(path, "family", plan, pd.Series(0.0, index=dates))
    assert result["registry_audit"]["available"]
    assert before == file_hash(path)
    retry = writer.start("study", {"alpha": 1})
    writer.finish(retry, "failed", {"error": "failed rerun"})
    with pytest.raises(ValueError, match="complete"):
        registered_robustness(path, "family", plan, pd.Series(0.0, index=dates))
    plan["planned"].append(retry)
    plan["statuses"][retry] = "failed"
    plan["parameters"][retry] = {"alpha": 1}
    assert not registered_robustness(path, "family", plan, None)["audit"]["available"]
    plan["parameters"][retry] = {"alpha": 100}
    with pytest.raises(ValueError, match="registered attempt"):
        registered_robustness(path, "family", plan, None)
