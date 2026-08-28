from __future__ import annotations

import pandas as pd
from quant_lab.contracts import write_standard_run
from quant_lab.contracts_v2 import (
    ARTIFACT_SCHEMAS_V2,
    RESEARCH_PROFILE,
    write_standard_run_v2,
)

from quant_agent.adapters.multifactor import MultifactorAdapter
from tests.fixtures import write_multifactor_fixture


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


def _v2_sample_value(column: str):
    if column == "event_time":
        return pd.Timestamp("2025-01-02T03:04:05Z")
    if column.endswith("_units"):
        return 10_000
    if column.endswith("_scale"):
        return 2
    if column in {"gross_return", "net_return", "value"}:
        return 1.0
    if column in {"currency", "base_currency"}:
        return "CNY"
    return f"sample-{column}"


def test_adapter_prefers_and_exposes_valid_standard_v2_contract(tmp_path) -> None:
    write_multifactor_fixture(tmp_path)
    artifact_names = ("returns", "positions", "portfolio_snapshots", "exposures")
    frames = {
        name: pd.DataFrame(
            [{column: _v2_sample_value(column) for column in ARTIFACT_SCHEMAS_V2[name]}],
            columns=ARTIFACT_SCHEMAS_V2[name],
        )
        for name in artifact_names
    }
    lineage = {name: ["dataset:prices"] for name in ("metrics", *artifact_names)}
    lineage["config"] = []
    write_standard_run_v2(
        tmp_path,
        project="a-share-multifactor",
        run_id="r2",
        strategy_ids=["Q5"],
        profile=RESEARCH_PROFILE,
        frames=frames,
        metrics={"total_return": 0.1},
        config={"base_currency": "CNY"},
        code_version="a" * 40,
        internal_dependencies={"quant-lab": "v0.3.0"},
        random_seed=7,
        dataset_snapshots={"prices": "sha256-demo-v2"},
        instrument_master_version="instruments-v1",
        execution_model_version="not-applicable",
        base_currency="CNY",
        lineage=lineage,
        created_at="2025-01-02T00:00:00+00:00",
    )

    contract = MultifactorAdapter().load(tmp_path).run_meta["standard_contract"]

    assert contract == {
        "schema_version": "2.0.0",
        "project": "a-share-multifactor",
        "run_id": "r2",
        "code_version": "a" * 40,
        "dataset_snapshots": {"prices": "sha256-demo-v2"},
        "profile": RESEARCH_PROFILE,
        "validated": True,
    }
