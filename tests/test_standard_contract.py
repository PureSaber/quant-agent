from __future__ import annotations

from quant_agent.adapters.multifactor import MultifactorAdapter
from quant_lab.contracts import write_standard_run
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
