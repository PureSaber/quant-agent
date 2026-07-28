from pathlib import Path

from quant_agent.adapters.multifactor import MultifactorAdapter
from tests.fixtures import write_multifactor_fixture


def test_multifactor_adapter_loads(tmp_path: Path) -> None:
    write_multifactor_fixture(tmp_path)
    ctx = MultifactorAdapter().load(tmp_path)
    assert ctx.project == "a-share-multifactor"
    assert len(ctx.ic_summary) == 3
    assert "forecast_score" in ctx.factor_list
    assert ctx.config_snapshot["filters"]["pit_fundamentals"] is True
