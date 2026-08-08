from quant_agent.config import load_agent_config
from quant_agent.nodes.pipeline import llm_enabled


def test_llm_disabled_without_env(monkeypatch) -> None:
    monkeypatch.delenv("QUANT_AGENT_LLM_OK", raising=False)
    cfg = {"enable_llm": True}
    assert llm_enabled(cfg) is False


def test_llm_enabled_with_env(monkeypatch) -> None:
    monkeypatch.setenv("QUANT_AGENT_LLM_OK", "1")
    cfg = {"enable_llm": True}
    assert llm_enabled(cfg) is True


def test_load_agent_config_has_rules_keys() -> None:
    cfg = load_agent_config()
    assert "rules" in cfg
    assert "thresholds" in cfg
