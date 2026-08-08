from pathlib import Path

from quant_agent.config import load_agent_config


def test_agent_config_loads_yaml() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "agent_review.yaml"
    cfg = load_agent_config(path)
    assert cfg.get("enable_llm") is False
    assert "thresholds" in cfg
