from pathlib import Path

from quant_agent.pitfalls import load_pitfalls


def test_load_pitfalls_from_file(tmp_path: Path) -> None:
    pitfalls = tmp_path / "pitfalls.md"
    pitfalls.write_text("- Avoid lookahead in fundamentals", encoding="utf-8")
    agent_config = {"pitfalls_path": "pitfalls.md"}
    text = load_pitfalls(agent_config, tmp_path / "agent_review.yaml")
    assert "lookahead" in text


def test_load_pitfalls_missing_returns_empty() -> None:
    assert load_pitfalls({"pitfalls_path": "missing.md"}, None) == ""
