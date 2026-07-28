"""Load optional pitfalls.md for LLM context."""

from __future__ import annotations

from pathlib import Path


def load_pitfalls(agent_config: dict, agent_config_path: Path | None = None) -> str:
    raw_path = agent_config.get("pitfalls_path")
    if not raw_path:
        return ""

    path = Path(raw_path)
    if not path.is_absolute() and agent_config_path:
        path = agent_config_path.parent / path
    elif not path.is_absolute():
        default_root = Path(__file__).resolve().parents[2]
        path = default_root / raw_path

    if not path.is_file():
        return ""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return f"\n\nKnown pitfalls:\n{text[:4000]}"
