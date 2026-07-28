"""Load agent_review.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_agent_config(path: Path | None = None) -> dict[str, Any]:
    default = Path(__file__).resolve().parents[2] / "configs" / "agent_review.yaml"
    target = path or default
    if not target.is_file():
        return {
            "enable_llm": False,
            "model": "gpt-4.1-mini",
            "temperature": 0.2,
            "thresholds": {"min_ic_abs": 0.02, "min_ic_positive_ratio": 0.52},
            "rules": {
                "flag_nan_factors": True,
                "check_pit_config": True,
                "check_alt_factors": True,
            },
            "output": {"experiment_log_dir": "experiment-log", "write_manifest": True},
            "pitfalls_path": None,
        }
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}
