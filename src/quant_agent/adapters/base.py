"""Run directory adapters for different quant projects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd
import yaml


@dataclass
class RunContext:
    project: str
    run_dir: Path
    ic_summary: list[dict] = field(default_factory=list)
    backtest_stats: list[dict] = field(default_factory=list)
    ic_decay: list[dict] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)
    factor_list: list[str] = field(default_factory=list)
    run_meta: dict = field(default_factory=dict)


class RunAdapter(Protocol):
    project: str

    def detect(self, run_dir: Path) -> bool: ...

    def load(self, run_dir: Path, config_path: Path | None = None) -> RunContext: ...


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


PROJECT_ALIASES = {
    "multifactor": "a-share-multifactor",
    "a-share-multifactor": "a-share-multifactor",
}


def get_adapter(project: str) -> RunAdapter:
    from quant_agent.adapters.multifactor import MultifactorAdapter

    canonical = PROJECT_ALIASES.get(project, project)
    registry: dict[str, RunAdapter] = {
        MultifactorAdapter.project: MultifactorAdapter(),
    }
    if canonical not in registry:
        known = sorted(set(PROJECT_ALIASES) | set(registry))
        raise ValueError(f"Unknown project {project!r}. Known: {known}")
    return registry[canonical]


def detect_project(run_dir: Path) -> str | None:
    from quant_agent.adapters.multifactor import MultifactorAdapter

    for adapter in (MultifactorAdapter(),):
        if adapter.detect(run_dir):
            return adapter.project
    return None
