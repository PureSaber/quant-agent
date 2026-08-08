"""Run directory adapters for different quant projects."""

from __future__ import annotations

import json
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


def _read_csv_rows(path: Path, *, max_read_mb: float = 32.0) -> list[dict]:
    if not path.is_file():
        return []
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_read_mb:
        raise ValueError(f"CSV {path} is {size_mb:.1f}MB; limit is {max_read_mb}MB")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_config_snapshot(run_dir: Path, config_path: Path | None = None) -> dict:
    if config_path and Path(config_path).is_file():
        return _read_yaml(Path(config_path))
    for candidate in (run_dir / "config.snapshot.yaml", run_dir.parent / "config.snapshot.yaml"):
        if candidate.is_file():
            return _read_yaml(candidate)
    return {}


def _load_run_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


PROJECT_ALIASES = {
    "multifactor": "a-share-multifactor",
    "a-share-multifactor": "a-share-multifactor",
    "futures-spread": "quant-futures-spread",
    "quant-futures-spread": "quant-futures-spread",
    "future_spread": "quant-futures-spread",
}


def get_adapter(project: str) -> RunAdapter:
    from quant_agent.adapters.futures_spread import FuturesSpreadAdapter
    from quant_agent.adapters.multifactor import MultifactorAdapter

    canonical = PROJECT_ALIASES.get(project, project)
    registry: dict[str, RunAdapter] = {
        MultifactorAdapter.project: MultifactorAdapter(),
        FuturesSpreadAdapter.project: FuturesSpreadAdapter(),
    }
    if canonical not in registry:
        known = sorted(set(PROJECT_ALIASES) | set(registry))
        raise ValueError(f"Unknown project {project!r}. Known: {known}")
    return registry[canonical]


def detect_project(run_dir: Path) -> str | None:
    from quant_agent.adapters.futures_spread import FuturesSpreadAdapter
    from quant_agent.adapters.multifactor import MultifactorAdapter

    for adapter in (MultifactorAdapter(), FuturesSpreadAdapter()):
        if adapter.detect(run_dir):
            return adapter.project
    return None
