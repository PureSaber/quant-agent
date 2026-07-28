"""Adapter for a-share-multifactor output layout."""

from __future__ import annotations

import json
from pathlib import Path

from quant_agent.adapters.base import RunContext, _read_csv_rows, _read_yaml


class MultifactorAdapter:
    project = "a-share-multifactor"

    def detect(self, run_dir: Path) -> bool:
        run_dir = Path(run_dir)
        if (run_dir / "ic_summary.csv").is_file():
            return True
        meta = run_dir / "run_meta.json"
        if meta.is_file():
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
                return payload.get("project") == self.project
            except json.JSONDecodeError:
                return False
        return False

    def load(self, run_dir: Path, config_path: Path | None = None) -> RunContext:
        run_dir = Path(run_dir)
        ic_rows = _read_csv_rows(run_dir / "ic_summary.csv")
        factors = [str(row.get("factor", "")) for row in ic_rows if row.get("factor")]

        config_snapshot: dict = {}
        if config_path and Path(config_path).is_file():
            config_snapshot = _read_yaml(Path(config_path))
        else:
            for candidate in (
                run_dir / "config.snapshot.yaml",
                run_dir.parent / "config.snapshot.yaml",
            ):
                if candidate.is_file():
                    config_snapshot = _read_yaml(candidate)
                    break

        run_meta: dict = {}
        meta_path = run_dir / "run_meta.json"
        if meta_path.is_file():
            run_meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if not factors and config_snapshot.get("factors"):
            factors = list(config_snapshot["factors"])

        return RunContext(
            project=self.project,
            run_dir=run_dir,
            ic_summary=ic_rows,
            backtest_stats=_read_csv_rows(run_dir / "backtest_stats.csv"),
            ic_decay=_read_csv_rows(run_dir / "ic_decay.csv"),
            config_snapshot=config_snapshot,
            factor_list=factors,
            run_meta=run_meta,
        )
