"""Adapter for a-share-multifactor output layout."""

from __future__ import annotations

import json
from pathlib import Path

from quant_agent.adapters.base import (
    RunContext,
    _load_config_snapshot,
    _load_run_meta,
    _read_csv_rows,
)


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

        config_snapshot = _load_config_snapshot(run_dir, config_path)
        run_meta = _load_run_meta(run_dir)

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
