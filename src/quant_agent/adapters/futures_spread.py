"""Adapter for quant-futures-spread output layout."""

from __future__ import annotations

import json
from pathlib import Path

from quant_agent.adapters.base import RunContext, _read_csv_rows, _read_yaml


class FuturesSpreadAdapter:
    project = "quant-futures-spread"

    def detect(self, run_dir: Path) -> bool:
        run_dir = Path(run_dir)
        summary = run_dir / "performance" / "summary.csv"
        if summary.is_file():
            return True
        meta = run_dir / "run_meta.json"
        if meta.is_file():
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
                return payload.get("project") in (self.project, "future_spread", "quant-futures-spread")
            except json.JSONDecodeError:
                return False
        return False

    def load(self, run_dir: Path, config_path: Path | None = None) -> RunContext:
        run_dir = Path(run_dir)
        summary_rows = _read_csv_rows(run_dir / "performance" / "summary.csv")

        backtest_stats = []
        if summary_rows:
            row = summary_rows[0]
            for key in ("total_return", "calmar", "max_drawdown", "sharpe", "annual_return"):
                if key in row:
                    backtest_stats.append({"metric": key, "value": row[key]})

        config_snapshot: dict = {}
        if config_path and Path(config_path).is_file():
            config_snapshot = _read_yaml(Path(config_path))
        else:
            for candidate in (run_dir / "config.snapshot.yaml", run_dir.parent / "config.snapshot.yaml"):
                if candidate.is_file():
                    config_snapshot = _read_yaml(candidate)
                    break

        run_meta: dict = {}
        meta_path = run_dir / "run_meta.json"
        if meta_path.is_file():
            run_meta = json.loads(meta_path.read_text(encoding="utf-8"))

        return RunContext(
            project=self.project,
            run_dir=run_dir,
            ic_summary=[],
            backtest_stats=backtest_stats,
            ic_decay=[],
            config_snapshot=config_snapshot,
            factor_list=list(config_snapshot.get("strategies") or []),
            run_meta=run_meta,
        )
