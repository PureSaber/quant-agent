"""Adapter for quant-futures-spread output layout."""

from __future__ import annotations

import json
from pathlib import Path

from quant_agent.adapters.base import (
    RunContext,
    _load_config_snapshot,
    _load_legacy_run_meta,
    _read_csv_rows,
)
from quant_agent.adapters.standard import (
    flat_metric_rows,
    load_standard_artifacts,
    metric_rows,
    string_list,
)

_SPREAD_METRICS = (
    "total_return",
    "calmar",
    "max_drawdown",
    "sharpe",
    "annual_return",
)


class FuturesSpreadAdapter:
    project = "quant-futures-spread"

    def detect(self, run_dir: Path) -> bool:
        run_dir = Path(run_dir)
        standard = load_standard_artifacts(run_dir)
        if standard is not None and standard.is_v2:
            return standard.project == self.project
        summary = run_dir / "performance" / "summary.csv"
        if summary.is_file():
            return True
        meta = run_dir / "run_meta.json"
        if meta.is_file():
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
                return payload.get("project") in (
                    self.project,
                    "future_spread",
                    "quant-futures-spread",
                )
            except json.JSONDecodeError:
                return False
        return False

    def load(self, run_dir: Path, config_path: Path | None = None) -> RunContext:
        run_dir = Path(run_dir)
        standard = load_standard_artifacts(run_dir)
        if standard is not None and standard.is_v2:
            if standard.project != self.project:
                raise ValueError(
                    f"standard/v2 project {standard.project!r} does not match {self.project!r}"
                )
            if config_path is not None:
                raise ValueError("--config cannot override validated standard/v2 config.json")
            backtest_stats = metric_rows(standard.metrics, "backtest_stats")
            if not backtest_stats:
                backtest_stats = flat_metric_rows(standard.metrics, _SPREAD_METRICS)
            strategies = string_list(standard.config.get("strategies"), "config.strategies")
            return RunContext(
                project=self.project,
                run_dir=run_dir,
                ic_summary=[],
                backtest_stats=backtest_stats,
                ic_decay=[],
                config_snapshot=standard.config,
                factor_list=strategies or list(standard.strategy_ids),
                run_meta={"standard_contract": standard.contract},
            )

        summary_rows = _read_csv_rows(run_dir / "performance" / "summary.csv")

        backtest_stats = []
        if summary_rows:
            row = summary_rows[0]
            for key in ("total_return", "calmar", "max_drawdown", "sharpe", "annual_return"):
                if key in row:
                    backtest_stats.append({"metric": key, "value": row[key]})

        config_snapshot = _load_config_snapshot(run_dir, config_path)
        run_meta = _load_legacy_run_meta(run_dir, standard)

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
