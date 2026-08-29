"""Adapter for a-share-multifactor output layout."""

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

_BACKTEST_METRICS = (
    "portfolio",
    "sharpe",
    "max_drawdown",
    "ann_return",
    "annual_return",
    "total_return",
    "calmar",
)


class MultifactorAdapter:
    project = "a-share-multifactor"

    def detect(self, run_dir: Path) -> bool:
        run_dir = Path(run_dir)
        standard = load_standard_artifacts(run_dir)
        if standard is not None and standard.is_v2:
            return standard.project == self.project
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
        standard = load_standard_artifacts(run_dir)
        if standard is not None and standard.is_v2:
            if standard.project != self.project:
                raise ValueError(
                    f"standard/v2 project {standard.project!r} does not match {self.project!r}"
                )
            if config_path is not None:
                raise ValueError("--config cannot override validated standard/v2 config.json")
            ic_rows = metric_rows(standard.metrics, "ic_summary")
            backtest_stats = metric_rows(standard.metrics, "backtest_stats")
            if not backtest_stats:
                backtest_stats = flat_metric_rows(standard.metrics, _BACKTEST_METRICS)
            factors = [str(row["factor"]) for row in ic_rows if row.get("factor")]
            if not factors:
                factors = string_list(standard.config.get("factors"), "config.factors")
            if not factors:
                factors = list(standard.strategy_ids)
            return RunContext(
                project=self.project,
                run_dir=run_dir,
                ic_summary=ic_rows,
                backtest_stats=backtest_stats,
                ic_decay=metric_rows(standard.metrics, "ic_decay"),
                config_snapshot=standard.config,
                factor_list=factors,
                run_meta={"standard_contract": standard.contract},
            )

        ic_rows = _read_csv_rows(run_dir / "ic_summary.csv")
        factors = [str(row.get("factor", "")) for row in ic_rows if row.get("factor")]

        config_snapshot = _load_config_snapshot(run_dir, config_path)
        run_meta = _load_legacy_run_meta(run_dir, standard)

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
