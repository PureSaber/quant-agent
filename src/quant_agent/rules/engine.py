"""Deterministic review rules."""

from __future__ import annotations

import math
from typing import Any, Literal

from quant_agent.adapters.base import RunContext
from quant_agent.state import Finding


SPREAD_PROJECTS = frozenset({"quant-futures-spread", "future_spread", "futures-spread"})


def _is_spread_project(project: str) -> bool:
    return project in SPREAD_PROJECTS


def _is_nan(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}


def _finding(
    severity: Literal["error", "warn", "info"],
    code: str,
    message: str,
    *,
    factor: str | None = None,
) -> Finding:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "factor": factor,
    }


def check_ic_quality(ctx: RunContext, thresholds: dict[str, Any]) -> list[Finding]:
    if _is_spread_project(ctx.project):
        return []
    findings: list[Finding] = []
    min_ic = float(thresholds.get("min_ic_abs", 0.02))
    min_pos = float(thresholds.get("min_ic_positive_ratio", 0.52))

    if not ctx.ic_summary:
        findings.append(
            _finding("error", "ic_missing", "ic_summary.csv is empty or missing")
        )
        return findings

    for row in ctx.ic_summary:
        factor = str(row.get("factor", ""))
        mean_ic = row.get("mean_ic")
        pos_ratio = row.get("ic_positive_ratio")

        if _is_nan(mean_ic):
            findings.append(
                _finding(
                    "warn",
                    "ic_nan",
                    f"Factor {factor} has NaN mean_ic — check data coverage",
                    factor=factor,
                )
            )
            continue

        mean_ic_f = float(mean_ic)
        if abs(mean_ic_f) < min_ic:
            findings.append(
                _finding(
                    "info",
                    "ic_weak",
                    f"Factor {factor} |mean_ic|={mean_ic_f:.4f} below {min_ic}",
                    factor=factor,
                )
            )

        if not _is_nan(pos_ratio) and float(pos_ratio) < min_pos:
            findings.append(
                _finding(
                    "info",
                    "ic_unstable",
                    f"Factor {factor} ic_positive_ratio={float(pos_ratio):.2f} below {min_pos}",
                    factor=factor,
                )
            )

    return findings


ALT_FACTORS = {"forecast_score", "northbound_chg_5d", "industry_rs_20d"}


def check_alt_data(ctx: RunContext) -> list[Finding]:
    findings: list[Finding] = []
    ic_by_factor = {str(r.get("factor")): r for r in ctx.ic_summary}

    for factor in ALT_FACTORS:
        if factor not in ctx.factor_list and factor not in ic_by_factor:
            continue
        row = ic_by_factor.get(factor, {})
        if _is_nan(row.get("mean_ic")):
            findings.append(
                _finding(
                    "warn",
                    "alt_data_nan",
                    f"Alt factor {factor} has no valid IC — seed alt data or run --fetch-alt",
                    factor=factor,
                )
            )

    return findings


def check_pit_config(ctx: RunContext) -> list[Finding]:
    findings: list[Finding] = []
    filters = ctx.config_snapshot.get("filters") or {}
    data = ctx.config_snapshot.get("data") or {}

    if filters.get("pit_fundamentals") is False:
        findings.append(
            _finding(
                "warn",
                "pit_off",
                "pit_fundamentals is false — fundamentals may look ahead",
            )
        )

    if filters.get("use_historical_universe") is False:
        findings.append(
            _finding(
                "info",
                "universe_current",
                "use_historical_universe is false — survivorship bias risk",
            )
        )

    for key in ("earnings_forecast", "northbound", "industry_returns"):
        path_key = key if key != "industry_returns" else "industry_returns"
        if path_key in data or key in data:
            continue
        if any(f in ALT_FACTORS for f in ctx.factor_list):
            findings.append(
                _finding("info", "alt_path_missing", f"No data.{key} in config snapshot")
            )

    return findings


def _spread_metrics(ctx: RunContext) -> dict[str, Any]:
    """Normalize spread adapter metric/value rows or flat summary dict."""
    if not ctx.backtest_stats:
        return {}
    rows = ctx.backtest_stats
    if rows and "metric" in rows[0] and "value" in rows[0]:
        return {str(row["metric"]): row["value"] for row in rows}
    return dict(rows[0])


def check_spread_performance(ctx: RunContext, thresholds: dict[str, Any]) -> list[Finding]:
    """Review futures-spread runs from performance/summary.csv."""
    findings: list[Finding] = []
    if not _is_spread_project(ctx.project):
        return findings

    metrics = _spread_metrics(ctx)
    if not metrics:
        findings.append(
            _finding(
                "error",
                "spread_perf_missing",
                "performance/summary.csv is empty or missing",
            )
        )
        return findings

    min_sharpe = float(thresholds.get("min_sharpe", 0.0))
    max_drawdown_limit = float(thresholds.get("max_drawdown_limit", -0.35))
    min_return = float(thresholds.get("min_spread_total_return", 0.0))
    min_calmar = float(thresholds.get("min_spread_calmar", 0.0))

    sharpe = metrics.get("sharpe")
    max_dd = metrics.get("max_drawdown")
    total_return = metrics.get("total_return")
    calmar = metrics.get("calmar")

    if not _is_nan(sharpe) and float(sharpe) < min_sharpe:
        findings.append(
            _finding(
                "info",
                "sharpe_low",
                f"spread sharpe={float(sharpe):.2f} below {min_sharpe}",
            )
        )
    if not _is_nan(max_dd) and float(max_dd) < max_drawdown_limit:
        findings.append(
            _finding(
                "warn",
                "drawdown_high",
                (
                    f"spread max_drawdown={float(max_dd):.2f} "
                    f"worse than {max_drawdown_limit}"
                ),
            )
        )
    if not _is_nan(total_return) and float(total_return) < min_return:
        findings.append(
            _finding(
                "warn",
                "spread_return_low",
                f"total_return={float(total_return):.4f} below {min_return}",
            )
        )
    if not _is_nan(calmar) and float(calmar) < min_calmar:
        findings.append(
            _finding(
                "info",
                "spread_calmar_low",
                f"calmar={float(calmar):.4f} below {min_calmar}",
            )
        )
    return findings


def check_backtest_stats(ctx: RunContext, thresholds: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    if not ctx.backtest_stats:
        findings.append(
            _finding(
                "warn",
                "backtest_stats_missing",
                "backtest_stats.csv is empty or missing",
            )
        )
        return findings

    min_sharpe = float(thresholds.get("min_sharpe", 0.0))
    max_drawdown_limit = float(thresholds.get("max_drawdown_limit", -0.35))

    for row in ctx.backtest_stats:
        portfolio = str(row.get("portfolio", ""))
        sharpe = row.get("sharpe")
        max_dd = row.get("max_drawdown")

        if portfolio == "long_short" or len(ctx.backtest_stats) == 1:
            if not _is_nan(sharpe) and float(sharpe) < min_sharpe:
                findings.append(
                    _finding(
                        "info",
                        "sharpe_low",
                        f"{portfolio} sharpe={float(sharpe):.2f} below {min_sharpe}",
                    )
                )
            if not _is_nan(max_dd) and float(max_dd) < max_drawdown_limit:
                findings.append(
                    _finding(
                        "warn",
                        "drawdown_high",
                        (
                            f"{portfolio} max_drawdown={float(max_dd):.2f} "
                            f"worse than {max_drawdown_limit}"
                        ),
                    )
                )
            break

    return findings


def check_ic_decay(ctx: RunContext, thresholds: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    if not ctx.ic_decay:
        return findings

    short_h = int(thresholds.get("ic_decay_short_horizon", 1))
    long_h = int(thresholds.get("ic_decay_long_horizon", 20))
    decay_ratio = float(thresholds.get("ic_decay_ratio_warn", 0.5))

    by_factor: dict[str, dict[int, float]] = {}
    for row in ctx.ic_decay:
        factor = str(row.get("factor", ""))
        horizon = row.get("horizon_days")
        mean_ic = row.get("mean_ic")
        if not factor or _is_nan(horizon) or _is_nan(mean_ic):
            continue
        by_factor.setdefault(factor, {})[int(horizon)] = float(mean_ic)

    for factor, horizons in by_factor.items():
        short_ic = horizons.get(short_h)
        long_ic = horizons.get(long_h)
        if short_ic is None or long_ic is None:
            continue
        if abs(short_ic) < 1e-6:
            continue
        ratio = abs(long_ic) / abs(short_ic)
        if ratio < decay_ratio and abs(short_ic) >= float(thresholds.get("min_ic_abs", 0.02)):
            findings.append(
                _finding(
                    "warn",
                    "ic_decay_fast",
                    (
                        f"Factor {factor} IC decays fast: h{short_h}={short_ic:.4f} "
                        f"→ h{long_h}={long_ic:.4f} (ratio={ratio:.2f})"
                    ),
                    factor=factor,
                )
            )

    return findings


def run_all_rules(ctx: RunContext, agent_config: dict) -> list[Finding]:
    from quant_agent.rules.run_compare import compare_with_previous

    rules_cfg = agent_config.get("rules") or {}
    thresholds = agent_config.get("thresholds") or {}
    findings: list[Finding] = []

    if _is_spread_project(ctx.project):
        if rules_cfg.get("check_backtest_stats", True):
            findings.extend(check_spread_performance(ctx, thresholds))
        if rules_cfg.get("compare_with_previous", True):
            findings.extend(compare_with_previous(ctx.run_dir, thresholds))
        return findings

    findings.extend(check_ic_quality(ctx, thresholds))

    if rules_cfg.get("check_backtest_stats", True):
        findings.extend(check_backtest_stats(ctx, thresholds))

    if rules_cfg.get("check_ic_decay", True):
        findings.extend(check_ic_decay(ctx, thresholds))

    if rules_cfg.get("check_alt_factors", True):
        findings.extend(check_alt_data(ctx))

    if rules_cfg.get("check_pit_config", True) and ctx.config_snapshot:
        findings.extend(check_pit_config(ctx))

    if rules_cfg.get("compare_with_previous", True):
        findings.extend(compare_with_previous(ctx.run_dir, thresholds))

    return findings
