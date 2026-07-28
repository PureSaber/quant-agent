"""Compare current run outputs with a previous timestamp run."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from quant_agent.state import Finding

_RUN_DIR_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def _find_previous_run_dir(run_dir: Path) -> Path | None:
    parent = run_dir.parent
    if not parent.is_dir():
        return None

    current_name = run_dir.name
    candidates: list[str] = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        if child.name in {"latest", current_name}:
            continue
        if _RUN_DIR_PATTERN.match(child.name):
            candidates.append(child.name)

    if not candidates:
        return None

    candidates.sort()
    if current_name in candidates:
        idx = candidates.index(current_name)
        if idx > 0:
            return parent / candidates[idx - 1]
        return None

    # Reviewing latest/ — compare against newest timestamp dir
    return parent / candidates[-1]


def _load_ic_map(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    result: dict[str, float] = {}
    for _, row in df.iterrows():
        factor = str(row.get("factor", ""))
        mean_ic = row.get("mean_ic")
        if factor and pd.notna(mean_ic):
            result[factor] = float(mean_ic)
    return result


def _load_long_short_sharpe(path: Path) -> float | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty or "portfolio" not in df.columns:
        return None
    row = df[df["portfolio"] == "long_short"]
    if row.empty:
        row = df.iloc[[0]]
    sharpe = row.iloc[0].get("sharpe")
    if pd.isna(sharpe):
        return None
    return float(sharpe)


def compare_with_previous(run_dir: Path, thresholds: dict) -> list[Finding]:
    findings: list[Finding] = []
    prev_dir = _find_previous_run_dir(run_dir)
    if prev_dir is None:
        findings.append(
            {
                "severity": "info",
                "code": "no_previous_run",
                "message": "No prior timestamp run found for comparison",
                "factor": None,
            }
        )
        return findings

    ic_delta_threshold = float(thresholds.get("ic_delta_warn", 0.02))
    current_ic = _load_ic_map(run_dir / "ic_summary.csv")
    prev_ic = _load_ic_map(prev_dir / "ic_summary.csv")

    for factor, prev_val in prev_ic.items():
        curr_val = current_ic.get(factor)
        if curr_val is None:
            continue
        delta = curr_val - prev_val
        if abs(delta) >= ic_delta_threshold:
            findings.append(
                {
                    "severity": "info",
                    "code": "ic_delta",
                    "message": (
                        f"Factor {factor} mean_ic changed {prev_val:.4f} → {curr_val:.4f} "
                        f"(Δ={delta:+.4f}) vs {prev_dir.name}"
                    ),
                    "factor": factor,
                }
            )

    curr_sharpe = _load_long_short_sharpe(run_dir / "backtest_stats.csv")
    prev_sharpe = _load_long_short_sharpe(prev_dir / "backtest_stats.csv")
    if curr_sharpe is not None and prev_sharpe is not None:
        delta = curr_sharpe - prev_sharpe
        if abs(delta) >= float(thresholds.get("sharpe_delta_warn", 0.3)):
            findings.append(
                {
                    "severity": "info",
                    "code": "sharpe_delta",
                    "message": (
                        f"long_short sharpe changed {prev_sharpe:.2f} → {curr_sharpe:.2f} "
                        f"(Δ={delta:+.2f}) vs {prev_dir.name}"
                    ),
                    "factor": None,
                }
            )

    return findings
