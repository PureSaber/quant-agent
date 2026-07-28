"""Shared LangGraph state for review runs."""

from __future__ import annotations

from typing import Literal, TypedDict


class Finding(TypedDict, total=False):
    severity: Literal["error", "warn", "info"]
    code: str
    message: str
    factor: str | None


class ReviewState(TypedDict, total=False):
    project: str
    run_dir: str
    config_path: str | None
    agent_config_path: str
    offline: bool

    ic_summary: list[dict]
    backtest_stats: list[dict]
    ic_decay: list[dict]
    config_snapshot: dict
    factor_list: list[str]
    run_meta: dict

    rule_findings: list[Finding]
    rule_passed: bool

    explain_draft: str
    skeptic_draft: str
    final_report: str

    report_path: str
    manifest_path: str
