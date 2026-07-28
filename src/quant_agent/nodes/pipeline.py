"""Graph nodes for the review pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from quant_agent.adapters.base import get_adapter
from quant_agent.config import load_agent_config
from quant_agent.pitfalls import load_pitfalls
from quant_agent.rules.engine import run_all_rules
from quant_agent.state import ReviewState


def load_run_node(state: ReviewState) -> ReviewState:
    project = state["project"]
    run_dir = Path(state["run_dir"])
    config_path = Path(state["config_path"]) if state.get("config_path") else None

    adapter = get_adapter(project)
    ctx = adapter.load(run_dir, config_path)

    return {
        **state,
        "ic_summary": ctx.ic_summary,
        "backtest_stats": ctx.backtest_stats,
        "ic_decay": ctx.ic_decay,
        "config_snapshot": ctx.config_snapshot,
        "factor_list": ctx.factor_list,
        "run_meta": ctx.run_meta,
    }


def rule_check_node(state: ReviewState) -> ReviewState:
    from quant_agent.adapters.base import RunContext

    agent_config = load_agent_config(
        Path(state["agent_config_path"]) if state.get("agent_config_path") else None
    )
    ctx = RunContext(
        project=state["project"],
        run_dir=Path(state["run_dir"]),
        ic_summary=state.get("ic_summary") or [],
        backtest_stats=state.get("backtest_stats") or [],
        ic_decay=state.get("ic_decay") or [],
        config_snapshot=state.get("config_snapshot") or {},
        factor_list=state.get("factor_list") or [],
        run_meta=state.get("run_meta") or {},
    )
    findings = run_all_rules(ctx, agent_config)
    has_error = any(f.get("severity") == "error" for f in findings)
    return {**state, "rule_findings": findings, "rule_passed": not has_error}


def _format_findings(findings: list[dict]) -> str:
    if not findings:
        return "- No rule findings."
    lines = []
    for f in findings:
        sev = f.get("severity", "info").upper()
        factor = f.get("factor")
        suffix = f" ({factor})" if factor else ""
        lines.append(f"- [{sev}] {f.get('code')}{suffix}: {f.get('message')}")
    return "\n".join(lines)


def _format_ic_table(rows: list[dict]) -> str:
    if not rows:
        return "_No IC summary._"
    header = "| factor | mean_ic | ic_ir | ic_positive_ratio |"
    sep = "| --- | ---: | ---: | ---: |"
    body = []
    for row in rows:
        body.append(
            f"| {row.get('factor', '')} | {row.get('mean_ic', '')} | "
            f"{row.get('ic_ir', '')} | {row.get('ic_positive_ratio', '')} |"
        )
    return "\n".join([header, sep, *body])


def _offline_explain(state: ReviewState) -> str:
    factors = state.get("factor_list") or []
    findings = state.get("rule_findings") or []
    warn_count = sum(1 for f in findings if f.get("severity") == "warn")
    return (
        f"Offline review for `{state['project']}` at `{state['run_dir']}`.\n\n"
        f"Factors: {', '.join(factors) or 'unknown'}.\n"
        f"Rule findings: {len(findings)} total ({warn_count} warnings).\n"
        "Enable LLM (`--llm` or `enable_llm: true`) for narrative interpretation."
    )


def _offline_skeptic(state: ReviewState) -> str:
    findings = state.get("rule_findings") or []
    if not findings:
        return "No red flags from deterministic rules. Still validate PIT and alt-data coverage manually."
    return (
        "Skeptic checklist (offline):\n"
        + _format_findings(findings)
        + "\n\nRe-run with real alt data before trusting alt-factor IC."
    )


def llm_explain_node(state: ReviewState) -> ReviewState:
    if state.get("offline"):
        return {**state, "explain_draft": _offline_explain(state)}

    agent_config = load_agent_config(
        Path(state["agent_config_path"]) if state.get("agent_config_path") else None
    )
    if not agent_config.get("enable_llm"):
        return {**state, "explain_draft": _offline_explain(state)}

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return {
            **state,
            "explain_draft": _offline_explain(state)
            + "\n\n(install `quant-agent[llm]` for LLM nodes)",
        }

    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "explain.md"
    system = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else "Explain IC results."
    cfg_path = Path(state["agent_config_path"]) if state.get("agent_config_path") else None
    system += load_pitfalls(agent_config, cfg_path)
    user = (
        f"Project: {state['project']}\nRun dir: {state['run_dir']}\n\n"
        f"IC summary:\n{_format_ic_table(state.get('ic_summary') or [])}\n\n"
        f"Rule findings:\n{_format_findings(state.get('rule_findings') or [])}\n"
    )
    model = ChatOpenAI(model=agent_config.get("model", "gpt-4.1-mini"), temperature=0.2)
    response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return {**state, "explain_draft": str(response.content)}


def llm_skeptic_node(state: ReviewState) -> ReviewState:
    if state.get("offline"):
        return {**state, "skeptic_draft": _offline_skeptic(state)}

    agent_config = load_agent_config(
        Path(state["agent_config_path"]) if state.get("agent_config_path") else None
    )
    if not agent_config.get("enable_llm"):
        return {**state, "skeptic_draft": _offline_skeptic(state)}

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return {**state, "skeptic_draft": _offline_skeptic(state)}

    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "skeptic.md"
    system = prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else "Be skeptical."
    cfg_path = Path(state["agent_config_path"]) if state.get("agent_config_path") else None
    system += load_pitfalls(agent_config, cfg_path)
    user = (
        f"Explain draft:\n{state.get('explain_draft', '')}\n\n"
        f"Rule findings:\n{_format_findings(state.get('rule_findings') or [])}\n"
    )
    model = ChatOpenAI(model=agent_config.get("model", "gpt-4.1-mini"), temperature=0.2)
    response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return {**state, "skeptic_draft": str(response.content)}


def write_report_node(state: ReviewState) -> ReviewState:
    agent_config = load_agent_config(
        Path(state["agent_config_path"]) if state.get("agent_config_path") else None
    )
    run_dir = Path(state["run_dir"])
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = run_dir.name

    report_body = f"""# Quant review — {state['project']}

- Run dir: `{run_dir}`
- Generated: {ts}
- Mode: {"offline" if state.get("offline") else "online"}

## IC summary

{_format_ic_table(state.get("ic_summary") or [])}

## Rule findings

{_format_findings(state.get("rule_findings") or [])}

## Interpretation

{state.get("explain_draft", "")}

## Skeptic notes

{state.get("skeptic_draft", "")}
"""

    log_dir_name = (agent_config.get("output") or {}).get("experiment_log_dir", "experiment-log")
    notes_root = _resolve_notes_root(run_dir)
    log_dir = notes_root / log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / f"review_{run_name}_{ts}.md"
    report_path.write_text(report_body, encoding="utf-8")

    manifest_path = run_dir / "review_manifest.json"
    findings = state.get("rule_findings") or []
    if (agent_config.get("output") or {}).get("write_manifest", True):
        manifest = {
            "project": state["project"],
            "run_dir": str(run_dir),
            "reviewed_at": ts,
            "offline": bool(state.get("offline")),
            "rule_passed": state.get("rule_passed", True),
            "finding_count": len(findings),
            "findings": findings,
            "report_path": str(report_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        **state,
        "final_report": report_body,
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
    }


def _resolve_notes_root(run_dir: Path) -> Path:
    """Walk up from run_dir to find quant-research-notes or fall back to run_dir parent."""
    for parent in [run_dir, *run_dir.parents]:
        if (parent / "experiment-log").is_dir() or parent.name == "quant-research-notes":
            return parent
        if parent.name == "quant_projects":
            candidate = parent / "quant-research-notes"
            if candidate.is_dir():
                return candidate
            break
    return run_dir.parent
