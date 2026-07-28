"""LangGraph review graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from quant_agent.nodes.pipeline import (
    llm_explain_node,
    llm_skeptic_node,
    load_run_node,
    rule_check_node,
    write_report_node,
)
from quant_agent.state import ReviewState


def build_review_graph():
    graph = StateGraph(ReviewState)
    graph.add_node("load", load_run_node)
    graph.add_node("rules", rule_check_node)
    graph.add_node("explain", llm_explain_node)
    graph.add_node("skeptic", llm_skeptic_node)
    graph.add_node("write", write_report_node)

    graph.add_edge(START, "load")
    graph.add_edge("load", "rules")
    graph.add_edge("rules", "explain")
    graph.add_edge("explain", "skeptic")
    graph.add_edge("skeptic", "write")
    graph.add_edge("write", END)

    return graph.compile()


def run_review(
    *,
    project: str,
    run_dir: str | Path,
    config_path: str | Path | None = None,
    agent_config_path: str | Path | None = None,
    offline: bool = True,
) -> dict[str, Any]:
    app = build_review_graph()
    initial: ReviewState = {
        "project": project,
        "run_dir": str(Path(run_dir).resolve()),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "agent_config_path": str(Path(agent_config_path).resolve()) if agent_config_path else "",
        "offline": offline,
    }
    return app.invoke(initial)
