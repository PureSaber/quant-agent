"""LangGraph review layer for quant research runs."""

__version__ = "0.3.2"

from importlib import import_module

from quant_agent.research_history import research_advice, research_history, verify_citations


def __getattr__(name):
    """Offline tools do not require importing an online-model dependency stack."""
    if name in {"build_review_graph", "run_review"}:
        module = import_module("quant_agent.graph")
    else:
        raise AttributeError(name)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "build_review_graph",
    "research_advice",
    "research_history",
    "run_review",
    "verify_citations",
]
