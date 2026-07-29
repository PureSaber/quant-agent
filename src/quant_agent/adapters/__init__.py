"""Adapter registry exports."""

from quant_agent.adapters.base import RunAdapter, RunContext, detect_project, get_adapter
from quant_agent.adapters.futures_spread import FuturesSpreadAdapter
from quant_agent.adapters.multifactor import MultifactorAdapter

__all__ = [
    "FuturesSpreadAdapter",
    "MultifactorAdapter",
    "RunAdapter",
    "RunContext",
    "detect_project",
    "get_adapter",
]
