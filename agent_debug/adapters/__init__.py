"""Trace adapters: convert SDK-specific formats to NormalizedTrace."""

from agent_debug.adapters.base import TraceAdapter
from agent_debug.adapters.claude_adapter import ClaudeAdapter
from agent_debug.adapters.langchain_adapter import LangChainAdapter
from agent_debug.adapters.openai_adapter import OpenAIAdapter

_ADAPTERS: list[TraceAdapter] = [
    ClaudeAdapter(),
    OpenAIAdapter(),
    LangChainAdapter(),
]


def auto_parse(raw: dict) -> "NormalizedTrace":  # noqa: F821
    """Try each adapter in order; raise ValueError if none can parse."""
    for adapter in _ADAPTERS:
        if adapter.can_parse(raw):
            return adapter.parse(raw)
    raise ValueError(
        "Unrecognized trace format. Supported: openai, claude, langchain. "
        "See https://github.com/Viktorsdb/agent-debug#trace-format for details."
    )


__all__ = [
    "TraceAdapter",
    "OpenAIAdapter",
    "ClaudeAdapter",
    "LangChainAdapter",
    "auto_parse",
]
