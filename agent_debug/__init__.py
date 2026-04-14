"""agent-debug: diagnose why your AI agent failed."""

__version__ = "0.2.0"

from agent_debug.pipeline import DiagnosisPipeline
from agent_debug.adapters import auto_parse

# Capture — three ways to record traces automatically
from agent_debug.capture import (
    trace,           # @agent_debug.trace decorator
    capture,         # with agent_debug.capture() context manager
    patch_openai,    # agent_debug.patch_openai() zero-touch patch
    patch_anthropic,
    unpatch_openai,
    unpatch_anthropic,
    get_last_trace,
    save_trace,
)

__all__ = [
    "DiagnosisPipeline",
    "auto_parse",
    # Capture
    "trace",
    "capture",
    "patch_openai",
    "patch_anthropic",
    "unpatch_openai",
    "unpatch_anthropic",
    "get_last_trace",
    "save_trace",
]
