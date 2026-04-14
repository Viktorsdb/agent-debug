"""Three ways to capture agent traces automatically."""

from agent_debug.capture.context import capture
from agent_debug.capture.decorators import trace
from agent_debug.capture.patches import (
    patch_openai,
    patch_anthropic,
    unpatch_openai,
    unpatch_anthropic,
    get_last_trace,
    save_trace,
)
from agent_debug.capture.recorder import TraceRecorder

__all__ = [
    # Method 1: decorator
    "trace",
    # Method 2: patch
    "patch_openai",
    "patch_anthropic",
    "unpatch_openai",
    "unpatch_anthropic",
    "get_last_trace",
    "save_trace",
    # Method 3: context manager
    "capture",
    # Core
    "TraceRecorder",
]
