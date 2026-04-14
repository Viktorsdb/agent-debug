"""agent-debug: diagnose why your AI agent failed."""

__version__ = "0.1.0"

from agent_debug.pipeline import DiagnosisPipeline
from agent_debug.adapters import auto_parse

__all__ = ["DiagnosisPipeline", "auto_parse"]
