"""Abstract base class for trace adapters."""

from abc import ABC, abstractmethod
from typing import Any

from agent_debug.models.types import NormalizedTrace


class TraceAdapter(ABC):
    """Convert SDK-specific trace format to NormalizedTrace."""

    @abstractmethod
    def can_parse(self, raw: dict[str, Any]) -> bool:
        """Return True if this adapter can handle the given raw trace."""
        ...

    @abstractmethod
    def parse(self, raw: dict[str, Any]) -> NormalizedTrace:
        """Parse raw trace dict into NormalizedTrace.

        Raises:
            ValueError: if the raw trace is missing required fields.
        """
        ...
