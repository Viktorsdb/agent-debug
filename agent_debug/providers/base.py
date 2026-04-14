"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Unified interface for any LLM backend."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> tuple[str, int, int]:
        """Send prompt, return (text, input_tokens, output_tokens).

        Raises:
            RuntimeError: on timeout or API error
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
