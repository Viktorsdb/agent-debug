"""Anthropic Claude provider."""

import os
from typing import Any

from agent_debug.providers.base import LLMProvider

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")

        kwargs: dict[str, Any] = {
            "api_key": api_key or os.environ.get("ANTHROPIC_API_KEY"),
        }
        url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        if url:
            kwargs["base_url"] = url

        self.client = anthropic.Anthropic(**kwargs)
        self.model = model

    def complete(self, prompt: str, system: str = "") -> tuple[str, int, int]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        msg = self.client.messages.create(**kwargs)
        text_blocks = [b for b in msg.content if getattr(b, "type", None) == "text"]
        if not text_blocks:
            raise RuntimeError("Anthropic returned no text content.")
        return (
            text_blocks[0].text,
            msg.usage.input_tokens,
            msg.usage.output_tokens,
        )
