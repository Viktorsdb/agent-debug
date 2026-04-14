"""OpenAI (and OpenAI-compatible) provider.

Works with:
  - OpenAI (gpt-4o, gpt-4o-mini)
  - DeepSeek (deepseek-chat)  → set base_url=https://api.deepseek.com
  - Any third-party OpenAI-compatible relay
"""

import os
from typing import Any

from agent_debug.providers.base import LLMProvider

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        kwargs: dict[str, Any] = {
            "api_key": api_key or os.environ.get("OPENAI_API_KEY"),
        }
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        if url:
            kwargs["base_url"] = url

        self.client = OpenAI(**kwargs)
        self.model = model

    def complete(self, prompt: str, system: str = "") -> tuple[str, int, int]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return (
            text,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
