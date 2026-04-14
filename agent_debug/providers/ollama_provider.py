"""Ollama provider for local models (completely free).

Requires Ollama running locally: https://ollama.ai
  ollama pull llama3
  ollama serve
"""

import os

from agent_debug.providers.base import LLMProvider

DEFAULT_MODEL = "llama3"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai  (used as Ollama client)")

        url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
        from openai import OpenAI
        self.client = OpenAI(base_url=f"{url}/v1", api_key="ollama")
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
