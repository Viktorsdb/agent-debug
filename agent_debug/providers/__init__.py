"""LLM provider factory — pick a backend via env var or config."""

import os

from agent_debug.providers.base import LLMProvider


def get_provider(
    provider: str | None = None,
    **kwargs,
) -> LLMProvider:
    """Return the appropriate LLMProvider.

    Priority:
      1. `provider` argument
      2. AGENT_DEBUG_PROVIDER env var
      3. Auto-detect from available API keys

    Supported values: anthropic, openai, deepseek, ollama
    """
    name = (provider or os.environ.get("AGENT_DEBUG_PROVIDER") or _autodetect()).lower()

    if name == "anthropic":
        from agent_debug.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(**kwargs)

    if name == "openai":
        from agent_debug.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(**kwargs)

    if name == "deepseek":
        from agent_debug.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=kwargs.pop("api_key", None) or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=kwargs.pop("base_url", None) or os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ),
            model=kwargs.pop("model", "deepseek-chat"),
            **kwargs,
        )

    if name == "ollama":
        from agent_debug.providers.ollama_provider import OllamaProvider
        return OllamaProvider(**kwargs)

    raise ValueError(
        f"Unknown provider: {name!r}. "
        "Supported: anthropic, openai, deepseek, ollama\n"
        "Set AGENT_DEBUG_PROVIDER env var or pass provider= argument."
    )


def _autodetect() -> str:
    """Guess provider from available env vars."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    # Default fallback
    return "anthropic"


__all__ = ["LLMProvider", "get_provider"]
