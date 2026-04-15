"""Base class for all analysis agents.

Three critical error handlers:
1. Unknown/invalid input format  → ValueError with friendly message
2. LLM returns invalid JSON      → retry once, then raise with partial output
3. API timeout (30s per call)    → RuntimeError
"""

import concurrent.futures
import json
import re
from typing import Any

from agent_debug.providers.base import LLMProvider
from agent_debug.providers import get_provider

TIMEOUT_SECONDS = 120

# Fallback cost estimate (claude-sonnet-4-6 pricing)
COST_PER_INPUT_TOKEN = 0.000003
COST_PER_OUTPUT_TOKEN = 0.000015


class BaseAgent:
    """Abstract base for all agents.

    Args:
        provider: LLMProvider instance. If None, auto-detected from env vars.
    """

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider or get_provider()
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def call(self, prompt: str, system: str = "") -> str:
        """Call the LLM with a 30-second timeout. Returns raw text response."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._sync_call, prompt, system)
            try:
                return future.result(timeout=TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise RuntimeError(
                    f"LLM call timed out after {TIMEOUT_SECONDS}s. "
                    "Check your network or increase timeout."
                )

    def _sync_call(self, prompt: str, system: str) -> str:
        text, input_tokens, output_tokens = self.provider.complete(prompt, system)
        self.last_input_tokens = input_tokens
        self.last_output_tokens = output_tokens
        return text

    def parse_json(self, text: str, prompt: str, system: str = "") -> dict[str, Any]:
        """Extract JSON from text. Retries once if parsing fails."""
        parsed = _try_extract_json(text)
        if parsed is not None:
            return parsed

        retry_prompt = (
            prompt
            + "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Reply with ONLY a valid JSON object, no markdown fences, no explanation."
        )
        retry_text = self.call(retry_prompt, system)
        parsed = _try_extract_json(retry_text)
        if parsed is not None:
            return parsed

        raise ValueError(
            f"LLM returned invalid JSON after retry.\n"
            f"Last response (first 500 chars): {retry_text[:500]}"
        )

    @property
    def last_cost_usd(self) -> float:
        return (
            self.last_input_tokens * COST_PER_INPUT_TOKEN
            + self.last_output_tokens * COST_PER_OUTPUT_TOKEN
        )


def _try_extract_json(text: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from text (handles markdown fences)."""
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
