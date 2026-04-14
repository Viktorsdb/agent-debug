"""Base class for all Claude-powered analysis agents.

Three critical error handlers:
1. Unknown/invalid input format  → ValueError with friendly message
2. Claude returns invalid JSON    → retry once, then raise with partial output
3. API timeout (30s per call)     → asyncio.TimeoutError wrapped as RuntimeError
"""

import asyncio
import json
import re
from typing import Any, TypeVar

import os

import anthropic

MODEL = "claude-sonnet-4-6"
TIMEOUT_SECONDS = 30


def _default_client() -> anthropic.Anthropic:
    """Create Anthropic client, respecting ANTHROPIC_BASE_URL if set."""
    kwargs: dict = {}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)

T = TypeVar("T")

# Approximate token cost for claude-sonnet-4-6:
# Input: $3 / 1M tokens = $0.000003 / token
# Output: $15 / 1M tokens = $0.000015 / token
COST_PER_INPUT_TOKEN = 0.000003
COST_PER_OUTPUT_TOKEN = 0.000015


class BaseAgent:
    """Abstract base for all agents. Subclasses implement `_build_prompt` and `_parse_output`."""

    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client or _default_client()
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    def call(self, prompt: str, system: str = "") -> str:
        """Call Claude with a 30-second timeout. Returns raw text response.

        Raises:
            RuntimeError: on timeout or API error
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — use a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self._sync_call, prompt, system)
                    return future.result(timeout=TIMEOUT_SECONDS)
            else:
                return loop.run_until_complete(
                    asyncio.wait_for(
                        asyncio.to_thread(self._sync_call, prompt, system),
                        timeout=TIMEOUT_SECONDS,
                    )
                )
        except (asyncio.TimeoutError, TimeoutError):
            raise RuntimeError(
                f"Claude API call timed out after {TIMEOUT_SECONDS}s. "
                "Check your network connection or try again."
            )

    def _sync_call(self, prompt: str, system: str) -> str:
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        msg = self.client.messages.create(**kwargs)
        self.last_input_tokens = msg.usage.input_tokens
        self.last_output_tokens = msg.usage.output_tokens
        # Filter out ThinkingBlock — only take TextBlock content
        text_blocks = [b for b in msg.content if getattr(b, "type", None) == "text"]
        if not text_blocks:
            raise RuntimeError("Claude returned no text content in response.")
        return text_blocks[0].text

    def parse_json(self, text: str, prompt: str, system: str = "") -> dict[str, Any]:
        """Extract JSON from text. Retries once if parsing fails.

        Raises:
            ValueError: if JSON cannot be extracted after retry
        """
        parsed = _try_extract_json(text)
        if parsed is not None:
            return parsed

        # Retry once with explicit JSON instruction
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
            f"Claude returned invalid JSON after retry.\n"
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
    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try to find the first { ... } block
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
