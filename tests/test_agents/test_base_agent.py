"""Tests for BaseAgent error handling — mocked, no real API calls."""

import pytest
from unittest.mock import MagicMock

from agent_debug.agents.base import BaseAgent, _try_extract_json
from agent_debug.providers.base import LLMProvider


# ─── _try_extract_json ────────────────────────────────────────────────────────

def test_extract_plain_json():
    text = '{"category": "wrong_tool", "confidence": 0.9}'
    result = _try_extract_json(text)
    assert result == {"category": "wrong_tool", "confidence": 0.9}


def test_extract_json_with_markdown_fence():
    text = '```json\n{"severity": 3}\n```'
    result = _try_extract_json(text)
    assert result == {"severity": 3}


def test_extract_json_embedded_in_text():
    text = 'Here is the answer: {"score": 5, "reason": "bad"} end.'
    result = _try_extract_json(text)
    assert result["score"] == 5


def test_extract_json_invalid_returns_none():
    result = _try_extract_json("This is not JSON at all")
    assert result is None


# ─── Mock provider ────────────────────────────────────────────────────────────

class MockProvider(LLMProvider):
    """Returns responses in order, then repeats the last one."""
    def __init__(self, responses: list[str]):
        self._responses = responses
        self._idx = 0

    def complete(self, prompt: str, system: str = "") -> tuple[str, int, int]:
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return text, 100, 50


class ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing."""
    pass


# ─── BaseAgent.parse_json with retry ─────────────────────────────────────────

def test_parse_json_success_first_try():
    agent = ConcreteAgent(MockProvider(['{"result": "ok"}']))
    result = agent.parse_json('{"result": "ok"}', "dummy prompt")
    assert result == {"result": "ok"}


def test_parse_json_retry_on_invalid():
    # parse_json receives the initial response as `text` (no API call for first attempt).
    # The retry is the first (and only) API call — it returns valid JSON.
    agent = ConcreteAgent(MockProvider(['{"result": "ok"}']))
    result = agent.parse_json("not json", "dummy prompt")
    assert result == {"result": "ok"}


def test_parse_json_raises_after_two_failures():
    agent = ConcreteAgent(MockProvider(["not json", "still not json"]))
    with pytest.raises(ValueError, match="invalid JSON after retry"):
        agent.parse_json("not json", "dummy prompt")


def test_last_cost_usd_calculated():
    agent = ConcreteAgent(MockProvider(['{"x": 1}']))
    agent._sync_call("hello", "")
    assert agent.last_cost_usd > 0
