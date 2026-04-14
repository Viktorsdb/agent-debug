"""Tests for BaseAgent error handling — mocked, no real API calls."""

import json
import pytest
from unittest.mock import MagicMock, patch

from agent_debug.agents.base import BaseAgent, _try_extract_json


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


# ─── BaseAgent.parse_json with retry ─────────────────────────────────────────

class ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing."""
    pass


def _make_mock_client(responses: list[str]):
    """Return a mock anthropic.Anthropic client that returns responses in order."""
    client = MagicMock()
    messages_mock = MagicMock()
    client.messages = messages_mock

    call_count = [0]
    def create(**kwargs):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        msg = MagicMock()
        msg.content = [MagicMock(text=responses[idx])]
        msg.usage = MagicMock(input_tokens=100, output_tokens=50)
        return msg
    messages_mock.create = create
    return client


def test_parse_json_success_first_try():
    client = _make_mock_client(['{"result": "ok"}'])
    agent = ConcreteAgent(client)
    result = agent.parse_json('{"result": "ok"}', "dummy prompt")
    assert result == {"result": "ok"}


def test_parse_json_retry_on_invalid():
    # parse_json receives the initial response as `text` (no API call for first attempt).
    # The retry is the first (and only) API call — it returns valid JSON.
    client = _make_mock_client(['{"result": "ok"}'])
    agent = ConcreteAgent(client)
    result = agent.parse_json("not json", "dummy prompt")
    assert result == {"result": "ok"}


def test_parse_json_raises_after_two_failures():
    client = _make_mock_client(["not json", "still not json"])
    agent = ConcreteAgent(client)
    with pytest.raises(ValueError, match="invalid JSON after retry"):
        agent.parse_json("not json", "dummy prompt")


def test_last_cost_usd_calculated():
    client = _make_mock_client(['{"x": 1}'])
    agent = ConcreteAgent(client)
    agent._sync_call("hello", "")
    assert agent.last_cost_usd > 0
