"""Tests for auto_parse dispatch."""

import pytest

from agent_debug.adapters import auto_parse


def test_auto_parse_openai():
    raw = {
        "choices": [{"message": {"role": "assistant", "content": "done"}}],
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "done"},
        ],
        "succeeded": True,
        "task_description": "Say hello",
    }
    trace = auto_parse(raw)
    assert trace["sdk_source"] == "openai"


def test_auto_parse_langchain():
    raw = {
        "intermediate_steps": [],
        "input": "Do something",
        "output": "Done",
        "succeeded": True,
    }
    trace = auto_parse(raw)
    assert trace["sdk_source"] == "langchain"


def test_auto_parse_unknown_raises():
    raw = {"completely": "unknown", "format": True}
    with pytest.raises(ValueError, match="Unrecognized trace format"):
        auto_parse(raw)
