"""Tests for ClaudeAdapter — deterministic, no API calls."""

import json
from pathlib import Path

import pytest

from agent_debug.adapters.claude_adapter import ClaudeAdapter

FIXTURES = Path(__file__).parent.parent / "fixtures" / "sample_traces"


@pytest.fixture
def adapter():
    return ClaudeAdapter()


@pytest.fixture
def premature_stop_trace():
    return json.loads((FIXTURES / "claude_premature_stop.json").read_text())


def test_can_parse_claude_trace(adapter, premature_stop_trace):
    assert adapter.can_parse(premature_stop_trace) is True


def test_cannot_parse_openai_trace(adapter):
    openai_trace = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "messages": [],
    }
    # Claude adapter checks for stop_reason or content blocks — this has neither
    assert adapter.can_parse(openai_trace) is False


def test_parse_trace_id(adapter, premature_stop_trace):
    result = adapter.parse(premature_stop_trace)
    assert result["trace_id"] == "test-claude-001"


def test_parse_sdk_source(adapter, premature_stop_trace):
    result = adapter.parse(premature_stop_trace)
    assert result["sdk_source"] == "claude"


def test_parse_has_tool_call(adapter, premature_stop_trace):
    result = adapter.parse(premature_stop_trace)
    tool_calls = [s for s in result["steps"] if "tool_name" in s]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "bash"


def test_parse_tool_call_input(adapter, premature_stop_trace):
    result = adapter.parse(premature_stop_trace)
    tool_calls = [s for s in result["steps"] if "tool_name" in s]
    assert tool_calls[0]["tool_input"] == {"command": "find /src -name '*.py'"}


def test_parse_tool_output_from_result_block(adapter, premature_stop_trace):
    result = adapter.parse(premature_stop_trace)
    tool_calls = [s for s in result["steps"] if "tool_name" in s]
    assert "main.py" in tool_calls[0]["tool_output"]


def test_parse_has_llm_completion(adapter, premature_stop_trace):
    result = adapter.parse(premature_stop_trace)
    completions = [s for s in result["steps"] if "response" in s]
    assert len(completions) == 1
    assert "3 Python files" in completions[0]["response"]


def test_parse_system_prompt(adapter, premature_stop_trace):
    result = adapter.parse(premature_stop_trace)
    assert "code analysis" in result["system_prompt"]


def test_parse_succeeded_false(adapter, premature_stop_trace):
    result = adapter.parse(premature_stop_trace)
    assert result["succeeded"] is False


def test_parse_step_indices_sequential(adapter, premature_stop_trace):
    result = adapter.parse(premature_stop_trace)
    indices = [s["index"] for s in result["steps"]]
    assert indices == list(range(len(indices)))
