"""Tests for OpenAIAdapter — deterministic, no API calls."""

import json
from pathlib import Path

import pytest

from agent_debug.adapters.openai_adapter import OpenAIAdapter

FIXTURES = Path(__file__).parent.parent / "fixtures" / "sample_traces"


@pytest.fixture
def adapter():
    return OpenAIAdapter()


@pytest.fixture
def wrong_tool_trace():
    return json.loads((FIXTURES / "openai_wrong_tool.json").read_text())


def test_can_parse_openai_trace(adapter, wrong_tool_trace):
    assert adapter.can_parse(wrong_tool_trace) is True


def test_cannot_parse_langchain_trace(adapter):
    langchain = {"intermediate_steps": [], "output": "done"}
    assert adapter.can_parse(langchain) is False


def test_parse_trace_id(adapter, wrong_tool_trace):
    result = adapter.parse(wrong_tool_trace)
    assert result["trace_id"] == "test-openai-001"


def test_parse_sdk_source(adapter, wrong_tool_trace):
    result = adapter.parse(wrong_tool_trace)
    assert result["sdk_source"] == "openai"


def test_parse_system_prompt(adapter, wrong_tool_trace):
    result = adapter.parse(wrong_tool_trace)
    assert "helpful assistant" in result["system_prompt"]


def test_parse_steps_has_tool_call(adapter, wrong_tool_trace):
    result = adapter.parse(wrong_tool_trace)
    tool_calls = [s for s in result["steps"] if "tool_name" in s]
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "web_search"


def test_parse_tool_call_input(adapter, wrong_tool_trace):
    result = adapter.parse(wrong_tool_trace)
    tool_calls = [s for s in result["steps"] if "tool_name" in s]
    assert tool_calls[0]["tool_input"] == {"query": "report.txt contents"}


def test_parse_tool_output(adapter, wrong_tool_trace):
    result = adapter.parse(wrong_tool_trace)
    tool_calls = [s for s in result["steps"] if "tool_name" in s]
    assert "No results" in tool_calls[0]["tool_output"]


def test_parse_succeeded_false(adapter, wrong_tool_trace):
    result = adapter.parse(wrong_tool_trace)
    assert result["succeeded"] is False


def test_parse_final_output(adapter, wrong_tool_trace):
    result = adapter.parse(wrong_tool_trace)
    assert "couldn't find" in result["final_output"]


def test_parse_auto_generates_trace_id_if_missing(adapter, wrong_tool_trace):
    del wrong_tool_trace["trace_id"]
    result = adapter.parse(wrong_tool_trace)
    assert len(result["trace_id"]) > 0


def test_parse_step_indices_are_sequential(adapter, wrong_tool_trace):
    result = adapter.parse(wrong_tool_trace)
    indices = [s["index"] for s in result["steps"]]
    assert indices == list(range(len(indices)))
