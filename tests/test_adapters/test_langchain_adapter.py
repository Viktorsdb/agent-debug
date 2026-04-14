"""Tests for LangChainAdapter — deterministic, no API calls."""

import json
from pathlib import Path

import pytest

from agent_debug.adapters.langchain_adapter import LangChainAdapter

FIXTURES = Path(__file__).parent.parent / "fixtures" / "sample_traces"


@pytest.fixture
def adapter():
    return LangChainAdapter()


@pytest.fixture
def misinterpretation_trace():
    return json.loads((FIXTURES / "langchain_tool_misinterpretation.json").read_text())


def test_can_parse_langchain_trace(adapter, misinterpretation_trace):
    assert adapter.can_parse(misinterpretation_trace) is True


def test_cannot_parse_openai_trace(adapter):
    openai = {"choices": [{"message": {"content": "hi"}}], "messages": []}
    assert adapter.can_parse(openai) is False


def test_parse_trace_id(adapter, misinterpretation_trace):
    result = adapter.parse(misinterpretation_trace)
    assert result["trace_id"] == "test-langchain-001"


def test_parse_sdk_source(adapter, misinterpretation_trace):
    result = adapter.parse(misinterpretation_trace)
    assert result["sdk_source"] == "langchain"


def test_parse_two_tool_calls(adapter, misinterpretation_trace):
    result = adapter.parse(misinterpretation_trace)
    tool_calls = [s for s in result["steps"] if "tool_name" in s]
    assert len(tool_calls) == 2


def test_parse_tool_errors_detected(adapter, misinterpretation_trace):
    result = adapter.parse(misinterpretation_trace)
    tool_calls = [s for s in result["steps"] if "tool_name" in s]
    assert all(tc["error"] is not None for tc in tool_calls)


def test_parse_final_output(adapter, misinterpretation_trace):
    result = adapter.parse(misinterpretation_trace)
    assert "182.50" in result["final_output"]


def test_parse_task_description_fallback(adapter, misinterpretation_trace):
    result = adapter.parse(misinterpretation_trace)
    assert result["task_description"] == "Get the current stock price of AAPL"


def test_parse_step_indices_sequential(adapter, misinterpretation_trace):
    result = adapter.parse(misinterpretation_trace)
    indices = [s["index"] for s in result["steps"]]
    assert indices == list(range(len(indices)))
