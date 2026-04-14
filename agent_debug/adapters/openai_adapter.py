"""Adapter for OpenAI function-calling traces.

Expected raw format (produced by the agent_debug SDK decorator or manual capture):
{
    "trace_id": "...",
    "task_description": "...",
    "system_prompt": "...",
    "tool_definitions": [...],   # list of OpenAI function defs
    "messages": [...],            # full messages array sent to the API
    "choices": [...],             # raw API response choices
    "succeeded": true,
    "metadata": {}
}
"""

import uuid
from typing import Any

from agent_debug.adapters.base import TraceAdapter
from agent_debug.models.types import LLMCompletion, NormalizedTrace, ToolCall


class OpenAIAdapter(TraceAdapter):
    def can_parse(self, raw: dict[str, Any]) -> bool:
        # OpenAI traces have "choices" with "message" objects
        return (
            isinstance(raw.get("choices"), list)
            and len(raw["choices"]) > 0
            and "message" in raw["choices"][0]
        )

    def parse(self, raw: dict[str, Any]) -> NormalizedTrace:
        trace_id = raw.get("trace_id") or str(uuid.uuid4())
        steps: list[ToolCall | LLMCompletion] = []
        step_index = 0

        messages: list[dict[str, Any]] = raw.get("messages", [])
        choices: list[dict[str, Any]] = raw.get("choices", [])

        # Walk messages to find assistant turns and tool results
        for msg in messages:
            role = msg.get("role", "")
            if role == "tool":
                # Tool result message — look for the matching call in previous step
                # We emit a ToolCall step when we see assistant + tool_calls,
                # then update it when we find the tool result.
                # For simplicity, emit tool call + output together by scanning ahead.
                pass

        # Reconstruct steps from (assistant message with tool_calls) + (tool messages)
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "assistant":
                content = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []

                if tool_calls:
                    # Each tool_call becomes a ToolCall step
                    # Find the matching tool result messages right after
                    tool_results: dict[str, str] = {}
                    j = i + 1
                    while j < len(messages) and messages[j].get("role") == "tool":
                        tool_msg = messages[j]
                        tool_results[tool_msg.get("tool_call_id", "")] = str(
                            tool_msg.get("content", "")
                        )
                        j += 1

                    for tc in tool_calls:
                        call_id = tc.get("id", "")
                        fn = tc.get("function", {})
                        import json
                        try:
                            tool_input = json.loads(fn.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            tool_input = {"raw_arguments": fn.get("arguments", "")}

                        output = tool_results.get(call_id, "")
                        # Detect error: if output starts with "Error" or "error"
                        error = None
                        if output.lower().startswith("error"):
                            error = output

                        steps.append(
                            ToolCall(
                                index=step_index,
                                tool_name=fn.get("name", "unknown"),
                                tool_input=tool_input,
                                tool_output=output,
                                error=error,
                            )
                        )
                        step_index += 1
                    i = j  # skip past tool result messages
                    continue
                else:
                    # Pure text completion
                    # Prompt summary: last user message
                    prompt_summary = _last_user_message(messages[:i])
                    steps.append(
                        LLMCompletion(
                            index=step_index,
                            prompt_summary=prompt_summary[:500],
                            response=content,
                            token_count=None,
                        )
                    )
                    step_index += 1

            i += 1

        # Final output: last choice's content
        final_output = ""
        if choices:
            last_choice = choices[-1]
            final_output = (last_choice.get("message") or {}).get("content") or ""

        # System prompt
        system_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break

        return NormalizedTrace(
            trace_id=trace_id,
            sdk_source="openai",
            task_description=raw.get("task_description", ""),
            system_prompt=system_prompt,
            tool_definitions=raw.get("tool_definitions", []),
            steps=steps,
            final_output=final_output,
            succeeded=bool(raw.get("succeeded", False)),
            metadata=raw.get("metadata", {}),
        )


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-part content
                parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                return " ".join(parts)
            return str(content)
    return ""
