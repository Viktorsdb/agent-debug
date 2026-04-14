"""Adapter for Anthropic Claude SDK traces.

Expected raw format:
{
    "trace_id": "...",
    "task_description": "...",
    "system_prompt": "...",
    "tool_definitions": [...],
    "messages": [...],         # list of {role, content} where content can be a list
    "final_response": {...},   # the last Message object from the SDK
    "succeeded": true,
    "metadata": {}
}

Claude content blocks look like:
  {"type": "text", "text": "..."}
  {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
  {"type": "tool_result", "tool_use_id": "...", "content": "..."}
"""

import uuid
from typing import Any

from agent_debug.adapters.base import TraceAdapter
from agent_debug.models.types import LLMCompletion, NormalizedTrace, ToolCall


class ClaudeAdapter(TraceAdapter):
    def can_parse(self, raw: dict[str, Any]) -> bool:
        # Claude traces have messages where content may be a list of blocks
        messages = raw.get("messages", [])
        if not messages:
            return False
        # Look for Claude-style content blocks
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list) and len(content) > 0:
                block = content[0]
                if isinstance(block, dict) and block.get("type") in (
                    "text", "tool_use", "tool_result"
                ):
                    return True
        # Also accept if final_response has stop_reason field (Claude SDK)
        return "stop_reason" in raw.get("final_response", {})

    def parse(self, raw: dict[str, Any]) -> NormalizedTrace:
        trace_id = raw.get("trace_id") or str(uuid.uuid4())
        steps: list[ToolCall | LLMCompletion] = []
        step_index = 0

        messages: list[dict[str, Any]] = raw.get("messages", [])

        # Build tool_result lookup: tool_use_id → output text
        tool_results: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_result":
                            tid = block.get("tool_use_id", "")
                            result_content = block.get("content", "")
                            if isinstance(result_content, list):
                                result_content = " ".join(
                                    b.get("text", "") for b in result_content
                                    if b.get("type") == "text"
                                )
                            tool_results[tid] = str(result_content)

        # Walk assistant messages for tool_use and text blocks
        last_text_response = ""
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            for block in content:
                btype = block.get("type")
                if btype == "tool_use":
                    tool_id = block.get("id", "")
                    output = tool_results.get(tool_id, "")
                    error = None
                    if output.lower().startswith("error"):
                        error = output
                    steps.append(
                        ToolCall(
                            index=step_index,
                            tool_name=block.get("name", "unknown"),
                            tool_input=block.get("input", {}),
                            tool_output=output,
                            error=error,
                        )
                    )
                    step_index += 1
                elif btype == "text":
                    text = block.get("text", "")
                    if text.strip():
                        prompt_summary = _last_user_text(messages)
                        steps.append(
                            LLMCompletion(
                                index=step_index,
                                prompt_summary=prompt_summary[:500],
                                response=text,
                                token_count=None,
                            )
                        )
                        step_index += 1
                        last_text_response = text

        # Final output from explicit final_response or last text
        final_response = raw.get("final_response", {})
        final_output = last_text_response
        if final_response:
            fr_content = final_response.get("content", [])
            if isinstance(fr_content, list):
                texts = [b.get("text", "") for b in fr_content if b.get("type") == "text"]
                if texts:
                    final_output = texts[-1]
            elif isinstance(fr_content, str):
                final_output = fr_content

        # Usage for token count (attach to last LLMCompletion)
        usage = final_response.get("usage", {})
        total_tokens = None
        if usage:
            total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if total_tokens and steps:
            for step in reversed(steps):
                if step.get("response") is not None:  # is LLMCompletion
                    step["token_count"] = total_tokens  # type: ignore[index]
                    break

        return NormalizedTrace(
            trace_id=trace_id,
            sdk_source="claude",
            task_description=raw.get("task_description", ""),
            system_prompt=raw.get("system_prompt", ""),
            tool_definitions=raw.get("tool_definitions", []),
            steps=steps,
            final_output=final_output,
            succeeded=bool(raw.get("succeeded", False)),
            metadata=raw.get("metadata", {}),
        )


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            if parts:
                return " ".join(parts)
    return ""
