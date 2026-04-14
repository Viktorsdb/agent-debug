"""Adapter for LangChain agent traces.

Expected raw format (from LangChain callbacks or AgentExecutor return):
{
    "trace_id": "...",
    "task_description": "...",
    "system_prompt": "...",
    "tool_definitions": [...],
    "intermediate_steps": [
        [{"tool": "...", "tool_input": "..."}, "tool_output_string"],
        ...
    ],
    "input": "...",         # original user input
    "output": "...",        # final agent output
    "succeeded": true,
    "metadata": {}
}

LangChain's AgentExecutor stores each step as (AgentAction, observation) tuple.
We serialize these as [[action_dict, observation_string], ...].
"""

import uuid
from typing import Any

from agent_debug.adapters.base import TraceAdapter
from agent_debug.models.types import LLMCompletion, NormalizedTrace, ToolCall


class LangChainAdapter(TraceAdapter):
    def can_parse(self, raw: dict[str, Any]) -> bool:
        return "intermediate_steps" in raw and isinstance(
            raw["intermediate_steps"], list
        )

    def parse(self, raw: dict[str, Any]) -> NormalizedTrace:
        trace_id = raw.get("trace_id") or str(uuid.uuid4())
        steps: list[ToolCall | LLMCompletion] = []
        step_index = 0

        intermediate_steps: list[Any] = raw.get("intermediate_steps", [])

        for item in intermediate_steps:
            # item is [action_dict_or_obj, observation]
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            action, observation = item[0], item[1]

            # action may be a dict or a serialized AgentAction
            if isinstance(action, dict):
                tool_name = action.get("tool") or action.get("tool_name", "unknown")
                tool_input_raw = action.get("tool_input", action.get("input", ""))
            else:
                # Fallback: stringify
                tool_name = str(getattr(action, "tool", "unknown"))
                tool_input_raw = str(getattr(action, "tool_input", ""))

            # Normalize tool_input to dict
            if isinstance(tool_input_raw, dict):
                tool_input = tool_input_raw
            elif isinstance(tool_input_raw, str):
                tool_input = {"input": tool_input_raw}
            else:
                tool_input = {"input": str(tool_input_raw)}

            output = str(observation) if observation is not None else ""
            error = None
            if output.lower().startswith("error") or output.lower().startswith("exception"):
                error = output

            steps.append(
                ToolCall(
                    index=step_index,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=output,
                    error=error,
                )
            )
            step_index += 1

        # Final LLM completion with the agent's answer
        final_output = raw.get("output", "")
        if final_output:
            steps.append(
                LLMCompletion(
                    index=step_index,
                    prompt_summary=str(raw.get("input", ""))[:500],
                    response=final_output,
                    token_count=None,
                )
            )

        return NormalizedTrace(
            trace_id=trace_id,
            sdk_source="langchain",
            task_description=raw.get("task_description") or raw.get("input", ""),
            system_prompt=raw.get("system_prompt", ""),
            tool_definitions=raw.get("tool_definitions", []),
            steps=steps,
            final_output=final_output,
            succeeded=bool(raw.get("succeeded", False)),
            metadata=raw.get("metadata", {}),
        )
