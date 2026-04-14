"""Core trace recorder — collects API calls into a NormalizedTrace-compatible dict."""

import json
import time
import uuid
from pathlib import Path
from typing import Any


class TraceRecorder:
    """Records LLM API calls into a trace dict that agent-debug can analyze.

    Not used directly — use patch_openai(), patch_anthropic(), @trace, or capture().
    """

    def __init__(
        self,
        task_description: str = "",
        output: str | Path | None = None,
        auto_save: bool = True,
    ):
        self.task_description = task_description
        self.output = Path(output) if output else None
        self.auto_save = auto_save

        self._trace_id = str(uuid.uuid4())
        self._messages: list[dict] = []
        self._tool_definitions: list[dict] = []
        self._system_prompt: str = ""
        self._final_response: dict | None = None
        self._sdk_source: str = "unknown"
        self._started_at = time.time()
        self._succeeded: bool | None = None
        self._metadata: dict[str, Any] = {}

    # ── Recording API ──────────────────────────────────────────────────────

    def record_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def record_tool_definitions(self, tools: list[dict]) -> None:
        self._tool_definitions = tools

    def record_message(self, message: dict) -> None:
        self._messages.append(message)

    def record_final_response(self, response: dict) -> None:
        self._final_response = response

    def set_sdk_source(self, source: str) -> None:
        self._sdk_source = source

    def set_succeeded(self, succeeded: bool) -> None:
        self._succeeded = succeeded

    def set_task_description(self, description: str) -> None:
        self.task_description = description

    # ── Output ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Build the raw trace dict (auto-detected format by adapters)."""
        elapsed = round(time.time() - self._started_at, 2)
        succeeded = self._succeeded if self._succeeded is not None else False

        base = {
            "trace_id": self._trace_id,
            "task_description": self.task_description,
            "system_prompt": self._system_prompt,
            "tool_definitions": self._tool_definitions,
            "messages": self._messages,
            "succeeded": succeeded,
            "metadata": {
                **self._metadata,
                "elapsed_seconds": elapsed,
                "sdk_source": self._sdk_source,
            },
        }

        # Add SDK-specific fields so adapters can auto-detect
        if self._sdk_source == "openai":
            base["choices"] = (
                [{"message": self._final_response}] if self._final_response else []
            )
        elif self._sdk_source == "claude":
            base["final_response"] = self._final_response or {
                "stop_reason": "end_turn",
                "content": [],
                "usage": {},
            }
        # langchain: intermediate_steps already in messages

        return base

    def save(self, path: str | Path | None = None) -> Path:
        """Save trace to JSON file. Returns the path written."""
        target = Path(path or self.output or f"agent_traces/{self._trace_id}.trace.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        return target

    def analyze(self) -> "DiagnosisReport":  # noqa: F821
        """Run agent-debug analysis immediately and return the report."""
        from agent_debug.pipeline import DiagnosisPipeline
        pipeline = DiagnosisPipeline()
        return pipeline.run(self.to_dict())
