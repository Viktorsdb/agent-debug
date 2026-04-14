"""Method 3: context manager — capture() with block.

Usage:
    import agent_debug

    with agent_debug.capture("agent_traces/run.json") as trace:
        result = my_agent.run(task)

    # trace.json is saved automatically when the with block exits.
    # Access the recorded data:
    print(trace.to_dict())

    # Or analyze immediately:
    with agent_debug.capture("traces/run.json", analyze=True) as trace:
        result = my_agent.run(task)
    # Prints diagnosis to terminal after the block.
"""

import traceback
from pathlib import Path
from typing import Any

from agent_debug.capture.patches import (
    patch_openai, patch_anthropic,
    unpatch_openai, unpatch_anthropic,
    get_last_trace,
)
from agent_debug.capture.recorder import TraceRecorder


class capture:
    """Context manager that records all LLM API calls within the block.

    Example:
        with agent_debug.capture("traces/my_run.json") as t:
            result = client.chat.completions.create(...)

        # File saved, access recorder:
        print(t.to_dict())
        report = t.analyze()
    """

    def __init__(
        self,
        output: str | Path | None = None,
        task_description: str = "",
        provider: str = "auto",
        save: bool = True,
        analyze: bool = False,
    ):
        """
        Args:
            output: Path to save the trace JSON.
                    Defaults to agent_traces/<uuid>.trace.json
            task_description: What the agent is trying to do.
            provider: Which SDK to intercept: "openai", "anthropic", or "auto".
            save: Automatically save trace when exiting the block.
            analyze: Run diagnosis immediately after the block and print summary.
        """
        self.output = Path(output) if output else None
        self.task_description = task_description
        self.provider = provider
        self.save = save
        self.analyze = analyze
        self._recorder: TraceRecorder | None = None

    def __enter__(self) -> TraceRecorder:
        # Start patching
        if self.provider in ("openai", "auto"):
            try:
                patch_openai()
            except ImportError:
                pass
        if self.provider in ("anthropic", "auto"):
            try:
                patch_anthropic()
            except ImportError:
                pass

        self._recorder = get_last_trace()
        if self._recorder:
            if self.task_description:
                self._recorder.set_task_description(self.task_description)
            if self.output:
                self._recorder.output = self.output

        # Return the recorder so users can do:
        #   with capture("out.json") as t:
        #       ...
        #   t.to_dict()
        return self._recorder or TraceRecorder()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        succeeded = exc_type is None

        if self._recorder:
            self._recorder.set_succeeded(succeeded)
            if exc_val:
                self._recorder._metadata["error"] = str(exc_val)
                self._recorder._metadata["traceback"] = traceback.format_exc()

            if self.save:
                saved = self._recorder.save(self.output)
                print(f"[agent-debug] Trace saved → {saved}")

            if self.analyze and succeeded:
                try:
                    report = self._recorder.analyze()
                    c = report["classification"]
                    s = report["severity"]
                    print(
                        f"[agent-debug] {c['subcategory']} | "
                        f"severity {s['severity']}/5 | "
                        f"confidence {c['confidence']:.0%}"
                    )
                except Exception as e:
                    print(f"[agent-debug] Analysis failed: {e}")

        # Stop patching
        if self.provider in ("openai", "auto"):
            unpatch_openai()
        if self.provider in ("anthropic", "auto"):
            unpatch_anthropic()

        return False  # don't suppress exceptions
