"""Method 1: @agent_debug.trace decorator.

Usage:
    import agent_debug

    @agent_debug.trace(output="agent_traces/my_agent.trace.json")
    def run_my_agent(task: str):
        client = openai.OpenAI()
        # ... your agent code ...
        return result

    # Or with auto-naming:
    @agent_debug.trace
    def run_my_agent(task):
        ...
"""

import functools
import traceback
from pathlib import Path
from typing import Any, Callable

from agent_debug.capture.patches import patch_openai, patch_anthropic, unpatch_openai, unpatch_anthropic, get_last_trace


def trace(
    func: Callable | None = None,
    *,
    output: str | Path | None = None,
    provider: str = "auto",
    task_description: str = "",
    save: bool = True,
    analyze: bool = False,
):
    """Decorator that captures all LLM API calls made inside the function.

    Can be used with or without arguments:
        @agent_debug.trace
        def run(): ...

        @agent_debug.trace(output="traces/run.json", analyze=True)
        def run(): ...

    Args:
        output: Where to save the trace JSON. Defaults to agent_traces/<func_name>.trace.json
        provider: Which SDK to patch: "openai", "anthropic", or "auto" (tries both)
        task_description: Human-readable description of what the agent is doing
        save: Whether to save the trace to disk (default True)
        analyze: Whether to immediately run agent-debug analysis after the call
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Determine output path
            out_path = Path(output) if output else Path(f"agent_traces/{fn.__name__}.trace.json")
            desc = task_description or fn.__name__

            # Patch SDK(s)
            _patch(provider)

            recorder = get_last_trace()
            if recorder:
                recorder.set_task_description(desc)
                recorder.output = out_path

            succeeded = False
            result = None
            try:
                result = fn(*args, **kwargs)
                succeeded = True
                return result
            except Exception as e:
                if recorder:
                    recorder._metadata["error"] = str(e)
                    recorder._metadata["traceback"] = traceback.format_exc()
                raise
            finally:
                if recorder:
                    recorder.set_succeeded(succeeded)
                    if save:
                        saved = recorder.save(out_path)
                        print(f"[agent-debug] Trace saved → {saved}")
                    if analyze:
                        try:
                            report = recorder.analyze()
                            _print_quick_summary(report)
                        except Exception as e:
                            print(f"[agent-debug] Analysis failed: {e}")
                _unpatch(provider)

        return wrapper

    # Support both @trace and @trace(...)
    if func is not None:
        return decorator(func)
    return decorator


def _patch(provider: str) -> None:
    if provider in ("openai", "auto"):
        try:
            patch_openai()
        except ImportError:
            pass
    if provider in ("anthropic", "auto"):
        try:
            patch_anthropic()
        except ImportError:
            pass


def _unpatch(provider: str) -> None:
    if provider in ("openai", "auto"):
        unpatch_openai()
    if provider in ("anthropic", "auto"):
        unpatch_anthropic()


def _print_quick_summary(report: dict) -> None:
    c = report["classification"]
    s = report["severity"]
    print(
        f"[agent-debug] {c['subcategory']} | severity {s['severity']}/5 | "
        f"confidence {c['confidence']:.0%}"
    )
