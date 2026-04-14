"""Method 2: monkey-patch OpenAI/Anthropic clients — zero code changes required.

Usage:
    import agent_debug
    agent_debug.patch_openai()          # patch default client
    agent_debug.patch_anthropic()       # patch default client

    # All subsequent API calls are recorded automatically.
    # Call agent_debug.get_last_trace() to get the recorded trace.
    # Call agent_debug.save_trace("out.json") to save it.
    # Call agent_debug.unpatch_openai() to stop recording.
"""

import functools
import json
from typing import Any

from agent_debug.capture.recorder import TraceRecorder

# Global recorder for patch-mode
_active_recorder: TraceRecorder | None = None
_openai_patched = False
_anthropic_patched = False
_original_openai_create = None
_original_anthropic_create = None


def get_last_trace() -> TraceRecorder | None:
    return _active_recorder


def save_trace(path: str, task_description: str = "") -> None:
    if _active_recorder is None:
        raise RuntimeError("No active trace. Call patch_openai() or patch_anthropic() first.")
    if task_description:
        _active_recorder.set_task_description(task_description)
    saved = _active_recorder.save(path)
    print(f"[agent-debug] Trace saved to {saved}")


# ── OpenAI patch ───────────────────────────────────────────────────────────

def patch_openai(output_dir: str = "agent_traces") -> None:
    """Monkey-patch OpenAI client to auto-record all chat.completions.create calls."""
    global _openai_patched, _active_recorder, _original_openai_create

    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    if _openai_patched:
        return

    _active_recorder = TraceRecorder(output=f"{output_dir}/trace.json")
    _active_recorder.set_sdk_source("openai")
    original = openai.resources.chat.completions.Completions.create

    @functools.wraps(original)
    def patched_create(self_inner, *args, **kwargs):
        # Extract and record the request
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools", [])

        for msg in messages:
            if msg.get("role") == "system":
                _active_recorder.record_system_prompt(msg.get("content", ""))
            _active_recorder.record_message(dict(msg))

        if tools:
            _active_recorder.record_tool_definitions(list(tools))

        # Call original
        response = original(self_inner, *args, **kwargs)

        # Record response
        resp_dict = _response_to_dict_openai(response)
        _active_recorder.record_final_response(resp_dict.get("message", {}))

        # Record assistant message + any tool results
        assistant_msg = resp_dict.get("message", {})
        if assistant_msg:
            _active_recorder.record_message(assistant_msg)

        return response

    _original_openai_create = original
    openai.resources.chat.completions.Completions.create = patched_create
    _openai_patched = True
    print("[agent-debug] OpenAI client patched. All chat.completions.create calls will be recorded.")


def unpatch_openai() -> None:
    global _openai_patched, _original_openai_create
    if not _openai_patched or _original_openai_create is None:
        return
    try:
        import openai
        openai.resources.chat.completions.Completions.create = _original_openai_create
        _openai_patched = False
        print("[agent-debug] OpenAI client unpatched.")
    except ImportError:
        pass


# ── Anthropic patch ────────────────────────────────────────────────────────

def patch_anthropic(output_dir: str = "agent_traces") -> None:
    """Monkey-patch Anthropic client to auto-record all messages.create calls."""
    global _anthropic_patched, _active_recorder, _original_anthropic_create

    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    if _anthropic_patched:
        return

    _active_recorder = TraceRecorder(output=f"{output_dir}/trace.json")
    _active_recorder.set_sdk_source("claude")
    original = anthropic.resources.messages.Messages.create

    @functools.wraps(original)
    def patched_create(self_inner, *args, **kwargs):
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools", [])
        system = kwargs.get("system", "")

        if system:
            _active_recorder.record_system_prompt(system)
        for msg in messages:
            _active_recorder.record_message(dict(msg))
        if tools:
            _active_recorder.record_tool_definitions(
                [t if isinstance(t, dict) else t.model_dump() for t in tools]
            )

        response = original(self_inner, *args, **kwargs)

        # Record response as Claude content blocks
        resp_dict = _response_to_dict_anthropic(response)
        _active_recorder.record_final_response(resp_dict)
        _active_recorder.record_message({
            "role": "assistant",
            "content": resp_dict.get("content", []),
        })

        return response

    _original_anthropic_create = original
    anthropic.resources.messages.Messages.create = patched_create
    _anthropic_patched = True
    print("[agent-debug] Anthropic client patched. All messages.create calls will be recorded.")


def unpatch_anthropic() -> None:
    global _anthropic_patched, _original_anthropic_create
    if not _anthropic_patched or _original_anthropic_create is None:
        return
    try:
        import anthropic
        anthropic.resources.messages.Messages.create = _original_anthropic_create
        _anthropic_patched = False
        print("[agent-debug] Anthropic client unpatched.")
    except ImportError:
        pass


# ── Helpers ────────────────────────────────────────────────────────────────

def _response_to_dict_openai(response: Any) -> dict:
    try:
        choice = response.choices[0]
        msg = choice.message
        result: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return {"message": result}
    except Exception:
        return {}


def _response_to_dict_anthropic(response: Any) -> dict:
    try:
        content = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return {
            "stop_reason": getattr(response, "stop_reason", "end_turn"),
            "content": content,
            "usage": {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            },
        }
    except Exception:
        return {"stop_reason": "end_turn", "content": [], "usage": {}}
