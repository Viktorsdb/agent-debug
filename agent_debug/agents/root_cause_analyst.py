"""RootCauseAnalyst: pinpoint the exact failing step and explain why."""

import json

import anthropic

from agent_debug.agents.base import BaseAgent
from agent_debug.models.types import (
    AnalystInput,
    AnalystOutput,
    NormalizedTrace,
)

SYSTEM = """You are an expert AI agent debugger performing root cause analysis.
Given a trace, its failure classification, and severity, identify:
1. The exact step where things went wrong (not just where the symptom appeared)
2. A clear explanation of WHY it went wrong (not just what happened)
3. A direct quote from the trace as evidence

Respond with a JSON object only."""

PROMPT_TEMPLATE = """Perform root cause analysis on this failed agent trace.

Task: {task_description}
Failure subcategory: {subcategory}
Severity: {severity}/5 — {severity_rationale}
Evidence step index: {evidence_step_index}

=== SYSTEM PROMPT ===
{system_prompt}

=== ALL STEPS ===
{steps}

=== FINAL OUTPUT ===
{final_output}

Return JSON:
{{
  "failing_step_index": <integer>,
  "failing_step_type": "<tool_call or llm_completion>",
  "root_cause_explanation": "<plain English, 2-4 sentences. Explain the underlying cause, not just what happened.>",
  "evidence_quote": "<direct quote from the trace proving the root cause>",
  "confidence": <float 0.0-1.0>
}}"""


def _format_steps(trace: NormalizedTrace) -> str:
    lines = []
    for step in trace["steps"]:
        idx = step["index"]
        if "tool_name" in step:
            err = f"\n   ERROR: {step['error']}" if step.get("error") else ""
            inp_str = json.dumps(step["tool_input"])[:300]
            out_str = step["tool_output"][:300]
            lines.append(
                f"[{idx}] TOOL_CALL tool={step['tool_name']}\n"
                f"   input={inp_str}\n"
                f"   output={out_str}{err}"
            )
        else:
            lines.append(
                f"[{idx}] LLM_COMPLETION\n"
                f"   prompt_summary={step['prompt_summary'][:200]}\n"
                f"   response={step['response'][:400]}"
            )
    return "\n\n".join(lines) if lines else "(no steps)"


class RootCauseAnalyst(BaseAgent):
    def __init__(self, client: anthropic.Anthropic | None = None):
        super().__init__(client)

    def analyze(self, inp: AnalystInput) -> AnalystOutput:
        trace = inp["trace"]
        classification = inp["classification"]
        severity = inp["severity"]

        prompt = PROMPT_TEMPLATE.format(
            task_description=trace["task_description"],
            subcategory=classification["subcategory"],
            severity=severity["severity"],
            severity_rationale=severity["rationale"],
            evidence_step_index=classification["evidence_step_index"],
            system_prompt=trace["system_prompt"][:600],
            steps=_format_steps(trace),
            final_output=trace["final_output"][:400],
        )

        raw = self.call(prompt, SYSTEM)
        data = self.parse_json(raw, prompt, SYSTEM)

        failing_step_type = data.get("failing_step_type", "tool_call")
        if failing_step_type not in ("tool_call", "llm_completion"):
            failing_step_type = "tool_call"

        return AnalystOutput(
            failing_step_index=int(data.get("failing_step_index", 0)),
            failing_step_type=failing_step_type,  # type: ignore[arg-type]
            root_cause_explanation=str(data.get("root_cause_explanation", "")),
            evidence_quote=str(data.get("evidence_quote", "")),
            confidence=float(data.get("confidence", 0.5)),
        )
