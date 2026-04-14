"""SeverityEstimator: rate failure severity 1-5."""

import json

import anthropic

from agent_debug.agents.base import BaseAgent
from agent_debug.models.types import (
    FailureCategory,
    NormalizedTrace,
    SeverityInput,
    SeverityOutput,
)

SYSTEM = """You are an expert AI agent reliability engineer. Given a failed agent trace and
its failure category, estimate how severe the failure is.

Severity scale:
1 - Minor: cosmetic issue, agent completed most of the task correctly
2 - Low: partial failure, missing some details but core task done
3 - Medium: significant failure, user must redo work
4 - High: task completely failed, user gets no useful output
5 - Critical: data loss, incorrect data written, security issue, or crash

Respond with a JSON object only."""

PROMPT_TEMPLATE = """Estimate the severity of this agent failure.

Task: {task_description}
Failure category: {category}
Succeeded: {succeeded}

=== FINAL OUTPUT ===
{final_output}

=== STEPS SUMMARY ===
{steps_summary}

Return JSON:
{{
  "severity": <integer 1-5>,
  "rationale": "<1-2 sentences explaining the severity rating>",
  "confidence": <float 0.0-1.0>
}}"""


def _steps_summary(trace: NormalizedTrace) -> str:
    tool_errors = sum(1 for s in trace["steps"] if s.get("error"))
    total_steps = len(trace["steps"])
    tool_calls = sum(1 for s in trace["steps"] if "tool_name" in s)
    return (
        f"{total_steps} total steps, {tool_calls} tool calls, "
        f"{tool_errors} tool errors"
    )


class SeverityEstimator(BaseAgent):
    def __init__(self, client: anthropic.Anthropic | None = None):
        super().__init__(client)

    def estimate(self, inp: SeverityInput) -> SeverityOutput:
        trace = inp["trace"]
        prompt = PROMPT_TEMPLATE.format(
            task_description=trace["task_description"],
            category=inp["category"],
            succeeded=trace["succeeded"],
            final_output=trace["final_output"][:600],
            steps_summary=_steps_summary(trace),
        )

        raw = self.call(prompt, SYSTEM)
        data = self.parse_json(raw, prompt, SYSTEM)

        severity = int(data.get("severity", 3))
        severity = max(1, min(5, severity))  # clamp to 1-5

        return SeverityOutput(
            severity=severity,
            rationale=str(data.get("rationale", "")),
            confidence=float(data.get("confidence", 0.5)),
        )
