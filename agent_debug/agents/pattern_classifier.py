"""PatternClassifier: classify the failure into one of 15 subcategories."""

import json
from typing import Any

import anthropic

from agent_debug.agents.base import BaseAgent
from agent_debug.models.types import (
    ClassifierInput,
    ClassifierOutput,
    FailureCategory,
    FailureSubcategory,
    NormalizedTrace,
)

SYSTEM = """You are an expert AI agent debugger. Given a failed agent trace, you classify
the root failure pattern into a precise subcategory.

Taxonomy:
- wrong_tool.similar_name     — agent called a tool with a similar name to the correct one
- wrong_tool.missing_guidance — no instruction on which tool to use for this task type
- wrong_tool.scope_confusion  — agent used a tool outside its intended scope
- hallucination.missing_retrieval  — agent stated a fact it should have looked up
- hallucination.domain_gap         — agent fabricated domain-specific details it lacks
- hallucination.format_pressure    — agent invented data to satisfy a required output format
- premature_stop.ambiguous_done    — unclear completion criteria; agent stopped too early
- premature_stop.error_avoidance   — agent stopped after a tool error instead of retrying
- premature_stop.max_steps_hit     — agent hit a step/token limit mid-task
- context_overflow.long_conversation — important early context forgotten
- context_overflow.large_tool_output — a large tool output crowded out other context
- tool_misinterpretation.schema_mismatch — agent called tool with wrong argument types/structure
- tool_misinterpretation.error_ignored  — tool returned an error; agent treated it as success
- tool_misinterpretation.partial_result — agent treated a partial result as complete
- prompt_ambiguity.conflicting_instructions — system prompt has contradictory rules
- prompt_ambiguity.underspecified_scope     — task is too vague for this input type

Respond with a JSON object only. No explanation outside the JSON."""

PROMPT_TEMPLATE = """Analyze this failed agent trace and classify the failure.

Task: {task_description}
SDK: {sdk_source}
Succeeded: {succeeded}

=== SYSTEM PROMPT ===
{system_prompt}

=== TOOL DEFINITIONS ===
{tool_definitions}

=== STEPS ===
{steps}

=== FINAL OUTPUT ===
{final_output}

Return JSON:
{{
  "category": "<one of the 6 parent categories>",
  "subcategory": "<one of the 15 subcategories above>",
  "evidence_step_index": <integer, which step index most clearly shows the failure>,
  "confidence": <float 0.0-1.0>
}}"""

_VALID_SUBCATEGORIES = set(FailureSubcategory.__args__)  # type: ignore[attr-defined]
_VALID_CATEGORIES = set(FailureCategory.__args__)  # type: ignore[attr-defined]


def _format_steps(trace: NormalizedTrace) -> str:
    lines = []
    for step in trace["steps"]:
        idx = step["index"]
        if "tool_name" in step:
            err = f" ERROR: {step['error']}" if step.get("error") else ""
            lines.append(
                f"[{idx}] TOOL_CALL {step['tool_name']}({json.dumps(step['tool_input'])}) "
                f"→ {step['tool_output'][:200]}{err}"
            )
        else:
            lines.append(
                f"[{idx}] LLM_RESPONSE: {step['response'][:300]}"
            )
    return "\n".join(lines) if lines else "(no steps)"


class PatternClassifier(BaseAgent):
    def __init__(self, client: anthropic.Anthropic | None = None):
        super().__init__(client)

    def classify(self, inp: ClassifierInput) -> ClassifierOutput:
        trace = inp["trace"]

        tool_defs_str = json.dumps(trace["tool_definitions"], indent=2)[:1000]
        prompt = PROMPT_TEMPLATE.format(
            task_description=trace["task_description"],
            sdk_source=trace["sdk_source"],
            succeeded=trace["succeeded"],
            system_prompt=trace["system_prompt"][:800],
            tool_definitions=tool_defs_str,
            steps=_format_steps(trace),
            final_output=trace["final_output"][:500],
        )

        raw = self.call(prompt, SYSTEM)
        data = self.parse_json(raw, prompt, SYSTEM)

        # Validate and coerce
        subcategory = data.get("subcategory", "")
        if subcategory not in _VALID_SUBCATEGORIES:
            # Try to find closest match
            subcategory = _closest_subcategory(subcategory) or "wrong_tool.similar_name"

        category = subcategory.split(".")[0]
        if category not in _VALID_CATEGORIES:
            category = "wrong_tool"

        return ClassifierOutput(
            category=category,  # type: ignore[arg-type]
            subcategory=subcategory,  # type: ignore[arg-type]
            evidence_step_index=int(data.get("evidence_step_index", 0)),
            confidence=float(data.get("confidence", 0.5)),
        )


def _closest_subcategory(raw: str) -> str | None:
    """Fuzzy match: find the subcategory whose suffix matches best."""
    raw_lower = raw.lower().replace(" ", "_").replace("-", "_")
    for sub in _VALID_SUBCATEGORIES:
        if raw_lower in sub or sub in raw_lower:
            return sub
    return None
