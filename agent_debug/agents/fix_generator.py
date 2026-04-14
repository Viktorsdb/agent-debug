"""FixGenerator: produce concrete before/after fix suggestions."""

import json

import anthropic

from agent_debug.agents.base import BaseAgent
from agent_debug.models.types import (
    FixInput,
    FixOutput,
    FixSuggestion,
    NormalizedTrace,
)

SYSTEM = """You are an expert AI agent prompt engineer. Given a failed trace and its root cause,
produce concrete, copy-paste-ready fixes.

Each fix must:
- Target one specific thing: system_prompt, tool_definition, tool_description, or retrieval_setup
- Show the EXACT text before (or a representative excerpt) and after
- Be actionable — no vague "improve clarity" suggestions
- Address the root cause directly

Respond with a JSON object only."""

PROMPT_TEMPLATE = """Generate fix suggestions for this agent failure.

Task: {task_description}
Failure subcategory: {subcategory}
Root cause: {root_cause_explanation}
Evidence: {evidence_quote}
Failing step: {failing_step_index} ({failing_step_type})

=== CURRENT SYSTEM PROMPT ===
{system_prompt}

=== CURRENT TOOL DEFINITIONS ===
{tool_definitions}

Return JSON:
{{
  "suggestions": [
    {{
      "target": "<system_prompt|tool_definition|tool_description|retrieval_setup>",
      "before": "<the exact current text that needs to change, or an excerpt>",
      "after": "<the replacement text>",
      "explanation": "<1-2 sentences: why this change addresses the root cause>",
      "confidence": <float 0.0-1.0>
    }}
  ],
  "disclaimer": "Test before deploying"
}}

Provide 1-3 suggestions ordered by expected impact (highest first).
The disclaimer field must always be exactly: "Test before deploying" """

_VALID_TARGETS = {"system_prompt", "tool_definition", "tool_description", "retrieval_setup"}


class FixGenerator(BaseAgent):
    def __init__(self, client: anthropic.Anthropic | None = None):
        super().__init__(client)

    def generate(self, inp: FixInput) -> FixOutput:
        trace = inp["trace"]
        classification = inp["classification"]
        root_cause = inp["root_cause"]

        tool_defs_str = json.dumps(trace["tool_definitions"], indent=2)[:1200]

        prompt = PROMPT_TEMPLATE.format(
            task_description=trace["task_description"],
            subcategory=classification["subcategory"],
            root_cause_explanation=root_cause["root_cause_explanation"],
            evidence_quote=root_cause["evidence_quote"][:400],
            failing_step_index=root_cause["failing_step_index"],
            failing_step_type=root_cause["failing_step_type"],
            system_prompt=trace["system_prompt"][:800],
            tool_definitions=tool_defs_str,
        )

        raw = self.call(prompt, SYSTEM)
        data = self.parse_json(raw, prompt, SYSTEM)

        suggestions_raw = data.get("suggestions", [])
        suggestions: list[FixSuggestion] = []
        for s in suggestions_raw:
            target = s.get("target", "system_prompt")
            if target not in _VALID_TARGETS:
                target = "system_prompt"
            suggestions.append(
                FixSuggestion(
                    target=target,  # type: ignore[arg-type]
                    before=str(s.get("before", "")),
                    after=str(s.get("after", "")),
                    explanation=str(s.get("explanation", "")),
                    confidence=float(s.get("confidence", 0.5)),
                )
            )

        return FixOutput(
            suggestions=suggestions,
            disclaimer="Test before deploying",
        )
