"""RiskScorer: static analysis of system prompt + tool definitions before deployment."""

import json

import anthropic

from agent_debug.agents.base import BaseAgent
from agent_debug.models.types import RiskFinding, RiskInput, RiskOutput

SYSTEM = """You are an expert AI agent reliability engineer performing pre-deployment static analysis.
Analyze the system prompt and tool definitions for common agent failure patterns BEFORE the agent runs.

Check for:
1. Ambiguous termination conditions — when should the agent stop? Is it clear?
2. Missing tool selection guidance — does the prompt explain which tool to use when?
3. Conflicting instructions — any rules that contradict each other?
4. No error handling specified — what should the agent do when a tool fails?
5. Underspecified scope — are there input types where the task is ambiguous?
6. Tool description gaps — do tool descriptions clearly state inputs, outputs, and when NOT to use them?

Respond with a JSON object only."""

PROMPT_TEMPLATE = """Analyze this agent configuration for failure risk.

=== SYSTEM PROMPT ===
{system_prompt}

=== TOOL DEFINITIONS ===
{tool_definitions}

Return JSON:
{{
  "overall_score": <integer 1-10, where 1=low risk and 10=high risk>,
  "findings": [
    {{
      "check": "<which check triggered: termination_conditions|tool_selection_guidance|conflicting_instructions|error_handling|scope_specification|tool_description_gaps>",
      "severity": "<high|medium|low>",
      "excerpt": "<the problematic text from the prompt or tool definition>",
      "suggestion": "<concrete fix in 1-2 sentences>"
    }}
  ],
  "confidence": <float 0.0-1.0>
}}

If no issues found, return an empty findings list and overall_score of 1."""

_VALID_SEVERITIES = {"high", "medium", "low"}


class RiskScorer(BaseAgent):
    def __init__(self, client: anthropic.Anthropic | None = None):
        super().__init__(client)

    def score(self, inp: RiskInput) -> RiskOutput:
        if not inp["system_prompt"] and not inp["tool_definitions"]:
            raise ValueError(
                "RiskScorer requires at least a system_prompt or tool_definitions. "
                "Both are empty."
            )

        tool_defs_str = json.dumps(inp["tool_definitions"], indent=2)[:1500]
        prompt = PROMPT_TEMPLATE.format(
            system_prompt=inp["system_prompt"][:1500],
            tool_definitions=tool_defs_str,
        )

        raw = self.call(prompt, SYSTEM)
        data = self.parse_json(raw, prompt, SYSTEM)

        findings_raw = data.get("findings", [])
        findings: list[RiskFinding] = []
        for f in findings_raw:
            severity = f.get("severity", "medium")
            if severity not in _VALID_SEVERITIES:
                severity = "medium"
            findings.append(
                RiskFinding(
                    check=str(f.get("check", "unknown")),
                    severity=severity,  # type: ignore[arg-type]
                    excerpt=str(f.get("excerpt", "")),
                    suggestion=str(f.get("suggestion", "")),
                )
            )

        overall_score = int(data.get("overall_score", 5))
        overall_score = max(1, min(10, overall_score))

        return RiskOutput(
            overall_score=overall_score,
            findings=findings,
            confidence=float(data.get("confidence", 0.5)),
        )
