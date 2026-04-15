"""FixGenerator: produce concrete before/after fix suggestions."""

import json

import anthropic

from agent_debug.agents.base import BaseAgent
from agent_debug.models.types import (
    CodeFixSuggestion,
    FixInput,
    FixOutput,
    FixSuggestion,
    NormalizedTrace,
)

SYSTEM = """You are an expert AI agent prompt engineer and Python developer. Given a failed trace and its root cause,
produce concrete, copy-paste-ready fixes.

Each fix must:
- Target one specific thing: system_prompt, tool_definition, tool_description, retrieval_setup, or code
- For "before": use a SHORT excerpt (max 120 chars) of the exact text to find — NOT the entire JSON blob
- For "after": show only the replacement for that excerpt — keep it concise
- Be actionable — no vague "improve clarity" suggestions
- Address the root cause directly
- For code fixes, include a file_hint field with the suggested filename to patch (e.g. "agent.py")

STRICT RULES for "before" and "after" fields:
- Must be a SINGLE LINE string — no newlines, no multi-line content
- Max 120 characters
- No triple quotes, no full variable assignments
- For a SYSTEM_PROMPT fix: use one KEY PHRASE from inside the prompt, e.g.:
    before: "Use the tools available to complete tasks."
    after:  "Use the tools available. For local files use read_file, NOT web_search."
- For a code fix: use one short identifier/value, e.g.:
    before: "max_retries: int = 1"
    after:  "max_retries: int = 3"
- For a tool description: use just the description string value, e.g.:
    before: "Parse data"
    after:  "Parse a local CSV file. Use for .csv/.tsv files, NOT web search."

When TEST FAILURES are provided:
- Generate a fix for EACH failing test
- Always use target: "code" when fixing Python source code (parameters, return values, etc.)
- The "before" must be an exact single-line substring found verbatim in the FILE CONTENTS

Respond with a JSON object only."""

PROMPT_TEMPLATE = """Generate fix suggestions for this agent failure.

Task: {task_description}
Failure subcategory: {subcategory}
Root cause: {root_cause_explanation}
Evidence: {evidence_quote}
Failing step: {failing_step_index} ({failing_step_type})

Current system prompt (from trace):
{system_prompt}

Current tool definitions (from trace):
{tool_definitions}
{file_contents_section}{agent_code_section}{test_failures_section}
CRITICAL RULE for "before" field:
- Must be a single-line substring (no \n, no multi-line) found verbatim in the file
- Copy it character-for-character from the FILE CONTENTS shown above
- Max 120 characters — use the shortest unique substring that identifies the bug
- Examples of GOOD before values: "max_retries: int = 1"  |  "return \"\""  |  "Parse data"
- Examples of BAD before values: anything with \n or triple quotes or > 120 chars

Return JSON:
{{
  "suggestions": [
    {{
      "target": "<system_prompt|tool_definition|tool_description|retrieval_setup|code>",
      "before": "<EXACT substring copied verbatim from the file content shown above>",
      "after": "<replacement for that exact substring>",
      "explanation": "<1-2 sentences: why this change addresses the root cause>",
      "confidence": <float 0.0-1.0>,
      "file_hint": "<filename to patch, only required when target is code, e.g. agent.py>"
    }}
  ],
  "disclaimer": "Test before deploying"
}}

Provide {suggestion_count} suggestions ordered by expected impact (highest first).
The disclaimer field must always be exactly: "Test before deploying"
When the root cause is in agent logic (not just prompts), generate code-level fixes targeting the code file."""

_VALID_TARGETS = {"system_prompt", "tool_definition", "tool_description", "retrieval_setup", "code"}


class FixGenerator(BaseAgent):
    def __init__(self, client: anthropic.Anthropic | None = None):
        super().__init__(client)

    def generate(
        self,
        inp: FixInput,
        agent_code: str = "",
        file_contents: dict[str, str] | None = None,
        test_failures: str = "",
    ) -> FixOutput:
        """Generate fix suggestions.

        Args:
            inp:           Standard FixInput (trace + classification + root cause).
            agent_code:    Raw source of the agent's Python file (legacy param).
            file_contents: Mapping of filename → full file text for files the
                           user passed via --code / --system-prompt.  These are
                           shown verbatim so the LLM can copy exact substrings.
            test_failures: Full pytest failure output from a pre-flight test run.
                           When provided, the LLM generates a fix for each failing test.
        """
        trace = inp["trace"]
        classification = inp["classification"]
        root_cause = inp["root_cause"]

        tool_defs_str = json.dumps(trace["tool_definitions"], indent=2)[:600]

        # Build a section that shows the actual file contents so the LLM
        # can produce "before" strings that exist verbatim in those files.
        if file_contents:
            parts = []
            for fname, content in file_contents.items():
                parts.append(f"\nFILE CONTENTS — {fname}:\n{content[:1500]}\n")
            file_contents_section = "\n".join(parts)
        else:
            file_contents_section = ""

        if agent_code and not file_contents:
            agent_code_section = f"\nAGENT CODE:\n{agent_code[:2000]}\n"
        else:
            agent_code_section = ""

        if test_failures:
            # Extract just the key lines: FAILED lines + AssertionError lines
            failure_lines = []
            for line in test_failures.splitlines():
                stripped = line.strip()
                if (stripped.startswith("FAILED")
                        or stripped.startswith("AssertionError")
                        or stripped.startswith("assert ")
                        or stripped.startswith("E ")):
                    failure_lines.append(stripped)
            compact_failures = "\n".join(failure_lines[:60])

            test_failures_section = (
                f"\nTEST FAILURES — generate a fix for each:\n"
                f"{compact_failures}\n\n"
                f"For each failure: find the exact single-line substring in FILE CONTENTS "
                f"that causes it, and provide the corrected replacement.\n"
            )
            suggestion_count = "exactly 3 (the 3 highest-impact)"
        else:
            test_failures_section = ""
            suggestion_count = "1-3"

        prompt = PROMPT_TEMPLATE.format(
            task_description=trace["task_description"],
            subcategory=classification["subcategory"],
            root_cause_explanation=root_cause["root_cause_explanation"],
            evidence_quote=root_cause["evidence_quote"][:400],
            failing_step_index=root_cause["failing_step_index"],
            failing_step_type=root_cause["failing_step_type"],
            system_prompt=trace["system_prompt"][:800],
            tool_definitions=tool_defs_str,
            file_contents_section=file_contents_section,
            agent_code_section=agent_code_section,
            test_failures_section=test_failures_section,
            suggestion_count=suggestion_count,
        )

        raw = self.call(prompt, SYSTEM)
        data = self.parse_json(raw, prompt, SYSTEM)

        suggestions_raw = data.get("suggestions", [])
        suggestions: list[FixSuggestion] = []
        for s in suggestions_raw:
            target = s.get("target", "system_prompt")
            if target not in _VALID_TARGETS:
                target = "system_prompt"
            if target == "code":
                suggestion: FixSuggestion = CodeFixSuggestion(
                    target=target,  # type: ignore[arg-type]
                    before=str(s.get("before", "")),
                    after=str(s.get("after", "")),
                    explanation=str(s.get("explanation", "")),
                    confidence=float(s.get("confidence", 0.5)),
                    file_hint=str(s.get("file_hint", "")),
                )
            else:
                suggestion = FixSuggestion(
                    target=target,  # type: ignore[arg-type]
                    before=str(s.get("before", "")),
                    after=str(s.get("after", "")),
                    explanation=str(s.get("explanation", "")),
                    confidence=float(s.get("confidence", 0.5)),
                )
            suggestions.append(suggestion)

        return FixOutput(
            suggestions=suggestions,
            disclaimer="Test before deploying",
        )
