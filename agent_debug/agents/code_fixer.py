"""CodeFixer: reads a file and applies targeted line-level fixes."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_debug.agents.base import BaseAgent

SYSTEM = """You are an expert Python developer. You will be given a Python file and failing tests.
Your task: identify the exact lines that need to change to fix the failing tests.

Output ONLY a JSON array of changes. Each change is:
  {"find": "<exact line content to find>", "replace": "<new line content>", "reason": "<one sentence>"}

Rules:
- "find" must be an EXACT substring from the file shown (copy verbatim, including indentation)
- Keep changes minimal and targeted — do NOT rewrite the whole file
- One change per bug
- Respond with ONLY the JSON array, no explanation, no markdown fences"""

PROMPT_TEMPLATE = """Find and fix the bugs in this Python file that cause these tests to fail.

FAILING TESTS:
{compact_test_failures}

FILE — {filename}:
{file_content}

Output ONLY a JSON array like:
[
  {{"find": "<exact text to find>", "replace": "<replacement>", "reason": "<why>"}},
  ...
]"""


@dataclass
class LineChange:
    find: str
    replace: str
    reason: str


@dataclass
class CodeFixResult:
    changes: list[LineChange]
    fixed_content: str
    changes_summary: str
    confidence: float = 0.85


class CodeFixer(BaseAgent):
    """Agent that fixes a Python file with targeted line replacements.

    Reads the full file, asks the LLM for a compact JSON list of
    find→replace changes, applies them one by one.  No full-file
    rewrite — output is tiny so it never gets truncated.
    """

    def fix_file(
        self,
        file_path: Path,
        test_failures: str,
        context: str = "",
    ) -> CodeFixResult:
        original = file_path.read_text()

        # Compact failures: FAILED + E lines only
        compact_lines = []
        for line in test_failures.splitlines():
            s = line.strip()
            if s.startswith("FAILED") or s.startswith("E ") or s.startswith("AssertionError"):
                compact_lines.append(s)
        compact = "\n".join(compact_lines[:50])

        prompt = PROMPT_TEMPLATE.format(
            compact_test_failures=compact,
            filename=file_path.name,
            file_content=original,
        )

        raw = self.call(prompt, SYSTEM)

        # Strip markdown fences
        raw = re.sub(r"```(?:json|python)?\s*", "", raw).strip().rstrip("`").strip()

        try:
            data: list[dict[str, Any]] = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"CodeFixer: LLM returned invalid JSON: {e}\nRaw: {raw[:300]}")

        if not isinstance(data, list) or not data:
            raise ValueError(f"CodeFixer: expected a non-empty JSON array, got: {raw[:200]}")

        # Apply changes
        changes: list[LineChange] = []
        content = original
        applied = []
        skipped = []

        for item in data:
            find = str(item.get("find", ""))
            replace = str(item.get("replace", ""))
            reason = str(item.get("reason", ""))
            if not find:
                continue
            lc = LineChange(find=find, replace=replace, reason=reason)
            changes.append(lc)
            if find in content:
                content = content.replace(find, replace, 1)
                applied.append(reason or find[:60])
            else:
                skipped.append(find[:60])

        if not applied:
            raise ValueError(
                f"CodeFixer: none of the {len(changes)} changes matched the file. "
                f"Skipped: {skipped}"
            )

        summary_parts = [f"Applied {len(applied)} fix(es): " + "; ".join(applied[:3])]
        if skipped:
            summary_parts.append(f"({len(skipped)} could not match)")
        changes_summary = " ".join(summary_parts)

        return CodeFixResult(
            changes=changes,
            fixed_content=content,
            changes_summary=changes_summary,
        )
