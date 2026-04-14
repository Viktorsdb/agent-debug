"""AutoFixer: applies FixSuggestions to actual files after human confirmation."""

import difflib
from pathlib import Path
from typing import Literal

from agent_debug.models.types import FixSuggestion


ApplyResult = Literal["applied", "skipped", "no_file", "error"]


class AutoFixer:
    """Applies fix suggestions to real files.

    Supports targets:
      - system_prompt  → patch a .txt / .md / .py file containing the prompt
      - tool_definition / tool_description → patch a JSON file
    """

    def __init__(
        self,
        system_prompt_file: Path | None = None,
        tools_file: Path | None = None,
    ):
        self.system_prompt_file = system_prompt_file
        self.tools_file = tools_file

    def apply(self, suggestion: FixSuggestion) -> ApplyResult:
        """Apply a single fix suggestion to the appropriate file.

        Returns:
            "applied"  — fix written to disk
            "skipped"  — user declined
            "no_file"  — no target file configured for this fix type
            "error"    — file write failed
        """
        target = suggestion["target"]
        before = suggestion["before"]
        after = suggestion["after"]

        # Determine which file to patch
        if target == "system_prompt":
            file = self.system_prompt_file
        elif target in ("tool_definition", "tool_description"):
            file = self.tools_file
        else:
            file = self.system_prompt_file  # best guess

        if file is None or not file.exists():
            return "no_file"

        content = file.read_text()

        if before not in content:
            # Try partial match — find the closest paragraph
            matched = _find_closest_block(content, before)
            if matched is None:
                return "no_file"
            before = matched

        new_content = content.replace(before, after, 1)

        try:
            file.write_text(new_content)
            return "applied"
        except OSError:
            return "error"

    def make_diff(self, suggestion: FixSuggestion, file: Path | None = None) -> str:
        """Return a unified diff string showing what would change."""
        before_lines = suggestion["before"].splitlines(keepends=True)
        after_lines = suggestion["after"].splitlines(keepends=True)
        fname = str(file) if file else "config"
        diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{fname}",
            tofile=f"b/{fname}",
            lineterm="",
        )
        return "\n".join(diff)

    def file_for(self, suggestion: FixSuggestion) -> Path | None:
        if suggestion["target"] == "system_prompt":
            return self.system_prompt_file
        return self.tools_file


def _find_closest_block(content: str, before: str) -> str | None:
    """Find the closest matching block in content using SequenceMatcher."""
    # Try matching on first 60 chars of before
    needle = before[:60].strip()
    if needle and needle in content:
        # Find the paragraph containing the needle
        idx = content.find(needle)
        start = content.rfind("\n\n", 0, idx)
        end = content.find("\n\n", idx)
        start = start + 2 if start != -1 else 0
        end = end if end != -1 else len(content)
        return content[start:end]
    return None
