"""AutoFixer: applies FixSuggestions to actual files after human confirmation."""

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agent_debug.models.types import FixSuggestion


@dataclass
class ApplyResult:
    status: Literal["applied", "skipped", "no_file", "error"]
    original_content: str | None = None  # set when status="applied"


class AutoFixer:
    """Applies fix suggestions to real files.

    Supports targets:
      - system_prompt  → patch a .txt / .md / .py file containing the prompt
      - tool_definition / tool_description → patch a JSON file
      - code → patch a Python code file (matched by file_hint or first code file)
    """

    def __init__(
        self,
        system_prompt_file: Path | None = None,
        tools_file: Path | None = None,
        code_files: list[Path] | None = None,
    ):
        self.system_prompt_file = system_prompt_file
        self.tools_file = tools_file
        self.code_files = code_files or []

    def apply(self, suggestion: FixSuggestion) -> ApplyResult:
        """Apply a single fix suggestion to the appropriate file.

        Returns an ApplyResult with:
            status="applied"  — fix written to disk, original_content set
            status="skipped"  — user declined
            status="no_file"  — no target file configured for this fix type
            status="error"    — file write failed
        """
        target = suggestion["target"]
        before = suggestion["before"]
        after = suggestion["after"]

        # Determine which file to patch
        file = self.file_for(suggestion)

        if file is None or not file.exists():
            return ApplyResult(status="no_file")

        content = file.read_text()
        original_content = content

        if before not in content:
            # Try partial match — find the closest paragraph
            matched = _find_closest_block(content, before)
            if matched is None:
                return ApplyResult(status="no_file")
            before = matched

        new_content = content.replace(before, after, 1)

        try:
            file.write_text(new_content)
            return ApplyResult(status="applied", original_content=original_content)
        except OSError:
            return ApplyResult(status="error")

    def revert(self, file: Path, backup: str) -> bool:
        """Restore a file to its original content from backup.

        Returns True if successful, False otherwise.
        """
        try:
            file.write_text(backup)
            return True
        except OSError:
            return False

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
        target = suggestion["target"]
        if target == "system_prompt":
            return self.system_prompt_file
        elif target == "code":
            return self._resolve_code_file(suggestion)
        elif target in ("tool_definition", "tool_description"):
            # Prefer dedicated tools file; fall back to code/system_prompt file
            # when tool definitions live inside a Python file
            if self.tools_file and self.tools_file.exists():
                return self.tools_file
            if self.code_files:
                return self.code_files[0]
            return self.system_prompt_file
        return self.tools_file

    def _resolve_code_file(self, suggestion: FixSuggestion) -> Path | None:
        """Return the best matching code file for a code fix suggestion."""
        if not self.code_files:
            return None
        # Try to match by file_hint if present
        file_hint: str = suggestion.get("file_hint", "")  # type: ignore[call-overload]
        if file_hint:
            for code_file in self.code_files:
                if code_file.name == file_hint or str(code_file).endswith(file_hint):
                    return code_file
        # Fall back to first code file if there's only one
        if len(self.code_files) == 1:
            return self.code_files[0]
        # Try partial name match
        if file_hint:
            for code_file in self.code_files:
                if file_hint in code_file.name:
                    return code_file
        return self.code_files[0] if self.code_files else None


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
