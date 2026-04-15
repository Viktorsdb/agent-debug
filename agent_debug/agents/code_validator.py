"""CodeValidator: runs a test command after a code fix and reports pass/fail."""

import subprocess
import time
from pathlib import Path
from typing import TypedDict


class ValidationResult(TypedDict):
    passed: bool
    output: str
    duration_sec: float


class CodeValidator:
    """Runs a shell test command and reports whether it passed."""

    def run(self, test_cmd: str, cwd: Path | None = None) -> ValidationResult:
        """Run test_cmd as a subprocess with a 120-second timeout.

        Args:
            test_cmd: The shell command to run (e.g. "pytest tests/").
            cwd: Working directory for the subprocess. Defaults to current dir.

        Returns:
            ValidationResult with passed, combined output, and duration_sec.
        """
        start = time.monotonic()
        try:
            result = subprocess.run(
                test_cmd,
                shell=True,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=120,
            )
            duration = time.monotonic() - start
            output = result.stdout
            if result.stderr:
                output = output + result.stderr if output else result.stderr
            return ValidationResult(
                passed=result.returncode == 0,
                output=output,
                duration_sec=duration,
            )
        except FileNotFoundError as e:
            duration = time.monotonic() - start
            return ValidationResult(
                passed=False,
                output=f"Command not found: {e}",
                duration_sec=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return ValidationResult(
                passed=False,
                output=f"Test command timed out after 120 seconds: {test_cmd}",
                duration_sec=duration,
            )
