"""run_tests — ARCHITECTURE.md §2.2.

Launches the repo's test suite through the sandbox and reports a typed
status line, so the model always sees pass/fail/timeout/error explicitly.
"""
from __future__ import annotations

import sys
import time
from typing import Any

from sandbox import SandboxError

from . import BaseTool, ToolError

DEFAULT_TEST_COMMAND: tuple[str, ...] = (sys.executable, "-m", "pytest", "-q")
TEST_TIMEOUT_S = 300.0
MAX_OUTPUT_CHARS = 8000


class RunTests(BaseTool):
    name = "run_tests"
    description = (
        "Run the repository test suite (or a subset) in the sandbox. "
        "Returns a status line: status=pass|fail|timeout|error."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Optional test file, directory or node id; defaults to the whole suite",
            },
        },
        "required": [],
    }

    def __init__(self, env) -> None:  # type: ignore[no-untyped-def]
        super().__init__(env)
        self.command = DEFAULT_TEST_COMMAND

    def run(self, target: str | None = None) -> str:
        argv = self._build_argv(target)
        if target is not None and not target.strip():
            raise ToolError("target must not be empty")

        started = time.monotonic()
        try:
            result = self.env.sandbox.run(
                argv,
                timeout=TEST_TIMEOUT_S,
                pythonpath=self.env.test_pythonpath,
            )
        except SandboxError as exc:  # pragma: no cover - run() returns errors instead
            raise ToolError(str(exc)) from exc
        duration = time.monotonic() - started

        body = _truncate(result.stdout, result.stderr)
        lines = [
            f"status={result.status}",
            f"duration={duration:.1f}s",
            f"returncode={result.returncode}",
            f"message={result.message}",
        ]
        if result.warnings:
            lines.append("warnings=" + " | ".join(result.warnings))
        if result.unsupported_limits:
            lines.append("unsupported_limits=" + ", ".join(result.unsupported_limits))
        if body:
            lines.append("")
            lines.append(body)
        return "\n".join(lines)

    def _build_argv(self, target: str | None) -> list[str]:
        """Task-configured suite when present, otherwise the default pytest run."""
        if target is not None and target.strip():
            return list(self.env.test_base or self.command) + [
                self._scope_checked_target(target.strip())
            ]
        if self.env.test_base is not None:
            return list(self.env.test_base) + list(self.env.test_targets or [])
        return list(self.command)

    def _scope_checked_target(self, target: str) -> str:
        """A target must not point outside the repository root."""
        base, sep, rest = target.partition("::")
        try:
            resolved = self.env.sandbox.resolve_path(base)
        except SandboxError as exc:
            raise ToolError(f"test target escapes the repository: {exc}") from exc
        scoped = resolved.relative_to(self.env.sandbox.root).as_posix()
        return scoped + (sep + rest if sep else "")


def _truncate(stdout: str, stderr: str) -> str:
    chunks = []
    if stdout:
        chunks.append(stdout)
    if stderr:
        chunks.append("[stderr]\n" + stderr)
    text = "\n".join(chunks).strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return text[:half] + "\n... output truncated ...\n" + text[-half:]
