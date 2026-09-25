"""search_repo — ARCHITECTURE.md §2.2.

Pure-Python literal/regex search: no shell, no grep subprocess, so there is
nothing to inject into and the allowlist stays small.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import BaseTool, ToolError

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "node_modules", ".venv", "venv", ".tox", ".eggs",
    "dist", "build", ".idea", ".vscode",
}
MAX_FILE_BYTES = 1_000_000
DEFAULT_MAX_RESULTS = 50
MAX_RESULTS_CAP = 200


class SearchRepo(BaseTool):
    name = "search_repo"
    description = (
        "Search the repository for a literal string or regular expression and "
        "return matching lines as path:line: text."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text or pattern to search for"},
            "is_regex": {"type": "boolean", "description": "Treat query as a regex (default false)"},
            "max_results": {"type": "integer", "description": "Cap on returned matches (default 50)"},
        },
        "required": ["query"],
    }

    def run(self, query: str, is_regex: bool = False, max_results: int = DEFAULT_MAX_RESULTS) -> str:
        if not query.strip():
            raise ToolError("query must not be empty")
        limit = max(1, min(int(max_results), MAX_RESULTS_CAP))
        pattern = self._compile(query, is_regex)

        matches: list[str] = []
        root = self.env.sandbox.root
        for path in sorted(root.rglob("*")):
            if len(matches) >= limit:
                break
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue  # binary
            text = raw.decode("utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    rel = path.relative_to(root).as_posix()
                    matches.append(f"{rel}:{lineno}: {line.rstrip()[:400]}")
                    if len(matches) >= limit:
                        break

        if not matches:
            return f"no matches for {query!r}"
        suffix = "\n... results truncated" if len(matches) >= limit else ""
        return "\n".join(matches) + suffix

    @staticmethod
    def _compile(query: str, is_regex: bool) -> re.Pattern[str]:
        if is_regex:
            try:
                return re.compile(query)
            except re.error as exc:
                raise ToolError(f"invalid regex: {exc}") from exc
        return re.compile(re.escape(query))
