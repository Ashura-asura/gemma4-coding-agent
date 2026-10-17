"""read_file — ARCHITECTURE.md §2.2."""
from __future__ import annotations

from typing import Any

from sandbox import SandboxError

from . import BaseTool, ToolError

#: hard cap on lines returned by one call (AUDIT C4: whole-file reads of
#: 2000-line modules made trajectories 10x longer than the SFT context)
MAX_LINES = 400
#: a read without a range returns this many lines and says how to get more
DEFAULT_WINDOW = 250
MAX_BYTES = 1_000_000


class ReadFile(BaseTool):
    name = "read_file"
    description = "View the contents of a file inside the repository. Lines are numbered."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root"},
            "start_line": {"type": "integer", "description": "1-based first line (optional)"},
            "end_line": {"type": "integer", "description": "1-based last line (optional)"},
        },
        "required": ["path"],
    }

    def run(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        try:
            resolved = self.env.sandbox.resolve_path(path, must_exist=True)
        except SandboxError as exc:
            raise ToolError(str(exc)) from exc
        if resolved.is_dir():
            raise ToolError(f"path is a directory, not a file: {path}")
        if resolved.stat().st_size > MAX_BYTES:
            raise ToolError(f"file is larger than {MAX_BYTES} bytes: {path}")

        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolError(f"cannot read {path}: {exc}") from exc

        lines = text.splitlines()
        first = 1 if start_line is None else int(start_line)
        ranged = start_line is not None or end_line is not None
        default_cap = None if ranged or len(lines) <= DEFAULT_WINDOW else DEFAULT_WINDOW
        last = (len(lines) if default_cap is None else default_cap) if end_line is None else int(end_line)
        if first < 1:
            raise ToolError("start_line must be >= 1")
        if last < first:
            raise ToolError("end_line must be >= start_line")

        window = lines[first - 1:last]
        truncated = False
        if len(window) > MAX_LINES:
            window = window[:MAX_LINES]
            truncated = True
        if not window:
            raise ToolError(f"no lines in range {first}-{last} for {path}")

        width = len(str(first + len(window) - 1))
        rendered = "\n".join(
            f"{first + i:>{width}}: {line}" for i, line in enumerate(window)
        )
        if truncated:
            rendered += f"\n... truncated at {MAX_LINES} lines"
        elif default_cap is not None:
            rendered += (
                f"\n... showing lines 1-{default_cap} of {len(lines)}; "
                "pass start_line/end_line to read more"
            )
        return rendered
