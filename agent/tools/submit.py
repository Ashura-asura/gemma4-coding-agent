"""submit — ARCHITECTURE.md §2.2.

Ends the episode with a final patch. Validity (does it apply, is it non-empty,
does it touch the repo) is scored by the reward function, not here — §2.5.
"""
from __future__ import annotations

from typing import Any

from . import BaseTool, ToolError


class Submit(BaseTool):
    name = "submit"
    description = "Submit the final patch for this issue and end the episode."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "patch": {"type": "string", "description": "Unified diff solving the issue"},
        },
        "required": ["patch"],
    }

    def run(self, patch: str) -> str:
        if not patch.strip():
            raise ToolError("patch must not be empty")
        self.env.submitted_patch = patch
        return f"submitted patch ({len(patch)} chars); episode complete"
