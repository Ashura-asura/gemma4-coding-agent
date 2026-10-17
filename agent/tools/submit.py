"""submit — ARCHITECTURE.md §2.2.

Ends the episode with a final patch. Validity (does it apply, is it non-empty,
does it touch the repo) is scored by the reward function, not here — §2.5.

``patch`` is optional: when omitted the tool submits the repository's own
``git diff``, so the model no longer has to re-type every edit it already made
(the old interface doubled trajectory length and added a transcription-error
failure mode; AUDIT H5).
"""
from __future__ import annotations

from typing import Any

from . import BaseTool, ToolError


#: scratch/cache paths the sandbox and test runs create; never part of a patch
_EXCLUDES = (
    ":(exclude).sandbox-tmp",
    ":(exclude).score_junit.xml",
    ":(exclude).pytest_cache",
    ":(exclude)**/__pycache__",
    ":(exclude)*.pyc",
)


class Submit(BaseTool):
    name = "submit"
    description = (
        "Submit the final patch for this issue and end the episode. "
        "Omit `patch` to submit all edits made so far."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "Optional unified diff; defaults to all edits made in the repository",
            },
        },
        "required": [],
    }

    def run(self, patch: str | None = None) -> str:
        if patch is None:
            patch = self._workspace_diff()
        elif not patch.strip():
            raise ToolError("patch must not be empty (omit it to submit all edits)")
        self.env.submitted_patch = patch
        return f"submitted patch ({len(patch)} chars); episode complete"

    def _workspace_diff(self) -> str:
        sandbox = self.env.sandbox
        # new files are invisible to `git diff` until marked intent-to-add
        sandbox.run(["git", "add", "--intent-to-add", "--all", "--", ".", *_EXCLUDES], timeout=60)
        result = sandbox.run(["git", "diff", "--no-color", "--no-ext-diff", "--", ".", *_EXCLUDES], timeout=60)
        if result.status != "pass":
            raise ToolError(
                "cannot compute the patch automatically "
                f"({result.message}); pass `patch` explicitly"
            )
        if not result.stdout.strip():
            raise ToolError("nothing to submit: no edits have been made")
        return result.stdout
