"""edit_file — ARCHITECTURE.md §2.2.

Applies a unified diff in-process (no `patch`/`git apply` subprocess, so
nothing to inject). A hunk that does not match the file exactly is rejected
with a message the model can act on — never a partially applied patch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sandbox import SandboxError

from . import BaseTool, ToolError

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FILE_HEADER = re.compile(r"^(?:---|\+\+\+) (?:[ab]/)?([^\t\n]+)")


@dataclass
class _Hunk:
    old_start: int
    old_count: int
    new_count: int
    body: list[tuple[str, str]]


def _parse_hunks(diff: str) -> list[_Hunk]:
    raw = diff.replace("\r\n", "\n").split("\n")
    hunks: list[_Hunk] = []
    i = 0
    while i < len(raw):
        line = raw[i]
        match = _HUNK_HEADER.match(line)
        if not match:
            i += 1
            continue
        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        new_count = int(match.group(4) or "1")
        i += 1
        body: list[tuple[str, str]] = []
        consumed_old = consumed_new = 0
        while not (consumed_old == old_count and consumed_new == new_count):
            if i >= len(raw):
                raise ToolError("diff is truncated inside a hunk")
            current = raw[i]
            i += 1
            if current.startswith("\\"):
                continue  # "\ No newline at end of file"
            if current == "":
                current = " "  # empty context line without the leading space
            tag, content = current[:1], current[1:]
            if tag == " ":
                consumed_old += 1
                consumed_new += 1
            elif tag == "-":
                consumed_old += 1
            elif tag == "+":
                consumed_new += 1
            else:
                raise ToolError(f"malformed hunk line: {current!r}")
            body.append((tag, content))
        hunks.append(_Hunk(old_start, old_count, new_count, body))
    if not hunks:
        raise ToolError("no unified-diff hunks found in the diff")
    return hunks


def _diff_targets(diff: str) -> list[str]:
    targets: list[str] = []
    for line in diff.replace("\r\n", "\n").split("\n"):
        match = _FILE_HEADER.match(line)
        if match:
            name = match.group(1).strip()
            if name and name not in targets:
                targets.append(name)
    return targets


class PatchError(Exception):
    """A patch could not be applied to a file tree."""


def split_multifile_patch(patch: str) -> dict[str, str]:
    """Split a multi-file unified diff into ``{path: diff}`` sections."""
    lines = patch.replace("\r\n", "\n").split("\n")
    sections: list[tuple[str | None, list[str]]] = []
    current: list[str] = []
    target: str | None = None
    for line in lines:
        if line.startswith("--- "):
            if current:
                sections.append((target, current))
            current, target = [line], None
            continue
        if not current:
            if line.strip():
                current = [line]  # "diff --git ..." preamble
            continue
        current.append(line)
        if target is None and line.startswith("+++ "):
            match = re.match(r"\+\+\+ (?:[ab]/)?(\S+)", line)
            if match and match.group(1) != "/dev/null":
                target = match.group(1)
    if current:
        sections.append((target, current))
    return {path: "\n".join(body) for path, body in sections if path}


def apply_patch_to_tree(patch: str, root: Path) -> list[str]:
    """Apply every section of ``patch`` under ``root``; returns changed paths.

    Paths are scope-checked against ``root`` (symlinks/`..` included), and a
    hunk that does not match raises :class:`PatchError` — all-or-nothing per
    file, never a half-applied patch.
    """
    sections = split_multifile_patch(patch)
    if not sections:
        raise PatchError("patch contains no file sections")
    applied: list[str] = []
    root_resolved = root.resolve()
    for rel, diff in sections.items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise PatchError(f"patch path escapes the repository: {rel}")
        target = (root / rel_path).resolve()
        if target != root_resolved and root_resolved not in target.parents:
            raise PatchError(f"patch path escapes the repository: {rel}")
        original = target.read_text(encoding="utf-8") if target.exists() else ""
        try:
            updated = apply_unified_diff(original, diff)
        except ToolError as exc:
            raise PatchError(f"{rel}: {exc}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated, encoding="utf-8", newline="")
        applied.append(rel)
    return applied


def apply_unified_diff(text: str, diff: str) -> str:
    """Apply ``diff`` to ``text`` or raise ToolError. Atomic: all-or-nothing."""
    hunks = _parse_hunks(diff)
    lines = text.splitlines()
    out: list[str] = []
    pos = 0
    for hunk in hunks:
        start = hunk.old_start - 1 if hunk.old_start > 0 else 0
        if start < pos:
            raise ToolError(f"overlapping hunks near line {hunk.old_start}")
        out.extend(lines[pos:start])
        pos = start
        for tag, content in hunk.body:
            if tag in (" ", "-"):
                if pos >= len(lines):
                    raise ToolError(
                        f"patch does not apply: expected {content!r} at line {pos + 1}, got <EOF>"
                    )
                if lines[pos] != content:
                    raise ToolError(
                        f"patch does not apply at line {pos + 1}: "
                        f"expected {content!r}, found {lines[pos]!r}"
                    )
                if tag == " ":
                    out.append(content)
                pos += 1
            else:
                out.append(content)
    out.extend(lines[pos:])
    trailing_newline = text.endswith("\n") if text else True
    if not out:
        return ""
    return "\n".join(out) + ("\n" if trailing_newline else "")


class EditFile(BaseTool):
    name = "edit_file"
    description = (
        "Apply a unified diff to one file in the repository. The patch must "
        "apply exactly; otherwise nothing is written."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the file to edit, relative to the repo root"},
            "diff": {"type": "string", "description": "Unified diff (@@ hunks) for this file"},
        },
        "required": ["path", "diff"],
    }

    def run(self, path: str, diff: str) -> str:
        if not diff.strip():
            raise ToolError("diff must not be empty")
        try:
            resolved = self.env.sandbox.resolve_path(path)
        except SandboxError as exc:
            raise ToolError(str(exc)) from exc

        targets = _diff_targets(diff)
        if len(targets) > 1:
            raise ToolError(
                f"diff touches {len(targets)} files ({', '.join(targets)}); "
                "edit one file per call"
            )
        if targets and Path(targets[0]).name != resolved.name:
            raise ToolError(
                f"diff is for {targets[0]!r} but path is {path!r}; refusing to apply"
            )

        exists = resolved.exists()
        if exists and resolved.is_dir():
            raise ToolError(f"path is a directory: {path}")
        if not exists and not resolved.parent.is_dir():
            raise ToolError(f"parent directory does not exist for: {path}")

        original = resolved.read_text(encoding="utf-8", errors="strict") if exists else ""
        try:
            updated = apply_unified_diff(original, diff)
        except ToolError:
            raise
        except UnicodeDecodeError as exc:
            raise ToolError(f"cannot decode {path} as utf-8: {exc}") from exc

        try:
            resolved.write_text(updated, encoding="utf-8", newline="")
        except OSError as exc:
            raise ToolError(f"cannot write {path}: {exc}") from exc
        return f"applied patch to {path} ({len(original.splitlines())} -> {len(updated.splitlines())} lines)"
