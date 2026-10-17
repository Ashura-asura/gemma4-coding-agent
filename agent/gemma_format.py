"""Gemma-4 native tool-call format — AUDIT.md finding C3.

Gemma 4's chat template renders (and the model therefore emits) tool calls as::

    <|tool_call>call:search_repo{query:<|"|>get_traceback_frames<|"|>,max_results:5}<tool_call|>

Strings are delimited by the ``<|"|>`` escape token, keys are bare, and the
whole call is *not* JSON. The original harness decoded with
``skip_special_tokens=True`` and only understood JSON, so a model trained on
the template could never be parsed. This module understands the native form
(also when the special tokens were stripped by a lossy decode) and leaves the
JSON fallback to ``eval.run_eval.parse_tool_call``.
"""
from __future__ import annotations

import json
import re
from json.decoder import scanstring
from typing import Any

ESC = '<|"|>'
CALL_OPEN = "<|tool_call>"
CALL_CLOSE = "<tool_call|>"

_NAME = re.compile(r"call:\s*([A-Za-z_][\w.\-]*)\s*")
_SPECIAL = re.compile(r"<\|[^<>|\n]{1,30}>|<[^<>|\n]{1,30}\|>|<bos>|<eos>|<pad>")
_THOUGHT = re.compile(r"<\|channel>.*?<channel\|>", re.S)
_BARE_STOP = ",}]"
_KEY = re.compile(r"[A-Za-z0-9_.\-]+")


class ToolCallParseError(ValueError):
    """The text contains no parseable tool call."""


class _Parser:
    def __init__(self, text: str, pos: int = 0) -> None:
        self.s, self.i = text, pos

    def _ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def _expect(self, ch: str) -> None:
        self._ws()
        if not self.s.startswith(ch, self.i):
            raise ToolCallParseError(f"expected {ch!r} at offset {self.i}")
        self.i += len(ch)

    def _esc_string(self) -> str:
        self.i += len(ESC)
        end = self.s.find(ESC, self.i)
        if end == -1:
            raise ToolCallParseError("unterminated <|\"|> string")
        value = self.s[self.i:end]
        self.i = end + len(ESC)
        return value

    def _json_string(self) -> str:
        try:
            value, end = scanstring(self.s, self.i + 1)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(f"bad quoted string: {exc}") from exc
        self.i = end
        return value

    def _key(self) -> str:
        self._ws()
        if self.s.startswith(ESC, self.i):
            return self._esc_string()
        if self.s.startswith('"', self.i):
            return self._json_string()
        match = _KEY.match(self.s, self.i)
        if not match:
            raise ToolCallParseError(f"expected a key at offset {self.i}")
        self.i = match.end()
        return match.group(0)

    def _bare(self) -> Any:
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in _BARE_STOP:
            self.i += 1
        token = self.s[start:self.i].strip()
        if token in ("true", "True"):
            return True
        if token in ("false", "False"):
            return False
        if token in ("null", "None", ""):
            return None
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            return token

    def value(self) -> Any:
        self._ws()
        if self.i >= len(self.s):
            raise ToolCallParseError("unexpected end of tool call")
        if self.s.startswith(ESC, self.i):
            return self._esc_string()
        ch = self.s[self.i]
        if ch == "{":
            return self.obj()
        if ch == "[":
            return self.arr()
        if ch == '"':
            return self._json_string()
        return self._bare()

    def arr(self) -> list[Any]:
        self._expect("[")
        items: list[Any] = []
        self._ws()
        if self.s.startswith("]", self.i):
            self.i += 1
            return items
        while True:
            items.append(self.value())
            self._ws()
            if self.s.startswith(",", self.i):
                self.i += 1
                continue
            self._expect("]")
            return items

    def obj(self) -> dict[str, Any]:
        self._expect("{")
        out: dict[str, Any] = {}
        self._ws()
        if self.s.startswith("}", self.i):
            self.i += 1
            return out
        while True:
            key = self._key()
            self._expect(":")
            out[key] = self.value()
            self._ws()
            if self.s.startswith(",", self.i):
                self.i += 1
                continue
            self._expect("}")
            return out


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Inverse of :func:`extract_tool_call` (used by tests and offline checks)."""

    def val(v: Any) -> str:
        if isinstance(v, str):
            return f"{ESC}{v}{ESC}"
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "null"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(val(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{" + ",".join(f"{k}:{val(x)}" for k, x in v.items()) + "}"
        raise TypeError(f"cannot render {type(v).__name__}")

    body = ",".join(f"{k}:{val(v)}" for k, v in arguments.items())
    return f"{CALL_OPEN}call:{name}{{{body}}}{CALL_CLOSE}"


def clean_reasoning(text: str, limit: int = 2000) -> str:
    """Prose the model wrote before the call, minus special tokens/thoughts."""
    text = _THOUGHT.sub("", text)
    text = _SPECIAL.sub("", text)
    return text.strip()[:limit]


def extract_tool_call(text: str) -> tuple[str, dict[str, Any], str]:
    """Return ``(name, arguments, reasoning)`` for the first native tool call."""
    start = text.find(CALL_OPEN)
    search_from = start + len(CALL_OPEN) if start != -1 else 0
    for match in _NAME.finditer(text, search_from):
        name = match.group(1)
        brace = match.end()
        try:
            if text.startswith("{", brace):
                args = _Parser(text, brace).obj()
            else:
                args = {}
        except ToolCallParseError:
            continue
        before = text[: match.start()]
        if start != -1:
            before = text[:start]
        return name, args, clean_reasoning(before)
    raise ToolCallParseError(f"no native tool call found in model output: {text[:200]!r}")


__all__ = ["extract_tool_call", "render_tool_call", "clean_reasoning", "ToolCallParseError", "ESC"]
