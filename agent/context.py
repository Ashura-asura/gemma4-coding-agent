"""Trajectory / context management — ARCHITECTURE.md §2.2, §3.2."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def make(cls, name: str, arguments: dict[str, Any], call_id: str | None = None) -> "ToolCall":
        return cls(id=call_id or "call_0", name=name, arguments=arguments)

    def to_message_part(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False, sort_keys=True),
            },
        }


@dataclass
class Observation:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False

    def to_message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
            "is_error": self.is_error,
        }


class Context:
    """Chat-style trajectory with a flat rendering for non-chat models."""

    def __init__(self, system: str = "", user: str = "") -> None:
        self.messages: list[dict[str, Any]] = []
        self.tool_calls: list[ToolCall] = []
        self.observations: list[Observation] = []
        if system:
            self.messages.append({"role": "system", "content": system})
        if user:
            self.messages.append({"role": "user", "content": user})

    # ------------------------------------------------------------------ adds
    @property
    def steps(self) -> int:
        return len(self.observations)

    def add_assistant(self, call: ToolCall, reasoning: str = "") -> None:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": reasoning or None,
            "tool_calls": [call.to_message_part()],
        }
        self.messages.append(message)
        self.tool_calls.append(call)

    def add_observation(self, observation: Observation) -> None:
        self.messages.append(observation.to_message())
        self.observations.append(observation)

    def add_note(self, text: str, role: str = "user") -> None:
        """Budget/step reminders appended by the loop (§2.2)."""
        self.messages.append({"role": role, "content": text})

    # --------------------------------------------------------------- render
    def render_flat(self) -> str:
        parts: list[str] = []
        for message in self.messages:
            role = message["role"]
            content = message.get("content") or ""
            if role == "assistant":
                calls = message.get("tool_calls") or []
                for call in calls:
                    fn = call["function"]
                    parts.append(f"[assistant] call {fn['name']}({fn['arguments']})")
                if content:
                    parts.append(f"[assistant] {content}")
            elif role == "tool":
                tag = "ERROR " if message.get("is_error") else ""
                parts.append(f"[tool {message.get('name')}] {tag}{content}")
            else:
                parts.append(f"[{role}] {content}")
        return "\n".join(parts)

    def estimate_tokens(self) -> int:
        # deliberately crude: used for budgeting only, never for training signal
        return max(1, len(self.render_flat()) // 4)

    # ---------------------------------------------------------------- persist
    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "steps": self.steps,
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ],
            "observations": [
                {
                    "tool_call_id": o.tool_call_id,
                    "name": o.name,
                    "content": o.content,
                    "is_error": o.is_error,
                }
                for o in self.observations
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Context":
        ctx = cls()
        ctx.messages = list(data.get("messages", []))
        ctx.tool_calls = [
            ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
            for c in data.get("tool_calls", [])
        ]
        ctx.observations = [
            Observation(
                tool_call_id=o["tool_call_id"],
                name=o["name"],
                content=o["content"],
                is_error=o.get("is_error", False),
            )
            for o in data.get("observations", [])
        ]
        return ctx

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.messages)
