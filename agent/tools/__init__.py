"""Tool interface — ARCHITECTURE.md §2.2.

Five fixed tools. Every tool is scope-checked against the sandbox root and
returns either a text observation or a :class:`ToolError` (never a silent
empty result — §2.3 applies to tools too).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from sandbox import Sandbox


class ToolError(Exception):
    """Tool rejected the call; surfaced to the model as an observation."""


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    def execute(self, arguments: dict[str, Any]) -> str: ...


@dataclass
class ToolEnv:
    """State shared by the tools of one episode.

    ``test_base`` / ``test_targets`` come from the task record so the agent
    runs *that* task's suite (§2.2). ``test_pythonpath`` puts the checkout's
    own packages (root, ``src/``) in front of site-packages. When unset,
    ``run_tests`` falls back to the default pytest command.
    """

    sandbox: Sandbox
    submitted_patch: str | None = None
    test_base: list[str] | None = None
    test_targets: list[str] | None = None
    test_pythonpath: list[str] | None = None


def validate_call(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Check the call against the tool's JSON schema before executing it."""
    if not isinstance(arguments, dict):
        raise ToolError("arguments must be a JSON object")
    schema = tool.parameters or {}
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    missing = [key for key in required if key not in arguments or arguments[key] is None]
    if missing:
        raise ToolError(f"missing required argument(s): {', '.join(missing)}")
    for key, value in arguments.items():
        if key not in properties:
            continue
        expected = properties[key].get("type")
        if expected == "string" and not isinstance(value, str):
            raise ToolError(f"argument {key!r} must be a string")
        if expected == "integer" and not isinstance(value, int):
            raise ToolError(f"argument {key!r} must be an integer")
    unknown = [key for key in arguments if key not in properties]
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(unknown)}")
    return arguments


class BaseTool:
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, env: ToolEnv) -> None:
        self.env = env

    def run(self, **kwargs: Any) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def execute(self, arguments: dict[str, Any]) -> str:
        return self.run(**validate_call(self, arguments))

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def build_toolset(env: ToolEnv) -> dict[str, BaseTool]:
    from .edit_file import EditFile
    from .read_file import ReadFile
    from .run_tests import RunTests
    from .search_repo import SearchRepo
    from .submit import Submit

    tools: list[BaseTool] = [
        ReadFile(env),
        SearchRepo(env),
        EditFile(env),
        RunTests(env),
        Submit(env),
    ]
    return {tool.name: tool for tool in tools}


def tool_schemas(tools: dict[str, BaseTool]) -> list[dict[str, Any]]:
    return [tools[name].schema() for name in sorted(tools)]


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


__all__ = [
    "BaseTool",
    "ToolEnv",
    "ToolError",
    "Tool",
    "build_toolset",
    "tool_schemas",
    "validate_call",
]
