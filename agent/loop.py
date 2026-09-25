"""plan → act → observe loop — ARCHITECTURE.md §2.2, §3.2.

Shared verbatim by RL rollouts and eval, so training and scoring exercise
the same code path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from .context import Context, Observation, ToolCall
from .tools import BaseTool, ToolEnv, ToolError

SYSTEM_PROMPT = """You are an autonomous software-engineering agent.
You are given an issue and a repository, and you must produce a patch that
makes the repository's tests pass.

Workflow: investigate with read_file and search_repo, make a minimal change
with edit_file (unified diff), verify with run_tests, then call submit.

Rules:
- Only the listed tools exist; their JSON arguments must be exact.
- Paths are relative to the repository root and cannot escape it.
- A tool result starting with "error:" means the call was rejected; fix the
  arguments and try again.
- Call submit exactly once, when you are confident the tests pass.
"""


class Policy(Protocol):
    def generate_tool_call(self, context: Context) -> ToolCall | None: ...


class ScriptedPolicy:
    """Deterministic policy for harness self-tests and Rung-0 smoke runs."""

    def __init__(self, calls: Sequence[ToolCall | Callable[[Context], ToolCall | None]]) -> None:
        self._calls = list(calls)
        self._index = 0

    def generate_tool_call(self, context: Context) -> ToolCall | None:
        if self._index >= len(self._calls):
            return None
        item = self._calls[self._index]
        self._index += 1
        return item(context) if callable(item) else item


@dataclass
class EpisodeResult:
    task_id: str
    status: str  # "submitted" | "budget_exhausted" | "error"
    steps: int
    submitted_patch: str | None
    error: str | None
    context: Context
    tool_env: ToolEnv | None = field(default=None, repr=False)

    @property
    def submitted(self) -> bool:
        return self.status == "submitted" and bool(self.submitted_patch)

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "steps": self.steps,
            "submitted_patch": self.submitted_patch,
            "error": self.error,
            "trajectory": self.context.to_dict(),
        }


def _next_call_id(context: Context) -> str:
    return f"call_{len(context.tool_calls)}"


def _coerce_call(candidate: Any, context: Context) -> ToolCall:
    """Normalise whatever the policy emitted into a ToolCall with our own id."""
    call_id = _next_call_id(context)
    if candidate is None:
        raise ValueError("policy returned no tool call")
    if isinstance(candidate, ToolCall):
        return ToolCall(id=call_id, name=candidate.name, arguments=candidate.arguments)
    if isinstance(candidate, tuple) and len(candidate) == 2:
        return ToolCall(id=call_id, name=candidate[0], arguments=candidate[1])
    if isinstance(candidate, dict):
        return ToolCall(
            id=call_id,
            name=str(candidate["name"]),
            arguments=dict(candidate.get("arguments") or {}),
        )
    if isinstance(candidate, str):  # tolerate a bare JSON tool call
        parsed = json.loads(candidate)
        if "function" in parsed:
            return ToolCall(
                id=call_id,
                name=parsed["function"]["name"],
                arguments=json.loads(parsed["function"].get("arguments") or "{}"),
            )
        return ToolCall(
            id=call_id,
            name=str(parsed["name"]),
            arguments=dict(parsed.get("arguments") or {}),
        )
    raise ValueError(f"cannot interpret policy output of type {type(candidate).__name__}")


def run_episode(
    policy: Policy,
    tools: dict[str, BaseTool],
    *,
    task_id: str,
    issue: str,
    repo_state: str = "",
    max_steps: int = 30,
    system_prompt: str = SYSTEM_PROMPT,
) -> EpisodeResult:
    """Execute one episode. Always returns a typed result (§2.3 spirit)."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if "submit" not in tools:
        raise ValueError("the tool set must include submit")

    prompt_parts = [f"Issue:\n{issue}"]
    if repo_state:
        prompt_parts.append(f"Repository state:\n{repo_state}")
    context = Context(system=system_prompt, user="\n\n".join(prompt_parts))
    env = tools["submit"].env

    status = "budget_exhausted"
    error: str | None = None

    for _ in range(max_steps):
        try:
            raw_call = policy.generate_tool_call(context)
        except Exception as exc:  # a crashing policy must not lose the episode record
            status, error = "error", f"policy raised {type(exc).__name__}: {exc}"
            break

        try:
            call = _coerce_call(raw_call, context)
        except Exception as exc:
            status, error = "error", f"unusable policy output: {exc}"
            break

        if call.name not in tools:
            content = (
                f"error: unknown tool {call.name!r}; "
                f"available tools: {', '.join(sorted(tools))}"
            )
            is_error = True
        else:
            try:
                content = tools[call.name].execute(call.arguments)
                is_error = False
            except ToolError as exc:
                content, is_error = f"error: {exc}", True
            except Exception as exc:
                content = f"error: tool crashed with {type(exc).__name__}: {exc}"
                is_error = True

        context.add_assistant(call)
        context.add_observation(Observation(call.id, call.name, content, is_error))

        if call.name == "submit" and not is_error:
            status = "submitted"
            break
    else:
        context.add_note(f"step budget of {max_steps} exhausted before submit")

    return EpisodeResult(
        task_id=task_id,
        status=status,
        steps=context.steps,
        submitted_patch=env.submitted_patch,
        error=error,
        context=context,
        tool_env=env,
    )
