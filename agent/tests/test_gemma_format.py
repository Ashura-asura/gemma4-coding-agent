"""Native Gemma-4 tool-call parsing and recoverable malformed output (AUDIT C3)."""
from __future__ import annotations

import pytest

from agent.context import ToolCall
from agent.gemma_format import (
    ESC,
    ToolCallParseError,
    extract_tool_call,
    render_tool_call,
)
from agent.loop import MalformedCall, run_episode
from agent.tools import ToolEnv, build_toolset, default_tool_specs
from eval.run_eval import parse_tool_call
from sandbox import Sandbox

DIFF = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,3 +1,3 @@\n def f():\n-    return {'a': [1, 2]}\n+    return {'a': [1, 2, 3]}  # } , <|\"|> tricky\n"


def test_native_call_roundtrip_including_braces_and_newlines():
    # the escape token cannot appear inside an escaped string, so drop it
    plain = DIFF.replace(ESC, "")
    text = render_tool_call("edit_file", {"path": "pkg/mod.py", "diff": plain})
    name, args, reasoning = extract_tool_call(text)
    assert (name, args) == ("edit_file", {"path": "pkg/mod.py", "diff": plain})
    assert reasoning == ""


def test_scalars_lists_and_nesting():
    text = render_tool_call(
        "search_repo", {"query": "x, y", "max_results": 5, "regex": False, "globs": ["*.py", "a b"], "opt": None}
    )
    _, args, _ = extract_tool_call(text)
    assert args == {"query": "x, y", "max_results": 5, "regex": False, "globs": ["*.py", "a b"], "opt": None}


def test_reasoning_and_thought_channels_are_separated():
    text = (
        "<|channel>thought\nprivate scratch<channel|>I'll read the file first.\n"
        '<|tool_call>call:read_file{path:<|"|>pkg/mod.py<|"|>,start_line:10}<tool_call|><turn|>'
    )
    name, args, reasoning = extract_tool_call(text)
    assert name == "read_file" and args == {"path": "pkg/mod.py", "start_line": 10}
    assert reasoning == "I'll read the file first."


def test_missing_closing_token_and_stripped_special_tokens():
    # generation stopped early: no <tool_call|>
    name, args, _ = extract_tool_call('<|tool_call>call:submit{}')
    assert (name, args) == ("submit", {})
    # a lossy decode (skip_special_tokens=True) leaves bare text
    name, args, _ = extract_tool_call('call:read_file{path:"a/b.py",start_line:3}')
    assert (name, args) == ("read_file", {"path": "a/b.py", "start_line": 3})


def test_first_call_wins_and_garbage_raises():
    text = render_tool_call("submit", {}) + render_tool_call("read_file", {"path": "x"})
    assert extract_tool_call(text)[0] == "submit"
    with pytest.raises(ToolCallParseError):
        extract_tool_call("I think the bug is in the parser.")


def test_parse_tool_call_prefers_native_and_still_accepts_json():
    call = parse_tool_call("Reading.\n" + render_tool_call("read_file", {"path": "a.py"}))
    assert (call.name, call.arguments, call.reasoning) == ("read_file", {"path": "a.py"}, "Reading.")
    call = parse_tool_call('```json\n{"name": "submit", "arguments": {}}\n```')
    assert call.name == "submit"
    with pytest.raises(MalformedCall):
        parse_tool_call("no call here")


def test_tool_specs_cover_the_five_tools():
    names = sorted(t["function"]["name"] for t in default_tool_specs())
    assert names == ["edit_file", "read_file", "run_tests", "search_repo", "submit"]


# ----------------------------------------------------------- loop recovery
class _Flaky:
    """Two unparseable outputs, then a good submit."""

    def __init__(self, bad: int):
        self.bad, self.calls = bad, 0

    def generate_tool_call(self, context):
        self.calls += 1
        if self.calls <= self.bad:
            raise MalformedCall("garbled")
        return ToolCall.make("submit", {}, None)


@pytest.fixture()
def tools(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    env = ToolEnv(sandbox=Sandbox(tmp_path))
    return build_toolset(env)


def test_malformed_output_is_recoverable_within_budget(tools):
    result = run_episode(_Flaky(2), tools, task_id="t", issue="i", max_steps=6, max_malformed=2)
    assert result.status != "error" and result.error is None
    notes = [m for m in result.context.messages if m["role"] == "user" and "not a valid tool call" in m["content"]]
    assert len(notes) == 2


def test_too_many_malformed_outputs_end_the_episode(tools):
    result = run_episode(_Flaky(5), tools, task_id="t", issue="i", max_steps=10, max_malformed=2)
    assert result.status == "error" and "unusable policy output" in result.error


def test_reasoning_survives_into_the_history(tools):
    class P:
        def generate_tool_call(self, context):
            return ToolCall(id="x", name="submit", arguments={}, reasoning="Done, tests pass.")

    result = run_episode(P(), tools, task_id="t", issue="i", max_steps=2)
    assistant = [m for m in result.context.messages if m["role"] == "assistant"][0]
    assert assistant["content"] == "Done, tests pass."
