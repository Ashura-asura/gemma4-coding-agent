"""Tool interface + agent loop tests — ARCHITECTURE.md §2.2."""
from __future__ import annotations

import sys

import pytest

from agent.context import Context, ToolCall
from agent.loop import ScriptedPolicy, run_episode
from agent.tools import ToolEnv, ToolError, build_toolset, validate_call
from sandbox import Limits, Sandbox


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (root / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (root / "notes.md").write_text("the magic number is 42\n", encoding="utf-8")
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("the magic number is 42\n", encoding="utf-8")
    sandbox = Sandbox(root, limits=Limits(cpu_seconds=10, wall_seconds=60))
    return root, ToolEnv(sandbox=sandbox)


FIXED_DIFF = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


# ------------------------------------------------------------------ read_file
def test_read_file_numbers_lines(repo):
    _, env = repo
    tools = build_toolset(env)
    out = tools["read_file"].execute({"path": "calc.py"})
    assert "1: def add(a, b):" in out
    assert "2:     return a - b" in out


def test_read_file_rejects_path_escape(repo):
    _, env = repo
    tools = build_toolset(env)
    with pytest.raises(ToolError, match="escapes sandbox root"):
        tools["read_file"].execute({"path": "../outside/secret.txt"})


def test_read_file_rejects_directory(repo):
    _, env = repo
    tools = build_toolset(env)
    with pytest.raises(ToolError, match="directory"):
        tools["read_file"].execute({"path": ".git"})


# --------------------------------------------------------------- search_repo
def test_search_repo_finds_literal_match(repo):
    _, env = repo
    tools = build_toolset(env)
    out = tools["search_repo"].execute({"query": "magic number"})
    assert "notes.md:1:" in out
    assert ".git/config" not in out, "search must not walk .git"


def test_search_repo_regex_and_no_match(repo):
    _, env = repo
    tools = build_toolset(env)
    out = tools["search_repo"].execute({"query": r"def \w+\(", "is_regex": True})
    assert "calc.py:1:" in out
    assert "no matches" in tools["search_repo"].execute({"query": "zzzz-not-there"})


# ----------------------------------------------------------------- edit_file
def test_edit_file_applies_unified_diff(repo):
    root, env = repo
    tools = build_toolset(env)
    out = tools["edit_file"].execute({"path": "calc.py", "diff": FIXED_DIFF})
    assert "applied patch" in out
    assert (root / "calc.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"


def test_edit_file_rejects_non_applying_patch_without_writing(repo):
    root, env = repo
    tools = build_toolset(env)
    bad = FIXED_DIFF.replace("return a - b", "return a * b")
    with pytest.raises(ToolError, match="does not apply"):
        tools["edit_file"].execute({"path": "calc.py", "diff": bad})
    assert "return a - b" in (root / "calc.py").read_text(encoding="utf-8"), "file must be untouched"


def test_edit_file_rejects_path_escape(repo):
    _, env = repo
    tools = build_toolset(env)
    with pytest.raises(ToolError, match="escapes sandbox root"):
        tools["edit_file"].execute({"path": "../outside/x.py", "diff": FIXED_DIFF})


def test_edit_file_rejects_multi_file_diff(repo):
    _, env = repo
    tools = build_toolset(env)
    multi = FIXED_DIFF + "\n" + FIXED_DIFF.replace("calc.py", "other.py")
    with pytest.raises(ToolError, match="one file per call"):
        tools["edit_file"].execute({"path": "calc.py", "diff": multi})


def test_edit_file_rejects_diff_for_a_different_file(repo):
    _, env = repo
    tools = build_toolset(env)
    with pytest.raises(ToolError, match="refusing to apply"):
        tools["edit_file"].execute({"path": "notes.md", "diff": FIXED_DIFF})


# ----------------------------------------------------------------- run_tests
def test_run_tests_reports_typed_status(repo):
    _, env = repo
    tools = build_toolset(env)
    out = tools["run_tests"].execute({})
    assert "status=fail" in out, out  # calc.add is wrong on purpose
    assert "1 failed" in out


def test_run_tests_after_fix_passes(repo):
    _, env = repo
    tools = build_toolset(env)
    tools["edit_file"].execute({"path": "calc.py", "diff": FIXED_DIFF})
    out = tools["run_tests"].execute({})
    assert "status=pass" in out, out
    assert "1 passed" in out


def test_run_tests_rejects_target_outside_repo(repo):
    _, env = repo
    tools = build_toolset(env)
    with pytest.raises(ToolError, match="escapes the repository"):
        tools["run_tests"].execute({"target": "../../../etc"})


# -------------------------------------------------------------------- submit
def test_submit_stores_patch(repo):
    _, env = repo
    tools = build_toolset(env)
    out = tools["submit"].execute({"patch": FIXED_DIFF})
    assert "submitted" in out
    assert env.submitted_patch == FIXED_DIFF


def test_submit_rejects_empty_patch(repo):
    _, env = repo
    tools = build_toolset(env)
    with pytest.raises(ToolError, match="empty"):
        tools["submit"].execute({"patch": "   "})
    assert env.submitted_patch is None


def test_validate_call_rejects_missing_and_unknown_args(repo):
    _, env = repo
    tools = build_toolset(env)
    with pytest.raises(ToolError, match="missing required"):
        validate_call(tools["read_file"], {})
    with pytest.raises(ToolError, match="unknown argument"):
        validate_call(tools["read_file"], {"path": "calc.py", "bogus": 1})
    with pytest.raises(ToolError, match="must be a string"):
        validate_call(tools["read_file"], {"path": 7})


# ---------------------------------------------------------------------- loop
def _happy_path_calls():
    return [
        ToolCall.make("read_file", {"path": "calc.py"}),
        ToolCall.make("edit_file", {"path": "calc.py", "diff": FIXED_DIFF}),
        ToolCall.make("run_tests", {}),
        ToolCall.make("submit", {"patch": FIXED_DIFF}),
    ]


def test_episode_happy_path_submits(repo):
    root, env = repo
    tools = build_toolset(env)
    policy = ScriptedPolicy(_happy_path_calls())
    result = run_episode(policy, tools, task_id="t1", issue="add() subtracts")
    assert result.status == "submitted"
    assert result.submitted is True
    assert result.steps == 4
    assert result.submitted_patch == FIXED_DIFF
    assert (root / "calc.py").read_text(encoding="utf-8").startswith("def add")
    # trajectory is internally consistent
    assert [c.id for c in result.context.tool_calls] == [o.tool_call_id for o in result.context.observations]
    assert "status=pass" in result.context.observations[2].content
    assert result.context.observations[3].name == "submit"


def test_episode_exhausts_step_budget(repo):
    _, env = repo
    tools = build_toolset(env)
    policy = ScriptedPolicy(
        [
            ToolCall.make("read_file", {"path": "calc.py"}),
            ToolCall.make("search_repo", {"query": "add"}),
        ]
    )
    result = run_episode(policy, tools, task_id="t2", issue="never submitted", max_steps=2)
    assert result.status == "budget_exhausted"
    assert result.submitted_patch is None
    assert result.steps == 2


def test_episode_handles_unknown_tool_then_recovers(repo):
    _, env = repo
    tools = build_toolset(env)
    policy = ScriptedPolicy(
        [
            ToolCall.make("drop_database", {"target": "prod"}),
            ToolCall.make("edit_file", {"path": "calc.py", "diff": FIXED_DIFF}),
            ToolCall.make("submit", {"patch": FIXED_DIFF}),
        ]
    )
    result = run_episode(policy, tools, task_id="t3", issue="fix add")
    assert result.status == "submitted"
    first = result.context.observations[0]
    assert first.is_error and "unknown tool" in first.content
    assert "read_file" in first.content, "error must list the real tools"


def test_episode_records_tool_rejection_without_crashing(repo):
    _, env = repo
    tools = build_toolset(env)
    policy = ScriptedPolicy(
        [
            ToolCall.make("read_file", {"path": "../secret"}),
            ToolCall.make("submit", {"patch": FIXED_DIFF}),
        ]
    )
    result = run_episode(policy, tools, task_id="t4", issue="escape attempt")
    assert result.status == "submitted"
    assert result.context.observations[0].is_error
    assert "escapes sandbox root" in result.context.observations[0].content


def test_episode_handles_policy_failure(repo):
    _, env = repo
    tools = build_toolset(env)

    class Boom:
        def generate_tool_call(self, context):
            raise RuntimeError("model server died")

    result = run_episode(Boom(), tools, task_id="t5", issue="x")
    assert result.status == "error"
    assert "model server died" in (result.error or "")


def test_episode_with_no_calls_is_an_error_not_a_silent_success(repo):
    _, env = repo
    tools = build_toolset(env)
    result = run_episode(ScriptedPolicy([]), tools, task_id="t6", issue="x")
    assert result.status == "error"
    assert result.submitted is False


def test_episode_output_serialises_to_json(repo):
    import json

    _, env = repo
    tools = build_toolset(env)
    result = run_episode(ScriptedPolicy(_happy_path_calls()), tools, task_id="t7", issue="x")
    record = result.to_record()
    assert json.loads(json.dumps(record))["status"] == "submitted"
    assert record["trajectory"]["steps"] == 4


def test_context_round_trip():
    ctx = Context(system="sys", user="do things")
    call = ToolCall.make("read_file", {"path": "a.py"})
    ctx.add_assistant(call, reasoning="looking")
    from agent.context import Observation

    ctx.add_observation(Observation(call.id, call.name, "contents"))
    restored = Context.from_dict(ctx.to_dict())
    assert restored.steps == 1
    assert restored.tool_calls[0].name == "read_file"
    assert "[tool read_file] contents" in restored.render_flat()
    assert restored.estimate_tokens() > 0
