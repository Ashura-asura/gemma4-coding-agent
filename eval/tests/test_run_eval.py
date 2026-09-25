"""Eval harness tests — ARCHITECTURE.md §2.6 (Rung-0 smoke, no model needed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.loop import ScriptedPolicy
from agent.context import ToolCall
from eval.report import categorize, format_report, summarize
from eval.run_eval import (
    PatchError,
    apply_patch_to_tree,
    evaluate_task,
    load_tasks,
    parse_tool_call,
    score_submission,
    split_multifile_patch,
)
from sandbox import Limits

FIX_DIFF = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""

NOOP_DIFF = """\
--- a/notes.md
+++ b/notes.md
@@ -1 +1 @@
-the magic number is 42
+the magic number is 43
"""

BAD_DIFF = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a * b
+    return a + b
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (root / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    (root / "notes.md").write_text("the magic number is 42\n", encoding="utf-8")
    return root


@pytest.fixture
def task(repo: Path) -> dict:
    return {"task_id": "t-1", "repo": str(repo), "issue": "add() subtracts"}


@pytest.fixture
def limits() -> Limits:
    return Limits(cpu_seconds=15, wall_seconds=300)


# -------------------------------------------------------------------- loading
def test_load_tasks_accepts_valid_and_rejects_incomplete(tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"task_id": "a", "repo": "r", "issue": "i"}) + "\n\n", encoding="utf-8"
    )
    assert len(load_tasks(good)) == 1

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"task_id": "a", "repo": "r"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_tasks(bad)


# --------------------------------------------------------------------- patches
def test_split_multifile_patch() -> None:
    patch = (
        "diff --git a/calc.py b/calc.py\n"
        + FIX_DIFF
        + "diff --git a/notes.md b/notes.md\n"
        + NOOP_DIFF
    )
    sections = split_multifile_patch(patch)
    assert set(sections) == {"calc.py", "notes.md"}
    assert "return a + b" in sections["calc.py"]


def test_apply_patch_to_tree_writes_files(repo: Path) -> None:
    applied = apply_patch_to_tree(FIX_DIFF, repo)
    assert applied == ["calc.py"]
    assert "return a + b" in (repo / "calc.py").read_text(encoding="utf-8")


def test_apply_patch_to_tree_rejects_escapes(repo: Path) -> None:
    escaping = """\
--- a/../evil.py
+++ b/../evil.py
@@ -0,0 +1 @@
+pwned
"""
    with pytest.raises(PatchError, match="escapes the repository"):
        apply_patch_to_tree(escaping, repo)
    assert not (repo.parent / "evil.py").exists()


def test_apply_patch_to_tree_rejects_non_applying_patch(repo: Path) -> None:
    with pytest.raises(PatchError, match="does not apply"):
        apply_patch_to_tree(BAD_DIFF, repo)
    assert "return a - b" in (repo / "calc.py").read_text(encoding="utf-8")


# -------------------------------------------------------------------- scoring
def test_score_empty_patch_is_invalid(task: dict, limits: Limits) -> None:
    result = score_submission(task, "", limits=limits)
    assert result["patch_valid"] is False and result["resolved"] is False
    assert categorize({**result, "status": "submitted"}) == "incomplete_patch"


def test_score_correct_patch_resolves(task: dict, limits: Limits) -> None:
    result = score_submission(task, FIX_DIFF, limits=limits)
    assert result["patch_valid"] is True
    assert result["resolved"] is True, result
    assert result["test_status"] == "pass"
    assert categorize({**result, "status": "submitted"}) == "resolved"


def test_score_valid_but_wrong_patch_reports_failure(task: dict, limits: Limits) -> None:
    result = score_submission(task, NOOP_DIFF, limits=limits)
    assert result["patch_valid"] is True
    assert result["resolved"] is False
    assert result["test_status"] == "fail"
    assert categorize({**result, "status": "submitted"}) == "wrong_file"


def test_score_non_applying_patch_is_invalid(task: dict, limits: Limits) -> None:
    result = score_submission(task, BAD_DIFF, limits=limits)
    assert result["patch_valid"] is False
    assert "does not apply" in result["reason"]


# ---------------------------------------------------------------- episode runs
def test_evaluate_task_resolves_with_fixing_policy(task: dict, limits: Limits) -> None:
    def factory(_task: dict) -> ScriptedPolicy:
        return ScriptedPolicy(
            [
                ToolCall.make("edit_file", {"path": "calc.py", "diff": FIX_DIFF}),
                ToolCall.make("run_tests", {}),
                ToolCall.make("submit", {"patch": FIX_DIFF}),
            ]
        )

    record = evaluate_task(task, factory, max_steps=5, limits=limits)
    assert record["status"] == "submitted"
    assert record["resolved"] is True, record
    assert record["steps"] == 3
    assert record["patch_valid"] is True


def test_evaluate_task_budget_exhaustion(task: dict, limits: Limits) -> None:
    def factory(_task: dict) -> ScriptedPolicy:
        return ScriptedPolicy([ToolCall.make("read_file", {"path": "calc.py"})])

    record = evaluate_task(task, factory, max_steps=1, limits=limits)
    assert record["status"] == "budget_exhausted"
    assert record["resolved"] is False
    assert categorize(record) == "step_budget_exhaustion"


def test_evaluate_task_missing_repo_is_reported(tmp_path: Path, limits: Limits) -> None:
    task = {"task_id": "x", "repo": str(tmp_path / "nope"), "issue": "i"}
    record = evaluate_task(task, lambda _t: ScriptedPolicy([]), limits=limits)
    assert record["status"] == "error"
    assert "not found" in record["reason"]


# ------------------------------------------------------------------- policies
def test_parse_tool_call_variants() -> None:
    call = parse_tool_call('Sure.\n```json\n{"tool": "read_file", "arguments": {"path": "a.py"}}\n```')
    assert call.name == "read_file" and call.arguments == {"path": "a.py"}

    call = parse_tool_call('{"name": "submit", "arguments": {"patch": "-- x\\n+y"}}')
    assert call.name == "submit"

    call = parse_tool_call('{"function": {"name": "run_tests", "arguments": "{}"}}')
    assert call.name == "run_tests"

    with pytest.raises(ValueError, match="no tool call"):
        parse_tool_call("I could not decide what to do.")


# -------------------------------------------------------------------- report
def test_summarize_and_format() -> None:
    records = [
        {"task_id": "a", "repo": "r1", "status": "submitted", "steps": 4,
         "resolved": True, "patch_valid": True, "test_status": "pass"},
        {"task_id": "b", "repo": "r1", "status": "submitted", "steps": 8,
         "resolved": False, "patch_valid": True, "test_status": "fail"},
        {"task_id": "c", "repo": "r2", "status": "budget_exhausted", "steps": 30,
         "resolved": False, "patch_valid": False, "test_status": "error"},
        {"task_id": "d", "repo": "r2", "status": "submitted", "steps": 2,
         "resolved": False, "patch_valid": False, "test_status": "error"},
    ]
    summary = summarize(records)
    assert summary["n_tasks"] == 4
    assert summary["n_resolved"] == 1
    assert summary["resolved_rate"] == 0.25
    assert summary["patch_validity_rate"] == 0.5
    assert summary["avg_steps_to_submit"] == pytest.approx(4.666, abs=0.01)
    assert summary["per_repo"]["r1"] == {"n": 2, "resolved": 1, "resolved_rate": 0.5}
    assert summary["failure_categories"]["step_budget_exhaustion"] == 1
    assert summary["failure_categories"]["incomplete_patch"] == 1
    assert summary["failure_categories"]["wrong_file"] == 1

    text = format_report(summary)
    assert "resolved-rate" in text
    assert "per repo:" in text
    assert "r2" in text


def test_categorize_test_environment_issue() -> None:
    record = {
        "status": "submitted",
        "resolved": False,
        "patch_valid": True,
        "test_status": "timeout",
    }
    assert categorize(record) == "test_environment_issue"
