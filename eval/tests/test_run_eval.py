"""Eval harness tests — ARCHITECTURE.md §2.6 (Rung-0 smoke, no model needed)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.loop import ScriptedPolicy
from agent.context import ToolCall
from eval.report import (
    categorize,
    format_report,
    paired_comparison,
    summarize,
    wilson_interval,
)
from eval.run_eval import (
    PatchError,
    apply_patch_to_tree,
    cap_prompt_tokens,
    evaluate_task,
    load_tasks,
    parse_tool_call,
    policy_dtype,
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
    assert summary["resolved_rate_ci95"][0] < summary["resolved_rate"] < summary["resolved_rate_ci95"][1]


def test_wilson_interval_is_wide_at_24_tasks() -> None:
    """3/24 resolved is ~12.5% with a 95% interval spanning roughly 4-31%."""
    low, high = wilson_interval(3, 24)
    assert 0.03 < low < 0.06
    assert 0.28 < high < 0.34


def test_paired_comparison_counts_discordant_tasks() -> None:
    treatment = [
        {"task_id": "a", "resolved": True},
        {"task_id": "b", "resolved": False},
        {"task_id": "c", "resolved": True},
    ]
    baseline = [
        {"task_id": "a", "resolved": False},
        {"task_id": "b", "resolved": True},
        {"task_id": "c", "resolved": True},
    ]
    paired = paired_comparison(treatment, baseline)
    assert paired["n_paired"] == 3
    assert paired["treatment_only"] == 1
    assert paired["baseline_only"] == 1
    assert paired["both"] == 1
    assert paired["neither"] == 0
    assert paired["mcnemar_exact_p"] == 1.0


def test_policy_dtype_matches_trainer_on_t4() -> None:
    """T4 is not bf16-capable; eval must use fp16 like SFT, not bf16."""
    fake = SimpleNamespace(
        float32="fp32",
        float16="fp16",
        bfloat16="bf16",
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False),
    )
    assert policy_dtype(fake) == "fp16"
    fake.cuda.is_bf16_supported = lambda: True
    assert policy_dtype(fake) == "bf16"
    fake.cuda.is_available = lambda: False
    assert policy_dtype(fake) == "fp32"


def test_categorize_test_environment_issue() -> None:
    record = {
        "status": "submitted",
        "resolved": False,
        "patch_valid": True,
        "test_status": "timeout",
    }
    assert categorize(record) == "test_environment_issue"


def test_categorize_environment_setup_failure_is_not_a_policy_error() -> None:
    """A broken task venv fails before inference; do not blame the policy."""
    record = {
        "status": "error",
        "resolved": False,
        "patch_valid": False,
        "test_status": "error",
        "error": "environment setup failed: RuntimeError: pip install failed for pydata/xarray",
    }
    assert categorize(record) == "test_environment_issue"


def test_categorize_cuda_oom_is_infrastructure_not_a_policy_error() -> None:
    """A 16 GiB T4 can run out of memory mid-episode; that is not a decision."""
    record = {
        "status": "error",
        "resolved": False,
        "patch_valid": False,
        "test_status": "error",
        "error": "policy raised OutOfMemoryError: CUDA out of memory. Tried to allocate 2.12 GiB",
    }
    assert categorize(record) == "infrastructure_error"


def test_main_dump_trajectories_survives_early_exit_records(tmp_path, monkeypatch) -> None:
    """Env/checkout early exits have no 'trajectory' key — the run must not crash
    (kernel eval v2 died with KeyError on the first Kaggle task)."""
    import eval.run_eval as run_eval

    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"task_id": "t1", "repo": "r1", "issue": "x"}) + "\n")
    out = tmp_path / "out.jsonl"
    traj = tmp_path / "traj.jsonl"

    def fake_evaluate(task, factory, **kwargs):
        return {
            "task_id": task["task_id"], "repo": "r1", "status": "error", "steps": 0,
            "resolved": False, "patch_valid": False, "test_status": "error",
            "reason": "environment setup failed",
        }

    monkeypatch.setattr(run_eval, "evaluate_task", fake_evaluate)
    code = run_eval.main([
        "--policy", "null", "--tasks", str(tasks), "--out", str(out),
        "--dump-trajectories", str(traj),
    ])
    assert code == 0
    assert json.loads(traj.read_text(encoding="utf-8").splitlines()[0]) is None
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["task_id"] == "t1"


def test_cap_prompt_tokens_keeps_head_and_tail() -> None:
    """Eval v4 OOMed once contexts grew past ~10k tokens; the cap must keep
    the issue/system head and the newest turns, and stay within the budget."""
    torch = pytest.importorskip("torch")
    ids = torch.arange(20).reshape(1, 20)
    mask = torch.ones_like(ids)
    out = cap_prompt_tokens({"input_ids": ids, "attention_mask": mask}, 10, torch)
    assert out["input_ids"].shape == (1, 10)
    assert out["input_ids"][0, 0].item() == 0
    assert out["input_ids"][0, -1].item() == 19
    assert out["attention_mask"].shape == (1, 10)


def test_cap_prompt_tokens_is_noop_under_limit() -> None:
    torch = pytest.importorskip("torch")
    ids = torch.arange(4).reshape(1, 4)
    out = cap_prompt_tokens({"input_ids": ids}, 10, torch)
    assert out["input_ids"] is ids
