"""Rollout worker tests — §2.5 rollouts, resumable groups."""
from __future__ import annotations

import json
from pathlib import Path

from train.rl.reward import reward_weights
from train.rl.rollout_worker import load_pending, rollout_task


def _pool(tmp_path: Path, ids: list[str]) -> Path:
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        "\n".join(json.dumps({"task_id": tid, "repo": "x/y"}) for tid in ids),
        encoding="utf-8",
    )
    return pool


def test_load_pending_resumes_and_limits(tmp_path: Path):
    pool = _pool(tmp_path, ["a", "b", "c"])
    out = tmp_path / "rollouts.jsonl"
    out.write_text(json.dumps({"task_id": "b"}) + "\n", encoding="utf-8")
    pending = load_pending(pool, out, None)
    assert [t["task_id"] for t in pending] == ["a", "c"]
    assert [t["task_id"] for t in load_pending(pool, out, 1)] == ["a"]
    assert load_pending(pool, tmp_path / "absent.jsonl", None)[0]["task_id"] == "a"


def test_rollout_task_scores_and_normalizes(monkeypatch, tmp_path: Path):
    calls = {"n": 0}
    outcomes = [
        {"resolved": True, "submitted": True, "steps": 3},
        {"resolved": False, "submitted": True, "steps": 10},
        {"resolved": False, "submitted": False, "steps": 30},
    ]

    def fake_evaluate(task, factory, **kwargs):
        spec = outcomes[calls["n"] % len(outcomes)]
        calls["n"] += 1
        return {
            "task_id": task["task_id"],
            "status": "submitted" if spec["submitted"] else "budget_exhausted",
            "steps": spec["steps"],
            "error": None,
            "submitted_patch": "--- a\n+++ b\n" if spec["submitted"] else None,
            "patch_valid": spec["submitted"],
            "resolved": spec["resolved"],
            "test_status": "pass" if spec["resolved"] else "fail",
            "reason": "ok",
            "trajectory": {"messages": [{"role": "user", "content": "i"}]},
            "duration_s": 1.0,
        }

    task = {"task_id": "t1", "repo": "x/y"}
    group = rollout_task(
        task,
        lambda t: None,
        k=3,
        max_steps=30,
        limits=None,
        weights=reward_weights({}),
        evaluate=fake_evaluate,
    )
    assert group["task_id"] == "t1"
    assert len(group["rollouts"]) == 3
    # defaults: test_pass 1.0, validity 0.15, step 0.005, no-submit 0.1
    assert group["rewards"][0] == round(1.0 + 0.15 - 0.005 * 3, 10)
    assert group["rewards"][2] == round(-0.005 * 30 - 0.1, 10)
    assert group["rollouts"][0]["messages"] == [{"role": "user", "content": "i"}]
    mean = sum(group["normalized"]) / len(group["normalized"])
    assert abs(mean) < 1e-6
    assert group["rollouts"][2]["submitted"] is False
