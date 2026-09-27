"""GRPO reward composition — ARCHITECTURE.md §2.5.

Composed of:
  1. primary: held-out tests pass on the submitted patch   (``test_pass``)
  2. patch validity: applies cleanly, non-empty, no-op-free (``patch_validity``)
  3. small step penalty against runaway tool loops         (``step_penalty``)
  4. penalty when the episode never submits                (``no_submit_penalty``)

The test pass/fail comes from :func:`eval.run_eval.score_submission` — the
same sandbox scorer the eval harness uses, so RL and eval share one reward
source (§2.3: "same sandbox, different task set").
"""
from __future__ import annotations

from typing import Any, Sequence

from eval.run_eval import score_submission

DEFAULT_WEIGHTS: dict[str, float] = {
    "test_pass": 1.0,
    "patch_validity": 0.15,
    "step_penalty": 0.005,
    "no_submit_penalty": 0.1,
}


def reward_weights(config: dict[str, Any]) -> dict[str, float]:
    """Merge the ``reward:`` section of ``configs/rl_config.yaml`` over defaults."""
    merged = dict(DEFAULT_WEIGHTS)
    for key, value in (config.get("reward") or {}).items():
        if key in merged:
            merged[key] = float(value)
    return merged


def episode_reward(
    task: dict[str, Any],
    *,
    patch: str | None,
    submitted: bool,
    steps: int,
    weights: dict[str, float] | None = None,
    limits: Any = None,
    test_timeout: float | None = None,
    score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one finished episode against the task's held-out tests.

    ``score`` passes through a precomputed ``score_submission`` result (the
    eval/rollout path already ran the tests — don't run them twice).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    if score is None:
        kwargs: dict[str, Any] = {}
        if limits is not None:
            kwargs["limits"] = limits
        if test_timeout is not None:
            kwargs["test_timeout"] = test_timeout
        score = score_submission(task, patch, **kwargs)

    test_bonus = w["test_pass"] if score.get("resolved") else 0.0
    validity_bonus = w["patch_validity"] if score.get("patch_valid") else 0.0
    step_cost = -w["step_penalty"] * max(0, int(steps))
    submit_cost = -w["no_submit_penalty"] if not submitted else 0.0

    reward = test_bonus + validity_bonus + step_cost + submit_cost
    return {
        "reward": reward,
        "submitted": bool(submitted),
        "steps": int(steps),
        "resolved": bool(score.get("resolved")),
        "patch_valid": bool(score.get("patch_valid")),
        "test_status": score.get("test_status"),
        "reason": score.get("reason"),
        "components": {
            "test_pass": test_bonus,
            "patch_validity": validity_bonus,
            "step_penalty": step_cost,
            "no_submit_penalty": submit_cost,
        },
    }


def group_normalize(rewards: Sequence[float], eps: float = 1e-8) -> list[float]:
    """Group-relative normalization within one task's k samples (§3.3)."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = variance ** 0.5
    if std < eps:
        return [0.0 for _ in rewards]
    return [(r - mean) / (std + eps) for r in rewards]
