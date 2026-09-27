"""RL rollouts — ARCHITECTURE.md §2.5 / §3.2.

    python -m train.rl.rollout_worker --config configs/rl_config.yaml [--limit N]

For every task in ``rollout.task_pool`` (``rl_task_pool.jsonl``): sample
*k* trajectories from the current policy through the **same agent loop and
tool interface as eval**, score each with the §2.5 reward (reusing the
score already computed inside the eval path — tests run once), group-
normalize within the task, and append one resumable group record per task.

Needs a GPU (§5): the policy is a 12B model — run this on Kaggle, not on
the local 2GB card.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from eval.run_eval import evaluate_task
from train.rl.reward import episode_reward, group_normalize, reward_weights


def load_pending(pool_path: Path, out_path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Task records from the RL pool, minus groups already written (resumable)."""
    if not pool_path.exists():
        raise SystemExit(f"missing task pool: {pool_path}")
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(str(json.loads(line).get("task_id")))
    tasks = []
    for line in pool_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record.get("task_id")) not in done:
            tasks.append(record)
    return tasks[:limit] if limit else tasks


def rollout_task(
    task: dict[str, Any],
    factory: Callable[[dict[str, Any]], Any],
    *,
    k: int,
    max_steps: int,
    limits: Any,
    weights: dict[str, float],
    evaluate: Callable[..., dict[str, Any]] = evaluate_task,
) -> dict[str, Any]:
    """k trajectories for one task, scored + group-normalized (§3.3)."""
    rollouts: list[dict[str, Any]] = []
    for _ in range(k):
        record = evaluate(
            task, factory, max_steps=max_steps, limits=limits, dump_trajectory=True
        )
        submitted = record.get("submitted_patch") is not None
        score = {
            "patch_valid": record.get("patch_valid"),
            "resolved": record.get("resolved"),
            "test_status": record.get("test_status"),
            "reason": record.get("reason"),
        }
        reward = episode_reward(
            task,
            patch=record.get("submitted_patch"),
            submitted=submitted,
            steps=int(record.get("steps") or 0),
            weights=weights,
            score=score,
        )
        trajectory = record.get("trajectory") or {}
        rollouts.append(
            {
                "messages": trajectory.get("messages"),
                "status": record.get("status"),
                "steps": reward["steps"],
                "submitted": submitted,
                "resolved": reward["resolved"],
                "patch_valid": reward["patch_valid"],
                "patch": record.get("submitted_patch"),
                "reward": reward["reward"],
                "components": reward["components"],
                "error": record.get("error"),
                "duration_s": record.get("duration_s"),
            }
        )
    normalized = group_normalize([r["reward"] for r in rollouts])
    for rollout, value in zip(rollouts, normalized):
        rollout["normalized"] = value
    return {
        "task_id": task["task_id"],
        "repo": task.get("repo"),
        "rewards": [r["reward"] for r in rollouts],
        "normalized": normalized,
        "rollouts": rollouts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample + score k rollouts per RL task (§2.5)")
    parser.add_argument("--config", default="configs/rl_config.yaml")
    parser.add_argument("--policy", help="model/adapter path (default: config policy_checkpoint)")
    parser.add_argument("--limit", type=int, help="only roll out the first N pending tasks")
    parser.add_argument("--out", default="data/rl_rollouts.jsonl")
    parser.add_argument("--quantize", action="store_true", default=True, help="4-bit policy load (default on)")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.add_argument(
        "--null-policy",
        action="store_true",
        help="ScriptedPolicy smoke run without loading a model",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rollout_cfg = config.get("rollout") or {}
    k = int(rollout_cfg.get("k", 8))
    max_steps = int(rollout_cfg.get("max_steps", 30))
    max_new_tokens = int(rollout_cfg.get("max_tokens_per_step", 2048))
    temperature = float(rollout_cfg.get("temperature", 1.0))

    from eval.run_eval import Limits, HFPolicy
    from agent.loop import ScriptedPolicy

    limits = Limits(wall_seconds=int(rollout_cfg.get("wall_seconds", 600)))
    weights = reward_weights(config)
    out_path = Path(args.out)
    pool_path = Path(rollout_cfg.get("task_pool") or "data/rl_task_pool.jsonl")
    pending = load_pending(pool_path, out_path, args.limit)
    if not pending:
        print(f"[rollout] nothing pending in {pool_path}", flush=True)
        return 0

    if args.null_policy:
        shared: Any = ScriptedPolicy([])
    else:
        model = args.policy or config.get("policy_checkpoint")
        if not model or not Path(model).exists():
            raise SystemExit(
                f"policy checkpoint not found: {model!r} — pass --policy or run SFT first"
            )
        print(f"[rollout] loading policy {model} (quantize={args.quantize})", flush=True)
        shared = HFPolicy(
            model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            quantize=args.quantize,
        )

    def factory(_task: dict[str, Any]) -> Any:
        return shared

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = -1e9
    for index, task in enumerate(pending, start=1):
        started = time.monotonic()
        try:
            group = rollout_task(
                task,
                factory,
                k=k,
                max_steps=max_steps,
                limits=limits,
                weights=weights,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch alive (§2.3 spirit)
            print(f"[rollout] {task['task_id']} crashed: {type(exc).__name__}: {exc}", flush=True)
            continue
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(group, ensure_ascii=False) + "\n")
        best = max(best, max(group["rewards"]))
        print(
            f"[rollout] {task['task_id']} ({index}/{len(pending)}) "
            f"rewards={[round(r, 3) for r in group['rewards']]} "
            f"{time.monotonic() - started:.0f}s",
            flush=True,
        )

    print(
        f"[rollout] done: {len(pending)} task group(s) -> {out_path} (best reward {best:.3f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
