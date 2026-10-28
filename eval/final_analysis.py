"""Reproduce the paper's held-out tables from the final eval records.

    python -m eval.final_analysis sft.jsonl base.jsonl \
        --sft-traj sft_traj.jsonl --base-traj base_traj.jsonl \
        --sft-data data/sft_trajectories.compact.jsonl

Two kinds of unsuccessful episode are *not* policy failures and are reported
separately:

``test_environment_issue``
    the task's venv could not be built, so the policy was never asked for a
    tool call (all four xarray tasks on Kaggle's Python 3.13).
``infrastructure_error``
    the episode hit CUDA OOM mid-generation. The model is truncated by the
    machine, so the episode says nothing about its policy.

The paper reports three nested samples:

* **attempted** — every task in the frozen subset (24);
* **effective** — attempted minus environment failures (20);
* **clean** — effective minus pairs where either arm OOM'd (8).

Only the clean sample supports a like-for-like comparison.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval.report import categorize, load_records, paired_comparison, wilson_interval

ENV_MARKERS = ("environment setup failed", "pip install failed")
OOM_MARKERS = ("outofmemoryerror", "cuda out of memory")


def kind(record: dict[str, Any]) -> str:
    """environment | oom | scored."""
    error = str(record.get("error") or "").lower()
    if any(marker in error for marker in ENV_MARKERS):
        return "environment"
    if any(marker in error for marker in OOM_MARKERS):
        return "oom"
    return "scored"


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} = {k / n:.1%} (95% Wilson {lo:.1%}-{hi:.1%})"


def sample(records: list[dict], ids: list[str]) -> list[dict]:
    by_id = {str(r.get("task_id")): r for r in records}
    return [by_id[i] for i in ids]


def repo_map(path: str | None) -> dict[str, str]:
    """task_id -> repo. Eval v6 predates the fix that preserved ``repo`` in the
    record (it stored the checkout path there), so map from the frozen subset."""
    if not path:
        return {}
    return {
        str(row["task_id"]): str(row.get("repo") or "unknown")
        for row in load_records(path)
    }


def report_arm(name: str, records: list[dict], ids: list[str], repos_of: dict[str, str]) -> None:
    rows = sample(records, ids)
    resolved = sum(bool(r.get("resolved")) for r in rows)
    submitted = sum(str(r.get("status")) == "submitted" for r in rows)
    valid = sum(bool(r.get("patch_valid")) for r in rows)
    cats = Counter(categorize(r) for r in rows)
    repos: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        repo = repos_of.get(str(r.get("task_id"))) or str(r.get("repo") or "unknown")
        repos[repo][0] += 1
        repos[repo][1] += int(bool(r.get("resolved")))
    print(f"  {name:<5} resolved {fmt_rate(resolved, len(rows))}")
    print(f"        submitted {submitted}/{len(rows)} | patch_valid {valid}/{len(rows)}")
    print("        categories: " + ", ".join(f"{c}={n}" for c, n in sorted(cats.items())))
    print("        per repo: " + ", ".join(f"{r} {v[1]}/{v[0]}" for r, v in sorted(repos.items())))


def trajectory_table(label: str, path: str | None) -> None:
    if not path:
        return
    results = load_records(path.replace("_traj", ""))
    trajs = load_records(path)
    if len(results) != len(trajs):
        print(f"  WARNING: {len(results)} results vs {len(trajs)} trajectories")
    calls = Counter()
    whole = ranged = 0
    episodes = submits = repeated = never_submit = stuck = long_episodes = 0
    tokens: list[int] = []
    for result, traj in zip(results, trajs):
        if not isinstance(traj, dict):
            continue
        tc = [c for c in (traj.get("tool_calls") or []) if isinstance(c, dict)]
        if not tc:
            continue
        episodes += 1
        names = [c.get("name") for c in tc]
        calls.update(names)
        if "submit" not in names:
            never_submit += 1
        submits += names.count("submit")
        run = best = 0
        prev = None
        for c in tc:
            sig = (c.get("name"), json.dumps(c.get("arguments"), sort_keys=True, default=str))
            run = run + 1 if sig == prev else 1
            prev = sig
            best = max(best, run)
        if best >= 3:
            repeated += 1
        # A periodic loop (edit, test, edit, test) has no long *consecutive*
        # run but is still stuck; count distinct actions instead.
        if len(tc) >= 4:
            long_episodes += 1
            if len({(c.get("name"), json.dumps(c.get("arguments"), sort_keys=True, default=str)) for c in tc}) <= 3:
                stuck += 1
        for c in tc:
            if c.get("name") != "read_file":
                continue
            args = c.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if args.get("start_line") and args.get("end_line"):
                ranged += 1
            else:
                whole += 1
        tokens.extend(len(c.get("completion_ids") or []) for c in tc)
    print(f"  {label}: episodes with calls {episodes}; calls {sum(calls.values())}")
    print("    tool mix: " + ", ".join(f"{k}={v}" for k, v in calls.most_common()))
    print(f"    read_file ranged={ranged} whole_file={whole}")
    print(f"    submits={submits} episodes_never_submitting={never_submit}/{episodes}")
    print(f"    identical-call run >=3: {repeated}/{episodes}")
    print(f"    episodes >=4 calls with <=3 distinct calls: {stuck}/{long_episodes}")
    if tokens:
        print(
            f"    completion tokens/call: median={statistics.median(tokens):.0f} "
            f"max={max(tokens)}"
        )


def supervision_table(path: str) -> None:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    calls: Counter = Counter()
    errors: Counter = Counter()
    with_error = 0
    submit_after_pass = 0
    for row in rows:
        messages = row["messages"]
        hit = False
        for message in messages:
            for call in message.get("tool_calls") or []:
                calls[(call.get("function", call) or {}).get("name")] += 1
            if message.get("role") != "tool":
                continue
            content = str(message.get("content", ""))
            if content.startswith("error:"):
                errors[content.split(":")[1].strip()] += 1
                hit = True
        if hit:
            with_error += 1
        for i, message in enumerate(messages):
            names = [
                (c.get("function", c) or {}).get("name") for c in (message.get("tool_calls") or [])
            ]
            if "submit" in names and i > 0:
                submit_after_pass += "status=pass" in str(messages[i - 1].get("content", ""))[:32]
                break
    print(f"  trajectories {len(rows)}; with >=1 tool error {with_error}")
    print("  tool mix: " + ", ".join(f"{k}={v}" for k, v in calls.most_common()))
    print("  error observations: " + (", ".join(f"{k}={v}" for k, v in errors.most_common()) or "none"))
    print(f"  submit immediately after a passing run_tests: {submit_after_pass}/{len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sft")
    parser.add_argument("base")
    parser.add_argument("--sft-traj")
    parser.add_argument("--base-traj")
    parser.add_argument("--sft-data", help="compacted SFT trajectories, for the supervision table")
    parser.add_argument("--subset", default="data/holdout_eval_subset.jsonl")
    args = parser.parse_args()

    sft = load_records(args.sft)
    base = load_records(args.base)
    sft_by = {str(r.get("task_id")): r for r in sft}
    base_by = {str(r.get("task_id")): r for r in base}
    ids = sorted(set(sft_by) & set(base_by))

    env = [i for i in ids if "environment" in (kind(sft_by[i]), kind(base_by[i]))]
    effective = [i for i in ids if i not in env]
    clean = [i for i in effective if kind(sft_by[i]) == "scored" and kind(base_by[i]) == "scored"]

    print(f"attempted pairs            : {len(ids)}")
    print(f"environment failures       : {len(env)}  {env}")
    print(f"effective pairs (no env)   : {len(effective)}")
    print(f"clean pairs (no env, no OOM): {len(clean)}")
    repos_of = repo_map(args.subset)
    print()
    for label, subset in (
        ("EFFECTIVE", effective),
        ("CLEAN", clean),
    ):
        print(f"== {label} (n={len(subset)})")
        report_arm("sft", sft, subset, repos_of)
        report_arm("base", base, subset, repos_of)
        print(
            "  paired: "
            + str(paired_comparison(sample(sft, subset), sample(base, subset)))
        )
        print()

    if args.sft_traj or args.base_traj:
        print("== trajectories")
        trajectory_table("sft", args.sft_traj)
        trajectory_table("base", args.base_traj)
        print()
    if args.sft_data:
        print("== SFT supervision")
        supervision_table(args.sft_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
