"""Build SFT trajectories — ARCHITECTURE.md §3.1, §4.

    python -m data.scripts.build_sft_trajectories [--limit N]

For every verified instance of the ``sft`` split, replay the gold patch
through the tool interface (read -> edit -> run_tests -> submit) and keep
the trajectory only if the sandboxed tests pass with no tool errors.
Output: ``data/sft_trajectories.jsonl`` (resumable — task ids already
written are skipped).

The teacher-model fallback of §3.1 (``N_TEACHER_ATTEMPTS``) is not run
here: no teacher is available in this environment. Instances whose gold
patch fails verification simply produce no trajectory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sys
from pathlib import Path as _Path

# allow `python data/scripts/<this>.py` from a plain checkout
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data.scripts.common import (
    REPO_SETUP,
    SFT_PATH,
    ensure_verified,
    ensure_venvs,
    jsonl_read,
    jsonl_write,
    load_instances,
    replay_trajectory,
    split_of,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", default="data/raw/swebench_test.parquet")
    parser.add_argument("--limit", type=int, help="verify at most N new instances")
    parser.add_argument("--replay-limit", type=int, help="replay at most N *new* trajectories")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--jobs", type=int, default=4, help="parallel verification workers")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)

    parquet = Path(args.parquet)
    if not parquet.exists():
        print(f"missing raw dataset: {parquet}", file=sys.stderr)
        return 1

    instances = load_instances(parquet, repos=list(REPO_SETUP))
    mine = [i for i in instances if split_of(i["instance_id"]) == "sft"]
    print(f"sft candidates: {len(mine)} of {len(instances)} runnable instances")

    ensure_venvs(i["repo"] for i in mine)
    verified = ensure_verified(
        mine,
        limit=args.limit,
        retry_failed=args.retry_failed,
        timeout=args.timeout,
        jobs=args.jobs,
    )

    records = [verified[i["instance_id"]] for i in mine if i["instance_id"] in verified]
    done = {r["task_id"] for r in jsonl_read(SFT_PATH)}
    new_records = [r for r in records if r["task_id"] not in done]

    kept: list[dict] = []
    replayed = 0
    for record in new_records:
        if args.replay_limit is not None and replayed >= args.replay_limit:
            break
        replayed += 1
        print(f"[replay] {record['task_id']} ({replayed}{'/' + str(args.replay_limit) if args.replay_limit else ''})", flush=True)
        trajectory = replay_trajectory(record, timeout=args.timeout)
        if trajectory is None:
            print("  skip  replay did not end in a clean test pass", flush=True)
            continue
        kept.append(trajectory)
        print(f"  ok    {trajectory['meta']['steps']} steps", flush=True)

    if kept and done:
        with open(SFT_PATH, "a", encoding="utf-8") as fh:
            for trajectory in kept:
                fh.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
        count = len(done) + len(kept)
    elif kept:
        count = jsonl_write(SFT_PATH, kept)
    else:
        count = len(done)
    print(f"wrote {count} trajectories -> {SFT_PATH}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
