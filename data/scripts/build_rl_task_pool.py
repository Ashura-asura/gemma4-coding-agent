"""Build the RL task pool — ARCHITECTURE.md §3.1, §4.

    python -m data.scripts.build_rl_task_pool [--limit N]

Reads the raw dataset, verifies the ``rl`` split fail-to-pass inside the
sandbox, and writes ``(repo, issue, tests)`` records with the gold patch
**withheld** to ``data/rl_task_pool.jsonl``. The sft / rl / holdout splits
are disjoint by construction (§2.1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sys
from pathlib import Path as _Path

# allow `python data/scripts/<this>.py` from a plain checkout
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from data.scripts.common import (
    POOL_PATH,
    REPO_SETUP,
    RL_PATH,
    ensure_verified,
    ensure_venvs,
    jsonl_append,
    jsonl_read,
    jsonl_write,
    load_instances,
    public_task,
    split_of,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", default="data/raw/swebench_test.parquet")
    parser.add_argument("--limit", type=int, help="verify at most N new instances")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--jobs", type=int, default=4, help="parallel verification workers")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)

    parquet = Path(args.parquet)
    if not parquet.exists():
        print(f"missing raw dataset: {parquet}", file=sys.stderr)
        return 1

    instances = load_instances(parquet, repos=list(REPO_SETUP))
    mine = [i for i in instances if split_of(i["instance_id"]) == "rl"]
    print(f"rl candidates: {len(mine)} of {len(instances)} runnable instances")

    ensure_venvs(i["repo"] for i in mine)
    verified = ensure_verified(
        mine,
        limit=args.limit,
        retry_failed=args.retry_failed,
        timeout=args.timeout,
        jobs=args.jobs,
    )
    records = [public_task(verified[i["instance_id"]]) for i in mine if i["instance_id"] in verified]

    already = {r["task_id"] for r in jsonl_read(RL_PATH)}
    fresh = [r for r in records if r["task_id"] not in already]
    if fresh and already:
        for record in fresh:
            jsonl_append(RL_PATH, record)
        count = len(already) + len(fresh)
    else:
        count = jsonl_write(RL_PATH, records)
    print(f"wrote {count} verified RL tasks (no gold patch) -> {RL_PATH}")
    print(f"pool: {sum(1 for e in jsonl_read(POOL_PATH) if e.get('verified'))} verified")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
