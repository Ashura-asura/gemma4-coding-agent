"""Build the frozen held-out split — ARCHITECTURE.md §4, §8.3.

Run this **once**. From that point ``data/holdout_tasks.jsonl`` is
read-only: no SFT or RL script may ever open it for writing (§8.3). The
split itself is decided by ``sha1(task_id)`` so it is stable as the
verified pool grows (§2.1: sft / rl / holdout disjoint).

    python -m data.scripts.build_holdout_split [--limit N] [--retry-failed]

The gold patch is stripped (``public_task``) — the holdout record carries
repo, issue and hidden tests only, exactly like the RL pool.
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
    HOLDOUT_PATH,
    POOL_PATH,
    REPO_SETUP,
    ensure_verified,
    ensure_venvs,
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
    parser.add_argument("--force", action="store_true", help="overwrite an existing holdout file (breaks §8.3)")
    args = parser.parse_args(argv)

    if HOLDOUT_PATH.exists() and not args.force:
        print(
            f"{HOLDOUT_PATH} already exists — it is read-only from now on (§8.3).\n"
            "Re-run with --force only if you intend to re-freeze the holdout.",
            file=sys.stderr,
        )
        return 1

    parquet = Path(args.parquet)
    if not parquet.exists():
        print(f"missing raw dataset: {parquet}", file=sys.stderr)
        return 1

    instances = load_instances(parquet, repos=list(REPO_SETUP))
    mine = [i for i in instances if split_of(i["instance_id"]) == "holdout"]
    print(f"holdout candidates: {len(mine)} of {len(instances)} runnable instances")

    ensure_venvs(i["repo"] for i in mine)
    verified = ensure_verified(
        mine,
        limit=args.limit,
        retry_failed=args.retry_failed,
        timeout=args.timeout,
        jobs=args.jobs,
    )
    records = [public_task(verified[i["instance_id"]]) for i in mine if i["instance_id"] in verified]
    count = jsonl_write(HOLDOUT_PATH, records)
    print(f"wrote {count} verified holdout tasks -> {HOLDOUT_PATH}")
    print(f"pool: {sum(1 for e in jsonl_read(POOL_PATH) if e.get('verified'))} verified")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
