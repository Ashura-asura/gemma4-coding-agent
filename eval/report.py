"""Eval reporting — ARCHITECTURE.md §2.6, §2.7 (LLM-friendly boilerplate).

Pure functions over the records `eval.run_eval` writes, so the numbers in
the paper are computed by code anyone can re-run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

FAILURE_CATEGORIES = (
    "wrong_file",
    "incomplete_patch",
    "test_environment_issue",
    "step_budget_exhaustion",
    "policy_error",
)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def categorize(record: dict[str, Any]) -> str:
    """Bucket an unresolved task into one of the §2.7 failure modes."""
    if record.get("resolved"):
        return "resolved"
    if record.get("status") == "error":
        return "policy_error"
    if record.get("status") == "budget_exhausted":
        return "step_budget_exhaustion"
    if not record.get("patch_valid"):
        return "incomplete_patch"
    if record.get("test_status") in ("timeout", "error"):
        return "test_environment_issue"
    return "wrong_file"


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    total = len(records)
    resolved = [r for r in records if r.get("resolved")]
    submitted = [r for r in records if r.get("status") == "submitted"]
    valid = [r for r in records if r.get("patch_valid")]
    steps = [r["steps"] for r in submitted if isinstance(r.get("steps"), int)]
    categories = Counter(categorize(r) for r in records)

    per_repo: dict[str, dict[str, Any]] = {}
    for record in records:
        repo = str(record.get("repo") or "unknown")
        bucket = per_repo.setdefault(repo, {"n": 0, "resolved": 0})
        bucket["n"] += 1
        bucket["resolved"] += int(bool(record.get("resolved")))

    return {
        "n_tasks": total,
        "n_resolved": len(resolved),
        "resolved_rate": (len(resolved) / total) if total else 0.0,
        "n_submitted": len(submitted),
        "patch_validity_rate": (len(valid) / total) if total else 0.0,
        "avg_steps_to_submit": (sum(steps) / len(steps)) if steps else None,
        "failure_categories": dict(categories),
        "per_repo": {
            name: {
                **counts,
                "resolved_rate": counts["resolved"] / counts["n"] if counts["n"] else 0.0,
            }
            for name, counts in sorted(per_repo.items())
        },
    }


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        "resolved-rate report",
        "====================",
        f"tasks            : {summary['n_tasks']}",
        f"resolved         : {summary['n_resolved']}",
        f"resolved-rate    : {summary['resolved_rate']:.1%}",
        f"patch validity   : {summary['patch_validity_rate']:.1%}",
        f"avg steps        : "
        + (f"{summary['avg_steps_to_submit']:.1f}" if summary["avg_steps_to_submit"] is not None else "n/a"),
        "",
        "failure modes:",
    ]
    for name in sorted(summary["failure_categories"], key=lambda k: -summary["failure_categories"][k]):
        count = summary["failure_categories"][name]
        share = count / summary["n_tasks"] if summary["n_tasks"] else 0.0
        lines.append(f"  {name:<24} {count:>4}  ({share:.1%})")

    if summary["per_repo"]:
        lines += ["", "per repo:"]
        width = max(len(name) for name in summary["per_repo"]) if summary["per_repo"] else 10
        for name, counts in summary["per_repo"].items():
            lines.append(
                f"  {name:<{width}}  {counts['resolved']}/{counts['n']} "
                f"({counts['resolved_rate']:.1%})"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise an eval run")
    parser.add_argument("records", help="JSONL produced by eval/run_eval.py")
    parser.add_argument("--out", help="Optional path for the summary JSON")
    args = parser.parse_args(argv)

    summary = summarize(load_records(args.records))
    print(format_report(summary))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
