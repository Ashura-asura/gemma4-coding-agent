"""Eval reporting — ARCHITECTURE.md §2.6, §2.7 (LLM-friendly boilerplate).

Pure functions over the records `eval.run_eval` writes, so the numbers in
the paper are computed by code anyone can re-run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb, sqrt
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


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — honest error bars for a 24-task subset."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def paired_comparison(
    treatment: Iterable[dict[str, Any]], baseline: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Per-task paired comparison of two policies on the same task set.

    At n=24 a difference in resolved-rate alone is too noisy; report the
    discordant tasks and an exact McNemar test instead of just two rates.
    """
    treated = {str(r.get("task_id")): bool(r.get("resolved")) for r in treatment}
    control = {str(r.get("task_id")): bool(r.get("resolved")) for r in baseline}
    ids = sorted(set(treated) & set(control))
    treatment_only = sum(1 for i in ids if treated[i] and not control[i])
    baseline_only = sum(1 for i in ids if control[i] and not treated[i])
    both = sum(1 for i in ids if treated[i] and control[i])
    neither = len(ids) - treatment_only - baseline_only - both
    discordant = treatment_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(comb(discordant, i) for i in range(min(treatment_only, baseline_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "n_paired": len(ids),
        "treatment_only": treatment_only,
        "baseline_only": baseline_only,
        "both": both,
        "neither": neither,
        "mcnemar_exact_p": p_value,
    }


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

    ci_low, ci_high = wilson_interval(len(resolved), total)

    return {
        "n_tasks": total,
        "n_resolved": len(resolved),
        "resolved_rate": (len(resolved) / total) if total else 0.0,
        "resolved_rate_ci95": [ci_low, ci_high],
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
        "  95% Wilson CI  : "
        f"{summary['resolved_rate_ci95'][0]:.1%} – {summary['resolved_rate_ci95'][1]:.1%}",
        f"patch validity   : {summary['patch_validity_rate']:.1%}",
        "avg steps        : "
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
    parser.add_argument(
        "--baseline",
        help="Optional second JSONL: paired per-task comparison (e.g. base vs SFT)",
    )
    args = parser.parse_args(argv)

    records = load_records(args.records)
    summary = summarize(records)
    print(format_report(summary))
    if args.baseline:
        paired = paired_comparison(records, load_records(args.baseline))
        print("\npaired comparison (this run vs baseline):")
        print(
            f"  n={paired['n_paired']} | treatment-only={paired['treatment_only']} | "
            f"baseline-only={paired['baseline_only']} | both={paired['both']} | "
            f"neither={paired['neither']} | McNemar exact p={paired['mcnemar_exact_p']:.3f}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
