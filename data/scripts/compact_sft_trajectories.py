"""Compact replayed SFT trajectories so they fit the training context (AUDIT C4/H5).

The original replays read whole files (median trajectory ~48k characters, far
beyond ``max_seq_len``), so training saw only a tail with no issue text. This
script rewrites each trajectory so that every observation is exactly what the
*current* tools would return:

* ``read_file``  -> a ranged read around the edited hunk(s) (+/- context);
                    unrelated reads are limited to the first lines
* ``search_repo`` -> at most ``DEFAULT_MAX_RESULTS`` results
* ``run_tests``  -> output capped at ``MAX_OUTPUT_CHARS``
* ``submit``     -> no re-typed patch (``submit`` now diffs the workspace)

Optionally (``--recovery-rate``) it inserts a *failed first edit* — the real
``apply_unified_diff`` error for a diff with one wrong context line — followed
by the original correct edit. The failed call is flagged ``"train": false`` so
the model sees the error and learns to recover from it without being trained
to emit the wrong call.

Run::

    python -m data.scripts.compact_sft_trajectories \
        --src data/sft_trajectories.jsonl --dst data/sft_trajectories.compact.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from agent.tools import ToolError
from agent.tools.edit_file import apply_unified_diff
from agent.tools.read_file import MAX_LINES
from agent.tools.run_tests import MAX_OUTPUT_CHARS
from agent.tools.search_repo import DEFAULT_MAX_RESULTS

CONTEXT = 40
UNRELATED_READ_LINES = 120
_NUMBERED = re.compile(r"^\s*(\d+): (.*)$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _call(message: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    fn = calls[0]["function"]
    args = fn.get("arguments")
    return fn["name"], (json.loads(args) if isinstance(args, str) else dict(args or {}))


def _set_args(message: dict[str, Any], args: dict[str, Any]) -> None:
    message["tool_calls"][0]["function"]["arguments"] = json.dumps(args, ensure_ascii=False, sort_keys=True)


def hunk_ranges(diff: str) -> list[tuple[int, int]]:
    out = []
    for line in diff.splitlines():
        m = _HUNK.match(line)
        if m:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            out.append((start, start + max(count, 1) - 1))
    return out


def parse_numbered(observation: str) -> dict[int, str]:
    lines: dict[int, str] = {}
    for raw in observation.split("\n"):
        m = _NUMBERED.match(raw)
        if m:
            lines[int(m.group(1))] = m.group(2)
    return lines


def render_window(lines: dict[int, str], first: int, last: int) -> str:
    width = len(str(last))
    return "\n".join(f"{n:>{width}}: {lines[n]}" for n in range(first, last + 1))


def _truncate_tests(observation: str) -> str:
    head, sep, body = observation.partition("\n\n")
    if not sep or len(body) <= MAX_OUTPUT_CHARS:
        return observation
    half = MAX_OUTPUT_CHARS // 2
    return head + sep + body[:half] + "\n... output truncated ...\n" + body[-half:]


def _corrupt(diff: str) -> str | None:
    """Break the first context/removed line so the hunk cannot match."""
    out, done = [], False
    for line in diff.split("\n"):
        if not done and line[:1] in (" ", "-") and not line.startswith(("--- ", "-- ")) and len(line) > 1:
            line, done = line + "  # stale", True
        out.append(line)
    return "\n".join(out) if done else None


def compact(record: dict[str, Any], *, rng: random.Random, recovery_rate: float) -> dict[str, Any]:
    messages = json.loads(json.dumps(record["messages"]))  # deep copy
    edit_ranges: dict[str, tuple[int, int]] = {}
    for m in messages:
        c = _call(m) if m["role"] == "assistant" else None
        if c and c[0] == "edit_file":
            rngs = hunk_ranges(c[1].get("diff", ""))
            if rngs:
                lo, hi = min(r[0] for r in rngs), max(r[1] for r in rngs)
                old = edit_ranges.get(c[1].get("path", ""))
                edit_ranges[c[1]["path"]] = (min(lo, old[0]), max(hi, old[1])) if old else (lo, hi)

    full_text: dict[str, dict[int, str]] = {}
    for i, m in enumerate(messages):
        c = _call(m) if m["role"] == "assistant" else None
        if not c or i + 1 >= len(messages) or messages[i + 1]["role"] != "tool":
            continue
        name, args = c
        obs = messages[i + 1]
        if name == "read_file" and not obs.get("is_error"):
            lines = parse_numbered(obs["content"])
            if not lines:
                continue
            total = max(lines)
            path = args.get("path", "")
            if "start_line" not in args and "end_line" not in args:
                full_text[path] = lines
            if path in edit_ranges:
                lo, hi = edit_ranges[path]
                first, last = max(1, lo - CONTEXT), min(total, hi + CONTEXT)
                if last - first + 1 > MAX_LINES:
                    last = first + MAX_LINES - 1
            else:
                first, last = 1, min(total, UNRELATED_READ_LINES)
            first = max(first, min(lines))
            last = min(last, max(lines))
            if all(n in lines for n in range(first, last + 1)) and (first, last) != (min(lines), max(lines)):
                _set_args(m, {"path": path, "start_line": first, "end_line": last})
                obs["content"] = render_window(lines, first, last)
        elif name == "search_repo" and not obs.get("is_error"):
            rows = obs["content"].split("\n")
            matches = [r for r in rows if not r.startswith("... results truncated")]
            if len(matches) > DEFAULT_MAX_RESULTS:
                obs["content"] = "\n".join(matches[:DEFAULT_MAX_RESULTS]) + "\n... results truncated"
        elif name == "run_tests":
            obs["content"] = _truncate_tests(obs["content"])
        elif name == "submit":
            _set_args(m, {})

    recovered = False
    if recovery_rate and rng.random() < recovery_rate:
        recovered = _insert_recovery(messages, full_text)

    meta = dict(record.get("meta") or {})
    meta["compacted"] = True
    meta["recovery_example"] = recovered
    return {**record, "messages": messages, "meta": meta}


def _insert_recovery(messages: list[dict[str, Any]], full_text: dict[str, dict[int, str]]) -> bool:
    for i, m in enumerate(messages):
        c = _call(m) if m["role"] == "assistant" else None
        if not c or c[0] != "edit_file":
            continue
        path, diff = c[1].get("path", ""), c[1].get("diff", "")
        lines = full_text.get(path)
        bad = _corrupt(diff)
        if not lines or not bad:
            return False
        original = "\n".join(lines[n] for n in sorted(lines)) + "\n"
        try:
            apply_unified_diff(original, bad)
        except ToolError as exc:
            error = f"error: {exc}"
        else:
            return False  # corrupted diff still applied: not a usable error example
        bad_msg = json.loads(json.dumps(m))
        _set_args(bad_msg, {"path": path, "diff": bad})
        bad_msg["tool_calls"][0]["id"] = "call_bad"
        bad_msg["train"] = False
        err = {"role": "tool", "tool_call_id": "call_bad", "name": "edit_file", "content": error, "is_error": True}
        messages[i:i] = [bad_msg, err]
        return True
    return False


def summarize(records: list[dict[str, Any]]) -> dict[str, float]:
    if len(records) < 2:  # quantiles need >= 2 points
        records = records * 2
    sizes = [
        sum(len(m.get("content") or "") + sum(len(c["function"]["arguments"]) for c in m.get("tool_calls") or [])
            for m in r["messages"])
        for r in records
    ]
    q = statistics.quantiles(sizes, n=10)
    return {"median_chars": statistics.median(sizes), "p90_chars": q[8], "max_chars": max(sizes)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="data/sft_trajectories.jsonl")
    ap.add_argument("--dst", default="data/sft_trajectories.compact.jsonl")
    ap.add_argument("--recovery-rate", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-chars",
        type=int,
        default=16000,
        help="drop trajectories still longer than this after compaction (~4k tokens; 0 = keep all)",
    )
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    src = [json.loads(row) for row in Path(args.src).read_text(encoding="utf-8").splitlines() if row.strip()]
    out = [compact(r, rng=rng, recovery_rate=args.recovery_rate) for r in src]
    if args.max_chars:
        before = len(out)
        out = [r for r in out if summarize([r])["max_chars"] <= args.max_chars]
        print(f"kept {len(out)}/{before} trajectories within {args.max_chars} chars")
    Path(args.dst).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    print(f"before: {summarize(src)}")
    print(f"after:  {summarize(out)}")
    print(f"recovery examples: {sum(1 for r in out if r['meta'].get('recovery_example'))}/{len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
