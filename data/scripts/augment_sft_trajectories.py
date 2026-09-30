"""Rebuild the SFT trajectories so they teach *finding* the file, not knowing it.

Problem with the raw scripted replays (``build_sft_trajectories.py``):

* 77% are the identical 4-step ``read_file -> edit_file -> run_tests -> submit``
  and every one opens by reading the gold file directly -- the model learns to
  guess the file with full confidence (``search_repo`` never appears);
* no assistant text at all;
* every ``run_tests`` observation carries the Windows host's "isolation
  degraded" warning and a ``C:\\Users\\<name>\\...`` path.

What this script does, per trajectory, WITHOUT changing the gold edit/test/
submit steps:

1. Inserts a real ``search_repo`` step before the first read of each gold file.
   The query is an identifier taken from the *issue text* (never from the gold
   patch), and the observation is what ``SearchRepo`` really returns on the
   base commit with the task's ``test_patch`` applied (emulated from a git
   mirror; validated against the real tool in ``data/tests``). Queries are only
   kept if the gold file is visible in the (untruncated) result list, alongside
   distractor files. About a quarter of trajectories get a broad first search
   followed by a narrowed second one.
2. Optionally adds one short, *truthful* reasoning line per assistant turn
   (templated from what actually happened: query, hit counts, diff size, test
   status). ``--no-reasoning`` disables it.
3. Cleans observations: drops host-isolation warnings, normalises ``\\r\\n``,
   rewrites the host path to ``/workspace/repo``.
4. Renumbers tool-call ids and refreshes ``meta``.

Trajectories whose gold file cannot be discovered from issue text are dropped
(or kept as-is with ``--keep-unaugmented``, not recommended).

Usage::

    python -m data.scripts.augment_sft_trajectories \\
        --mirrors /tmp/mirrors --out data/sft_trajectories_v2.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.tools.edit_file import apply_patch_to_tree, split_multifile_patch  # noqa: E402
from agent.tools.search_repo import (  # noqa: E402
    DEFAULT_MAX_RESULTS,
    MAX_FILE_BYTES,
    SKIP_DIRS,
)

WORKSPACE_ROOT = "/workspace/repo"
MAX_QUERY_HITS = 400          # skip identifiers too generic to be a sane first query
MAX_DISPLAY_FILES = 25        # keep result lists readable
BROAD_MIN_LINES = 8           # a "broad" first search returns at least this many lines
TWO_STEP_FRACTION = 0.25      # share of tasks that get broad-then-narrow searches

_STOPWORDS = {
    "this", "that", "with", "from", "have", "when", "then", "there", "these", "those",
    "should", "would", "could", "which", "where", "while", "about", "after", "before",
    "into", "also", "only", "some", "same", "each", "here", "does", "doesn", "didn",
    "issue", "error", "errors", "test", "tests", "bug", "problem", "expected", "actual",
    "result", "results", "value", "values", "code", "file", "files", "line", "lines",
    "none", "true", "false", "self", "return", "import", "class", "def", "print",
    "python", "django", "sympy", "sphinx", "pytest", "pylint", "xarray", "flask",
    "seaborn", "traceback", "call", "last", "most", "recent", "example", "description",
    "using", "used", "uses", "use", "make", "made", "like", "just", "seems", "works",
    "work", "fails", "fail", "failed", "raises", "raise", "raised", "instead", "output",
    "input", "type", "types", "name", "names", "data", "object", "objects", "string",
    "http", "https", "html", "github", "com", "org", "www", "md", "rst", "txt",
}

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_BACKTICK = re.compile(r"`([^`\n]{2,80})`")


# --------------------------------------------------------------------------- text
def clean_observation(text: str) -> str:
    """Strip host-specific noise from a tool observation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    kept = [
        line
        for line in text.split("\n")
        if not line.startswith(("warnings=", "unsupported_limits="))
    ]
    text = "\n".join(kept)

    # normalise doubled/escaped backslashes (nested traceback reprs) before matching
    text = re.sub(r"\\{2,}", "\\\\", text)

    def _path(match: re.Match[str]) -> str:
        tail = match.group("tail") or ""
        return WORKSPACE_ROOT + tail.replace("\\", "/")

    # full form, with drive+user prefix
    text = re.sub(
        r"[A-Za-z]:\\Users\\[^\\\s'\"]+\\Desktop\\gemma4-coding-agent\\data\\work"
        r"\\replay\\[^\\\s'\"]+\\repo(?P<tail>(?:\\[^\s'\"]*)?)",
        _path,
        text,
    )
    # prefix-independent form: the sandbox's own output-truncation marker can cut a
    # long warnings block mid-path, dropping the "C:\Users\<name>\" lead-in and
    # leaving a fragment that starts straight at "Desktop\..." -- still a leak.
    text = re.sub(
        r"(?:[A-Za-z]:\\Users\\[^\\\s'\"]+\\)?Desktop\\gemma4-coding-agent\\data\\work"
        r"\\replay\\[^\\\s'\"]+\\repo(?P<tail>(?:\\[^\s'\"]*)?)",
        _path,
        text,
    )
    # any leftover user-profile path
    text = re.sub(
        r"[A-Za-z]:\\Users\\[^\\\s'\"]+(?P<tail>(?:\\[^\s'\"]*)?)",
        lambda m: "/home/user" + (m.group("tail") or "").replace("\\", "/"),
        text,
    )
    # absolute safety net: a stray fragment of the host username can survive a
    # truncation cut that this function has never seen before. Scrub the literal
    # token wherever it appears, independent of path context.
    text = re.sub(r"(?i)\bbisha\b", "user", text)
    return text


def _is_codelike(token: str, in_backticks: bool) -> bool:
    if in_backticks:
        return True
    return (
        "_" in token
        or "." in token
        or bool(re.search(r"[a-z][A-Z]", token))          # camelCase
        or bool(re.match(r"^[A-Z][a-z]+[A-Z]", token))    # PascalCase
    )


def extract_candidates(issue: str) -> list[str]:
    """Search-worthy identifiers from the issue, best first (most code-like)."""
    scored: dict[str, tuple[int, int]] = {}
    order = 0

    def _add(token: str, in_backticks: bool, extra: int = 0) -> None:
        nonlocal order
        token = token.strip(".")
        if len(token) < 4 or len(token) > 60 or token.lower() in _STOPWORDS:
            return
        if token.isdigit() or not re.search(r"[A-Za-z]", token):
            return
        if not _is_codelike(token, in_backticks):
            return
        score = 2 * int(in_backticks) + int("_" in token) + int("." in token) + extra
        order += 1
        if token not in scored or scored[token][0] < score:
            scored[token] = (score, order if token not in scored else scored[token][1])

    for span in _BACKTICK.findall(issue):
        for tok in _TOKEN.findall(span):
            _add(tok, True)
            for part in tok.split("."):
                _add(part, True)
    for tok in _TOKEN.findall(issue):
        _add(tok, False)
        if "." in tok:
            for part in tok.split("."):
                _add(part, False)
    ranked = sorted(scored, key=lambda t: (-scored[t][0], scored[t][1]))
    return ranked


def diff_stats(diff: str) -> tuple[int, int]:
    added = removed = 0
    for line in diff.replace("\r\n", "\n").split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _no_braces(text: str) -> str:
    """Reasoning text must never contain JSON-ish braces (eval parser scans for them)."""
    return text.replace("{", "(").replace("}", ")")


# ------------------------------------------------------------------ search emulator
@dataclass
class SearchResult:
    query: str
    text: str                       # exactly what SearchRepo.run would return
    first_line: dict[str, int] = field(default_factory=dict)   # path -> first match line shown
    n_lines: int = 0
    truncated: bool = False

    @property
    def files(self) -> list[str]:
        return list(self.first_line)


class MirrorSearch:
    """Emulates ``SearchRepo`` on ``<base_commit> + test_patch`` from a bare git mirror.

    Much faster than materialising every checkout. Fidelity against the real
    tool is asserted by ``--validate N`` (which extracts real trees and diffs
    the outputs) and was run at build time (17/17 queries matched).
    """

    def __init__(self, mirrors: Path) -> None:
        self.mirrors = Path(mirrors)
        self._big: dict[tuple[str, str], set[str]] = {}

    def _git_dir(self, repo: str) -> Path:
        return self.mirrors / (repo.replace("/", "__") + ".git")

    def _git(self, repo: str, *args: str, check: bool = True) -> str:
        proc = subprocess.run(
            ["git", "--git-dir", str(self._git_dir(repo)), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if check and proc.returncode not in (0, 1):
            raise RuntimeError(f"git {' '.join(args[:3])} failed: {proc.stderr[:200]}")
        return proc.stdout

    def _big_files(self, repo: str, commit: str) -> set[str]:
        key = (repo, commit)
        if key not in self._big:
            big: set[str] = set()
            for line in self._git(repo, "ls-tree", "-r", "-l", commit).splitlines():
                meta, _, path = line.partition("\t")
                parts = meta.split()
                if len(parts) >= 4 and parts[3].isdigit() and int(parts[3]) > MAX_FILE_BYTES:
                    big.add(path)
            self._big = {key: big}   # keep one entry: memory-bounded
        return self._big[key]

    def count_hits(self, repo: str, commit: str, query: str) -> int:
        out = self._git(repo, "grep", "-c", "-I", "-F", "--no-color", "-e", query, commit, "--")
        total = 0
        for line in out.splitlines():
            _, _, count = line.rpartition(":")
            if count.isdigit():
                total += int(count)
        return total

    def search(
        self, repo: str, commit: str, test_patch: str, query: str, limit: int = DEFAULT_MAX_RESULTS
    ) -> SearchResult:
        hits: dict[str, list[tuple[int, str]]] = {}
        out = self._git(repo, "grep", "-n", "-I", "-F", "--no-color", "-e", query, commit, "--")
        big = self._big_files(repo, commit)
        prefix = commit + ":"
        for raw in out.splitlines():
            if not raw.startswith(prefix):
                continue
            body = raw[len(prefix):]
            match = re.match(r"^(.*?):(\d+):(.*)$", body)
            if not match:
                continue
            path, lineno, text = match.group(1), int(match.group(2)), match.group(3)
            if path in big or any(p in SKIP_DIRS for p in PurePosixPath(path).parts[:-1]):
                continue
            hits.setdefault(path, []).append((lineno, text))

        # files the test patch touches are searched in their patched form
        if test_patch:
            touched = list(split_multifile_patch(test_patch))
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for rel in touched:
                    blob = self._git(repo, "show", f"{commit}:{rel}", check=False)
                    target = root / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if blob:
                        target.write_text(blob, encoding="utf-8", newline="")
                apply_patch_to_tree(test_patch, root)
                pattern = re.compile(re.escape(query))
                for rel in touched:
                    hits.pop(rel, None)
                    target = root / rel
                    if not target.is_file() or any(
                        p in SKIP_DIRS for p in PurePosixPath(rel).parts[:-1]
                    ):
                        continue
                    raw_bytes = target.read_bytes()
                    if len(raw_bytes) > MAX_FILE_BYTES or b"\x00" in raw_bytes[:8192]:
                        continue
                    for lineno, line in enumerate(
                        raw_bytes.decode("utf-8", errors="replace").splitlines(), start=1
                    ):
                        if pattern.search(line):
                            hits.setdefault(rel, []).append((lineno, line))

        matches: list[str] = []
        first_line: dict[str, int] = {}
        for path in sorted(hits, key=lambda p: PurePosixPath(p).parts):
            for lineno, text in hits[path]:
                if len(matches) >= limit:
                    break
                matches.append(f"{path}:{lineno}: {text.rstrip()[:400]}")
                first_line.setdefault(path, lineno)
            if len(matches) >= limit:
                break

        if not matches:
            return SearchResult(query, f"no matches for {query!r}")
        truncated = len(matches) >= limit
        text = "\n".join(matches) + ("\n... results truncated" if truncated else "")
        return SearchResult(query, text, first_line, len(matches), truncated)


SearchFn = Callable[[str], SearchResult]   # query -> result (task already bound)


# ------------------------------------------------------------------ query planning
def plan_searches(
    issue: str,
    read_paths: list[str],
    search: SearchFn,
    count_hits: Callable[[str], int],
    rng_two_step: bool,
) -> list[SearchResult] | None:
    """Pick real searches for as many ``read_paths`` as the issue text supports.

    The *first* read must be search-discoverable -- that is the step that was
    teaching blind confidence and is worth dropping the trajectory over if it
    can't be fixed. Later reads (e.g. a sibling backend module) are allowed to
    stay a direct ``read_file`` when no clean query exists for them; that is a
    realistic action too, and requiring a query for every single read threw
    away good trajectories over files an issue simply never names.
    """
    candidates = extract_candidates(issue)[:14]
    usable: list[SearchResult] = []
    for query in candidates:
        total = count_hits(query)
        if total == 0 or total > MAX_QUERY_HITS:
            continue
        result = search(query)
        if result.truncated or len(result.files) > MAX_DISPLAY_FILES:
            continue
        if any(path in result.first_line for path in read_paths):
            usable.append(result)
    if not usable or not any(read_paths[0] in r.first_line for r in usable):
        return None

    chosen: list[SearchResult] = []
    found_paths: set[str] = set()
    for path in read_paths:
        options = [r for r in usable if path in r.first_line and path not in found_paths]
        if not options:
            continue  # leave this read as a direct read, no search inserted
        best = min(options, key=lambda r: (r.n_lines, candidates.index(r.query)))
        if rng_two_step and not chosen:
            broad = [r for r in options if r.n_lines >= BROAD_MIN_LINES and r.query != best.query
                     and r.n_lines > best.n_lines]
            if broad:
                chosen.append(max(broad, key=lambda r: r.n_lines))
        chosen.append(best)
        found_paths.add(path)
    return chosen


# ------------------------------------------------------------------ trajectory rewrite
def _call(idx: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{idx}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _read_path(call: dict[str, Any]) -> str | None:
    fn = call["function"]
    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    return args.get("path") if fn["name"] == "read_file" else None


def rewrite_trajectory(
    record: dict[str, Any],
    searches: list[SearchResult],
    reasoning: bool = True,
) -> dict[str, Any]:
    """Insert search steps + reasoning + cleaned observations. Gold steps untouched."""
    messages = record["messages"]
    head = [dict(m) for m in messages if m["role"] in ("system", "user")]
    body = [m for m in messages if m["role"] in ("assistant", "tool")]

    # (assistant_msg, tool_msg) pairs; every scripted turn has exactly one call
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i in range(0, len(body), 2):
        pairs.append((body[i], body[i + 1]))

    out: list[dict[str, Any]] = list(head)
    counter = 0
    shown: set[str] = set()
    pending = list(searches)
    prev_search: SearchResult | None = None

    def emit(name: str, arguments: dict[str, Any], observation: str, note: str) -> None:
        nonlocal counter
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": _no_braces(note) if (reasoning and note) else None,
            "tool_calls": [_call(counter, name, arguments)],
        }
        tool = {
            "role": "tool",
            "tool_call_id": f"call_{counter}",
            "name": name,
            "content": observation,
            "is_error": False,
        }
        out.extend([assistant, tool])
        counter += 1

    def run_searches_until(path: str) -> None:
        nonlocal prev_search
        while pending and path not in shown:
            result = pending.pop(0)
            if prev_search is None:
                note = (f"The issue mentions `{result.query}`. I'll search the repo to find "
                        "where it is defined or used.")
            else:
                note = (f"The previous search matched {prev_search.n_lines} lines in "
                        f"{len(prev_search.files)} files, which is too broad. Narrowing down "
                        f"with `{result.query}`.")
            emit("search_repo", {"query": result.query}, result.text, note)
            shown.update(result.files)
            prev_search = result

    for assistant, tool in pairs:
        call = assistant["tool_calls"][0]
        fn = call["function"]
        name = fn["name"]
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
        observation = clean_observation(tool["content"])

        if name == "read_file":
            path = args["path"]
            run_searches_until(path)
            where = ""
            if prev_search and path in prev_search.first_line:
                where = f" (line {prev_search.first_line[path]})"
            note = (f"The search matched `{path}`{where}. Reading the file to understand the "
                    "code around the match.")
            emit(name, args, observation, note)
        elif name == "edit_file":
            added, removed = diff_stats(args.get("diff", ""))
            target = args.get("path", "the file")
            note = (f"Making a minimal change to `{target}` (+{added}/-{removed} lines) "
                    "to address the issue.")
            emit(name, args, observation, note)
        elif name == "run_tests":
            emit(name, args, observation, "Running the task's tests to verify the change.")
        elif name == "submit":
            passed = observation.startswith("status=pass") or any(
                m["role"] == "tool" and m["name"] == "run_tests"
                and m["content"].startswith("status=pass") for m in out
            )
            note = ("The tests pass, so I'm submitting the patch." if passed
                    else "Submitting the patch.")
            emit(name, args, observation, note)
        else:  # unknown scripted tool: keep verbatim
            emit(name, args, observation, "")

    new = dict(record)
    new["messages"] = out
    meta = dict(record.get("meta", {}))
    meta.update(
        steps=counter,
        augmented="v2",
        search_queries=[s.query for s in searches],
        reasoning=bool(reasoning),
    )
    new["meta"] = meta
    return new


# ------------------------------------------------------------------ driver
def _two_step(task_id: str) -> bool:
    digest = hashlib.sha256(task_id.encode()).digest()
    return digest[0] / 255.0 < TWO_STEP_FRACTION


def augment_record(
    record: dict[str, Any], engine: MirrorSearch, reasoning: bool = True
) -> tuple[dict[str, Any] | None, str]:
    read_paths: list[str] = []
    for m in record["messages"]:
        if m["role"] == "assistant":
            for call in m.get("tool_calls") or []:
                p = _read_path(call)
                if p and p not in read_paths:
                    read_paths.append(p)
    if not read_paths:
        return None, "no_read_step"

    repo, commit, patch = record["repo"], record["base_commit"], record.get("test_patch", "")
    searches = plan_searches(
        record["issue"],
        read_paths,
        search=lambda q: engine.search(repo, commit, patch, q),
        count_hits=lambda q: engine.count_hits(repo, commit, q),
        rng_two_step=_two_step(record["task_id"]),
    )
    if not searches:
        return None, "no_discoverable_query"
    rewritten = rewrite_trajectory(record, searches, reasoning=reasoning)
    if "bisha" in json.dumps(rewritten).lower():
        return None, "host_path_leak"
    return rewritten, "ok"


def _validate(records: list[dict[str, Any]], engine: MirrorSearch, n: int) -> int:
    """Diff emulator output against the *real* SearchRepo on extracted trees."""
    import os
    from agent.tools import ToolEnv
    from agent.tools.search_repo import SearchRepo

    class _Box:  # minimal stand-in: SearchRepo only needs .root
        def __init__(self, root: Path) -> None:
            self.root = root

    bad = checked = 0
    for record in records[:n]:
        repo, commit = record["repo"], record["base_commit"]
        queries = record["meta"].get("search_queries") or []
        if not queries:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"), GIT_WORK_TREE=str(root))
            git = ["git", "--git-dir", str(engine._git_dir(repo))]
            subprocess.run([*git, "read-tree", commit], env=env, check=True)
            subprocess.run([*git, "checkout-index", "-a", f"--prefix={root}/"], env=env, check=True)
            if record.get("test_patch"):
                apply_patch_to_tree(record["test_patch"], root)
            tool = SearchRepo(ToolEnv(sandbox=_Box(root)))  # type: ignore[arg-type]
            for query in queries:
                real = tool.run(query=query)
                fake = engine.search(repo, commit, record.get("test_patch", ""), query).text
                checked += 1
                if real != fake:
                    bad += 1
                    print(f"[validate] MISMATCH {record['task_id']} q={query!r}", file=sys.stderr)
    print(f"[validate] {checked} searches compared, {bad} mismatches")
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="data/sft_trajectories.jsonl")
    parser.add_argument("--out", default="data/sft_trajectories_v2.jsonl")
    parser.add_argument("--mirrors", required=True, help="dir of bare clones: <owner>__<repo>.git")
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--keep-unaugmented", action="store_true",
                        help="keep (cleaned) trajectories that could not get a search step")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0, help="skip this many input records first")
    parser.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    parser.add_argument("--validate", type=int, default=0,
                        help="after building, compare N records' searches to the real tool")
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.offset:
        records = records[args.offset:]
    if args.limit:
        records = records[: args.limit]
    engine = MirrorSearch(Path(args.mirrors))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and out.exists() else "w"
    kept = 0
    stats: Counter[str] = Counter()
    validate_pool: list[dict[str, Any]] = []
    with out.open(mode, encoding="utf-8", newline="\n") as fh:
        for i, record in enumerate(records, 1):
            try:
                new, status = augment_record(record, engine, reasoning=not args.no_reasoning)
            except Exception as exc:  # one bad record must not kill an hour-long build
                new, status = None, f"error:{type(exc).__name__}"
            stats[status] += 1
            if new is None and args.keep_unaugmented:
                new = json.loads(json.dumps(record))
                for m in new["messages"]:
                    if m["role"] == "tool":
                        m["content"] = clean_observation(m["content"])
                new["meta"]["augmented"] = "cleaned-only"
            if new is not None:
                fh.write(json.dumps(new, ensure_ascii=False) + "\n")
                fh.flush()
                kept += 1
                if status == "ok" and args.validate and len(validate_pool) < args.validate:
                    validate_pool.append(new)
            if i % 10 == 0:
                print(f"[augment] {i}/{len(records)}  {dict(stats)}", flush=True)

    print(f"[augment] {'appended' if mode == 'a' else 'wrote'} {kept} / {len(records)} -> {out}")
    print(f"[augment] outcomes: {dict(stats)}")

    if args.validate:
        return 1 if _validate(validate_pool, engine, args.validate) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
