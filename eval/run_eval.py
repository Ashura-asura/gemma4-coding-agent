"""Evaluation harness — ARCHITECTURE.md §2.6.

Runs the frozen held-out split only. Same sandbox, same agent loop as RL
rollouts (§3.2), no gradient updates.

Task record (one JSON object per line, e.g. ``data/holdout_tasks.jsonl``)::

    {"task_id": "repo-123",
     "repo": "path/to/checkout",      # local checkout; cloning lands in Phase 2
     "issue": "text of the issue",
     "test_target": "tests/test_x.py"}   # optional

Scoring: the submitted patch is applied to a *fresh* copy of the checkout
and the task's tests are run there. A model therefore cannot score by
editing or deleting its own tests — the primary reward (§2.5) always comes
from an untouched tree.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from agent.context import ToolCall
from agent.loop import Policy, ScriptedPolicy, run_episode
from agent.tools import ToolEnv, build_toolset
from agent.tools.edit_file import apply_unified_diff
from agent.tools.run_tests import DEFAULT_TEST_COMMAND, TEST_TIMEOUT_S
from eval.report import format_report, summarize
from sandbox import Limits, Sandbox

DEFAULT_MAX_STEPS = 30
DEFAULT_TEMPERATURE = 0.2


class PatchError(Exception):
    """The submitted patch could not be applied to a fresh checkout."""


# --------------------------------------------------------------------- tasks
def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = [key for key in ("task_id", "repo", "issue") if not record.get(key)]
            if missing:
                raise ValueError(f"{path}:{lineno} missing keys: {', '.join(missing)}")
            tasks.append(record)
    return tasks


def describe_repo(root: Path, max_files: int = 80) -> str:
    """Compact file listing handed to the model as repo_state."""
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] == ".git":
            continue
        files.append(rel.as_posix())
        if len(files) >= max_files:
            files.append("...")
            break
    return "\n".join(files)


# --------------------------------------------------------------------- patch
def split_multifile_patch(patch: str) -> dict[str, str]:
    """Split a multi-file unified diff into {path: diff} sections."""
    lines = patch.replace("\r\n", "\n").split("\n")
    sections: list[tuple[str | None, list[str]]] = []
    current: list[str] = []
    target: str | None = None
    for line in lines:
        if line.startswith("--- "):
            if current:
                sections.append((target, current))
            current, target = [line], None
            continue
        if not current:
            if line.strip():
                current = [line]  # "diff --git ..." preamble
            continue
        current.append(line)
        if target is None and line.startswith("+++ "):
            match = re.match(r"\+\+\+ (?:[ab]/)?(\S+)", line)
            if match and match.group(1) != "/dev/null":
                target = match.group(1)
    if current:
        sections.append((target, current))
    return {path: "\n".join(body) for path, body in sections if path}


def apply_patch_to_tree(patch: str, root: Path) -> list[str]:
    """Apply every section of ``patch`` under ``root``. Atomic per file."""
    sections = split_multifile_patch(patch)
    if not sections:
        raise PatchError("patch contains no file sections")
    applied: list[str] = []
    root_resolved = root.resolve()
    for rel, diff in sections.items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise PatchError(f"patch path escapes the repository: {rel}")
        target = (root / rel_path).resolve()
        if target != root_resolved and root_resolved not in target.parents:
            raise PatchError(f"patch path escapes the repository: {rel}")
        original = target.read_text(encoding="utf-8") if target.exists() else ""
        try:
            updated = apply_unified_diff(original, diff)
        except Exception as exc:  # ToolError and friends
            raise PatchError(f"{rel}: {exc}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated, encoding="utf-8", newline="")
        applied.append(rel)
    return applied


# ------------------------------------------------------------------- scoring
def score_submission(
    task: dict[str, Any],
    patch: str | None,
    *,
    limits: Limits | None = None,
    test_timeout: float = TEST_TIMEOUT_S,
) -> dict[str, Any]:
    """Apply the patch to a fresh checkout and run the task's tests there."""
    if not patch or not patch.strip():
        return {
            "patch_valid": False,
            "resolved": False,
            "test_status": "error",
            "reason": "empty or missing patch",
        }

    repo = Path(task["repo"])
    if not repo.is_dir():
        return {
            "patch_valid": False,
            "resolved": False,
            "test_status": "error",
            "reason": f"repo checkout not found: {repo}",
        }

    with tempfile.TemporaryDirectory(prefix="score-") as tmp:
        fresh = Path(tmp) / "repo"
        shutil.copytree(repo, fresh)
        try:
            applied = apply_patch_to_tree(patch, fresh)
        except PatchError as exc:
            return {
                "patch_valid": False,
                "resolved": False,
                "test_status": "error",
                "reason": f"patch does not apply: {exc}",
            }
        if not applied:
            return {
                "patch_valid": False,
                "resolved": False,
                "test_status": "error",
                "reason": "patch touched no files",
            }

        argv = list(DEFAULT_TEST_COMMAND)
        target = task.get("test_target")
        if target:
            base, sep, rest = str(target).partition("::")
            base_path = Path(base)
            if base_path.is_absolute() or ".." in base_path.parts:
                return {
                    "patch_valid": False,
                    "resolved": False,
                    "test_status": "error",
                    "reason": f"task test_target escapes the repo: {target}",
                }
            argv.append(base + (sep + rest if sep else ""))

        sandbox = Sandbox(fresh, limits=limits)
        result = sandbox.run(argv, timeout=test_timeout)
        return {
            "patch_valid": True,
            "resolved": result.status == "pass",
            "test_status": result.status,
            "test_output": (result.stdout + "\n" + result.stderr)[-4000:],
            "applied_files": applied,
            "isolation": result.isolation,
            "warnings": list(result.warnings),
            "reason": "tests passed" if result.status == "pass" else result.message,
        }


# ------------------------------------------------------------------ policies
def parse_tool_call(text: str) -> ToolCall:
    """Pull a tool call out of a raw model completion."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except ValueError:
            idx = text.find("{", idx + 1)
            continue
        candidates.append(text[idx:idx + end])
        idx = text.find("{", idx + end)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("tool") or payload.get("name")
        arguments = payload.get("arguments", payload.get("params", payload.get("parameters")))
        if isinstance(payload.get("function"), dict):
            name = name or payload["function"].get("name")
            arguments = arguments if arguments is not None else payload["function"].get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                continue
        if name and isinstance(arguments, dict):
            return ToolCall(id="call_0", name=str(name), arguments=arguments)
    raise ValueError(f"no tool call found in model output: {text[:200]!r}")


class HFPolicy:
    """Zero-shot (Rung 0) or adapter-loaded (Rungs 1-3) transformers policy."""

    def __init__(self, model_path: str, *, max_new_tokens: int = 768, temperature: float = 0.2) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - needs .[train]
            raise SystemExit(
                "the 'hf' policy needs the training extras: pip install -e '.[train]'"
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate_tool_call(self, context) -> ToolCall:  # type: ignore[no-untyped-def]
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                context.messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = context.render_flat()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return parse_tool_call(text)


def build_policy_factory(name: str, args: argparse.Namespace) -> Callable[[dict[str, Any]], Policy]:
    if name == "null":

        def _null(task: dict[str, Any]) -> Policy:
            return ScriptedPolicy([])  # immediately stops; harness smoke test

        return _null
    if name == "hf":

        def _hf(task: dict[str, Any]) -> Policy:
            return HFPolicy(
                args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

        return _hf
    raise SystemExit(f"unknown policy: {name}")


# ----------------------------------------------------------------- one task
def evaluate_task(
    task: dict[str, Any],
    policy_factory: Callable[[dict[str, Any]], Policy],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    limits: Limits | None = None,
    dump_trajectory: bool = False,
) -> dict[str, Any]:
    repo = Path(task["repo"])
    record: dict[str, Any] = {
        "task_id": task["task_id"],
        "repo": str(repo),
        "status": "error",
        "steps": 0,
        "error": None,
        "submitted_patch": None,
        "patch_valid": False,
        "resolved": False,
        "test_status": "error",
        "reason": "not run",
    }
    if not repo.is_dir():
        record["error"] = f"repo checkout not found: {repo}"
        record["reason"] = record["error"]
        return record

    with tempfile.TemporaryDirectory(prefix="episode-") as tmp:
        workdir = Path(tmp) / "repo"
        shutil.copytree(repo, workdir)
        sandbox = Sandbox(workdir, limits=limits)
        env = ToolEnv(sandbox=sandbox)
        tools = build_toolset(env)
        started = time.monotonic()
        episode = run_episode(
            policy_factory(task),
            tools,
            task_id=str(task["task_id"]),
            issue=str(task["issue"]),
            repo_state=describe_repo(workdir),
            max_steps=max_steps,
        )
        record["duration_s"] = round(time.monotonic() - started, 2)

    record["status"] = episode.status
    record["steps"] = episode.steps
    record["error"] = episode.error
    record["submitted_patch"] = episode.submitted_patch
    if episode.submitted:
        record.update(
            score_submission(task, episode.submitted_patch, limits=limits)
        )
    else:
        record["reason"] = episode.error or "no patch submitted"
        if episode.status == "budget_exhausted":
            record["test_status"] = "error"
    if dump_trajectory:
        record["trajectory"] = episode.context.to_dict()
    return record


# ----------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the held-out evaluation")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--tasks", help="JSONL of task records (defaults to config holdout_path)")
    parser.add_argument("--policy", choices=("hf", "null"), default="null")
    parser.add_argument("--model", help="model path/id for --policy hf")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--limit", type=int, help="only run the first N tasks")
    parser.add_argument("--out", default="eval/results/run.jsonl")
    parser.add_argument("--dump-trajectories", help="write full trajectories to this JSONL")
    parser.add_argument("--dry-run", action="store_true", help="list tasks and exit")
    args = parser.parse_args(argv)

    config: dict[str, Any] = {}
    if args.config and Path(args.config).exists():
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    tasks_path = args.tasks or config.get("holdout_path") or "data/holdout_tasks.jsonl"
    rollout_cfg = config.get("rollout", {}) or {}
    max_steps = args.max_steps or int(rollout_cfg.get("max_steps", DEFAULT_MAX_STEPS))
    limits = Limits(wall_seconds=int(rollout_cfg.get("wall_seconds", 600)))
    model = args.model or config.get("model_checkpoint")
    if args.policy == "hf" and not model:
        raise SystemExit("--policy hf requires --model or model_checkpoint in the config")

    tasks = load_tasks(tasks_path)
    if args.limit:
        tasks = tasks[: args.limit]
    if args.dry_run:
        for task in tasks:
            print(f"{task['task_id']}\t{task['repo']}")
        print(f"{len(tasks)} task(s)")
        return 0
    if not tasks:
        raise SystemExit(f"no tasks in {tasks_path}")

    args.model = model
    policy_factory = build_policy_factory(args.policy, args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_fh = open(args.dump_trajectories, "w", encoding="utf-8") if args.dump_trajectories else None

    records: list[dict[str, Any]] = []
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            for index, task in enumerate(tasks, start=1):
                record = evaluate_task(
                    task,
                    policy_factory,
                    max_steps=max_steps,
                    limits=limits,
                    dump_trajectory=trajectory_fh is not None,
                )
                records.append(record)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                if trajectory_fh:
                    trajectory_fh.write(json.dumps(record.pop("trajectory"), ensure_ascii=False) + "\n")
                    trajectory_fh.flush()
                mark = "resolved" if record["resolved"] else record["status"]
                print(f"[{index}/{len(tasks)}] {task['task_id']}: {mark}", flush=True)
    finally:
        if trajectory_fh:
            trajectory_fh.close()

    summary = summarize(records)
    summary["policy"] = args.policy
    summary["model"] = model
    print()
    print(format_report(summary))
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
