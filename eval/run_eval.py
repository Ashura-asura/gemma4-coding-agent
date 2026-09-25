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
from agent.tools.edit_file import PatchError, apply_patch_to_tree, split_multifile_patch
from agent.tools.run_tests import DEFAULT_TEST_COMMAND, TEST_TIMEOUT_S
from eval.report import format_report, summarize
from sandbox import Limits, Sandbox

__all__ = [
    "PatchError",
    "apply_patch_to_tree",
    "split_multifile_patch",
    "load_tasks",
    "score_submission",
    "evaluate_task",
    "parse_tool_call",
]

DEFAULT_MAX_STEPS = 30
DEFAULT_TEMPERATURE = 0.2


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


def task_test_config(task: dict[str, Any], *, python: str | None = None) -> tuple[list[str], list[str]]:
    """Build (argv base, targets) for this task's test suite (§2.2, §2.6)."""
    interpreter = python or task.get("python") or _venv_python(task) or sys.executable
    runner = str(task.get("test_runner", "pytest"))
    raw = task.get("test_target") or []
    targets = [raw] if isinstance(raw, str) else [str(t) for t in raw]
    if runner == "django":
        base = [interpreter, "tests/runtests.py", "--verbosity", "1", "--parallel", "1"]
    elif runner == "pytest-k":
        base = [interpreter, "-m", "pytest", "-q"]
        expr = str(task.get("test_k_expr", "") or "")
        if expr:
            base += ["-k", expr]
    else:
        base = [interpreter, "-m", "pytest", "-q"]
    return base, targets


def _venv_python(task: dict[str, Any]) -> str | None:
    """Interpreter for this task's repo: record override, its venv, or none."""
    if task.get("python"):
        return str(task["python"])
    if task.get("venv"):
        from data.scripts.common import venv_python

        return str(venv_python(str(task["venv"])))
    return None


def task_project(task: dict[str, Any]) -> str | None:
    """GitHub ``owner/name`` behind the task, if the record carries a URL."""
    url = task.get("repo_url")
    if not url:
        return None
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", str(url))
    return match.group(1) if match else None


def task_checkout_dir(task: dict[str, Any]) -> Path:
    """Where this task's persistent checkout lives.

    Phase 2 records use ``repo`` for the GitHub slug and ``checkout`` for the
    local path; older records stored the path directly in ``repo``.
    """
    if task.get("checkout"):
        return Path(str(task["checkout"]))
    local = Path(str(task["repo"]))
    if local.is_dir() or not task.get("repo_url"):
        return local
    from data.scripts.common import CHECKOUT_DIR, slugify

    name = str(task.get("task_id") or slugify(str(task["repo"])))
    return CHECKOUT_DIR / name / "repo"


def ensure_task_environment(task: dict[str, Any]) -> str | None:
    """Create the task's venv (and its repo's deps) once. Error or None."""
    try:
        project = task_project(task)
        if project:
            from data.scripts.common import ensure_venv

            ensure_venv(project)
        elif task.get("venv"):
            from data.scripts.common import ensure_venv_by_slug

            ensure_venv_by_slug(str(task["venv"]))
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed eval error
        return f"environment setup failed: {type(exc).__name__}: {exc}"
    return None


def materialize_repo(task: dict[str, Any]) -> Path:
    """Return a local checkout for the task, archiving it in if needed."""
    repo = task_checkout_dir(task)
    if repo.is_dir():
        return repo
    if not task.get("repo_url") or not task.get("base_commit"):
        raise FileNotFoundError(f"repo checkout not found: {repo}")
    from data.scripts.common import checkout_instance

    checkout_instance(str(task["repo_url"]), str(task["base_commit"]), repo)
    return repo


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


# ------------------------------------------------------------------- scoring
def score_submission(
    task: dict[str, Any],
    patch: str | None,
    *,
    limits: Limits | None = None,
    test_timeout: float = TEST_TIMEOUT_S,
) -> dict[str, Any]:
    """Apply the task's test patch *and* the submitted patch to a fresh
    checkout, then run the task's tests there (§2.5 primary reward source)."""
    if not patch or not patch.strip():
        return {
            "patch_valid": False,
            "resolved": False,
            "test_status": "error",
            "reason": "empty or missing patch",
        }

    env_error = ensure_task_environment(task)
    if env_error:
        return {
            "patch_valid": False,
            "resolved": False,
            "test_status": "error",
            "reason": env_error,
        }

    try:
        repo = materialize_repo(task)
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed eval error
        return {
            "patch_valid": False,
            "resolved": False,
            "test_status": "error",
            "reason": f"checkout failed: {type(exc).__name__}: {exc}",
        }

    from data.scripts.common import prepare_checkout, pythonpath_for

    with tempfile.TemporaryDirectory(prefix="score-") as tmp:
        fresh = Path(tmp) / "repo"
        shutil.copytree(repo, fresh)

        test_patch = str(task.get("test_patch") or "")
        if test_patch:
            try:
                apply_patch_to_tree(test_patch, fresh)
            except PatchError as exc:
                return {
                    "patch_valid": False,
                    "resolved": False,
                    "test_status": "error",
                    "reason": f"task test patch does not apply: {exc}",
                }

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

        project = task_project(task)
        if project:
            try:
                prepare_checkout(project, fresh)
            except Exception as exc:  # noqa: BLE001
                return {
                    "patch_valid": False,
                    "resolved": False,
                    "test_status": "error",
                    "reason": f"checkout prep failed: {type(exc).__name__}: {exc}",
                }

        base, targets = task_test_config(task)
        runner = str(task.get("test_runner", "pytest"))
        if not targets and runner != "pytest":
            return {
                "patch_valid": True,
                "resolved": False,
                "test_status": "error",
                "reason": "task defines no test targets",
            }

        sandbox = Sandbox(fresh, limits=limits)
        result = sandbox.run(base + targets, timeout=test_timeout, pythonpath=pythonpath_for(fresh))
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
    from data.scripts.common import prepare_checkout, pythonpath_for

    record: dict[str, Any] = {
        "task_id": task["task_id"],
        "repo": str(task["repo"]),
        "status": "error",
        "steps": 0,
        "error": None,
        "submitted_patch": None,
        "patch_valid": False,
        "resolved": False,
        "test_status": "error",
        "reason": "not run",
    }

    env_error = ensure_task_environment(task)
    if env_error:
        record["error"] = env_error
        record["reason"] = env_error
        return record

    try:
        repo = materialize_repo(task)
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed eval error
        record["error"] = f"checkout failed: {type(exc).__name__}: {exc}"
        record["reason"] = record["error"]
        return record
    record["repo"] = str(repo)

    base, targets = task_test_config(task)

    with tempfile.TemporaryDirectory(prefix="episode-") as tmp:
        workdir = Path(tmp) / "repo"
        shutil.copytree(repo, workdir)

        # tests only exist after the task's test patch (§2.6)
        if task.get("test_patch"):
            try:
                apply_patch_to_tree(str(task["test_patch"]), workdir)
            except PatchError as exc:
                record["error"] = f"task test patch does not apply: {exc}"
                record["reason"] = record["error"]
                return record

        project = task_project(task)
        if project:
            try:
                prepare_checkout(project, workdir)
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"checkout prep failed: {type(exc).__name__}: {exc}"
                record["reason"] = record["error"]
                return record

        sandbox = Sandbox(workdir, limits=limits)
        env = ToolEnv(
            sandbox=sandbox,
            test_base=base,
            test_targets=targets,
            test_pythonpath=pythonpath_for(workdir),
        )
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
