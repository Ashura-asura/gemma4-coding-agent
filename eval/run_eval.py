"""Evaluation harness — ARCHITECTURE.md §2.6.

Runs the frozen held-out split only. Same sandbox, same agent loop as RL
rollouts (§3.2), no gradient updates.

Task record (one JSON object per line, e.g. ``data/holdout_tasks.jsonl``)::

    {"task_id": "repo-123",
     "repo": "path/to/checkout",      # local checkout; cloning lands in Phase 2
     "issue": "text of the issue",
     "test_target": "tests/test_x.py"}   # optional

Scoring: the submitted patch is applied to a *fresh* copy of the checkout
and the task's tests are run there. Sections of the patch that touch test or
harness files are dropped first (``eval.scoring_guard``), and an exit code
alone is never trusted: the run must have executed and passed at least one
test with nothing skipped. A model therefore cannot score by editing,
skipping or deleting its own tests.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import tokenize
from pathlib import Path
from typing import Any, Callable

import yaml

from agent.context import ToolCall
from agent.gemma_format import ToolCallParseError, clean_reasoning, extract_tool_call
from agent.loop import MalformedCall, Policy, ScriptedPolicy, run_episode
from agent.tools import ToolEnv, build_toolset, default_tool_specs
from agent.tools.edit_file import PatchError, apply_patch_to_tree, split_multifile_patch
from agent.tools.run_tests import TEST_TIMEOUT_S
from eval.report import format_report, summarize
from eval.scoring_guard import is_infra_kill, sanitize_patch, verify_outcome
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
#: Generation prompt budget. Each episode appends a tool result per step, so
#: an uncapped 30-step context eventually OOMs a 16 GiB T4 even in 4-bit
#: (eval v4: 14-21 steps -> CUDA OOM). Keep the system prompt / issue head
#: and the most recent turns; older observations are safe to drop.
MAX_PROMPT_TOKENS = int(os.environ.get("EVAL_MAX_PROMPT_TOKENS", "12288"))


def cap_prompt_tokens(inputs: dict[str, Any], limit: int, torch: Any) -> dict[str, Any]:
    """Left-truncate a tokenized prompt to ``limit`` ids, keeping head + tail."""
    ids = inputs.get("input_ids")
    if limit <= 0 or ids is None or ids.shape[1] <= limit:
        return inputs
    head = min(2048, limit // 4)
    tail = limit - head
    for key in ("input_ids", "attention_mask"):
        value = inputs.get(key)
        if value is not None:
            inputs[key] = torch.cat([value[:, :head], value[:, -tail:]], dim=1)
    return inputs
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
        return _fail("empty or missing patch")

    test_patch = str(task.get("test_patch") or "")
    try:
        patch, dropped = sanitize_patch(patch, test_patch)
    except Exception as exc:  # noqa: BLE001 - malformed diff text
        return _fail(f"patch is not a valid diff: {type(exc).__name__}: {exc}")
    if not patch:
        return _fail(
            "patch touched no non-test files",
            dropped_protected=dropped,
        )

    env_error = ensure_task_environment(task)
    if env_error:
        return _fail(env_error, infra=True)

    try:
        repo = materialize_repo(task)
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed eval error
        return _fail(f"checkout failed: {type(exc).__name__}: {exc}", infra=True)

    from data.scripts.common import prepare_checkout, pythonpath_for

    with tempfile.TemporaryDirectory(prefix="score-") as tmp:
        fresh = Path(tmp) / "repo"
        shutil.copytree(repo, fresh)

        if test_patch:
            try:
                apply_patch_to_tree(test_patch, fresh)
            except PatchError as exc:
                return _fail(f"task test patch does not apply: {exc}", infra=True)

        try:
            applied = apply_patch_to_tree(patch, fresh)
        except PatchError as exc:
            return _fail(f"patch does not apply: {exc}", dropped_protected=dropped)
        if not applied:
            return _fail("patch touched no files", dropped_protected=dropped)
        if not _changes_code(repo, fresh, applied):
            return _fail("patch is a no-op (whitespace or comments only)", dropped_protected=dropped)

        project = task_project(task)
        if project:
            try:
                prepare_checkout(project, fresh)
            except Exception as exc:  # noqa: BLE001
                return _fail(f"checkout prep failed: {type(exc).__name__}: {exc}", infra=True)

        base, targets = task_test_config(task)
        runner = str(task.get("test_runner", "pytest"))
        # regression guard: pass-to-pass targets, when the record carries them
        extra = [str(t) for t in (task.get("p2p_targets") or []) if str(t) not in targets]
        if extra:
            targets = targets + extra
        if not targets and runner != "pytest":
            return {
                "patch_valid": True,
                "resolved": False,
                "test_status": "error",
                "infra_error": True,
                "reason": "task defines no test targets",
            }

        junit_path = fresh / ".score_junit.xml"
        if runner != "django":
            base = base + [f"--junitxml={junit_path}", "-p", "no:cacheprovider"]

        run_limits = limits if limits is not None else Limits.for_tests(int(test_timeout))
        sandbox = Sandbox(fresh, limits=run_limits)
        result = sandbox.run(base + targets, timeout=test_timeout, pythonpath=pythonpath_for(fresh))
        output = result.stdout + "\n" + result.stderr

        infra = is_infra_kill(result.returncode, result.status, result.message)
        resolved, reason = False, result.message
        if result.status == "pass":
            junit_text = None
            if junit_path.exists():
                junit_text = junit_path.read_text(encoding="utf-8", errors="replace")
            resolved, reason = verify_outcome(runner, output, junit_xml=junit_text)
        return {
            "patch_valid": True,
            "resolved": resolved,
            "test_status": result.status if not (result.status == "pass" and not resolved) else "fail",
            "infra_error": bool(infra and not resolved),
            "test_output": output[-4000:],
            "applied_files": applied,
            "dropped_protected": dropped,
            "isolation": result.isolation,
            "warnings": list(result.warnings),
            "reason": reason,
        }


def _fail(reason: str, *, infra: bool = False, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "patch_valid": False,
        "resolved": False,
        "test_status": "error",
        "infra_error": infra,
        "reason": reason,
    }
    out.update(extra)
    return out


_COMMENT_OR_BLANK = re.compile(r"^\s*(#.*)?$")
_IGNORED_TOKENS = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
}


def _code_signature(text: str) -> list[Any]:
    """Token stream without comments/whitespace; line-based fallback if it
    does not tokenize (e.g. a half-edited file)."""
    try:
        return [
            (tok.type, tok.string)
            for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            if tok.type not in _IGNORED_TOKENS
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [" ".join(row.split()) for row in text.splitlines() if not _COMMENT_OR_BLANK.match(row)]


def _changes_code(original_root: Path, patched_root: Path, applied: list[str]) -> bool:
    """False when every touched file differs only by whitespace/comments."""
    for rel in applied:
        new = patched_root / rel
        old = original_root / rel
        new_text = new.read_text(encoding="utf-8", errors="replace") if new.exists() else ""
        old_text = old.read_text(encoding="utf-8", errors="replace") if old.exists() else ""
        if not rel.endswith(".py"):
            if new_text != old_text:
                return True
            continue
        if _code_signature(new_text) != _code_signature(old_text):
            return True
    return False


# ------------------------------------------------------------------ policies
def parse_tool_call(text: str) -> ToolCall:
    """Pull a tool call out of a raw model completion.

    Gemma 4's native ``<|tool_call>call:name{...}<tool_call|>`` form is tried
    first (that is what a model trained on the chat template emits); bare
    JSON tool calls remain supported for zero-shot / other chat templates.
    Raises :class:`MalformedCall` (a ``ValueError``) when nothing parses.
    """
    try:
        name, arguments, reasoning = extract_tool_call(text)
        return ToolCall(id="call_0", name=name, arguments=arguments, reasoning=reasoning)
    except ToolCallParseError:
        pass
    return _parse_json_tool_call(text)


def _parse_json_tool_call(text: str) -> ToolCall:
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
            cut = text.find(candidate)
            reasoning = clean_reasoning(text[:cut]) if cut > 0 else ""
            return ToolCall(id="call_0", name=str(name), arguments=arguments, reasoning=reasoning)
    raise MalformedCall(f"no tool call found in model output: {text[:200]!r}")


def policy_dtype(torch: Any) -> Any:
    """Match the trainer: fp16 on T4/P100, bf16 only where the GPU supports it.

    ``torch.cuda.is_available()`` alone selected bf16 on a T4, where the
    trainer had correctly fallen back to fp16. The mismatch made eval slower
    and different from the dtype the adapter was trained under.
    """
    if not torch.cuda.is_available():
        return torch.float32
    supported = getattr(torch.cuda, "is_bf16_supported", None)
    if supported is not None and supported():
        return torch.bfloat16
    return torch.float16


class HFPolicy:
    """Zero-shot (Rung 0) or adapter-loaded (Rungs 1-3) transformers policy."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        max_new_tokens: int = 768,
        temperature: float = 0.2,
        quantize: bool = False,
        preloaded: tuple[Any, Any] | None = None,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    ) -> None:
        if preloaded is not None:
            # GRPO shares its trainable model with rollouts instead of
            # loading a second copy (§5: one 12B resident at a time)
            import torch

            self._torch = torch
            self.model, self.tokenizer = preloaded
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            self.max_prompt_tokens = max_prompt_tokens
            self._quantized = False
            return
        if model_path is None:
            raise ValueError("HFPolicy needs model_path or preloaded")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - needs .[train]
            raise SystemExit(
                "the 'hf' policy needs the training extras: pip install -e '.[train]'"
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = policy_dtype(torch)
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        self._quantized = bool(quantize and torch.cuda.is_available())
        if self._quantized:
            from transformers import BitsAndBytesConfig

            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                ),
                "device_map": "auto",
            }
        from train.rl.checkpoints import resolve_adapter_dir

        adapter_dir = resolve_adapter_dir(model_path)
        if adapter_dir is not None:
            from peft import PeftConfig, PeftModel

            peft_cfg = PeftConfig.from_pretrained(str(adapter_dir))
            base = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path, **kwargs)
            self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if torch.cuda.is_available() and not self._quantized:
            self.model = self.model.cuda()
        # E4B's per-layer embedding table is ~5.2 GiB and bitsandbytes keeps
        # it in fp16: the same host-RAM offload SFT uses is what gives eval
        # enough KV-cache headroom on a 16 GiB T4 (eval v4 OOMed at 14 steps).
        try:
            from train.sft.trainer import offload_per_layer_embedding

            offload_per_layer_embedding(self.model, torch)
        except Exception as exc:  # noqa: BLE001 - optimization, not correctness
            print(f"[eval] PLE offload skipped: {type(exc).__name__}: {exc}", flush=True)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_prompt_tokens = max_prompt_tokens

    def _stop_ids(self) -> set[int]:
        ids: set[int] = set()
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, (list, tuple)):
            ids.update(int(i) for i in eos)
        unk = getattr(self.tokenizer, "unk_token_id", None)
        for token in ("<turn|>", "<end_of_turn>", "<tool_call|>"):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(token)
            except Exception:  # noqa: BLE001 - tokenizers differ
                continue
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                ids.add(tid)
        return ids

    def build_prompt(self, context) -> str:  # type: ignore[no-untyped-def]
        """Chat-template prompt for ``context`` incl. tool schemas (AUDIT C3)."""
        from train.sft.data_collator import deserialize_tool_args

        # Gemma-4's template validates tool arguments as JSON objects; our
        # messages carry them as strings (same as the stored trajectories)
        messages = deserialize_tool_args(context.messages)
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tools=default_tool_specs(),
                tokenize=False,
                add_generation_prompt=True,
            )
        return context.render_flat()

    def generate_tool_call(self, context) -> ToolCall:  # type: ignore[no-untyped-def]
        prompt = self.build_prompt(context)
        # the template already contains <bos>; adding it again degrades output
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = cap_prompt_tokens(inputs, self.max_prompt_tokens, self._torch)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=sorted(self._stop_ids()) or None,
            )
        generated = output[0][inputs["input_ids"].shape[1]:].tolist()
        del output
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()  # release the KV cache before parsing
        stops = self._stop_ids()
        for index, token_id in enumerate(generated):
            if token_id in stops:
                generated = generated[: index + 1]
                break
        # keep special tokens: the native tool-call delimiters are special
        # tokens and skipping them (the old behaviour) destroyed the format
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        call = parse_tool_call(text)  # raises MalformedCall: the loop retries (AUDIT C3)
        call.completion_ids = generated
        return call


def build_policy_factory(name: str, args: argparse.Namespace) -> Callable[[dict[str, Any]], Policy]:
    if name == "null":

        def _null(task: dict[str, Any]) -> Policy:
            return ScriptedPolicy([])  # immediately stops; harness smoke test

        return _null
    if name == "hf":
        shared: dict[str, Policy] = {}

        def _hf(task: dict[str, Any]) -> Policy:
            # one load per run: reloading a 12B model per task would dominate
            if "policy" not in shared:
                shared["policy"] = HFPolicy(
                    args.model,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    quantize=bool(getattr(args, "quantize", False)),
                )
            return shared["policy"]

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

        sandbox = Sandbox(workdir, limits=limits or Limits.for_tests(int(TEST_TIMEOUT_S)))
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
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="4-bit load for --policy hf (required to fit 12B on T4/P100, §5)",
    )
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
    limits = Limits.for_tests(int(rollout_cfg.get("wall_seconds", 600)))
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
                    # early-exit records (env/checkout failure) never ran an
                    # episode, so they carry no "trajectory" key — write a null
                    # placeholder rather than crashing the whole eval run
                    trajectory = record.pop("trajectory", None)
                    trajectory_fh.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
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
