"""Guarded local smoke test: base vs SFT on two tasks via llama.cpp.

The 4-bit transformers path needs ~10-11 GB resident (the 5.25 GiB PLE table
is not quantized by bitsandbytes), which does not fit a 15.7 GB laptop. A
Q3_K_M GGUF is 3.78 GB and stays mmap'd, so the OS can evict pages instead
of swapping.

Same two tasks, same prompt construction (HF tokenizer + chat template +
tool specs), same tool-call parser as the paper eval. Only the backend
changes: llama-server /completion instead of transformers.generate.

This is a side check, not the headline evaluation (see LOCAL_SMOKE.md).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import psutil  # noqa: E402
import requests  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer  # noqa: E402

from agent.tools import default_tool_specs  # noqa: E402
from eval.run_eval import evaluate_task, load_tasks, parse_tool_call  # noqa: E402
from sandbox.limits import Limits  # noqa: E402
from train.sft.data_collator import deserialize_tool_args  # noqa: E402

ROOT = Path(os.environ.get("GEMMA_LOCAL", r"C:\Users\bisha\.cache\gemma-local"))
GGUF = Path(os.environ.get("GEMMA_GGUF", ROOT / "gguf" / "gemma-4-E4B-it-Q3_K_M.gguf"))
LORA = Path(os.environ.get("GEMMA_LORA", ROOT / "lora" / "sft-lora-f16.gguf"))
SERVER = Path(os.environ.get("GEMMA_SERVER", ROOT / "llama" / "llama-server.exe"))
ADAPTER = Path(
    os.environ.get(
        "GEMMA_ADAPTER",
        r"C:\Users\bisha\.cache\kagglehub\datasets\ashuraasura\sft-adapter\versions\1\sft",
    )
)
TASKS = REPO / "data" / "holdout_tasks.jsonl"
TASK_IDS = ("django__django-10390", "django__django-10531")
OUT = Path(os.environ.get("GEMMA_OUT", r"C:\Users\bisha\AppData\Local\Temp\local_two_problems"))

PORT = int(os.environ.get("GEMMA_PORT", "8091"))
URL = f"http://127.0.0.1:{PORT}"
MIN_AVAIL_MB = 500
MAX_RSS_MB = 5500


class Guard:
    """Abort before the machine is pushed into swap."""

    def __init__(self, proc: subprocess.Popen | None = None) -> None:
        self.proc = proc
        self.peak_rss = 0

    def check(self, where: str, kill_on_fail: bool = True) -> None:
        vm = psutil.virtual_memory()
        rss = 0
        if self.proc is not None and self.proc.poll() is None:
            try:
                rss = psutil.Process(self.proc.pid).memory_info().rss
            except Exception:  # noqa: BLE001
                rss = 0
        self.peak_rss = max(self.peak_rss, rss)
        print(
            f"  [mem] {where}: server_rss={rss / 2**30:.2f} GB "
            f"avail={vm.available / 2**30:.2f} GB",
            flush=True,
        )
        if vm.available < MIN_AVAIL_MB * 2**20 or rss > MAX_RSS_MB * 2**20:
            if kill_on_fail and self.proc is not None:
                self.proc.kill()
            raise MemoryError(
                f"aborted at {where}: avail={vm.available / 2**30:.2f} GB, "
                f"rss={rss / 2**30:.2f} GB"
            )


class LlamaCppPolicy:
    def __init__(
        self,
        tokenizer: Any,
        guard: Guard,
        *,
        max_new_tokens: int,
        max_prompt_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.guard = guard
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens

    def build_prompt(self, context) -> str:  # type: ignore[no-untyped-def]
        messages = deserialize_tool_args(context.messages)
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tools=default_tool_specs(),
                tokenize=False,
                add_generation_prompt=True,
            )
        return context.render_flat()

    def _cap(self, prompt: str) -> str:
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        limit = self.max_prompt_tokens
        if limit <= 0 or len(ids) <= limit:
            return prompt
        head = min(2048, limit // 4)
        tail = limit - head
        ids = ids[:head] + ids[-tail:]
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def generate_tool_call(self, context):  # type: ignore[no-untyped-def]
        prompt = self._cap(self.build_prompt(context))
        self.guard.check(f"policy call ({len(prompt)} chars)")
        t0 = time.time()
        response = requests.post(
            URL + "/completion",
            json={
                "prompt": prompt,
                "n_predict": self.max_new_tokens,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "repeat_penalty": 1.0,
                "cache_prompt": False,
                "stream": False,
            },
            timeout=600,
        )
        response.raise_for_status()
        body = response.json()
        text = body.get("content", "")
        for marker in ("<turn|>", "<end_of_turn>", "<tool_call|>"):
            index = text.find(marker)
            if index != -1:
                text = text[: index + len(marker)]
                break
        print(
            f"  [gen] {time.time() - t0:.1f}s, {body.get('tokens_predicted')} tok, "
            f"{len(body.get('content', ''))} chars",
            flush=True,
        )
        call = parse_tool_call(text)
        call.completion_ids = []
        return call


def start_server(model_kind: str, guard: Guard) -> subprocess.Popen:
    if not SERVER.exists():
        raise SystemExit(f"missing {SERVER}")
    if not GGUF.exists():
        raise SystemExit(f"missing {GGUF}")
    cmd = [
        str(SERVER), "-m", str(GGUF),
        "-c", "4096", "-t", "4", "-tb", "4", "-np", "1",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--special", "--no-warmup",
    ]
    if model_kind == "sft":
        if not LORA.exists():
            raise SystemExit(f"missing {LORA}")
        cmd += ["--lora", str(LORA)]
    log = OUT / f"llama_server_{model_kind}.log"
    print(f"  launching llama-server ({model_kind})", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=str(SERVER.parent), stdout=log.open("w"), stderr=subprocess.STDOUT
    )
    guard.proc = proc
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited early (see {log})")
        try:
            health = requests.get(URL + "/health", timeout=2)
            if health.status_code == 200 and health.json().get("status") == "ok":
                print(f"  server ready in {300 - (deadline - time.time()):.0f}s", flush=True)
                return proc
        except requests.RequestException:
            pass
        guard.check("loading")
        time.sleep(1)
    proc.kill()
    raise SystemExit(f"llama-server did not become ready (see {log})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("base", "sft"), required=True)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()

    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass

    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in load_tasks(TASKS) if t["task_id"] in TASK_IDS]
    if len(tasks) != 2:
        raise SystemExit(f"expected 2 tasks, found {[t['task_id'] for t in tasks]}")
    print(f"model={args.model} tasks={[t['task_id'] for t in tasks]}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
    guard = Guard()
    proc = start_server(args.model, guard)
    try:
        if args.load_only:
            # one tiny generation proves the graph + lora path works
            response = requests.post(
                URL + "/completion",
                json={"prompt": "def add(a, b):", "n_predict": 8, "temperature": 0.0},
                timeout=300,
            )
            response.raise_for_status()
            print("load-only OK:", repr(response.json().get("content"))[:80], flush=True)
            return 0

        policy = LlamaCppPolicy(
            tokenizer,
            guard,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        out_path = OUT / f"gguf_{args.model}.jsonl"
        traj_path = OUT / f"gguf_{args.model}_traj.jsonl"
        with out_path.open("w", encoding="utf-8") as fh, \
                traj_path.open("w", encoding="utf-8") as tfh:
            for task in tasks:
                print(f"\n== {task['task_id']}", flush=True)
                t0 = time.time()
                try:
                    record = evaluate_task(
                        task,
                        lambda _task: policy,
                        max_steps=args.max_steps,
                        limits=Limits.for_tests(300),
                        dump_trajectory=True,
                    )
                except MemoryError as exc:
                    print(f"ABORT: {exc}", flush=True)
                    return 2
                trajectory = record.pop("trajectory", None)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                tfh.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
                tfh.flush()
                print(
                    f"  status={record['status']} steps={record['steps']} "
                    f"resolved={record['resolved']} "
                    f"error={str(record.get('error'))[:120]} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
                guard.check(f"after {task['task_id']}")
        print(f"\npeak server RSS {guard.peak_rss / 2**30:.2f} GB", flush=True)
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
