# Gemma-4 Coding Agent

> **Audit branch (`audit-fixes`)** — scoring, sandbox, tool-call parsing, SFT data and GRPO were
> reviewed and fixed; see [AUDIT.md](AUDIT.md) for findings, verification and residual risks.
> Regenerate the SFT file with `python -m data.scripts.compact_sft_trajectories` before training.


Post-trained Gemma 4 into an autonomous software-engineering agent that
resolves real repo issues offline, on consumer hardware. Entry for the
Kaggle Developer Agent Competition (Paper Track).

**Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first** — it is the source of
truth for scope, phase order, decision gates, and what deliberately is *not*
being built.

## Status

| Phase | State |
|---|---|
| 1. Sandbox + tool interface | **done** — adversarial suite green on win32 + Linux (WSL); 5 tools + loop wired; eval harness runs end-to-end with a null policy |
| 2. Data pipeline | **done** — 9 repos end-to-end; runnable pool 1786; holdout frozen at 291; RL pool 291; SFT build finished: **870 test-verified trajectories** (§8.2 gate 500–1000 ✓), mirrored to the `gemma4-coding-agent-repo` Kaggle dataset |
| 3. SFT (QLoRA) | **done** — Kaggle kernel v22 on E4B: 33/33 steps, 515 trajectories, loss 0.648→0.301→0.264 (no collapse), grad_norm 0.41→0.19, adapter `ashuraasura/sft-adapter`. OOM fixed by per-layer-embedding offload (~5.25 GiB to host RAM) + CE chunk 128. |
| 4. RL (GRPO) | **full run in flight (kernel v11)** — reward + rollout worker + GRPO trainer (group-normalized PG, KL to the frozen SFT baseline, shared-policy rollouts, resumable checkpoints). RL v10 smoke: 2 rollouts × 6 steps, update 0 took 619 s, no OOM, `mean_reward=-0.128`; `loss=0` because both rollouts failed (k=2). Full config: `k=4`, `batch_tasks=1`, `max_steps=8`, 20 updates. |
| 5. Eval + ablations | **running (eval v5)** — `kaggle/eval.ipynb` scores SFT adapter vs zero-shot base on the *same* tasks. Measured: ~50–80 s/step on T4; full 291×2 holdout would be ~100+ GPU-hours, so the headline eval uses the committed 24-task stratified subset (`data/holdout_eval_subset.jsonl`, round-robin by repo), with the 23 holdout tasks sharing a base commit/issue with SFT/RL (AUDIT M5) excluded before sampling. Final comparison is greedy (temperature 0) for a paired McNemar test. Eval v4 exposed a CUDA OOM at 14–21 steps; fixed by PLE host-RAM offload + 12k prompt cap. |
| 6. Write-up | **draft** — [`paper/paper.md`](paper/paper.md): method, compute accounting, failure analysis and honest scope; result tables fill from the final eval/RL runs. |

Rung 0 (`ARCHITECTURE.md` §8.1) is measured by the eval kernel on the
Kaggle GPU host (base/instruct zero-shot vs SFT adapter, same tasks).
Sandbox isolation on Kaggle is `best_effort` (unshare is seccomp-blocked) and
is disclosed with every number. The 24-task subset is sampled deterministically:
holdout rows are bucketed by repo and taken round-robin (4 django, 4 xarray,
4 pylint, 3 pytest, 3 sphinx, 3 sympy, 2 flask, 1 requests).

Kaggle notebooks (private kernels, `kaggle/` is the source of truth):
SFT → <https://www.kaggle.com/code/ashuraasura/gemma-4-qlora-sft-architecture-2-4>,
GRPO → <https://www.kaggle.com/code/ashuraasura/gemma-4-grpo-rl>,
Eval → <https://www.kaggle.com/code/ashuraasura/gemma-4-eval-holdout-architecture-2-6>.
The GRPO and eval kernels mount the SFT adapter (`ashuraasura/sft-adapter`).

## Layout

See `ARCHITECTURE.md` §4. Short version: `sandbox/` (executor + adversarial
tests), `agent/` (tools + loop), `data/` (pipeline), `train/` (sft, rl),
`eval/` (harness), `paper/` (write-up).

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

Sandbox tests must pass before any reward or eval number is trusted
(`ARCHITECTURE.md` §2.3, §8.2 Week-1 gate).
