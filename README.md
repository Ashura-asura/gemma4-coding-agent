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
| 4. RL (GRPO) | **validated on Kaggle (dry-run + probe)** — `train/rl/` reward + rollout worker + GRPO trainer (group-normalized PG, k3-KL to the frozen SFT baseline, shared-policy rollouts, resumable every-25-update checkpoints), `configs/rl_config.yaml`, `scripts/run_rl.sh`; smoke run in flight post-SFT |
| 5. Eval + ablations | **in progress** — `kaggle/eval.ipynb` scores SFT adapter vs zero-shot base on the frozen 291-task holdout (smoke `EVAL_LIMIT=10`, full `EVAL_LIMIT=0`) |
| 6. Write-up | not started |

Rung 0 (`ARCHITECTURE.md` §8.1) is being measured by the eval kernel on the
Kaggle GPU host (base/instruct zero-shot vs SFT adapter, same holdout slice).
Sandbox isolation on Kaggle is `best_effort` (unshare is seccomp-blocked) and
is disclosed with every number.

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
