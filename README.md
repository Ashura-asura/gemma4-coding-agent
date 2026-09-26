# Gemma-4 Coding Agent

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
| 2. Data pipeline | **verification running** — 9 repos end-to-end; runnable pool 1786; holdout frozen at 291; RL pool 291 (238 verified, 34 skipped); SFT build (~1092 candidates → test-verified scripted replays) in flight; §8.2 gate (500–1000 SFT trajectories) measured when it lands |
| 3. SFT (QLoRA) | **code ready** — `train/sft/trainer.py` + completion-only collator, `configs/sft_config.yaml`, `kaggle/sft.ipynb` (T4/P100, resumable `--resume auto`); runs once the SFT build finishes |
| 4. RL (GRPO) | not started |
| 5. Eval + ablations | not started |
| 6. Write-up | not started |

Rung 0 (`ARCHITECTURE.md` §8.1) still needs the base/instruct model on the
Kaggle GPU host — nothing is trusted as a reward signal until the sandbox
suite and a baseline eval have been run there, not just on this Windows box.

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
