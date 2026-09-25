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
| 2. Data pipeline | not started |
| 3. SFT (QLoRA) | not started |
| 4. RL (GRPO) | not started |
| 5. Eval + ablations | not started |
| 6. Write-up | not started |

Phase 1 gate (`ARCHITECTURE.md` §8.2, Week 1) is only half met: the sandbox
passes its adversarial tests on both hosts, but Rung 0 cannot produce a real
resolved-rate until Phase 2 builds `data/holdout_tasks.jsonl` and a base
model is available. Nothing is trusted as a reward signal until the suite has
been run on the *training* host (Kaggle), not just here.

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
