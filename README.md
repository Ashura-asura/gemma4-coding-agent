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
| 2. Data pipeline | **done** — 9 repos end-to-end; runnable pool 1786; holdout frozen at 291; RL pool 291; SFT build finished: **870 test-verified trajectories** built, **515 retained** after compaction to the ~4k-token budget (355 excluded for length; §8.2 gate 500–1000 ✓), mirrored to the `gemma4-coding-agent-repo` Kaggle dataset |
| 3. SFT (QLoRA) | **done** — Kaggle kernel v22 on E4B: 33/33 steps, 515 trajectories, loss 0.648→0.301→0.264 (no collapse), grad_norm 0.41→0.19, adapter `ashuraasura/sft-adapter`. OOM fixed by per-layer-embedding offload (~5.25 GiB to host RAM) + CE chunk 128. |
| 4. RL (GRPO) | **stopped — negative result** — kernel v11 ran 4/20 updates; `loss=0.0000`, `pg=0.0000`, `kl=0.0000` on every update and mean group reward stayed at `-0.136`, `-0.134`, `-0.140`, `-0.139` (best `-0.130`, `-0.120`, `-0.140`, `-0.135`). All rollouts failed, so advantages were zero; update 1 also produced an unparseable tool call. Checkpoints are every 5 updates, so no RL adapter exists. The SFT adapter is the final policy. |
| 5. Eval + ablations | **done — null result** — eval v6 completed (11 h 19 m, one T4) on the committed 24-task stratified subset. **Both arms resolve 0/20 scored tasks** (0.0%, 95% Wilson 0.0–16.1%); 0 discordant pairs, exact McNemar p=1.0. Four xarray tasks fail environment setup identically in both arms (pandas 1.5.3 has no Python 3.13 wheel), leaving 20 scored pairs; the base arm additionally OOMs 12/20 because it reads whole files (0/48 ranged reads) and sits at the 12k prompt cap, leaving **8 clean pairs, also 0/8 vs 0/8**. SFT holds the tool protocol (1 unparseable output vs 6 for base) but collapses into repeating one rejected action (18/19 long episodes use ≤3 distinct calls) and never submits; its data contains only one error class (`patch does not apply`, 125 observations) and no path-resolution errors, which cause 11/20 failures. Reproduce with `python -m eval.final_analysis`. |
| 6. Write-up | **done** — [`paper/paper.md`](paper/paper.md) with the measured null result, failure taxonomy, trajectory analysis and compute accounting; [`paper/paper.pdf`](paper/paper.pdf) rendered by `paper/build_pdf.py`. |

Rung 0 (`ARCHITECTURE.md` §8.1) is measured by the eval kernel on the
Kaggle GPU host (base/instruct zero-shot vs SFT adapter, same tasks).
Sandbox isolation on Kaggle is `best_effort` (unshare is seccomp-blocked) and
is disclosed with every number. The 24-task subset is sampled deterministically:
holdout rows are bucketed by repo and taken round-robin (4 django, 4 xarray,
4 pylint, 3 pytest, 3 sphinx, 3 sympy, 2 flask, 1 requests). Before sampling,
the 23 holdout tasks sharing a base commit or issue text with the SFT/RL pools
(AUDIT M5) are excluded, so the headline subset is clean at commit/issue level;
public SWE-bench pre-training exposure remains a disclosed limitation.

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
