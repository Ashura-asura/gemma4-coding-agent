# Post-training Gemma 4 into an offline software-engineering agent on a free T4

**Paper Track submission draft — work in progress.**
All headline numbers marked `TBD` are filled from the final Kaggle runs; the
method, compute accounting and failure analysis are final unless noted.

## Abstract

We post-train Gemma 4 E4B into a repository-editing agent that runs entirely
offline inside a CPU-sandboxed tool loop. The pipeline is QLoRA supervised
fine-tuning (SFT) on 870 test-verified tool-use trajectories, followed by
group-relative policy optimization (GRPO) on a disjoint 267-task pool with a
test-execution reward. All training and evaluation run on a single free-tier
Kaggle T4 (16 GiB). We report resolved rate, patch validity and failure modes
on a frozen, repo-stratified 24-task subset of a 291-task held-out split, and
we state the compute-driven reductions explicitly. `TBD: one-sentence result.`

## 1. Introduction

Software-engineering agents must do three things at once: read an unfamiliar
repository, edit it correctly, and verify the edit without network access.
Frontier results rely on large models and large-scale execution feedback. We
ask what is achievable with a 4B-parameter model, 4-bit quantization, one
consumer GPU, and a hard wall-clock budget.

Our contributions are:

1. An end-to-end, reproducible pipeline (data construction → QLoRA SFT →
   GRPO → held-out evaluation) that fits in free-tier Kaggle sessions.
2. A memory-bounded agent runtime: host-RAM offload of Gemma 4's 5.25 GiB
   per-layer embedding table plus a bounded generation context, which turns
   long episodes that OOMed on a T4 into runs that complete.
3. An honest, compute-bounded evaluation on a frozen stratified subset, with
   sandbox-isolation caveats reported alongside every number.

## 2. Related work

- **SWE-bench / SWE-bench Verified** established repository-level issue
  resolution with hidden tests as the metric.
- **SWE-agent** showed the importance of a small, fixed tool interface
  (view/edit/search/run) for agent-computer interaction.
- **QLoRA** makes 4-bit fine-tuning of large models feasible on a single GPU.
- **GRPO** removes the value network by normalizing rewards within a group of
  rollouts for the same prompt, which suits a tight compute budget.

Our work is an independent, small-scale reproduction of this recipe under
explicit hardware constraints, not a new algorithm.

## 3. Method

### 3.1 Data and splits

- 9 Python repositories are prepared as local checkouts.
- Runnable task pool: 1,786 issue–test pairs whose environments build.
- SFT: 870 trajectories kept only if the final patch passes the task's tests
  in the sandbox. Prompts and observations are masked; loss is computed on
  the model's own reasoning and tool-call tokens only.
- RL: 267 tasks, strictly disjoint from SFT; the policy sees the issue and
  repository but never the gold patch.
- Evaluation: a frozen 291-task holdout, untouched by SFT and RL. Because a
  full 291 × 2-policy evaluation costs ~100+ GPU-hours at the measured T4
  speed, the headline evaluation uses a deterministic 24-task subset:
  holdout rows are bucketed by repository and sampled round-robin
  (4 django, 4 xarray, 4 pylint, 3 pytest, 3 sphinx, 3 sympy, 2 flask,
  1 requests). The subset file is committed at
  `data/holdout_eval_subset.jsonl`.

### 3.2 Agent runtime and sandbox

The policy emits exactly one tool call per step from a fixed interface:
`read_file`, `search_repo`, `edit_file`, `run_tests`, `submit`. There is no
shell: argv is executed directly, binaries are allowlisted, and repository
paths are scope-checked. Each test run has CPU-time, memory and process
limits and returns a typed result (`pass`, `fail`, `timeout`, `error`).

On Kaggle, `unshare` is blocked by seccomp, so filesystem/network isolation
degrades to `best_effort`; every reported number carries that caveat.

### 3.3 QLoRA SFT

Base model: Gemma 4 E4B (the 12B variant exceeds 16 GiB once full-vocabulary
logits, activations and optimizer state are added). The base is loaded in
4-bit NF4 with double quantization; LoRA rank 16 / alpha 32 is applied to all
attention and MLP projections. Training uses one epoch, batch size 1 with 16
gradient-accumulation steps, LR 2e-4 with cosine decay, max sequence length
4096, and gradient checkpointing.

Two engineering changes were required to fit the T4:

- **PLE offload.** Gemma 4's `embed_tokens_per_layer` table is
  `[262144, 10752]` bfloat16 ≈ 5.25 GiB and is never trained. It is moved to
  host RAM; the forward pass gathers on CPU and copies only the
  `[B, S, 10752]` result to the GPU. This recovers roughly a third of the
  card.
- **Chunked sliced cross-entropy.** The LM head is applied only to supervised
  positions in chunks of 128, avoiding a full-vocabulary logit tensor.

Final SFT loss fell from 0.648 to 0.301 to 0.264 over the logged run
(adapter: `ashuraasura/sft-adapter`).

### 3.4 GRPO

The SFT adapter initializes both the trainable policy and a frozen reference
(they share one 4-bit base). For each task we sample `k=4` rollouts through
the same agent loop and score each with the sandbox:

```
reward = 1.0 * tests_pass
       + 0.15 * patch_valid
       - 0.005 * steps_taken
       - 0.1 * (no submission)
```

Advantages are group-centered within a task (no std scaling, following the
Dr. GRPO correction), validity/step shaping is gated on at least one rollout
resolving the task, and a KL penalty (coefficient 0.01) anchors the policy to
the frozen SFT adapter. Budget after measurement on the T4:
`k=4`, one task per update, 8 steps per episode, 384 new tokens per step,
20 updates, checkpoint every 5 updates, 8-hour wall clock.

## 4. Experimental setup

| Item | Value |
|---|---|
| Hardware | 1× Kaggle T4, 16 GiB, bfloat16 unsupported → fp16 compute |
| Model | Gemma 4 E4B, 4-bit NF4 + LoRA |
| SFT data | 870 test-verified trajectories |
| RL pool | 267 tasks |
| Eval | 24-task frozen, repo-stratified subset; 12 steps; 512 new tokens |
| Metrics | resolved rate, patch validity, steps-to-submit, failure category |
| Sandbox | `best_effort` isolation on Kaggle (seccomp blocks `unshare`) |

Decoding for evaluation is temperature 0.2. A task counts as resolved only
if the submitted patch applies to a fresh checkout, touches no test/harness
file, and makes the task's tests pass with at least one test executed and
none skipped. This blocks the obvious reward hack of editing the tests.

## 5. Results

`TBD(final-eval): adapter vs base on the 24-task subset.`

Table 1. Held-out results (24 tasks; 12 steps; 512 new tokens).

| Policy | Resolved | Resolved rate | Patch validity | Avg steps to submit |
|---|---|---|---|---|
| Base Gemma 4 E4B (zero-shot) | TBD | TBD | TBD | TBD |
| + QLoRA SFT | TBD | TBD | TBD | TBD |
| + GRPO (if gate passes) | TBD | TBD | TBD | TBD |

`TBD(rl): final mean reward / best reward, updates completed, whether the
mid-RL dev gate passed, and whether the RL adapter was evaluated.`

SFT loss curve (logged): 0.648 → 0.301 → 0.264.

## 6. Ablations

We report only ablations the compute budget actually allows; anything not run
is listed as not run rather than implied.

- **SFT vs zero-shot base** — same 24 tasks, same harness. `TBD`
- **SFT vs SFT+GRPO** — same 24 tasks. Enabled only if the RL dev gate
  accepts the adapter; otherwise the SFT adapter is the final policy and we
  report the negative result.
- **Group size k** — RL v10 used k=2 and produced zero gradient because both
  rollouts failed (advantages ≈ ±0.0025); the final run uses k=4. This is a
  measured debugging result, not a clean ablation.
- **Reward shaping** — not ablated; shaping is gated and small by design.
- **Model size (12B vs E4B)** — not a choice: 12B OOMs on a T4. Reported as a
  constraint, not an ablation.

## 7. Failure modes

Categories are assigned from the rollout record and the test log:

| Category | Definition | Observed |
|---|---|---|
| `wrong_file` | edits a file unrelated to the failing test | TBD |
| `incomplete_patch` | patch applies but does not fix the failure | TBD |
| `test_environment_issue` | dependency/venv failure before policy action | TBD |
| `step_budget_exhaustion` | no `submit` within 12 steps | TBD |
| `policy_error` | generation/parse error, including CUDA OOM | eval v4: 3/4 episodes |

Eval v4 (pre-fix) was diagnostic: the SFT adapter OOMed after 14–21 steps
because the 5.25 GiB PLE table stayed resident while the KV cache grew; the
base policy OOMed at step 3 on one task and submitted at step 11 on another.
The fix (PLE host offload, 12k-token head+tail prompt cap, per-step cache
release) is validated by eval v5 and used for all final numbers.

## 8. Limitations and honest scope

- **Compute-bounded evaluation.** The full 291-task × 2-policy holdout is
  ~100+ GPU-hours at measured speed and was not run. Headline numbers are on
  the committed 24-task stratified subset. Confidence intervals at n=24 are
  wide; we report counts, not just rates.
- **Sandbox isolation.** Kaggle blocks `unshare`; filesystem/network
  isolation is `best_effort`. A malicious policy could in principle reach the
  host filesystem. Reward and eval numbers are therefore not fully isolated.
- **Model size.** E4B, not 12B: 12B does not fit a 16 GiB T4 with
  full-vocabulary logits and activations.
- **Single seed, single run.** No error bars across seeds.
- **RL scale.** 20 updates with one task each is small; the RL result, if
  any, is a lower bound on what the recipe can do.
- **No teacher-model distillation.** SFT trajectories are replayed from
  test-passing gold patches and filtered rollouts, not generated by a larger
  teacher.

## 9. Reproducibility

- Code: this repository (`gemma4-coding-agent`), commit `TBD(final)`.
- Kaggle kernels: `ashuraasura/gemma-4-qlora-sft-architecture-2-4`,
  `ashuraasura/gemma-4-grpo-rl`,
  `ashuraasura/gemma-4-eval-holdout-architecture-2-6`.
- Adapter: `ashuraasura/sft-adapter`.
- Evaluation artifacts: per-task JSONL plus trajectories from the eval
  kernel; summary JSON contains resolved rate, patch validity, steps to
  submit and per-repo breakdown.
- Deterministic subset: `data/holdout_eval_subset.jsonl`.

## Appendix A. Compute accounting

| Stage | Measured |
|---|---|
| SFT | 22 kernel versions; PLE offload frees 5.25 GiB; CE chunk 128 |
| Eval (pre-fix) | ~50–80 s/step; 19.5 min for a 21-step SFT episode |
| Eval OOM | 14–21 steps (SFT), 3 steps (base, one task) |
| RL rollout | 619 s for 2 rollouts × 6 steps + one update (k=2 smoke) |
| Kaggle quota | 24 h 17 m available of 30 h at the start of the final runs |
