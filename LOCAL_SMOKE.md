# Local two-problem smoke check (base vs SFT)

This is a **side check**, not the headline evaluation. It exists to show the
same harness, prompt template and tool-call parser running base and SFT on
exactly two held-out tasks on a 15.7 GB laptop without swapping.

The headline result in `paper/` remains the 24-task Kaggle evaluation
(0/20 vs 0/20, McNemar p = 1.0; 8 clean pairs 0/8 vs 0/8).

## Why not the transformers path

Gemma 4 E4B keeps a per-layer embedding (PLE) table of
262,144 × 256 × 42 × 2 B = **5.25 GiB**. bitsandbytes quantizes `nn.Linear`
only, so a 4-bit load still needs roughly 10-11 GB resident. This machine has
15.7 GB total and ~5 GB free; the guarded run would have thrashed or frozen.
A Q3_K_M GGUF (3.78 GB) stays mmap'd, so pages are evictable instead.

## Setup

- Base: `unsloth/gemma-4-E4B-it-GGUF`, `gemma-4-E4B-it-Q3_K_M.gguf`,
  4,058,137,728 bytes,
  sha256 `0980898d0e7d9c75cea05141bf4010393312101ef6cf5b31f8197e0de18655f4`.
- Runtime: `llama.cpp` b11968, `llama-b11968-bin-win-cpu-x64`,
  `-c 4096 -t 4 -tb 4 -np 1 --special --no-warmup`.
- Adapter: the paper's PEFT SFT adapter, converted to a GGUF LoRA with
  `scripts/make_lora_gguf.py` (258 pairs, f16, r = 16, alpha = 32,
  69,798,464 bytes). The converter refuses to run unless every adapter
  target exists in the base GGUF.
- Prompt: the same tokenizer, chat template, `default_tool_specs()` and
  `parse_tool_call()` as `eval/run_eval.py`; prompt capped at 2,048 tokens
  (head 512 + tail 1,536, matching `cap_prompt_tokens`); greedy decoding.
- Budget: 3 steps, 64 new tokens per call, 300 s test limit. The paper used
  12 steps / 512 tokens; the smaller budget is deliberate (protect the
  laptop) and makes this descriptive only.
- Arms were run one at a time, base first, then SFT.

Commands:

```bash
python scripts/make_lora_gguf.py
python scripts/local_gguf_smoke.py --model base
python scripts/local_gguf_smoke.py --model sft
```

## Results

| task | base | SFT |
| --- | --- | --- |
| `django__django-10390` | `budget_exhausted`, 2 steps, 0 resolved, 157 s | `budget_exhausted`, 1 step, 0 resolved, 170 s |
| `django__django-10531` | `budget_exhausted`, 2 steps, 0 resolved, 161 s | `budget_exhausted`, 2 steps, 0 resolved, 179 s |
| **resolved** | **0/2** | **0/2** |

Peak `llama-server` RSS was 3.64 GB (base) and 3.89 GB (SFT); the minimum
system-available RAM observed was 1.80 GB (base) and 1.96 GB (SFT). No swap
was entered and both processes shut down cleanly.

Qualitative notes from the three allowed generations per task:

- `10390`: base searched for `test_truncate_to_dst_ends_stockholm` (no
  matches), then `AmbiguousTimeError`; SFT read
  `django/utils/timezone.py` lines 121-225 on its first call.
- `10531`: base called a nonexistent `search` tool, recovered with
  `search_repo`, then malformed; SFT searched `verbose_name` and read
  `django/apps/config.py`.
- Neither arm submitted a patch before the budget ran out.

## Caveat

This uses a Q3_K_M GGUF and a runtime LoRA, not the paper's 4-bit
transformers run. The 0/2 vs 0/2 outcome agrees with the headline null
result but does not replace or extend it; n = 2 and the budget is smaller.
