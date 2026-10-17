# Audit report and fixes

Branch `audit-fixes`. Every finding was reproduced against this repo's own code
before being fixed; "verified" below means a test in this repo now covers it and
passes (run `python -m pytest -q`, see *How to verify*).

## Findings and status

| ID | Severity | Finding | Fix | Verification |
|----|----------|---------|-----|--------------|
| C1 | Critical | Reward hackable: patch applied *after* the test patch with no path guard — editing the target test to `assert True` or adding a `conftest.py` that skips everything scored `resolved=True` (reproduced). Exit code alone was trusted. | `eval/scoring_guard.py`: protected-path sanitising, junit/`Ran N tests … OK` verification (≥1 executed, 0 skipped), no-op (whitespace/comment) patches invalid. | `eval/tests/test_scoring_guard.py` (real sandbox, toy repo) |
| C2 | Critical | Sandbox leaked: payload could read the trainer's secrets via `/proc/<ppid>/environ`, write anywhere (shared venvs/checkouts → cross-episode poisoning), see all host PIDs, survive timeouts via `setsid`; loopback was down (reproduced). | `sandbox/fsisolation.py`: PID namespace + fresh `/proc`, PID-1 supervisor, read-only root with writable sandbox root, empty capability set + `no_new_privs`, loopback up. `SANDBOX_FS_ISOLATION=required` fails closed (default in RL/eval scripts). | `sandbox/tests/test_fs_isolation.py` |
| C3 | Critical | Model output unparseable: Gemma 4 emits `<|tool_call>call:name{k:<\|"\|>v<\|"\|>}<tool_call|>`; harness decoded with `skip_special_tokens=True` and parsed JSON only. `tools=` was never passed to the template; double `<bos>` in `HFPolicy`. | `agent/gemma_format.py` native parser (+JSON fallback), special tokens kept, `tools=` rendered everywhere, `add_special_tokens=False`, malformed output is retried (≤2) instead of ending the episode. | `agent/tests/test_gemma_format.py` (**not** run against the real gated tokenizer — see residual risk) |
| C4 | Critical | ~100 % of SFT trajectories exceeded `max_seq_len`; `input_ids[-N:]` cut off system prompt, tool schemas and the issue. | Head-preserving `fit_messages`; bounded tools (`read_file` windows, 3 k test output, 25 search hits); `data/scripts/compact_sft_trajectories.py` (median 47.8 k → 11.2 k chars; 515/870 kept ≤16 k chars); truncation is reported by the trainer. | `train/tests/test_fit_messages.py`, `data/tests/test_pipeline.py` |
| H1 | High | Oracle-replay SFT data: no failures, no recovery. | 25 % of trajectories get a *real* failed first edit (actual `apply_unified_diff` error) flagged `train:false`, then the correct edit. Reasoning is now kept in the rollout history. | compaction tests |
| H2 | High | GRPO std-normalisation turned shaping terms into unit advantages in all-fail groups (trains "submit a valid-looking patch fast"). | Gated shaping, mean-centred advantages (no std), infra-killed rollouts excluded. | `train/rl/tests/test_audit_fixes.py` |
| H3 | High | RL trained on a re-rendered canonical tool call, not the sampled tokens; reasoning dropped. | Per-step samples = prompt + **exact sampled ids** (`train/rl/samples.py`). | `test_audit_fixes.py` (fake template; **GPU path not run**) |
| H4 | High | Score noise: 30 s CPU default killed real suites and counted as failure; infra kills were indistinguishable from test failures. | `Limits.for_tests`, `infra_error` flag, excluded from advantages. | scoring-guard tests |
| H5 | High | `submit` required re-typing the whole patch (doubled length, transcription errors). | `patch` optional; submits `git diff` (scratch paths excluded). | `agent/tests/test_tools_and_loop.py` |
| M1 | Medium | Compute plan infeasible (serial HF generate, 200 updates × 32 long episodes). | Re-sized defaults (k=8, 2 tasks, 60 updates, 768 tokens/step), wall-clock budget with clean checkpoint. vLLM **not** added. | config only |
| M2 | Medium | Regression (pass-to-pass) tests missing for Django/sympy (77 % of holdout). | Scorer accepts `p2p_targets`; **data must be rebuilt** to populate it. | code path only |
| M3 | Medium | PEFT nested adapter layout → silent no-resume / unloadable final adapter. | `train/rl/checkpoints.py` flattens; resume refuses to start from scratch silently. | `test_audit_fixes.py` (fake PEFT; verify once with real PEFT) |
| M4 | Medium | Dev gate never ran (`dev_slice: null`), 4 tasks from the training pool. | `dev_slice: auto` carves 24 held-out tasks, 12 per eval, patience-based stop, `best` adapter. | `test_audit_fixes.py` |
| M5 | Medium | Holdout validity: SWE-bench is public; instance-level split (17 holdout tasks share a base commit with SFT, 2 share issue text). | **Not changed** — report a repo-disjoint / post-cutoff slice and disclose possible pre-training contamination. | — |
| M6 | Medium | No seeding of torch. | `torch.manual_seed`. | — |
| M7 | Medium | Notebook put `GH_TOKEN` in the clone URL (persisted in `.git/config`, visible to payloads). | Transient auth header, token removed from env; no LICENSE → added placeholder; see `DATA_LICENSES.md`. | — |

## How to verify

```bash
python -m pytest -q --ignore=train/rl/tests/test_grpo.py     # torch-free suite
python -m pytest -q                                          # everything (needs torch)
python -m data.scripts.compact_sft_trajectories               # regenerate the SFT file
```

Known: `data/tests/test_pipeline.py::test_verify_instance_fail_to_pass` fails on the
unmodified baseline in the environment used for this audit (pre-existing, unrelated).

## Residual risks / not done

* **Nothing GPU- or model-dependent was executed.** The native tool-call parser follows
  the documented Gemma 4 format but has never seen the real tokenizer; run one
  `apply_chat_template → generate → decode → parse_tool_call` round trip first
  (`train/tests/test_gemma_processor.py` is gated on network access).
* `SANDBOX_FS_ISOLATION=required` needs `CAP_SYS_ADMIN` or unprivileged user namespaces.
  If Kaggle's kernel image forbids both, the run aborts (by design); `best_effort` runs
  but results are **not trustworthy**. Reads of host files and seccomp are not covered —
  keep secrets out of files.
* A forged "all tests passed" produced from inside the tested process is hard but not
  impossible; the junit/summary checks raise the bar, they are not a proof.
* The SFT data is still oracle-derived (search query and edit come from the gold patch).
  Real exploration trajectories from a teacher model, and failing-test→fix examples, would
  help most. `p2p_targets` need a data rebuild from raw SWE-bench.
* Malformed generations are retried but not penalised in the RL loss (their ids are not
  recorded).
* Throughput: still serial HF `generate`; moving rollouts to vLLM is the largest remaining
  speed-up.
