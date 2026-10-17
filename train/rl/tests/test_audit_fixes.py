"""torch-free tests for the GRPO audit fixes (H2, H3, M3, M4)."""
from __future__ import annotations

import random

import pytest

from train.rl.checkpoints import resolve_adapter_dir, save_policy_adapter
from train.rl.devgate import DevGate, hours_left, split_dev
from train.rl.reward import effective_rewards, group_advantages, group_normalize
from train.rl.rollout_worker import rollout_task
from train.rl.samples import build_step_samples, select_steps


# ----------------------------------------------------------- H2: reward shaping
COMP_FAIL = {"test_pass": 0.0, "patch_validity": 0.15, "step_penalty": -0.05, "no_submit_penalty": 0.0}
COMP_FAIL_FAST = {"test_pass": 0.0, "patch_validity": 0.15, "step_penalty": -0.01, "no_submit_penalty": 0.0}
COMP_WIN = {"test_pass": 1.0, "patch_validity": 0.15, "step_penalty": -0.03, "no_submit_penalty": 0.0}
COMP_NOSUBMIT = {"test_pass": 0.0, "patch_validity": 0.0, "step_penalty": -0.15, "no_submit_penalty": -0.1}


def test_old_std_normalisation_amplified_shaping_noise():
    # nobody solved the task; the only difference is how fast they submitted
    raw = [0.15 - 0.05, 0.15 - 0.01]
    adv = group_normalize(raw)
    assert abs(adv[1] - 1.0) < 1e-6 and abs(adv[0] + 1.0) < 1e-6  # unit-sized push toward "submit fast"


def test_gated_shaping_gives_no_gradient_when_nobody_resolves():
    rewards = effective_rewards([COMP_FAIL, COMP_FAIL_FAST], [False, False], [False, False])
    assert rewards == [0.0, 0.0]
    assert group_advantages(rewards) == [0.0, 0.0]


def test_shaping_applies_once_someone_resolves_and_no_submit_always_counts():
    rewards = effective_rewards([COMP_WIN, COMP_FAIL, COMP_NOSUBMIT], [True, False, False], [False] * 3)
    assert rewards[0] == pytest.approx(1.0 + 0.15 - 0.03)
    assert rewards[1] == pytest.approx(0.15 - 0.05)
    assert rewards[2] == pytest.approx(-0.1 + -0.15)  # validity 0 + step -0.15 + no-submit -0.1
    only_format = effective_rewards([COMP_NOSUBMIT, COMP_FAIL], [False, False], [False, False])
    assert only_format == [-0.1, 0.0]


def test_always_mode_restores_old_shaping():
    assert effective_rewards([COMP_FAIL], [False], [False], shaping="always") == [pytest.approx(0.10)]
    with pytest.raises(ValueError):
        effective_rewards([COMP_FAIL], [False], [False], shaping="nope")


def test_advantages_are_centred_not_variance_scaled_and_skip_infra_kills():
    adv = group_advantages([1.1, 0.0, 0.0, 0.0], [False, False, False, True])
    assert adv[3] == 0.0  # infra kill: no gradient, excluded from the baseline
    assert adv[0] == pytest.approx(1.1 - 1.1 / 3) and adv[1] == pytest.approx(-1.1 / 3)
    scaled = group_advantages([1.0, 0.0], scale_by_std=True)
    assert scaled == [pytest.approx(1.0, abs=1e-6), pytest.approx(-1.0, abs=1e-6)]
    assert group_advantages([1.0, 2.0], [True, True]) == [0.0, 0.0]


def _record(resolved: bool, steps: int, infra: bool = False, submitted: bool = True):
    return {
        "task_id": "t", "status": "submitted", "steps": steps, "error": None,
        "submitted_patch": "p" if submitted else None, "patch_valid": submitted,
        "resolved": resolved, "test_status": "pass" if resolved else "fail",
        "reason": "x", "infra_error": infra,
        "trajectory": {"messages": [], "tool_calls": [{"id": "c", "name": "submit", "arguments": {}, "completion_ids": [5, 6]}]},
    }


def test_rollout_task_uses_gated_advantages_and_carries_completions():
    records = iter([_record(False, 3), _record(False, 9)])
    group = rollout_task({"task_id": "t"}, None, k=2, max_steps=5, limits=None,
                         weights={"test_pass": 1.0, "patch_validity": 0.15, "step_penalty": 0.005, "no_submit_penalty": 0.1},
                         evaluate=lambda *a, **k: next(records))
    assert group["normalized"] == [0.0, 0.0]
    assert group["rollouts"][0]["completions"] == [[5, 6]]


def test_rollout_task_ignores_resource_killed_rollouts():
    records = iter([_record(True, 3), _record(False, 4), _record(False, 4, infra=True)])
    group = rollout_task({"task_id": "t"}, None, k=3, max_steps=5, limits=None,
                         weights={"test_pass": 1.0, "patch_validity": 0.15, "step_penalty": 0.005, "no_submit_penalty": 0.1},
                         evaluate=lambda *a, **k: next(records))
    assert group["normalized"][2] == 0.0
    assert group["normalized"][0] > 0 > group["normalized"][1]


# --------------------------------------------------- H3: exact on-policy samples
class Tok:
    """Concatenative template that honours add_generation_prompt and tools."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, tools=None):
        t = ("<tools>" + ",".join(x["function"]["name"] for x in tools) + "</tools>") if tools else ""
        t += "".join(f"<{m['role']}>{m.get('content') or ''}{[c['function']['name'] for c in m.get('tool_calls') or []] or ''}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            t += "<assistant>"
        ids = [ord(c) % 30000 + 2 for c in t]
        return ids if tokenize else t


def _msgs(turns=3):
    out = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "ISSUE text"}]
    for i in range(turns):
        out.append({"role": "assistant", "content": f"r{i}", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
        out.append({"role": "tool", "content": "obs" * 5, "tool_call_id": f"c{i}", "name": "read_file"})
    return out


def test_samples_train_exactly_the_sampled_ids_and_nothing_else():
    tok = Tok()
    completions = [[900 + i, 901 + i, 902 + i] for i in range(3)]
    samples = build_step_samples(tok, _msgs(3), completions, max_seq_len=4096, tools=None, max_steps=0)
    assert len(samples) == 3
    for n, s in enumerate(samples):
        sup = [row for row in s["labels"] if row != -100]
        assert sup == completions[n]                                   # exactly the sampled tokens
        assert s["input_ids"][-3:] == completions[n]
        assert all(row == -100 for row in s["labels"][: len(s["labels"]) - 3])
    # prompt of step 2 contains steps 0 and 1 (their reasoning is part of the history)
    prompt2 = tok.apply_chat_template(_msgs(3)[:6], tokenize=True, add_generation_prompt=True)
    assert samples[2]["input_ids"][: len(prompt2)] == prompt2


def test_samples_keep_the_issue_when_the_prompt_must_be_fitted():
    tok = Tok()
    long = _msgs(10)
    samples = build_step_samples(tok, long, [[7, 8]] * 10, max_seq_len=260, tools=None, max_steps=0)
    last = samples[-1]
    text = "".join(chr(i - 2) for i in last["input_ids"][:-2])
    assert "ISSUE text" in text and len(last["input_ids"]) <= 260


def test_select_steps_always_includes_the_final_step_and_respects_the_cap():
    rng = random.Random(1)
    assert select_steps(3, 6, rng) == [0, 1, 2]
    picked = select_steps(20, 4, rng)
    assert len(picked) == 4 and picked[-1] == 19 and picked == sorted(set(picked))


def test_steps_without_recorded_ids_are_skipped():
    assert build_step_samples(Tok(), _msgs(2), [None, None], max_seq_len=1000, tools=None) == []


# --------------------------------------------------------- M3: checkpoint layout
class FakePeft:
    """Mimics PEFT: a non-'default' adapter lands in <dir>/<name>/."""

    def save_pretrained(self, path, selected_adapters=None):
        from pathlib import Path

        for name in selected_adapters or ["policy", "ref"]:
            d = Path(path) / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "adapter_config.json").write_text("{}")
            (d / "adapter_model.safetensors").write_text("w")


def test_policy_adapter_is_flattened_to_the_checkpoint_root(tmp_path):
    target = save_policy_adapter(FakePeft(), tmp_path / "checkpoint-25")
    assert (target / "adapter_config.json").exists() and (target / "adapter_model.safetensors").exists()
    assert not (target / "policy").exists() and not (target / "ref").exists()
    assert resolve_adapter_dir(target) == target


def test_resolve_adapter_dir_finds_legacy_nested_layouts(tmp_path):
    FakePeft().save_pretrained(tmp_path / "old")
    assert resolve_adapter_dir(tmp_path / "old") == tmp_path / "old" / "policy"
    assert resolve_adapter_dir(tmp_path / "missing") is None


# ------------------------------------------------------------- M4: dev gate
def test_dev_split_is_deterministic_disjoint_and_leaves_training_data():
    tasks = [{"task_id": f"t{i}"} for i in range(50)]
    train, dev = split_dev(tasks, 10, seed=3)
    assert len(dev) == 10 and len(train) == 40
    assert not {t["task_id"] for t in train} & {t["task_id"] for t in dev}
    assert split_dev(tasks, 10, seed=3) == (train, dev)
    assert split_dev(tasks, 0)[1] == []
    assert len(split_dev(tasks[:3], 10)[0]) >= 1  # never starve training


def test_dev_gate_tracks_best_and_stops_after_patience():
    gate = DevGate(patience=2)
    assert gate.update(0.1) is True
    assert gate.update(0.2) is True and not gate.should_stop()
    assert gate.update(0.15) is False and not gate.should_stop()
    assert gate.update(0.2) is False and gate.should_stop()
    assert gate.best == 0.2 and gate.best_index == 1


def test_time_budget_helper():
    assert hours_left(0.0, None) is None
    assert hours_left(0.0, 11.0, now=3600.0) == pytest.approx(10.0)
    assert hours_left(0.0, 1.0, now=7200.0) < 0
