"""Unit tests for GRPO math and bookkeeping (ARCHITECTURE §2.5, §3.3) — no GPU needed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from train.rl.grpo import (
    _dev_tasks,
    _last_checkpoint,
    _load_tasks,
    _scaled_loss,
    grpo_loss,
    k3_kl,
    token_logprobs,
)


def _fake_logits_labels(vocab: int = 6, seq: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(1, seq, vocab)
    labels = torch.tensor([[0, 1, 2, 3, 4]])  # next-token targets shift left
    labels[0, 2] = -100  # one masked position
    return logits, labels


def test_token_logprobs_shift_and_mask():
    logits, labels = _fake_logits_labels()
    logp, mask = token_logprobs(logits, labels)

    # HF shift convention: length seq-1; output[t] scores labels[t+1] from logits[t]
    assert logp.shape == torch.Size([1, 4])
    assert labels.shape == torch.Size([1, 5])
    expected_0 = torch.log_softmax(logits[0, 0], dim=-1)[1].item()
    assert logp[0, 0] == pytest.approx(expected_0, abs=1e-5)
    # labels = [0, 1, -100, 3, 4] -> scored targets are labels[1:] = [1, -100, 3, 4]
    assert mask.tolist() == [[1, 0, 1, 1]]
    assert logp[0, 1].item() == 0.0
    expected_2 = torch.log_softmax(logits[0, 2], dim=-1)[3].item()
    assert logp[0, 2] == pytest.approx(expected_2, abs=1e-5)


def test_token_logprobs_greedy_consistency():
    logits, labels = _fake_logits_labels()
    # labels: dummy first token, then the argmax of each logits position
    targets = logits[0, :-1].argmax(dim=-1)
    labels = torch.cat([torch.tensor([[99]]), targets.unsqueeze(0)], dim=1)
    logp, mask = token_logprobs(logits, labels)
    assert mask.sum() == 4
    # output[t] scores argmax(logits[t]) -> must equal the max probability
    for t in range(4):
        probs = torch.softmax(logits[0, t], dim=-1)
        assert torch.exp(logp[0, t]).item() == pytest.approx(probs.max().item(), rel=1e-3)


def test_k3_kl_properties():
    mask = torch.ones(1, 4)
    p = torch.zeros(1, 4)
    r = torch.zeros(1, 4)
    assert k3_kl(p, r, mask).abs().max().item() == 0.0

    r = torch.full((1, 4), -0.5)  # ref logprob lower than policy
    kl = k3_kl(p, r, mask)
    assert (kl >= 0).all()
    delta = -0.5
    assert kl[0, 0].item() == pytest.approx(torch.exp(torch.tensor(delta)) - delta - 1.0)

    # asymmetric: KL(policy || ref) direction matters
    kl_rev = k3_kl(r, p, mask)
    assert not torch.allclose(kl, kl_rev)


def test_grpo_loss_signs_and_zero_advantage():
    mask = torch.ones(1, 3)
    r = torch.full((1, 3), -2.0)

    p = torch.full((1, 3), -2.0)  # identical to ref -> k3 KL term exactly 0
    loss, stats = grpo_loss(p, r, mask, advantage=0.0, kl_coeff=0.01)
    assert stats["kl"] == pytest.approx(0.0, abs=1e-6)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    # pg = -adv * mean(logp); logp=-2, adv=1 -> +2
    loss_pos, stats_pos = grpo_loss(p, r, mask, advantage=1.0, kl_coeff=0.0)
    assert stats_pos["pg"] == pytest.approx(2.0, abs=1e-5)
    assert loss_pos.item() == pytest.approx(2.0, abs=1e-5)

    # negative advantage flips the gradient direction exactly
    loss_neg, _ = grpo_loss(p, r, mask, advantage=-1.0, kl_coeff=0.0)
    assert loss_neg.item() == pytest.approx(-loss_pos.item(), rel=1e-5)

    # descent on the pg term must push logp up when adv > 0
    p_grad = torch.full((1, 3), -2.0, requires_grad=True)
    loss_grad, _ = grpo_loss(p_grad, r, mask, advantage=1.0, kl_coeff=0.0)
    loss_grad.backward()
    assert p_grad.grad is not None
    assert torch.allclose(p_grad.grad, torch.full((1, 3), -1.0 / 3.0))

    # KL pulls back toward the reference when it drifts
    r_drift = torch.full((1, 3), -4.0)
    loss_kl, stats_kl = grpo_loss(p, r_drift, mask, advantage=0.0, kl_coeff=1.0)
    assert stats_kl["kl"] > 0 and loss_kl.item() > 0


def test_scaled_loss_applies_batch_token_scale():
    mask = torch.ones(1, 10)
    p = torch.zeros(1, 10)
    r = torch.zeros(1, 10)
    full, _ = _scaled_loss(p, r, mask, advantage=1.0, kl_coeff=0.0, scale=1.0)
    half, _ = _scaled_loss(p, r, mask, advantage=1.0, kl_coeff=0.0, scale=0.5)
    assert half.item() == pytest.approx(full.item() * 0.5, rel=1e-5)


def test_last_checkpoint_numeric_order(tmp_path: Path):
    assert _last_checkpoint(tmp_path / "nope") is None
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-5").mkdir()
    (tmp_path / "other").mkdir()
    assert _last_checkpoint(tmp_path).name == "checkpoint-10"


def test_dev_tasks_null_path_and_ids(tmp_path: Path):
    tasks = [{"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"}]
    assert _dev_tasks(None, tasks) == []
    assert _dev_tasks("", tasks) == []
    assert _dev_tasks(["a", "c"], tasks) == [{"task_id": "a"}, {"task_id": "c"}]
    assert _dev_tasks(["zz"], tasks) == []

    pool = tmp_path / "dev.jsonl"
    pool.write_text(json.dumps({"task_id": "x"}) + "\n", encoding="utf-8")
    assert _dev_tasks(str(pool), tasks) == [{"task_id": "x"}]


def test_load_tasks(tmp_path: Path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps({"task_id": "t1"}) + "\n" + "\n" + json.dumps({"task_id": "t2"}) + "\n",
        encoding="utf-8",
    )
    assert [t["task_id"] for t in _load_tasks(pool)] == ["t1", "t2"]

    with pytest.raises(SystemExit):
        _load_tasks(tmp_path / "missing.jsonl")
