"""Sliced lm_head loss ≡ full-sequence CE on a tiny Gemma-4.

The collator emits ``logits_to_keep``/``shift_labels`` so the model projects
only the K supervised rows through the lm_head instead of the whole padded
sequence — with Gemma-4's vocab 262144, full [S, V] logits are ~2 GiB fp32
at S=2048 and OOM a T4 (ARCHITECTURE §0.3). This pins the two paths equal,
batch path (union + -100 fillers) included, and checks gradients flow.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import (  # noqa: E402
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
)

from train.sft.data_collator import TrajectoryCollator  # noqa: E402

VOCAB = 512


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [ord(c) % 300 + 3 for m in messages for c in (m.get("content") or "")]
        return ids if tokenize else "".join(m.get("content") or "" for m in messages)


@pytest.fixture(scope="module")
def tiny_gemma4():
    torch.manual_seed(0)
    text = Gemma4TextConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=VOCAB,
        hidden_size_per_layer_input=16,
        max_position_embeddings=512,
    )
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=text)).eval()
    yield model
    model.zero_grad(set_to_none=True)


def _inputs(batch: int = 1, seq: int = 24, seed: int = 1):
    torch.manual_seed(seed)
    input_ids = torch.randint(3, VOCAB, (batch, seq))
    labels = input_ids.clone()
    for t in range(seq):
        if t % 5 in (0, 1, 2) or t < 6:
            labels[:, t] = -100
    labels[:, 0] = -100  # position 0 is never a target
    return input_ids, labels


def test_sliced_loss_matches_full_ce_single_row(tiny_gemma4):
    input_ids, labels = _inputs()
    seq = input_ids.shape[1]
    keep = [t - 1 for t in range(1, seq) if labels[0, t] != -100]
    shift = torch.tensor([labels[0, t] for t in range(1, seq) if labels[0, t] != -100])
    assert keep  # supervised targets exist
    with torch.no_grad():
        full = tiny_gemma4(input_ids=input_ids, labels=labels).loss
        slim = tiny_gemma4(
            input_ids=input_ids,
            labels=labels,
            logits_to_keep=torch.tensor(keep),
            shift_labels=shift,
        ).loss
    assert torch.isfinite(full) and torch.isfinite(slim)
    assert torch.allclose(full, slim, atol=1e-5)


def test_sliced_loss_matches_full_ce_batch_union(tiny_gemma4):
    """Collator batch (union rows + per-row -100 fillers) == full-sequence CE."""
    tok = FakeTokenizer()
    feats = [
        {"input_ids": [3, 4, 5, 6, 7, 8, 9, 10],
         "labels": [-100, -100, 5, -100, 7, 8, -100, 10]},
        {"input_ids": [3, 4, 5, 6, 7, 8, 9, 10, 11],
         "labels": [-100, 4, -100, -100, -100, 8, -100, -100, -100]},
    ]
    batch = TrajectoryCollator(tok, pad_to_multiple_of=8)(feats)
    assert batch["input_ids"].shape[1] % 8 == 0
    with torch.no_grad():
        full = tiny_gemma4(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
        ).loss
        slim = tiny_gemma4(**batch).loss
    assert torch.allclose(full, slim, atol=1e-5)


def test_sliced_backward_flows_gradients(tiny_gemma4):
    input_ids, labels = _inputs()
    seq = input_ids.shape[1]
    keep = [t - 1 for t in range(1, seq) if labels[0, t] != -100]
    shift = torch.tensor([labels[0, t] for t in range(1, seq) if labels[0, t] != -100])
    tiny_gemma4.train()
    try:
        loss = tiny_gemma4(
            input_ids=input_ids,
            labels=labels,
            logits_to_keep=torch.tensor(keep),
            shift_labels=shift,
        ).loss
        loss.backward()
        grads = [p.grad for p in tiny_gemma4.parameters() if p.grad is not None]
        assert grads, "no gradients reached any parameter"
        assert any(g.abs().sum() > 0 for g in grads)
    finally:
        tiny_gemma4.zero_grad(set_to_none=True)
        tiny_gemma4.eval()


def _supervised(input_ids, labels):
    seq = input_ids.shape[1]
    keep = [t - 1 for t in range(1, seq) if labels[0, t] != -100]
    shift = torch.tensor([labels[0, t] for t in range(1, seq) if labels[0, t] != -100])
    return torch.tensor(keep), shift


def test_chunked_ce_matches_model_loss(tiny_gemma4):
    """chunked_sliced_ce (chunk=3, forced boundaries) == model-internal loss."""
    from train.sft.trainer import chunked_sliced_ce

    input_ids, labels = _inputs()
    keep, shift = _supervised(input_ids, labels)
    with torch.no_grad():
        builtin = tiny_gemma4(
            input_ids=input_ids, labels=labels, logits_to_keep=keep, shift_labels=shift
        ).loss
        logits = tiny_gemma4(input_ids=input_ids, logits_to_keep=keep).logits
        chunked = chunked_sliced_ce(logits, shift, chunk=3)
    assert torch.allclose(builtin, chunked, atol=1e-5)


def test_chunked_ce_matches_with_softcap():
    """E4B sets final_logit_softcapping=30 — chunked CE matches that path too."""
    from train.sft.trainer import chunked_sliced_ce

    torch.manual_seed(0)
    text = Gemma4TextConfig(
        num_hidden_layers=2, hidden_size=64, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, vocab_size=VOCAB,
        hidden_size_per_layer_input=16, max_position_embeddings=512,
        final_logit_softcapping=30.0,
    )
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=text)).eval()
    input_ids, labels = _inputs()
    keep, shift = _supervised(input_ids, labels)
    with torch.no_grad():
        builtin = model(
            input_ids=input_ids, labels=labels, logits_to_keep=keep, shift_labels=shift
        ).loss
        logits = model(input_ids=input_ids, logits_to_keep=keep).logits
        chunked = chunked_sliced_ce(logits, shift, chunk=5)
    assert torch.allclose(builtin, chunked, atol=1e-5)


def test_sliced_loss_trainer_compute_loss(tiny_gemma4, tmp_path):
    """SlicedLossTrainer pops label keys, skips the model loss, matches built-in."""
    from transformers import TrainingArguments

    from train.sft.trainer import SlicedLossTrainer

    input_ids, labels = _inputs()
    keep, shift = _supervised(input_ids, labels)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "logits_to_keep": keep,
        "shift_labels": shift,
    }
    with torch.no_grad():
        builtin = tiny_gemma4(
            input_ids=input_ids, labels=labels, logits_to_keep=keep, shift_labels=shift
        ).loss
    args = TrainingArguments(output_dir=str(tmp_path / "t"), use_cpu=True, report_to=[])
    trainer = SlicedLossTrainer(model=tiny_gemma4, args=args, train_dataset=[])
    inputs = dict(batch)
    loss = trainer.compute_loss(tiny_gemma4, inputs)
    assert torch.allclose(builtin, loss, atol=1e-5)
    assert "labels" not in inputs and "shift_labels" not in inputs
    assert "logits_to_keep" in inputs  # forward still gets the slice
