"""Collator tests — completion-only masking (ARCHITECTURE §2.4)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train.sft.data_collator import TrajectoryCollator, as_ids, encode_trajectory, sliced_supervision


class FakeTokenizer:
    """Concatenative turn template: ``<role>content</role>`` per message."""

    pad_token_id = 0
    eos_token_id = 1

    def _render(self, messages) -> str:
        return "".join(
            f"<{m['role']}>{m.get('content') or ''}</{m['role']}>"
            for m in messages
        )

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = self._render(messages)
        # one id per character (deterministic, order-preserving)
        ids = [ord(c) % 30000 + 2 for c in text]
        return ids if tokenize else text


MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "issue"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "edit_file", "arguments": "{}"}}]},
    {"role": "tool", "content": "observation", "tool_call_id": "c1", "name": "edit_file"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c2", "type": "function",
         "function": {"name": "submit", "arguments": "{}"}}]},
]


def test_assistant_spans_unmasked_others_masked():
    tok = FakeTokenizer()
    enc = encode_trajectory(tok, MESSAGES, max_seq_len=4096)
    ids, labels = enc["input_ids"], enc["labels"]

    def assistant_spans() -> set[int]:
        indices: set[int] = set()
        start = 0
        for m in MESSAGES:
            seg = f"<{m['role']}>{m.get('content') or ''}</{m['role']}>"
            if m["role"] == "assistant":
                indices.update(range(start, start + len(seg)))
            start += len(seg)
        return indices

    covered = assistant_spans()
    for index, (token, label) in enumerate(zip(ids, labels)):
        assert (label == token) == (index in covered), f"token {index}"
    assert any(label != -100 for label in labels)


def test_left_truncation_keeps_tail():
    tok = FakeTokenizer()
    enc = encode_trajectory(tok, MESSAGES, max_seq_len=40)
    assert len(enc["input_ids"]) == 40
    assert len(enc["labels"]) == 40
    # the final (submit) assistant turn survives truncation
    assert any(label != -100 for label in enc["labels"])


def test_collator_padding_shapes():
    tok = FakeTokenizer()
    collator = TrajectoryCollator(tok, pad_to_multiple_of=8)
    batch = collator([
        encode_trajectory(tok, MESSAGES, max_seq_len=4096),
        encode_trajectory(tok, MESSAGES[:2], max_seq_len=4096),
    ])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    rows, cols = batch["input_ids"].shape
    assert cols % 8 == 0 and rows == 2
    # padding never contributes to loss
    short = 1
    padded_from = len(encode_trajectory(tok, MESSAGES[:2], max_seq_len=4096)["input_ids"])
    assert (batch["labels"][short, padded_from:] == -100).all()
    assert (batch["attention_mask"][short, padded_from:] == 0).all()


def test_as_ids_handles_batch_encoding_userdict():
    # bare AutoTokenizer wraps tokenize=True output in BatchEncoding (UserDict,
    # NOT a dict) — must unwrap ["input_ids"] instead of iterating the keys
    from transformers import BatchEncoding

    be = BatchEncoding({"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]})
    assert not isinstance(be, dict)
    assert as_ids(be) == [7, 8, 9]
    # processor-style batched nested list still unwraps
    assert as_ids([[4, 5], [6]]) == [4, 5]
    assert as_ids([4, 5, 6]) == [4, 5, 6]


def test_sliced_supervision_pairs_positions_with_targets():
    labels = [-100, -100, 7, -100, 9, 10]
    keep, targets = sliced_supervision(labels)
    # t=2 -> logit row 1, t=4 -> row 3, t=5 -> row 4; label[0] is never a
    # target (nothing predicts position 0 — the shift drops it)
    assert keep == [1, 3, 4]
    assert targets == [7, 9, 10]
    # unsupervised-everything yields no slicing keys (full-logit fallback)
    assert sliced_supervision([-100, -100, -100]) == ([], [])
    assert sliced_supervision([9, -100, -100]) == ([], [])


def test_collator_emits_sliced_keys_single_row():
    tok = FakeTokenizer()
    batch = TrajectoryCollator(tok, pad_to_multiple_of=8)([
        encode_trajectory(tok, MESSAGES, max_seq_len=4096),
    ])
    keep, shift = batch["logits_to_keep"], batch["shift_labels"]
    assert keep.ndim == 1 and shift.shape == (1, keep.numel())
    labels = batch["labels"][0]
    # each kept row p pairs with the supervised label at p+1, none masked
    assert (shift[0] == labels[keep + 1]).all()
    assert (shift[0] != -100).all()
    assert (keep >= 0).all()


def test_collator_sliced_union_fills_unsupervised_rows():
    tok = FakeTokenizer()
    a = encode_trajectory(tok, MESSAGES, max_seq_len=4096)
    b = encode_trajectory(tok, MESSAGES[:2], max_seq_len=4096)
    batch = TrajectoryCollator(tok, pad_to_multiple_of=8)([a, b])
    keep, shift = batch["logits_to_keep"], batch["shift_labels"]
    assert shift.shape == (2, keep.numel())
    for r in range(2):
        labels = batch["labels"][r]
        expected = torch.where(
            labels[keep + 1] != -100, labels[keep + 1], torch.full_like(keep, -100)
        )
        assert torch.equal(shift[r], expected)
    # union covers at least each row's own targets
    assert shift.shape[1] >= len(sliced_supervision(list(a["labels"]))[0])


def test_collator_omits_sliced_keys_when_no_targets():
    # encode_trajectory guarantees >=1 supervised token; this covers the
    # guard: without targets the batch falls back to full-sequence logits
    batch = TrajectoryCollator(FakeTokenizer(), pad_to_multiple_of=None)([
        {"input_ids": [5, 6, 7], "labels": [-100, -100, -100]},
    ])
    assert "logits_to_keep" not in batch
    assert "shift_labels" not in batch
