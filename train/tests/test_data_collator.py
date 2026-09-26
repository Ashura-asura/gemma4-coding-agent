"""Collator tests — completion-only masking (ARCHITECTURE §2.4)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train.sft.data_collator import TrajectoryCollator, encode_trajectory


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
