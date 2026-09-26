"""Completion-only encoding + padding for trajectory SFT (ARCHITECTURE §2.4).

The loss must cover only the model's own tokens — tool calls and reasoning.
System/user prompts and tool observations are masked with -100.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch


def _prefix_lengths(tokenizer: Any, messages: Sequence[dict[str, Any]]) -> list[int]:
    """Token count of each message prefix under the chat template.

    Turn-based templates (Gemma/Qwen) render prefixes concatenatively, so
    prefix[i] is the end offset of message i in the full rendering; a
    non-concatenative template still yields monotone-clamped usable bounds.
    """
    lengths: list[int] = []
    for index in range(1, len(messages) + 1):
        rendered = tokenizer.apply_chat_template(
            list(messages[:index]), tokenize=True, add_generation_prompt=False
        )
        lengths.append(len(rendered))
    return lengths


def encode_trajectory(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    max_seq_len: int,
) -> dict[str, list[int]]:
    """Render one trajectory to ``input_ids``/``labels`` with assistant-only loss."""
    if not messages:
        raise ValueError("trajectory has no messages")
    input_ids = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=False
    )
    if not input_ids:
        raise ValueError("chat template produced no tokens")

    labels = [-100] * len(input_ids)
    start = 0
    for message, end in zip(messages, _prefix_lengths(tokenizer, messages)):
        # clamp: non-concatenative templates can misalign, never go backwards
        end = max(start, min(end, len(input_ids)))
        if message.get("role") == "assistant":
            labels[start:end] = input_ids[start:end]
        start = end

    if not any(label != -100 for label in labels):
        # never train on an all-masked example; the final assistant turn is
        # the minimal signal (the submit call in a replayed trajectory)
        for index in range(len(labels) - 1, -1, -1):
            if input_ids[index] is not None:
                labels[index] = input_ids[index]
                break

    if len(input_ids) > max_seq_len:
        # keep the most recent turns: the edit/test/submit tail matters most
        input_ids = input_ids[-max_seq_len:]
        labels = labels[-max_seq_len:]

    return {"input_ids": list(input_ids), "labels": list(labels)}


class TrajectoryCollator:
    """Pad a pre-tokenized batch (labels pad with -100, ids with eos)."""

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer needs pad_token_id or eos_token_id")
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_len = -(-max_len // multiple) * multiple

        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention: list[list[int]] = []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * pad)
            labels.append(list(feature["labels"]) + [-100] * pad)
            attention.append([1] * len(feature["input_ids"]) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
