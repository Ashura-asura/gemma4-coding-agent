"""Completion-only encoding + padding for trajectory SFT (ARCHITECTURE §2.4).

The loss must cover only the model's own tokens — tool calls and reasoning.
System/user prompts and tool observations are masked with -100.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Sequence

import torch


def deserialize_tool_args(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gemma-4's template validates ``arguments`` as a JSON object (intentional
    per the merged model-card fix); our OpenAI-style records store strings."""
    rendered: list[dict[str, Any]] = []
    for message in messages:
        message = dict(message)
        calls = message.get("tool_calls")
        if calls:
            converted = []
            for call in calls:
                call = dict(call)
                function = dict(call.get("function") or {})
                args = function.get("arguments")
                if isinstance(args, str):
                    function["arguments"] = json.loads(args)
                call["function"] = function
                converted.append(call)
            message["tool_calls"] = converted
        rendered.append(message)
    return rendered


def as_ids(rendered: Any) -> list[int]:
    """Normalize apply_chat_template(tokenize=True) output to a flat id list.

    Tokenizers return ``list[int]`` (sometimes wrapped in a ``BatchEncoding``,
    which is a UserDict — not a dict); Gemma-4's processor returns a batched
    ``list[list[int]]``.
    """
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], (list, tuple)):
        rendered = rendered[0]
    if rendered and isinstance(rendered[0], (list, tuple)):
        rendered = rendered[0]
    return list(rendered)


def _prefix_lengths(template_owner: Any, messages: Sequence[dict[str, Any]]) -> list[int]:
    """Token count of each message prefix under the chat template.

    Turn-based templates (Gemma/Qwen) render prefixes concatenatively, so
    prefix[i] is the end offset of message i in the full rendering; a
    non-concatenative template still yields monotone-clamped usable bounds.
    """
    lengths: list[int] = []
    for index in range(1, len(messages) + 1):
        rendered = template_owner.apply_chat_template(
            list(messages[:index]), tokenize=True, add_generation_prompt=False
        )
        lengths.append(len(as_ids(rendered)))
    return lengths


def encode_trajectory(
    template_owner: Any,
    messages: Sequence[dict[str, Any]],
    max_seq_len: int,
) -> dict[str, list[int]]:
    """Render one trajectory to ``input_ids``/``labels`` with assistant-only loss.

    ``template_owner`` is a processor (Gemma-4) or tokenizer exposing
    ``apply_chat_template``.
    """
    if not messages:
        raise ValueError("trajectory has no messages")
    messages = deserialize_tool_args(messages)
    input_ids = as_ids(
        template_owner.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=False
        )
    )
    if not input_ids:
        raise ValueError("chat template produced no tokens")

    labels = [-100] * len(input_ids)
    start = 0
    for message, end in zip(messages, _prefix_lengths(template_owner, messages)):
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


def sliced_supervision(labels: Sequence[int]) -> tuple[list[int], list[int]]:
    """(logit positions, target ids) for the completion-only loss.

    Logits row ``p`` predicts token ``p+1``. Full-sequence CE drops label[0]
    (nothing predicts position 0), so targets are supervised labels at t >= 1
    and the matching logit rows are t - 1. Pairing them lets the model
    project only the K supervised rows through the lm_head instead of the
    whole padded sequence — with Gemma-4's vocab 262144 that is the
    difference between fitting and OOMing a T4 (ARCHITECTURE §0.3).
    """
    keep = [t - 1 for t in range(1, len(labels)) if labels[t] != -100]
    targets = [labels[t] for t in range(1, len(labels)) if labels[t] != -100]
    return keep, targets


class TrajectoryCollator:
    """Pad a pre-tokenized batch (labels pad with -100, ids with eos).

    ``tokenizer`` may be a processor — Gemma-4 is multimodal and exposes the
    chat template on ``AutoProcessor``, with the tokenizer underneath.

    The batch also carries ``logits_to_keep``/``shift_labels``: the union of
    supervised logit rows across the batch (1-D, shared seq index) plus the
    per-row target ids (-100 where a row does not supervise that position),
    which ``ForCausalLMLoss`` consumes via the model's ``**kwargs``.
    """

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8) -> None:
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
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
            **self._sliced([sliced_supervision(row) for row in labels]),
        }

    @staticmethod
    def _sliced(rows: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
        positions = sorted({p for keep, _ in rows for p in keep})
        if not positions:
            return {}
        index = {p: i for i, p in enumerate(positions)}
        shift = torch.full((len(rows), len(positions)), -100, dtype=torch.long)
        for r, (keep, targets) in enumerate(rows):
            for p, target in zip(keep, targets):
                shift[r, index[p]] = target
        return {
            "logits_to_keep": torch.tensor(positions, dtype=torch.long),
            "shift_labels": shift,
        }
