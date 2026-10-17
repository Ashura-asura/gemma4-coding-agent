"""Completion-only encoding + padding for trajectory SFT (ARCHITECTURE §2.4).

The loss must cover only the model's own tokens — tool calls and reasoning.
System/user prompts and tool observations are masked with -100.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Sequence

try:  # torch is only needed by the collator; encoding works without it
    import torch
except ImportError:  # pragma: no cover - exercised on CPU-only CI
    torch = None  # type: ignore[assignment]

#: marker value: render with the repo's default five-tool schemas
DEFAULT_TOOLS = "default"


def _resolve_tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools == DEFAULT_TOOLS:
        from agent.tools import default_tool_specs

        return default_tool_specs()
    return tools or None


def strip_training_flags(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop bookkeeping keys (``train``) the chat template must never see."""
    return [{k: v for k, v in dict(m).items() if k != "train"} for m in messages]


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


def _render(template_owner: Any, messages: Sequence[dict[str, Any]], tools: Any) -> list[int]:
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": False}
    if tools:
        kwargs["tools"] = tools
    return as_ids(template_owner.apply_chat_template(list(messages), **kwargs))


def _prefix_lengths(template_owner: Any, messages: Sequence[dict[str, Any]], tools: Any = None) -> list[int]:
    """Token count of each message prefix under the chat template.

    Turn-based templates (Gemma/Qwen) render prefixes concatenatively, so
    prefix[i] is the end offset of message i in the full rendering; a
    non-concatenative template still yields monotone-clamped usable bounds.
    """
    return [
        len(_render(template_owner, messages[:index], tools))
        for index in range(1, len(messages) + 1)
    ]


def _shrink_text(text: str, keep_head: float = 0.65, keep_tail: float = 0.15, factor: float = 0.5) -> str:
    """Cut the middle out of a long observation, keeping both ends."""
    target = max(200, int(len(text) * factor))
    if len(text) <= target:
        return text
    head = int(target * keep_head / (keep_head + keep_tail))
    tail = target - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...[{omitted} chars truncated]...\n{text[len(text) - tail:]}"


def fit_messages(
    template_owner: Any,
    messages: Sequence[dict[str, Any]],
    max_seq_len: int,
    tools: Any = None,
    *,
    shrink_floor: int = 3000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Make a trajectory fit ``max_seq_len`` without losing the task.

    The old behaviour sliced ``input_ids[-max_seq_len:]``, which removed the
    system prompt, tool schemas and the issue text — the model was trained to
    write fixes for a problem it never saw (AUDIT C4). Instead, in order:

    1. shrink very large observations (> ``shrink_floor`` chars) in the
       middle, keeping both ends — they usually carry the file view the next
       edit depends on, so they are trimmed before anything is dropped;
    2. drop the *oldest* complete assistant/tool turns, always keeping the
       head (system + first user message = the task) and the final turn;
    3. shrink any remaining observation down to a small floor;
    4. report ``hard_truncated`` if it still does not fit.
    """
    info = {"dropped_turns": 0, "shrunk_messages": 0, "hard_truncated": 0}
    msgs = [dict(m) for m in messages]

    def fits() -> bool:
        return len(_render(template_owner, msgs, tools)) <= max_seq_len

    def shrink_largest(floor: int) -> bool:
        candidates = [
            (len(m.get("content") or ""), i)
            for i, m in enumerate(msgs)
            if m.get("role") in ("tool", "user") and i != len(msgs) - 1
        ]
        candidates = [c for c in candidates if c[0] > floor]
        if not candidates:
            return False
        _, index = max(candidates)
        msgs[index]["content"] = _shrink_text(msgs[index]["content"])
        info["shrunk_messages"] += 1
        return True

    if fits():
        return msgs, info

    for _ in range(16):  # phase 1
        if fits():
            return msgs, info
        if not shrink_largest(shrink_floor):
            break

    head_end = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), 0)
    starts = [i for i in range(head_end + 1, len(msgs)) if msgs[i].get("role") == "assistant"]
    if starts:  # phase 2
        lengths = _prefix_lengths(template_owner, msgs, tools)
        total = lengths[-1]
        chosen = starts[-1]
        for start in starts:  # ascending: first fit keeps the most history
            if lengths[head_end] + (total - lengths[start - 1]) <= max_seq_len:
                chosen = start
                break
        if chosen > head_end + 1:
            info["dropped_turns"] = sum(1 for i in starts if i < chosen)
            msgs = msgs[: head_end + 1] + msgs[chosen:]

    for _ in range(16):  # phase 3
        if fits():
            return msgs, info
        if not shrink_largest(200):
            break
    info["hard_truncated"] = 1
    return msgs, info


def encode_trajectory_ex(
    template_owner: Any,
    messages: Sequence[dict[str, Any]],
    max_seq_len: int,
    tools: Any = DEFAULT_TOOLS,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """:func:`encode_trajectory` plus a report of what had to be cut."""
    if not messages:
        raise ValueError("trajectory has no messages")
    tools = _resolve_tools(tools)
    trainable = [bool(m.get("train", True)) for m in messages]
    messages = deserialize_tool_args(strip_training_flags(messages))
    # carry the per-message loss flags through any dropping/shrinking
    for flag, message in zip(trainable, messages):
        message["_train"] = flag
    fitted, info = fit_messages(_StripOwner(template_owner), messages, max_seq_len, tools)
    flags = [m.pop("_train", True) for m in fitted]

    input_ids = _render(template_owner, fitted, tools)
    if not input_ids:
        raise ValueError("chat template produced no tokens")

    labels = [-100] * len(input_ids)
    start = 0
    for message, flag, end in zip(fitted, flags, _prefix_lengths(template_owner, fitted, tools)):
        # clamp: non-concatenative templates can misalign, never go backwards
        end = max(start, min(end, len(input_ids)))
        if message.get("role") == "assistant" and flag:
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
        # last resort only (reported): keep the most recent tokens
        info["hard_truncated"] = 1
        input_ids = input_ids[-max_seq_len:]
        labels = labels[-max_seq_len:]

    return {"input_ids": list(input_ids), "labels": list(labels)}, info


class _StripOwner:
    """Hide the private ``_train`` flag from the chat template."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001, ANN201
        clean = [{k: v for k, v in m.items() if k != "_train"} for m in messages]
        return self._owner.apply_chat_template(clean, **kwargs)


def encode_trajectory(
    template_owner: Any,
    messages: Sequence[dict[str, Any]],
    max_seq_len: int,
    tools: Any = DEFAULT_TOOLS,
) -> dict[str, list[int]]:
    """Render one trajectory to ``input_ids``/``labels`` with assistant-only loss.

    ``template_owner`` is a processor (Gemma-4) or tokenizer exposing
    ``apply_chat_template``. Tool schemas are rendered into the prompt
    (``tools="default"``; pass ``None`` to omit). Over-long trajectories are
    fitted by :func:`fit_messages` rather than sliced from the front. A
    message with ``"train": False`` is shown to the model but never trained on.
    """
    encoded, _ = encode_trajectory_ex(template_owner, messages, max_seq_len, tools)
    return encoded


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
