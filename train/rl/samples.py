"""Exact on-policy training samples for GRPO (AUDIT H3).

The original update re-rendered the *parsed* tool call through the chat
template and trained on that text: not the tokens the policy actually
sampled (canonical JSON instead of Gemma's native format, reasoning dropped,
retokenisation drift). Here each assistant step becomes its own sample::

    input_ids = prompt(messages[:i], generation prompt, tools) + completion_ids
    labels    = -100 over the prompt, the sampled ids over the completion

so the log-probs are those of the sampled sequence. The prompt is fitted to
the budget with :func:`train.sft.data_collator.fit_messages` (task kept,
oldest turns dropped) instead of slicing off the front.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from train.sft.data_collator import (
    DEFAULT_TOOLS,
    _resolve_tools,
    as_ids,
    deserialize_tool_args,
    fit_messages,
    strip_training_flags,
)


def _prompt_ids(owner: Any, messages: Sequence[dict[str, Any]], tools: Any) -> list[int]:
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    return as_ids(owner.apply_chat_template(list(messages), **kwargs))


def select_steps(n_steps: int, limit: int, rng: random.Random) -> list[int]:
    """Which assistant steps to train on: always the last, others at random."""
    if limit <= 0 or n_steps <= limit:
        return list(range(n_steps))
    chosen = {n_steps - 1}
    pool = list(range(n_steps - 1))
    rng.shuffle(pool)
    chosen.update(pool[: limit - 1])
    return sorted(chosen)


def build_step_samples(
    owner: Any,
    messages: Sequence[dict[str, Any]],
    completions: Sequence[Sequence[int] | None],
    *,
    max_seq_len: int,
    tools: Any = DEFAULT_TOOLS,
    max_steps: int = 6,
    rng: random.Random | None = None,
) -> list[dict[str, list[int]]]:
    """One ``{input_ids, labels}`` sample per selected assistant step.

    ``completions[j]`` are the sampled token ids of the j-th assistant message.
    Steps without recorded ids (scripted policies) are skipped.
    """
    rng = rng or random.Random(0)
    tools = _resolve_tools(tools)
    msgs = deserialize_tool_args(strip_training_flags(messages))
    assistant_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    usable = [
        (i, list(completions[j]))
        for j, i in enumerate(assistant_idx)
        if j < len(completions) and completions[j]
    ]
    if not usable:
        return []
    samples: list[dict[str, list[int]]] = []
    for pick in select_steps(len(usable), max_steps, rng):
        index, completion = usable[pick]
        completion = completion[: max(1, max_seq_len // 2)]
        budget = max_seq_len - len(completion) - 16  # generation-prompt slack
        prompt_msgs, _ = fit_messages(owner, msgs[:index], max(budget, 64), tools)
        prompt = _prompt_ids(owner, prompt_msgs, tools)
        if len(prompt) + len(completion) > max_seq_len:
            prompt = prompt[: max(0, max_seq_len - len(completion))]  # last resort
        samples.append(
            {
                "input_ids": prompt + completion,
                "labels": [-100] * len(prompt) + completion,
            }
        )
    return samples
