"""Real Gemma-4 processor tests: template render, tool-arg deserialization,
turn-tag balance across a multi-turn tool chain (reasoning-loop regression).

Skipped offline / when the gated tokenizer can't be fetched.
"""
from __future__ import annotations

import json

import pytest

from train.sft.data_collator import deserialize_tool_args, encode_trajectory

MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Fix the failing test in tests/test_app.py"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "edit_file", "arguments": json.dumps({"path": "a.py", "old": "x", "new": "y"})}},
        ],
    },
    {"role": "tool", "name": "edit_file", "tool_call_id": "c1", "content": "ok", "is_error": False},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "submit", "arguments": json.dumps({"patch": "--- a\\n+++ b\\n"})}},
        ],
    },
]


@pytest.fixture(scope="module")
def processor():
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained("google/gemma-4-12B-it")
    except Exception as exc:  # gated/offline
        pytest.skip(f"Gemma-4 processor unavailable: {exc}")


def test_deserialize_tool_args():
    converted = deserialize_tool_args(MESSAGES)
    assert isinstance(converted[2]["tool_calls"][0]["function"]["arguments"], dict)
    assert isinstance(MESSAGES[2]["tool_calls"][0]["function"]["arguments"], str)  # input untouched


def test_render_and_balance(processor):
    messages = deserialize_tool_args(MESSAGES)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert "edit_file" in rendered and "submit" in rendered
    # Gemma-4 canonical turn tags: at most ONE unclosed terminal model turn
    # (the template leaves the "awaiting tool response" state open by design);
    # anything more means the multi-turn re-injection loop ("zombie loop")
    opens, closes = rendered.count("<|turn>"), rendered.count("<turn|>")
    assert 0 <= opens - closes <= 1, f"unbalanced turns: {opens} open vs {closes} close"
    # no verbatim repetition of a unique marker (multi-turn re-injection loop)
    assert rendered.count("Fix the failing test") == 1


def test_encode_assistant_only(processor):
    enc = encode_trajectory(processor, MESSAGES, max_seq_len=8192)
    ids, labels = enc["input_ids"], enc["labels"]
    assert len(ids) == len(labels)
    trained = [i for i, label in enumerate(labels) if label != -100]
    assert trained, "assistant spans must carry loss"
    # system/user/tool spans never trained: their tokens equal labels only in assistant turns,
    # so verify the first trained token is not at position 0 (system header)
    assert trained[0] > 0
