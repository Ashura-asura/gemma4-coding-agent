"""Head-preserving fitting of long trajectories (AUDIT C4). No torch needed."""
from __future__ import annotations

from train.sft.data_collator import encode_trajectory_ex, fit_messages


class Tok:
    """Concatenative template; one id per character; renders tool schemas."""

    def __init__(self):
        self.seen_tools: list = []

    def _text(self, messages, tools):
        head = ("<tools>" + ",".join(t["function"]["name"] for t in tools) + "</tools>") if tools else ""
        body = "".join(
            f"<{m['role']}>{m.get('content') or ''}"
            f"{[c['function']['name'] for c in m.get('tool_calls') or []] or ''}</{m['role']}>"
            for m in messages
        )
        return head + body

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, tools=None):
        assert all("train" not in m and "_train" not in m for m in messages), "flags leaked into the template"
        self.seen_tools.append(tools)
        text = self._text(messages, tools)
        ids = [ord(c) % 30000 + 2 for c in text]
        return ids if tokenize else text

    def decode(self, ids):
        return "".join(chr((i - 2)) for i in ids)


def trajectory(turns: int, obs: int = 80):
    msgs = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "ISSUE: the parser crashes on empty input"},
    ]
    for i in range(turns):
        msgs.append({"role": "assistant", "content": f"step{i}", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "content": f"OBS{i}" + "o" * obs, "tool_call_id": f"c{i}", "name": "read_file"})
    msgs.append({"role": "assistant", "content": "FINAL", "tool_calls": [
        {"id": "z", "type": "function", "function": {"name": "submit", "arguments": "{}"}}]})
    return msgs


def test_short_trajectory_is_untouched():
    msgs = trajectory(2)
    fitted, info = fit_messages(Tok(), msgs, 10_000)
    assert fitted == msgs and info == {"dropped_turns": 0, "shrunk_messages": 0, "hard_truncated": 0}


def test_task_and_final_action_survive_and_oldest_turns_go():
    tok = Tok()
    msgs = trajectory(8)
    enc, info = encode_trajectory_ex(tok, msgs, max_seq_len=420, tools=None)
    text = tok.decode(enc["input_ids"])
    assert "SYSTEM" in text and "ISSUE: the parser crashes" in text      # head kept
    assert "FINAL" in text                                              # final submit kept
    assert "step0" not in text and "step7" in text                      # oldest dropped, newest kept
    assert info["dropped_turns"] > 0 and info["hard_truncated"] == 0
    assert len(enc["input_ids"]) <= 420


def test_old_behaviour_would_have_lost_the_issue():
    tok = Tok()
    msgs = trajectory(8)
    full = tok.apply_chat_template(msgs, tokenize=True)
    tail = tok.decode(full[-420:])
    assert "ISSUE" not in tail  # documents the bug the fit logic fixes


def test_one_huge_observation_is_shrunk_in_the_middle_not_cut_from_the_front():
    tok = Tok()
    msgs = trajectory(1, obs=5000)
    enc, info = encode_trajectory_ex(tok, msgs, max_seq_len=3000, tools=None)
    text = tok.decode(enc["input_ids"])
    assert "ISSUE" in text and "FINAL" in text and "chars truncated" in text
    assert "step0" in text  # the turn itself is kept; only its bulk was trimmed
    assert info["shrunk_messages"] >= 1 and info["dropped_turns"] == 0 and info["hard_truncated"] == 0


def test_tools_are_rendered_into_the_prompt_by_default():
    tok = Tok()
    enc, _ = encode_trajectory_ex(tok, trajectory(1), max_seq_len=10_000)
    assert "<tools>edit_file,read_file,run_tests,search_repo,submit</tools>" in tok.decode(enc["input_ids"])
    tok2 = Tok()
    encode_trajectory_ex(tok2, trajectory(1), max_seq_len=10_000, tools=None)
    assert all(t is None for t in tok2.seen_tools)


def test_train_false_turns_are_visible_but_unsupervised():
    tok = Tok()
    msgs = trajectory(2)
    msgs[2]["train"] = False  # first assistant turn: e.g. a deliberately wrong call
    enc, _ = encode_trajectory_ex(tok, msgs, max_seq_len=10_000, tools=None)
    ids, labels = enc["input_ids"], enc["labels"]
    text_labeled = tok.decode([i for i, row in zip(ids, labels) if row != -100])
    assert "step0" not in text_labeled and "step1" in text_labeled and "FINAL" in text_labeled
    assert "step0" in tok.decode(ids)
