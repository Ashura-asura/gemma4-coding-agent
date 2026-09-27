"""Reward composition tests — ARCHITECTURE §2.5 / §3.3."""
from __future__ import annotations

import yaml
from pathlib import Path

from train.rl import reward as reward_mod
from train.rl.reward import episode_reward, group_normalize, reward_weights

TASK = {"instance_id": "pallets__flask-1"}


def _score(patch_valid: bool, resolved: bool):
    return {
        "patch_valid": patch_valid,
        "resolved": resolved,
        "test_status": "pass" if resolved else "fail",
        "reason": "tests passed" if resolved else "boom",
    }


def test_weights_pass_and_valid(monkeypatch):
    monkeypatch.setattr(reward_mod, "score_submission", lambda *a, **k: _score(True, True))
    out = episode_reward(TASK, patch="--- a\n+++ b\n", submitted=True, steps=4)
    w = reward_mod.DEFAULT_WEIGHTS
    assert out["components"]["test_pass"] == w["test_pass"]
    assert out["components"]["patch_validity"] == w["patch_validity"]
    assert out["components"]["step_penalty"] == -w["step_penalty"] * 4
    assert out["components"]["no_submit_penalty"] == 0.0
    assert out["reward"] == pytest_approx(w["test_pass"] + w["patch_validity"] - w["step_penalty"] * 4)
    assert out["resolved"] and out["patch_valid"]


def test_weights_fail_no_submit(monkeypatch):
    monkeypatch.setattr(reward_mod, "score_submission", lambda *a, **k: _score(False, False))
    out = episode_reward(TASK, patch=None, submitted=False, steps=30)
    w = reward_mod.DEFAULT_WEIGHTS
    assert out["components"]["test_pass"] == 0.0
    assert out["components"]["patch_validity"] == 0.0
    assert out["components"]["no_submit_penalty"] == -w["no_submit_penalty"]
    assert out["reward"] == pytest_approx(-w["step_penalty"] * 30 - w["no_submit_penalty"])
    assert not out["resolved"] and not out["patch_valid"]


def test_invalid_patch_no_validity_bonus(monkeypatch):
    monkeypatch.setattr(reward_mod, "score_submission", lambda *a, **k: _score(False, False))
    out = episode_reward(TASK, patch="garbage", submitted=True, steps=1)
    assert out["components"]["patch_validity"] == 0.0
    assert out["components"]["no_submit_penalty"] == 0.0


def test_config_weights_override_defaults():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "configs" / "rl_config.yaml").read_text(encoding="utf-8")
    )
    w = reward_weights(config)
    assert w["test_pass"] == float(config["reward"]["test_pass"])
    assert set(w) == set(reward_mod.DEFAULT_WEIGHTS)


def test_group_normalize():
    out = group_normalize([1.0, 0.0, -1.0])
    mean = sum(out) / len(out)
    var = sum((x - mean) ** 2 for x in out) / len(out)
    assert abs(mean) < 1e-6
    assert abs(var ** 0.5 - 1.0) < 1e-6
    assert group_normalize([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]
    assert group_normalize([]) == []


def pytest_approx(value: float, tol: float = 1e-9):
    class _Approx:
        def __eq__(self, other):
            return abs(float(other) - value) < tol

        def __repr__(self):
            return f"approx({value})"

    return _Approx()
