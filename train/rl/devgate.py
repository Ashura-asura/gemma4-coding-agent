"""Held-out dev split and mid-RL gate (AUDIT M4, ARCHITECTURE §8.2).

The shipped config had ``dev_slice: null`` (gate never ran) and, when set, it
sampled 4 tasks *from the training pool* at k=1 — a measurement of training
reward on 4 draws. Now a deterministic slice of the pool is removed from
training and the gate stops the run when it stops improving.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Sequence


def split_dev(
    tasks: Sequence[dict[str, Any]], dev_size: int, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(train_tasks, dev_tasks)``; stable across runs and resumes."""
    if dev_size <= 0:
        return list(tasks), []
    ranked = sorted(
        tasks,
        key=lambda t: hashlib.sha1(f"{seed}:{t['task_id']}".encode()).hexdigest(),
    )
    dev = ranked[: min(dev_size, max(0, len(ranked) - 1))]
    dev_ids = {t["task_id"] for t in dev}
    return [t for t in tasks if t["task_id"] not in dev_ids], dev


@dataclass
class DevGate:
    """Tracks dev scores; ``should_stop`` once ``patience`` evals brought no gain."""

    patience: int = 3
    min_delta: float = 0.0
    history: list[float] = field(default_factory=list)
    best: float = float("-inf")
    best_index: int = -1

    def update(self, score: float) -> bool:
        """Record a score; True when it is a new best (save a ``best`` adapter)."""
        self.history.append(score)
        if score > self.best + self.min_delta:
            self.best, self.best_index = score, len(self.history) - 1
            return True
        return False

    def should_stop(self) -> bool:
        return self.patience > 0 and len(self.history) - 1 - self.best_index >= self.patience


def hours_left(started_monotonic: float, budget_hours: float | None, now: float | None = None) -> float | None:
    """Remaining wall-clock budget in hours, or None when unlimited."""
    if not budget_hours:
        return None
    now = time.monotonic() if now is None else now
    return budget_hours - (now - started_monotonic) / 3600.0
