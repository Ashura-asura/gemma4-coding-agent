"""Adapter checkpoint layout helpers (AUDIT M3).

PEFT writes an adapter that is *not* named ``default`` into a sub-folder named
after it (``<dir>/policy/adapter_config.json``). The trainer's resume check and
the eval loader both looked for ``<dir>/adapter_config.json``, so a resumed run
silently started from scratch and ``--model checkpoints/rl`` could not load.
We always flatten the policy adapter into the checkpoint root.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

POLICY_ADAPTER = "policy"
_MARKER = "adapter_config.json"


def save_policy_adapter(model: Any, target: str | Path, adapter: str = POLICY_ADAPTER) -> Path:
    """Save only ``adapter`` and leave ``adapter_config.json`` at ``target`` root."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target), selected_adapters=[adapter])
    nested = target / adapter
    if (nested / _MARKER).exists() and not (target / _MARKER).exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(target / item.name))
        nested.rmdir()
    if not (target / _MARKER).exists():
        raise RuntimeError(f"adapter was not written to {target}")
    return target


def resolve_adapter_dir(path: str | Path, adapter: str = POLICY_ADAPTER) -> Path | None:
    """Directory that actually holds the adapter (root or ``<adapter>/``), if any."""
    path = Path(path)
    for candidate in (path, path / adapter):
        if (candidate / _MARKER).exists():
            return candidate
    return None
