"""Sandbox package — ARCHITECTURE.md §2.3."""
from .executor import DEFAULT_ALLOWLIST, ENV_ALLOWLIST, ExecResult, Sandbox, SandboxError, Status
from .limits import LimitError, Limits

__all__ = [
    "DEFAULT_ALLOWLIST",
    "ENV_ALLOWLIST",
    "ExecResult",
    "LimitError",
    "Limits",
    "Sandbox",
    "SandboxError",
    "Status",
]
