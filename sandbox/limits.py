"""Resource limits for sandboxed processes — ARCHITECTURE.md §2.3.

`apply()` is executed inside the sandbox launcher process, before the
payload is exec'd. Any failure to apply a limit is fatal (fail closed):
a silently un-limited child would invalidate every reward number.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

IS_POSIX = os.name == "posix"

if IS_POSIX:
    import resource as _resource
else:  # pragma: no cover - exercised on win32
    _resource = None  # type: ignore[assignment]


class LimitError(RuntimeError):
    """A resource limit could not be applied."""


#: field name -> (resource module constant name, description)
_POSIX_LIMITS: tuple[tuple[str, str], ...] = (
    ("cpu_seconds", "RLIMIT_CPU"),
    ("address_space_bytes", "RLIMIT_AS"),
    ("process_count", "RLIMIT_NPROC"),
    ("file_size_bytes", "RLIMIT_FSIZE"),
    ("open_files", "RLIMIT_NOFILE"),
    ("core_size", "RLIMIT_CORE"),
)


@dataclass(frozen=True)
class Limits:
    """RLIMIT_* ceilings plus the parent-enforced wall-clock budget.

    wall_seconds is not a setrlimit — the executor enforces it with a
    kill on the process group.
    """

    cpu_seconds: int = 30
    address_space_bytes: int = 2 * 1024**3
    process_count: int = 128
    file_size_bytes: int = 1024**3
    open_files: int = 512
    core_size: int = 0
    wall_seconds: int = 60

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    def supported(self) -> tuple[str, ...]:
        if not IS_POSIX:
            return ("wall_seconds",)
        return ("wall_seconds", *(name for name, _ in _POSIX_LIMITS))

    def unsupported(self) -> tuple[str, ...]:
        if not IS_POSIX:
            return tuple(name for name, _ in _POSIX_LIMITS)
        return ()

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "Limits":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: int(v) for k, v in data.items() if k in known})


def _clamp(value: int, hard: int) -> int:
    """Never ask to raise a hard limit — that needs privileges and would fail."""
    if hard == _resource.RLIM_INFINITY:
        return value
    return min(value, int(hard))


def apply(limits: Limits) -> dict[str, int]:
    """Apply every rlimit in the current process. Raises LimitError on any failure."""
    if not IS_POSIX:
        raise LimitError("resource limits (setrlimit) are not available on this platform")

    applied: dict[str, int] = {}
    for field_name, const_name in _POSIX_LIMITS:
        res = getattr(_resource, const_name, None)
        if res is None:
            raise LimitError(f"{const_name} is not available on this platform")
        try:
            value = int(getattr(limits, field_name))
            soft, hard = _resource.getrlimit(res)
            # only ever lower a limit: raising a hard limit needs privileges
            target = _clamp(_clamp(value, hard), soft)
            _resource.setrlimit(res, (target, target))
        except (ValueError, OSError) as exc:
            raise LimitError(f"could not set {const_name}: {exc}") from exc
        applied[field_name] = target
    return applied
