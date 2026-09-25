"""Network / mount namespace isolation — ARCHITECTURE.md §2.3.

Everything here runs inside the sandbox *launcher* process, before the
payload is exec'd. That ordering is the whole point: if isolation cannot
be established, the payload is never started (fail closed), and the reason
is written to a status file the parent reads.

Degraded mode (no network namespace, e.g. win32) is only ever entered
when the caller explicitly allows it, and it is reported in the status
file so the result can never look like a fully isolated run.
"""
from __future__ import annotations

import ctypes
import errno
import json
import os
import sys
from pathlib import Path

from .limits import LimitError, Limits, apply as apply_limits

IS_POSIX = os.name == "posix"

# linux clone flags / mount flags
CLONE_NEWNS = 0x00020000
CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000
MS_REC = 16384
MS_PRIVATE = 1 << 18


class IsolationError(RuntimeError):
    """Isolation could not be established; the payload must not run."""


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def _unshare(flags: int) -> None:
    libc = _libc()
    if libc.unshare(flags) != 0:
        err = ctypes.get_errno()
        raise IsolationError(f"unshare({flags:#x}) failed: {err} {errno.errorcode.get(err, '')}")


def _supports_netns() -> bool:
    return IS_POSIX


def _enter_user_namespace() -> None:
    """Map our own uid/gid into a fresh user namespace so CLONE_NEWNET is permitted.

    The host uid/gid must be captured *before* unshare(): once the new user
    namespace exists our own ids are unmapped, getuid() reports the overflow
    id (65534), and the kernel rejects the mapping with EPERM.
    """
    uid, gid = os.getuid(), os.getgid()
    _unshare(CLONE_NEWUSER)
    try:
        with open("/proc/self/setgroups", "w") as fh:
            fh.write("deny")
    except FileNotFoundError:
        pass  # kernels without setgroups support
    try:
        with open("/proc/self/uid_map", "w") as fh:
            fh.write(f"0 {uid} 1")
        with open("/proc/self/gid_map", "w") as fh:
            fh.write(f"0 {gid} 1")
    except OSError as exc:
        raise IsolationError(
            f"could not write user-namespace id maps (host uid {uid}): {exc}"
        ) from exc


def _isolate_network() -> str:
    """Fresh network namespace with no interfaces up => no egress."""
    if not _supports_netns():
        raise IsolationError("network namespaces are not available on this platform")
    try:
        _unshare(CLONE_NEWNET)
        return "ok"
    except IsolationError:
        pass
    _enter_user_namespace()
    _unshare(CLONE_NEWNET)
    return "ok"


def _isolate_mounts() -> str:
    """Private mount propagation; best effort (netns is the hard requirement)."""
    if not IS_POSIX:
        raise IsolationError("mount namespaces are not available on this platform")
    try:
        _unshare(CLONE_NEWNS)
        libc = _libc()
        rc = libc.mount(None, b"/", None, MS_REC | MS_PRIVATE, None)
        if rc != 0:
            err = ctypes.get_errno()
            return f"degraded: could not mark mounts private ({errno.errorcode.get(err, err)})"
        return "ok"
    except IsolationError as exc:
        return f"degraded: {exc}"


def prepare(status_path: Path, limits: Limits, *, allow_degraded: bool) -> dict:
    """Isolate the current process, then write a status file.

    Returns the status dict on success. On any hard failure the status file
    is still written (ok=False) and IsolationError is raised so the launcher
    exits non-zero *before* exec'ing the payload.
    """
    status: dict = {
        "ok": False,
        "platform": sys.platform,
        "netns": "not_attempted",
        "mountns": "not_attempted",
        "rlimits": {},
        "unsupported_rlimits": list(limits.unsupported()),
        "warnings": [],
    }

    def _flush() -> None:
        Path(status_path).write_text(json.dumps(status), encoding="utf-8")

    try:
        if IS_POSIX:
            try:
                os.setsid()
            except OSError as exc:
                status["warnings"].append(f"setsid failed: {exc}")

            try:
                status["netns"] = _isolate_network()
            except IsolationError as exc:
                status["netns"] = "failed"
                if not allow_degraded:
                    raise IsolationError(f"network namespace unavailable: {exc}") from exc
                status["netns"] = "degraded"
                status["warnings"].append(f"network isolation degraded: {exc}")

            status["mountns"] = _isolate_mounts()
            if status["mountns"].startswith("degraded"):
                status["warnings"].append(status["mountns"])

            status["rlimits"] = apply_limits(limits)
        else:
            status["netns"] = "unavailable"
            status["mountns"] = "unavailable"
            if not allow_degraded:
                raise IsolationError(
                    "network namespace is unavailable on this platform and degraded "
                    "mode is not allowed"
                )
            status["warnings"].append(
                "no network namespace on this platform: isolation is degraded, "
                "reward/eval numbers from this host are not trustworthy"
            )

        status["ok"] = True
        _flush()
        return status
    except IsolationError as exc:
        status["ok"] = False
        status["error"] = str(exc)
        _flush()
        raise
    except LimitError as exc:
        status["ok"] = False
        status["error"] = f"resource limits not applied: {exc}"
        _flush()
        raise IsolationError(status["error"]) from exc
