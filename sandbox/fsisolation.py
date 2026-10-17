"""Process / filesystem / privilege hardening — AUDIT.md finding C2.

The original sandbox gave a payload a private network and rlimits, but it
could still (a) read the trainer's secrets from ``/proc/<ppid>/environ``,
(b) write anywhere the uid could, including shared venvs and checkouts used
by later episodes, and (c) see and signal every host process. This module
closes those holes. Everything runs in the launcher, *before* the payload is
exec'd; with ``mode="required"`` any failure keeps the payload from starting.

Layers (each reported individually in the isolation status):

``pidns``
    New PID namespace + fresh ``/proc``: the host is invisible, the trainer's
    environment unreadable, and killing the namespace's PID 1 reaps every
    descendant (no daemonised leftovers after a timeout).
``readonly_root``
    The whole mount tree becomes read-only except the sandbox root, which is
    re-bound writable. Venvs, shared checkouts, the status file and the rest
    of the host can no longer be modified by test code.
``caps``
    ``no_new_privs`` and an empty capability set, so the payload cannot
    remount the tree writable or otherwise undo the above.
``loopback``
    ``lo`` is brought up inside the private netns so tests that talk to
    127.0.0.1 still work (no egress either way).

Not covered (documented residual risk): reads of host files are still
possible (keep secrets out of files), and there is no seccomp filter.
"""
from __future__ import annotations

import ctypes
import errno
import os
import socket
import struct
from pathlib import Path

try:  # Windows dev hosts run tests only; the sandbox itself targets Linux
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

CLONE_NEWPID = 0x20000000

MS_RDONLY = 1
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_NOSUID, MS_NODEV, MS_NOEXEC = 2, 4, 8

AT_FDCWD = -100
AT_RECURSIVE = 0x8000
MOUNT_ATTR_RDONLY = 0x1
SYS_MOUNT_SETATTR = 442  # same number on x86_64 and aarch64

PR_CAPBSET_DROP = 24
PR_SET_NO_NEW_PRIVS = 38
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4

SIOCGIFFLAGS, SIOCSIFFLAGS = 0x8913, 0x8914
IFF_UP = 0x1

MODES = ("off", "best_effort", "required")


class FsIsolationError(RuntimeError):
    """A hardening step failed."""


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def _err() -> str:
    e = ctypes.get_errno()
    return f"{errno.errorcode.get(e, e)}"


# ---------------------------------------------------------------- loopback
def bring_up_loopback() -> str:
    """Raise ``lo`` inside the current (private) network namespace."""
    if fcntl is None:
        return "degraded: no fcntl on this host"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError as exc:
        return f"degraded: {exc}"
    try:
        req = struct.pack("16sH22s", b"lo", 0, b"\0" * 22)
        flags = struct.unpack("16sH22s", fcntl.ioctl(sock, SIOCGIFFLAGS, req))[1]
        fcntl.ioctl(sock, SIOCSIFFLAGS, struct.pack("16sH22s", b"lo", flags | IFF_UP, b"\0" * 22))
        return "ok"
    except OSError as exc:
        return f"degraded: {exc}"
    finally:
        sock.close()


# --------------------------------------------------------------------- pid
def unshare_pid() -> None:
    """Children forked after this call live in a fresh PID namespace."""
    if _libc().unshare(CLONE_NEWPID) != 0:
        raise FsIsolationError(f"unshare(CLONE_NEWPID) failed: {_err()}")


def mount_fresh_proc() -> None:
    rc = _libc().mount(b"proc", b"/proc", b"proc", MS_NOSUID | MS_NODEV | MS_NOEXEC, None)
    if rc != 0:
        raise FsIsolationError(f"could not mount a fresh /proc: {_err()}")


# --------------------------------------------------------- read-only rootfs
class _MountAttr(ctypes.Structure):
    _fields_ = [
        ("attr_set", ctypes.c_uint64),
        ("attr_clr", ctypes.c_uint64),
        ("propagation", ctypes.c_uint64),
        ("userns_fd", ctypes.c_uint64),
    ]


def _readonly_recursive_setattr() -> bool:
    libc = _libc()
    libc.syscall.restype = ctypes.c_long
    attr = _MountAttr(MOUNT_ATTR_RDONLY, 0, 0, 0)
    rc = libc.syscall(
        ctypes.c_long(SYS_MOUNT_SETATTR),
        ctypes.c_int(AT_FDCWD),
        ctypes.c_char_p(b"/"),
        ctypes.c_uint(AT_RECURSIVE),
        ctypes.byref(attr),
        ctypes.c_size_t(ctypes.sizeof(attr)),
    )
    if rc == 0:
        return True
    if ctypes.get_errno() == errno.ENOSYS:
        return False
    raise FsIsolationError(f"mount_setattr(RDONLY) failed: {_err()}")


def _readonly_each_mount(skip_prefix: str) -> list[str]:
    """Fallback for kernels < 5.12: remount every mount read-only, keeping flags."""
    libc = _libc()
    failures: list[str] = []
    with open("/proc/self/mountinfo", encoding="utf-8") as fh:
        points = [line.split()[4] for line in fh]
    for mp in sorted(set(points), key=len, reverse=True):
        mp = mp.replace("\\040", " ")
        if mp.startswith(skip_prefix) or mp.startswith("/proc"):
            continue
        try:
            keep = os.statvfs(mp).f_flag & (MS_NOSUID | MS_NODEV | MS_NOEXEC)
        except OSError:
            continue
        if libc.mount(None, mp.encode(), None, MS_REMOUNT | MS_BIND | MS_RDONLY | keep, None) != 0:
            failures.append(f"{mp}: {_err()}")
    return failures


def make_root_readonly(writable: Path) -> str:
    """Everything read-only except ``writable`` (the sandbox root).

    The process cwd keeps pointing at the *original* mount, which becomes
    read-only; callers re-enter it by path afterwards (see ``harden_child``).
    """
    libc = _libc()
    target = str(writable).encode()
    if libc.mount(target, target, None, MS_BIND | MS_REC, None) != 0:
        raise FsIsolationError(f"could not bind-mount the sandbox root: {_err()}")
    note = "ok"
    if not _readonly_recursive_setattr():
        failures = _readonly_each_mount(str(writable))
        if failures:
            raise FsIsolationError("could not remount read-only: " + "; ".join(failures[:3]))
        note = "ok (legacy remount)"
    keep = os.statvfs(str(writable)).f_flag & (MS_NOSUID | MS_NODEV | MS_NOEXEC)
    if libc.mount(None, target, None, MS_REMOUNT | MS_BIND | keep, None) != 0:
        raise FsIsolationError(f"could not re-open the sandbox root for writing: {_err()}")
    return note


# --------------------------------------------------------------- privileges
class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def drop_privileges() -> str:
    """no_new_privs + empty bounding/ambient/effective/inheritable sets."""
    libc = _libc()
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise FsIsolationError(f"PR_SET_NO_NEW_PRIVS failed: {_err()}")
    try:
        last = int(Path("/proc/sys/kernel/cap_last_cap").read_text())
    except (OSError, ValueError):
        last = 40
    for cap in range(last + 1):
        if libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0) != 0 and ctypes.get_errno() != errno.EINVAL:
            raise FsIsolationError(f"could not drop capability {cap}: {_err()}")
    libc.prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0)
    header = _CapHeader(0x20080522, 0)
    data = (_CapData * 2)()
    libc.syscall.restype = ctypes.c_long
    rc = libc.syscall(ctypes.c_long(126), ctypes.byref(header), ctypes.byref(data))  # capset
    if rc != 0:
        raise FsIsolationError(f"capset failed: {_err()}")
    return "ok"


# ------------------------------------------------------------ orchestration
def harden_child(root: Path, *, mode: str, pid_namespace: bool) -> tuple[dict[str, str], list[str]]:
    """Run inside the (already forked) PID-namespace child, right before exec.

    Returns ``(per-layer results, warnings)``. In ``required`` mode any
    failing layer raises :class:`FsIsolationError`.
    """
    results: dict[str, str] = {}
    warnings: list[str] = []

    def layer(name: str, fn) -> None:
        try:
            results[name] = fn() or "ok"
        except FsIsolationError as exc:
            if mode == "required":
                raise FsIsolationError(f"{name}: {exc}") from exc
            results[name] = f"degraded: {exc}"
            warnings.append(f"fs isolation degraded ({name}): {exc}")

    cwd = os.getcwd()
    if pid_namespace:
        layer("proc", lambda: mount_fresh_proc())
    layer("readonly_root", lambda: make_root_readonly(root))
    try:  # re-resolve the cwd through the writable bind mount
        os.chdir(cwd)
    except OSError as exc:
        raise FsIsolationError(f"could not re-enter working directory {cwd}: {exc}") from exc
    layer("caps", drop_privileges)
    return results, warnings
