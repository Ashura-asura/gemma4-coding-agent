"""Sandbox launcher: isolate first, then exec the payload.

Spawned by :mod:`sandbox.executor` as::

    python sandbox/launcher.py --status <file> [--allow-degraded] \
        [--limits '<json>'] -- <payload argv...>

The payload is only exec'd after isolation succeeds. There is no shell
anywhere in this path — argv is handed to execve verbatim.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

# allow `python .../sandbox/launcher.py` regardless of the payload's cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sandbox import fsisolation as fs  # noqa: E402
from sandbox import namespace as ns  # noqa: E402
from sandbox.limits import Limits  # noqa: E402

FAIL_ISOLATION = 3
FAIL_PAYLOAD = 127


def _split(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        raise SystemExit("launcher: expected '--' before the payload argv")
    idx = argv.index("--")
    return argv[:idx], argv[idx + 1:]


def main(argv: list[str] | None = None) -> int:
    head, payload = _split(list(sys.argv[1:] if argv is None else argv))
    if not payload:
        print("launcher: empty payload", file=sys.stderr)
        return FAIL_PAYLOAD

    status_path = None
    allow_degraded = False
    fs_mode = "off"
    root: Path | None = None
    limits = Limits()
    it = iter(head)
    for tok in it:
        if tok == "--status":
            status_path = Path(next(it))
        elif tok == "--limits":
            limits = Limits.from_dict(json.loads(next(it)))
        elif tok == "--allow-degraded":
            allow_degraded = True
        elif tok == "--fs-mode":
            fs_mode = next(it)
            if fs_mode not in fs.MODES:
                raise SystemExit(f"launcher: bad --fs-mode {fs_mode!r}")
        elif tok == "--root":
            root = Path(next(it))
        else:
            raise SystemExit(f"launcher: unexpected argument {tok!r}")
    if status_path is None:
        raise SystemExit("launcher: --status is required")

    if fs_mode != "off" and root is None:
        raise SystemExit("launcher: --root is required with --fs-mode")

    try:
        status = ns.prepare(status_path, limits, allow_degraded=allow_degraded, fs_mode=fs_mode)
    except ns.IsolationError as exc:
        print(f"launcher: isolation failed: {exc}", file=sys.stderr)
        return FAIL_ISOLATION

    if os.name == "posix" and fs_mode != "off":
        return _run_hardened(payload, status, status_path, fs_mode, root)

    if os.name == "posix":
        try:
            os.execvpe(payload[0], payload, os.environ)
        except OSError as exc:
            print(f"launcher: cannot exec {payload[0]!r}: {exc}", file=sys.stderr)
            return FAIL_PAYLOAD
        return 0  # unreachable

    # win32: os.exec* semantics are spawn-and-wait; keep the same contract
    try:
        return subprocess.run(payload, env=dict(os.environ)).returncode
    except OSError as exc:
        print(f"launcher: cannot run {payload[0]!r}: {exc}", file=sys.stderr)
        return FAIL_PAYLOAD


def _write_status(path: Path, status: dict) -> None:
    path.write_text(json.dumps(status), encoding="utf-8")


def _read_all(fd: int) -> bytes:
    chunks = []
    while True:
        part = os.read(fd, 65536)
        if not part:
            break
        chunks.append(part)
    os.close(fd)
    return b"".join(chunks)


def _payload_process(payload, root, fs_mode, pid_ns, report_w) -> None:
    """Runs as the payload's own process: private mount namespace, harden, exec."""
    report: dict = {}
    try:
        if fs._libc().unshare(ns.CLONE_NEWNS) != 0:
            raise fs.FsIsolationError("could not create a private mount namespace")
        fs._libc().mount(None, b"/", None, ns.MS_REC | ns.MS_PRIVATE, None)
        results, warnings = fs.harden_child(root, mode=fs_mode, pid_namespace=pid_ns)
        report = {"fs": results, "warnings": warnings}
    except fs.FsIsolationError as exc:
        if fs_mode == "required":
            report = {"error": f"filesystem isolation failed: {exc}"}
        else:
            # hosts without mount namespaces (Kaggle: unshare is seccomp-blocked)
            # run the payload unhardened, marked degraded in the status file
            report = {"fs": {}, "warnings": [f"filesystem isolation degraded: {exc}"]}
    os.write(report_w, json.dumps(report).encode("utf-8"))
    if "error" in report:
        print(f"launcher: {report['error']}", file=sys.stderr)
        os._exit(FAIL_ISOLATION)
    try:
        os.execvpe(payload[0], payload, os.environ)
    except OSError as exc:
        print(f"launcher: cannot exec {payload[0]!r}: {exc}", file=sys.stderr)
        os._exit(FAIL_PAYLOAD)


def _supervisor(payload, root, fs_mode, pid_ns, report_r, report_w, result_r, result_w) -> None:
    """PID 1 of the sandbox namespace (a tiny init).

    The payload runs as PID 2 so signals it sends itself behave normally;
    this process reaps orphans, relays how the payload ended, and on exit
    takes every remaining process of the namespace down with it.
    """
    os.close(report_r)
    os.close(result_r)
    kid = os.fork()
    if kid == 0:
        os.close(result_w)
        _payload_process(payload, root, fs_mode, pid_ns, report_w)
        os._exit(FAIL_PAYLOAD)
    os.close(report_w)
    outcome: dict = {"code": FAIL_PAYLOAD}
    while True:
        try:
            pid, st = os.waitpid(-1, 0)
        except ChildProcessError:
            break
        if pid == kid:
            if os.WIFSIGNALED(st):
                outcome = {"signal": os.WTERMSIG(st)}
            else:
                outcome = {"code": os.WEXITSTATUS(st)}
            break
    os.write(result_w, json.dumps(outcome).encode("utf-8"))
    os._exit(0)


def _run_hardened(payload: list[str], status: dict, status_path: Path, fs_mode: str, root: Path) -> int:
    """launcher -> supervisor (PID 1 of the namespace) -> payload (PID 2).

    The payload gets a mount namespace of its own, so making the tree
    read-only there never affects the launcher, which can therefore still
    write the final status file. Results travel over close-on-exec pipes.
    """
    pid_ns = status.get("pidns") == "ok"
    report_r, report_w = os.pipe()
    result_r, result_w = os.pipe()
    init = os.fork()
    if init == 0:
        _supervisor(payload, root, fs_mode, pid_ns, report_r, report_w, result_r, result_w)
        os._exit(FAIL_PAYLOAD)
    os.close(report_w)
    os.close(result_w)

    raw = _read_all(report_r)  # EOF = payload exec'd (or died during setup)
    try:
        report = json.loads(raw.decode("utf-8") or "{}")
    except ValueError:
        report = {"error": "filesystem isolation reported garbage"}
    if not report:
        report = {"error": "filesystem isolation did not report"}
    if "error" in report:
        status["ok"] = False
        status["error"] = report["error"]
    else:
        status["fs"] = report.get("fs", {})
        status["warnings"] = list(status.get("warnings", [])) + list(report.get("warnings", []))
    _write_status(status_path, status)

    raw_outcome = _read_all(result_r)  # EOF = supervisor (and the namespace) gone
    os.waitpid(init, 0)
    try:
        outcome = json.loads(raw_outcome.decode("utf-8") or "{}")
    except ValueError:
        outcome = {}
    if "signal" in outcome:
        sig = int(outcome["signal"])
        try:
            signal.signal(sig, signal.SIG_DFL)  # SIGKILL/SIGSTOP cannot be reset
        except (OSError, ValueError):
            pass
        os.kill(os.getpid(), sig)
        return 128 + sig
    return int(outcome.get("code", FAIL_PAYLOAD))


if __name__ == "__main__":
    raise SystemExit(main())
