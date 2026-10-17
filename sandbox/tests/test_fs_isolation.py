"""PID / filesystem / privilege isolation (AUDIT.md C2).

Skipped automatically where the host cannot provide the namespaces (the
sandbox then reports ``degraded`` and ``SANDBOX_FS_ISOLATION=required`` makes
real runs fail closed instead).
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from sandbox import Limits, Sandbox
from sandbox.executor import SandboxError

PY = sys.executable

pytestmark = pytest.mark.skipif(os.name != "posix", reason="linux namespaces only")


@pytest.fixture()
def box(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    sb = Sandbox(root, fs_isolation="required", limits=Limits(wall_seconds=30))
    probe = sb.run([PY, "-c", "print('ok')"])
    if probe.status != "pass":
        pytest.skip(f"host cannot provide filesystem isolation: {probe.message}")
    return sb


def run_code(box: Sandbox, code: str, **kw):
    return box.run([PY, "-c", textwrap.dedent(code)], **kw)


def test_all_layers_reported(box):
    res = run_code(box, "print(1)")
    assert res.isolation["pidns"] == "ok"
    assert res.isolation["fs"] == {"proc": "ok", "readonly_root": "ok", "caps": "ok"}
    assert res.isolation["loopback"] == "ok"


def test_trainer_secrets_are_not_readable(box, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "ghp_SUPERSECRET")
    monkeypatch.setenv("KAGGLE_KEY", "kk_SUPERSECRET")
    res = run_code(
        box,
        """
        import os
        found = []
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                if b"SUPERSECRET" in open(f"/proc/{pid}/environ", "rb").read():
                    found.append(pid)
            except OSError:
                pass
        print("LEAKED" if found else "clean", len(os.listdir("/proc")))
        """,
    )
    assert res.status == "pass", res.stderr
    assert res.stdout.startswith("clean")


def test_host_processes_are_invisible(box):
    res = run_code(box, "import os; print(sum(p.isdigit() for p in os.listdir('/proc')))")
    assert res.status == "pass", res.stderr
    assert int(res.stdout.strip()) <= 3  # the payload (and at most its parent)


def test_cannot_write_outside_the_sandbox_root(box):
    outside = Path(tempfile.gettempdir()) / "fs_iso_outside.txt"
    outside.unlink(missing_ok=True)
    res = run_code(
        box,
        f"""
        try:
            open({str(outside)!r}, "w").write("escaped")
            print("WROTE")
        except OSError as e:
            print("blocked", e.errno)
        """,
    )
    assert "blocked" in res.stdout, res.stdout + res.stderr
    assert not outside.exists()


def test_shared_directories_cannot_be_tampered_with(box, tmp_path):
    shared = tmp_path / "shared_venv"
    shared.mkdir()
    (shared / "lib.py").write_text("x = 1\n")
    res = run_code(
        box,
        f"""
        import pathlib
        p = pathlib.Path({str(shared)!r})
        for action in (
            lambda: (p / "lib.py").write_text("x = 'poisoned'"),
            lambda: (p / "evil.pth").write_text("import os"),
            lambda: (p / "lib.py").unlink(),
        ):
            try:
                action(); print("TAMPERED")
            except OSError as e:
                print("ro")
        """,
    )
    assert res.stdout.split() == ["ro", "ro", "ro"], res.stdout + res.stderr
    assert (shared / "lib.py").read_text() == "x = 1\n"


def test_root_cwd_and_tmpdir_are_writable(box):
    res = run_code(
        box,
        """
        import os, pathlib, tempfile
        pathlib.Path("made_in_cwd.txt").write_text("ok")
        with tempfile.NamedTemporaryFile(dir=tempfile.gettempdir()) as fh:
            fh.write(b"scratch")
        pathlib.Path(os.path.expanduser("~/home_ok")).write_text("ok")
        print("fine", tempfile.gettempdir())
        """,
    )
    assert res.status == "pass", res.stderr
    assert (box.root / "made_in_cwd.txt").read_text() == "ok"
    assert ".sandbox-tmp" in res.stdout


def test_loopback_works_but_egress_is_blocked(box):
    res = run_code(
        box,
        """
        import socket
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen()
        cli = socket.socket(); cli.settimeout(2); cli.connect(srv.getsockname())
        print("loopback ok")
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=1)
            print("EGRESS")
        except OSError:
            print("egress blocked")
        """,
    )
    assert res.stdout.split("\n")[:2] == ["loopback ok", "egress blocked"], res.stdout + res.stderr


def test_capabilities_are_dropped_and_mounts_cannot_be_undone(box):
    res = run_code(
        box,
        """
        import ctypes, os
        status = dict(l.split(":\\t", 1) for l in open("/proc/self/status").read().splitlines() if ":\\t" in l)
        print("CapEff", int(status["CapEff"], 16), "NoNewPrivs", status["NoNewPrivs"])
        libc = ctypes.CDLL(None, use_errno=True)
        MS_REMOUNT, MS_BIND = 32, 4096
        rc = libc.mount(None, b"/", None, MS_REMOUNT | MS_BIND, None)
        print("remount rc", rc)
        """,
    )
    lines = res.stdout.strip().splitlines()
    assert lines[0] == "CapEff 0 NoNewPrivs 1", res.stdout + res.stderr
    assert lines[1] == "remount rc -1"


def test_a_payload_can_signal_itself(box):
    res = run_code(box, "import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
    assert res.status == "fail" and "signal 9" in res.message


def test_daemonised_children_do_not_outlive_the_run(box):
    marker = f"fs-iso-marker-{os.getpid()}-{int(time.time())}"
    res = run_code(
        box,
        f"""
        import os, subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)  # {marker}"],
                         start_new_session=True)
        time.sleep(0.2)
        """,
    )
    assert res.status == "pass", res.stderr
    time.sleep(0.3)
    survivors = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if marker.encode() in Path(f"/proc/{pid}/cmdline").read_bytes():
                survivors.append(pid)
        except OSError:
            pass
    assert survivors == []


def test_invalid_mode_is_rejected(tmp_path):
    with pytest.raises(SandboxError):
        Sandbox(tmp_path, fs_isolation="yolo")


def test_mode_off_keeps_the_old_behaviour(tmp_path):
    sb = Sandbox(tmp_path, fs_isolation="off")
    res = sb.run([PY, "-c", "import os; print(os.getppid() > 1)"])
    assert res.status == "pass"
    assert res.isolation["pidns"] == "off"
