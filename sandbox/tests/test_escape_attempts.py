"""Adversarial sandbox tests — ARCHITECTURE.md §2.3, §8.2 Week-1 gate.

These are the tests that decide whether any reward or eval number is
trustworthy. They are written against the *attack*, not the implementation:
shell injection, allowlist bypass, path/symlink escape, resource exhaustion,
network egress, and silent (empty) results.

Note on scope: the sandbox does not filesystem-jail the payload — running a
repo's test suite *requires* reading the repo. What is enforced is that
cwd and tool paths stay inside the root, that the payload's own network/
resource state is isolated, and that only allowlisted binaries launch.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from sandbox import DEFAULT_ALLOWLIST, ExecResult, Limits, Sandbox, SandboxError
from sandbox import namespace as ns
from sandbox.limits import LimitError

PY = sys.executable
IS_POSIX = os.name == "posix"
IS_WINDOWS = sys.platform == "win32"

FORBIDDEN_BINARIES = [
    "sh", "bash", "zsh", "dash", "ksh", "cmd", "powershell", "powershell_ise",
    "curl", "wget", "ftp", "nc", "ncat", "telnet",
    "ssh", "scp", "sftp", "rsync",
    "sudo", "su", "doas", "pkexec",
    "chmod", "chown", "chroot", "mount", "umount",
    "rm", "dd", "mkfs", "shutdown", "reboot", "kill", "killall",
    "python2", "perl", "ruby", "php",
]


@pytest.fixture
def sandbox(tmp_path: object) -> Sandbox:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "ok.txt").write_text("hello\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("classified\n", encoding="utf-8")
    return Sandbox(
        root,
        limits=Limits(
            cpu_seconds=10,
            wall_seconds=20,
            address_space_bytes=1024**3,
            process_count=64,
        ),
    )


def _assert_typed(res: ExecResult) -> None:
    assert res.status in ("pass", "fail", "timeout", "error")
    assert isinstance(res.stdout, str)
    assert isinstance(res.stderr, str)
    assert isinstance(res.duration_s, float)
    if res.status != "pass":
        assert res.message, "a non-pass result must carry a message"


# --------------------------------------------------------------------- injection
def test_shell_metacharacters_are_inert(sandbox: Sandbox) -> None:
    payload = "a; touch ESCAPED1 && touch ESCAPED2 | cat `id` $(id) > ESCAPED3"
    res = sandbox.run([PY, "-c", "import sys, pathlib; pathlib.Path('out.txt').write_text(sys.argv[1])", payload])
    _assert_typed(res)
    assert res.status == "pass"
    assert (sandbox.root / "out.txt").read_text(encoding="utf-8") == payload
    for marker in ("ESCAPED1", "ESCAPED2", "ESCAPED3"):
        assert not (sandbox.root / marker).exists(), f"metacharacter executed: {marker}"


def test_no_shell_interpreter_is_used(sandbox: Sandbox) -> None:
    arg = "&& echo pwned > SHELL_OUT || echo pwned >> SHELL_OUT"
    res = sandbox.run([PY, "-c", "import sys; print(sys.argv[1])", arg])
    _assert_typed(res)
    assert res.status == "pass"
    assert res.stdout.strip() == arg
    assert not (sandbox.root / "SHELL_OUT").exists()


def test_newline_and_null_byte_in_arg_do_not_split_commands(sandbox: Sandbox) -> None:
    arg = "one\ntwo\nimport os; os.system('touch NL_PWNED')"
    res = sandbox.run([PY, "-c", "import sys, pathlib; pathlib.Path('nl.txt').write_text(sys.argv[1])", arg])
    _assert_typed(res)
    assert res.status == "pass"
    assert (sandbox.root / "nl.txt").read_text(encoding="utf-8") == arg
    assert not (sandbox.root / "NL_PWNED").exists()


# -------------------------------------------------------------------- allowlist
@pytest.mark.parametrize("binary", FORBIDDEN_BINARIES)
def test_forbidden_binary_is_rejected(sandbox: Sandbox, binary: str) -> None:
    res = sandbox.run([binary, "-c", "print('hi')"])
    _assert_typed(res)
    assert res.status == "error"
    assert "allowlist" in res.message
    assert res.returncode is None, "rejected binaries must never be launched"


def test_allowlist_applies_to_path_qualified_programs(sandbox: Sandbox) -> None:
    rogue = sandbox.root / "sh"
    rogue.write_text("#!/bin/sh\ntouch ROGUE_PWNED\n", encoding="utf-8")
    rogue.chmod(0o755)
    res = sandbox.run([str(rogue)])
    _assert_typed(res)
    assert res.status == "error"
    assert "allowlist" in res.message
    assert not (sandbox.root / "ROGUE_PWNED").exists()


def test_allowlisted_but_missing_binary_is_an_error(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    allow = set(DEFAULT_ALLOWLIST) | {"ghost_binary_xyz"}
    sb = Sandbox(root, allowlist=allow)
    res = sb.run(["ghost_binary_xyz", "--version"])
    _assert_typed(res)
    assert res.status == "error"
    assert "not found" in res.message


# ----------------------------------------------------------------------- paths
def test_relative_path_escape_is_rejected(sandbox: Sandbox) -> None:
    with pytest.raises(SandboxError):
        sandbox.resolve_path("../outside/secret.txt")
    with pytest.raises(SandboxError):
        sandbox.resolve_path("a/../../outside/secret.txt")


def test_absolute_path_outside_root_is_rejected(sandbox: Sandbox) -> None:
    with pytest.raises(SandboxError):
        sandbox.resolve_path(sandbox.root.parent / "outside" / "secret.txt")


def test_cwd_escape_is_rejected(sandbox: Sandbox) -> None:
    res = sandbox.run([PY, "-c", "print('hi')"], cwd="..")
    _assert_typed(res)
    assert res.status == "error"
    assert "escapes sandbox root" in res.message


def test_symlink_escape_is_rejected(sandbox: Sandbox, tmp_path) -> None:
    target = tmp_path / "outside" / "secret.txt"
    link = sandbox.root / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    with pytest.raises(SandboxError):
        sandbox.resolve_path("link.txt")


def test_symlinked_directory_escape_is_rejected(sandbox: Sandbox, tmp_path) -> None:
    target_dir = tmp_path / "outside"
    link_dir = sandbox.root / "linkdir"
    try:
        link_dir.symlink_to(target_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    with pytest.raises(SandboxError):
        sandbox.resolve_path("linkdir/secret.txt")


def test_valid_in_root_path_resolves(sandbox: Sandbox) -> None:
    resolved = sandbox.resolve_path("ok.txt", must_exist=True)
    assert resolved == (sandbox.root / "ok.txt").resolve()


# --------------------------------------------------------------------- resources
def test_timeout_is_typed_and_kills_the_process(sandbox: Sandbox) -> None:
    res = sandbox.run([PY, "-c", "import time; time.sleep(30)"], timeout=2)
    _assert_typed(res)
    assert res.status == "timeout"
    assert "exceeded" in res.message
    assert res.duration_s < 15


def test_nonzero_exit_is_fail_not_error(sandbox: Sandbox) -> None:
    res = sandbox.run([PY, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])
    _assert_typed(res)
    assert res.status == "fail"
    assert res.returncode == 3
    assert "boom" in res.stderr
    assert "3" in res.message


def test_signal_death_is_fail_with_message(sandbox: Sandbox) -> None:
    if not IS_POSIX:
        pytest.skip("posix signals only")
    res = sandbox.run([PY, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"])
    _assert_typed(res)
    assert res.status == "fail"
    assert "signal" in res.message


def test_empty_output_still_yields_typed_result(sandbox: Sandbox) -> None:
    res = sandbox.run([PY, "-c", "pass"])
    _assert_typed(res)
    assert res.status == "pass"
    assert res.stdout == ""
    assert res.stderr == ""
    assert res.message


def test_invalid_utf8_output_does_not_crash(sandbox: Sandbox) -> None:
    res = sandbox.run([PY, "-c", "import sys; sys.stdout.buffer.write(b'\\xff\\xfe raw')"])
    _assert_typed(res)
    assert res.status == "pass"
    assert "raw" in res.stdout


def test_empty_argv_is_rejected(sandbox: Sandbox) -> None:
    res = sandbox.run([])
    _assert_typed(res)
    assert res.status == "error"


def test_non_string_argv_is_rejected(sandbox: Sandbox) -> None:
    res = sandbox.run([PY, "-c", "pass", 42])  # type: ignore[list-item]
    _assert_typed(res)
    assert res.status == "error"


@pytest.mark.skipif(not IS_POSIX, reason="RLIMIT_CPU is posix-only")
def test_cpu_limit_enforced(sandbox: Sandbox) -> None:
    sb = Sandbox(
        sandbox.root,
        limits=Limits(cpu_seconds=1, wall_seconds=20, address_space_bytes=1024**3),
    )
    res = sb.run([PY, "-c", "while True: pass"], timeout=20)
    _assert_typed(res)
    assert res.status == "fail"
    assert res.duration_s < 10
    assert "cpu_seconds" in res.enforced_limits


@pytest.mark.skipif(not IS_POSIX, reason="RLIMIT_AS is posix-only")
def test_memory_limit_enforced(sandbox: Sandbox) -> None:
    sb = Sandbox(
        sandbox.root,
        limits=Limits(cpu_seconds=10, wall_seconds=20, address_space_bytes=512 * 1024**2),
    )
    res = sb.run([PY, "-c", "x = bytearray(2 * 1024**3); print(len(x))"], timeout=20)
    _assert_typed(res)
    assert res.status in ("fail", "timeout")
    assert not res.unsupported_limits


@pytest.mark.skipif(not IS_POSIX, reason="RLIMIT_NPROC is posix-only")
def test_process_count_limit_stops_fork_bomb(sandbox: Sandbox) -> None:
    bomb = (
        "import os, sys\n"
        "n = 0\n"
        "while True:\n"
        "    try:\n"
        "        pid = os.fork()\n"
        "    except Exception:\n"
        "        print('blocked after', n); sys.exit(7)\n"
        "    if pid == 0:\n"
        "        os._exit(0)\n"
        "    n += 1\n"
    )
    sb = Sandbox(sandbox.root, limits=Limits(cpu_seconds=5, wall_seconds=10, process_count=16))
    res = sb.run([PY, "-c", bomb], timeout=10)
    _assert_typed(res)
    assert res.status in ("fail", "timeout")
    assert res.duration_s < 15


# ----------------------------------------------------------------------- network
@pytest.mark.skipif(not IS_POSIX, reason="requires network namespace (Kaggle/Linux)")
def test_no_network_egress(sandbox: Sandbox) -> None:
    probe = (
        "import socket, sys\n"
        "try:\n"
        "    s = socket.create_connection(('1.1.1.1', 443), timeout=3)\n"
        "    s.close()\n"
        "    print('EGRESS_ALLOWED')\n"
        "    sys.exit(0)\n"
        "except Exception as exc:\n"
        "    print('blocked', type(exc).__name__)\n"
        "    sys.exit(9)\n"
    )
    res = sandbox.run([PY, "-c", probe], timeout=15)
    _assert_typed(res)
    assert "EGRESS_ALLOWED" not in res.stdout, "payload reached the network"
    assert res.returncode == 9, f"expected egress to be blocked, got: {res.status} {res.message}"
    assert res.isolation.get("netns") == "ok"


# ---------------------------------------------------------- isolation is reported
def test_isolation_quality_is_always_reported(sandbox: Sandbox) -> None:
    res = sandbox.run([PY, "-c", "print('x')"])
    _assert_typed(res)
    assert res.status == "pass"
    if IS_POSIX:
        assert res.isolation.get("netns") == "ok"
        assert not res.unsupported_limits, "posix host must enforce every rlimit"
        assert not res.warnings, f"unexpected isolation warnings: {res.warnings}"
    else:
        assert res.warnings, "degraded isolation must never be silent"
        assert res.unsupported_limits, "win32 must report which limits it cannot enforce"


def test_environment_is_scrubbed(sandbox: Sandbox, monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_LEAK_CANARY", "hunter2")
    monkeypatch.setenv("PYTHONSTARTUP", "/tmp/evil.py")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/evil.so")
    res = sandbox.run([PY, "-c", "import os, json; print(json.dumps(sorted(os.environ)))"])
    _assert_typed(res)
    assert res.status == "pass"
    seen = json.loads(res.stdout)
    for leaked in ("SANDBOX_LEAK_CANARY", "PYTHONSTARTUP", "LD_PRELOAD"):
        assert leaked not in seen, f"{leaked} leaked into the sandbox"
    assert "PATH" in seen


def test_extra_env_cannot_smuggle_disallowed_keys(sandbox: Sandbox) -> None:
    res = sandbox.run(
        [PY, "-c", "import os; print(os.environ.get('PYTHONPATH', 'unset'))"],
        extra_env={"PYTHONPATH": "/tmp/evil"},
    )
    _assert_typed(res)
    assert res.status == "pass"
    assert res.stdout.strip() == "unset"


# ------------------------------------------------------- fail-closed isolation
@pytest.mark.skipif(not IS_POSIX, reason="posix isolation branch only")
def test_prepare_fails_closed_without_netns(tmp_path, monkeypatch) -> None:
    status_path = tmp_path / "status.json"

    def _boom() -> str:
        raise ns.IsolationError("simulated unshare failure")

    monkeypatch.setattr(ns, "_isolate_network", _boom)
    with pytest.raises(ns.IsolationError):
        ns.prepare(status_path, Limits(), allow_degraded=False)
    data = json.loads(status_path.read_text(encoding="utf-8"))
    assert data["ok"] is False
    assert "network namespace" in data["error"]


@pytest.mark.skipif(not IS_POSIX, reason="posix isolation branch only")
def test_prepare_degraded_mode_is_explicit_and_warned(tmp_path, monkeypatch) -> None:
    status_path = tmp_path / "status.json"

    def _boom() -> str:
        raise ns.IsolationError("simulated unshare failure")

    monkeypatch.setattr(ns, "_isolate_network", _boom)
    status = ns.prepare(status_path, Limits(), allow_degraded=True)
    assert status["ok"] is True
    assert status["netns"] == "degraded"
    assert status["warnings"], "degraded mode must leave a warning"
    assert json.loads(status_path.read_text(encoding="utf-8"))["netns"] == "degraded"


@pytest.mark.skipif(IS_POSIX, reason="win32 isolation branch only")
def test_prepare_win32_reports_unavailable_and_warns(tmp_path) -> None:
    status_path = tmp_path / "status.json"
    with pytest.raises(ns.IsolationError):
        ns.prepare(status_path, Limits(), allow_degraded=False)
    data = json.loads(status_path.read_text(encoding="utf-8"))
    assert data["ok"] is False
    assert "namespace" in data["error"]

    status = ns.prepare(status_path, Limits(), allow_degraded=True)
    assert status["ok"] is True
    assert status["netns"] == "unavailable"
    assert status["warnings"], "win32 degraded mode must never be silent"


def test_prepare_without_netns_api_fails_closed(tmp_path, monkeypatch) -> None:
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(ns, "_supports_netns", lambda: False)
    if not IS_POSIX:
        pytest.skip("win32 branch reported by the test above")
    with pytest.raises(ns.IsolationError):
        ns.prepare(status_path, Limits(), allow_degraded=False)
    data = json.loads(status_path.read_text(encoding="utf-8"))
    assert data["ok"] is False
    assert "namespace" in data["error"]


@pytest.mark.skipif(not IS_POSIX, reason="rlimits only exist on posix")
def test_prepare_fatal_when_rlimits_cannot_be_applied(tmp_path, monkeypatch) -> None:
    status_path = tmp_path / "status.json"

    def _boom(limits: Limits) -> dict:
        raise LimitError("simulated setrlimit failure")

    monkeypatch.setattr(ns, "apply_limits", _boom)
    with pytest.raises(ns.IsolationError):
        ns.prepare(status_path, Limits(), allow_degraded=True)
    data = json.loads(status_path.read_text(encoding="utf-8"))
    assert data["ok"] is False
    assert "resource limits" in data["error"]


def test_missing_status_file_is_an_error(sandbox: Sandbox, monkeypatch) -> None:
    """If we never learn whether isolation worked, the run must not score."""
    monkeypatch.setattr(Sandbox, "_read_status", staticmethod(lambda path: None))
    res = sandbox.run([PY, "-c", "print('should not count')"])
    _assert_typed(res)
    assert res.status == "error"
    assert "isolation status" in res.message


# ------------------------------------------------------------------ test runner
def test_python_module_test_runner_works(sandbox: Sandbox) -> None:
    """run_tests() has to be able to launch a real test runner (§2.2)."""
    (sandbox.root / "test_sanity.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    res = sandbox.run([PY, "-m", "pytest", "-q"], timeout=120)
    _assert_typed(res)
    assert res.status == "pass", f"{res.status}: {res.message}\n{res.stdout}\n{res.stderr}"
    assert "1 passed" in res.stdout
