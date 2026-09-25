"""Sandboxed command executor — ARCHITECTURE.md §2.3.

Guarantees:
  * no shell interpreter: argv is handed to execve verbatim, so
    metacharacters (``;``, ``|``, ``&&``, backticks) are inert
  * binary allowlist: ``sh``, ``curl``, ``ssh``, ``sudo``, ... never launch
  * fresh network namespace per run (fail closed; degraded mode must be
    requested explicitly and is reported in the result)
  * rlimits for cpu / address space / process count / file size
  * working directory and file paths scope-checked against the repo root,
    symlinks included
  * every run yields a typed ``pass | fail | timeout | error`` result with
    a message — never a silent empty result
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from .limits import Limits

Status = Literal["pass", "fail", "timeout", "error"]

#: Every binary the sandbox is allowed to exec, matched on basename only
#: (after stripping an .exe/.com/.bat/.cmd suffix).
DEFAULT_ALLOWLIST: frozenset[str] = frozenset(
    {
        # python / test runners
        "python", "python3", "pytest", "pip", "pip3", "uv", "poetry",
        "ruff", "mypy", "flake8", "black", "coverage", "tox",
        # vcs + build
        "git", "make", "cmake", "ninja", "pkg-config",
        "gcc", "g++", "cc", "c++", "clang", "clang++",
        "cargo", "rustc", "go",
        "java", "javac", "mvn", "gradle", "dotnet",
        # js toolchain
        "node", "npm", "npx", "yarn", "pnpm",
    }
)

#: Environment variables a sandboxed process is allowed to see. Anything
#: else in the parent environment is dropped (LD_PRELOAD, PYTHONPATH,
#: PYTHONSTARTUP, ... never make it through).
ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH", "HOME", "USERPROFILE", "LOGNAME", "USER",
    "TMP", "TEMP", "TMPDIR", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR",
    "LANG", "LC_ALL", "LC_CTYPE",
    "PYTHONIOENCODING", "PYTHONHASHSEED", "PYTHONNOUSERSITE",
    "VIRTUAL_ENV", "CONDA_PREFIX",
)

_EXE_SUFFIXES = (".exe", ".com", ".bat", ".cmd")


class SandboxError(Exception):
    """A path or argument was rejected before anything was executed."""


@dataclass(frozen=True)
class ExecResult:
    status: Status
    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    message: str
    isolation: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    enforced_limits: tuple[str, ...] = ()
    unsupported_limits: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "message": self.message,
            "isolation": self.isolation,
            "warnings": list(self.warnings),
            "enforced_limits": list(self.enforced_limits),
            "unsupported_limits": list(self.unsupported_limits),
        }


def _executable_stem(name: str) -> str:
    """Basename of ``name`` with any executable suffix removed, lower-cased."""
    stem = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in _EXE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _within(path: Path, root: Path) -> bool:
    a = os.path.normcase(str(path))
    b = os.path.normcase(str(root))
    return a == b or a.startswith(b + os.sep)


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # launcher called setsid()
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    else:  # win32: taskkill walks the child tree
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


class Sandbox:
    """Executes argv inside a scope-checked, resource-limited, isolated child."""

    def __init__(
        self,
        root: str | Path,
        *,
        limits: Limits | None = None,
        allowlist: Iterable[str] | None = None,
        allow_degraded: bool | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise SandboxError(f"sandbox root is not a directory: {self.root}")
        self.limits = limits or Limits()
        names = DEFAULT_ALLOWLIST if allowlist is None else allowlist
        self.allowlist = frozenset(_executable_stem(n) for n in names)
        # degraded isolation is opt-in everywhere except win32, which has no
        # network namespace API at all (dev host only — see ARCHITECTURE §5)
        self.allow_degraded = (sys.platform == "win32") if allow_degraded is None else allow_degraded
        self.launcher = Path(__file__).resolve().parent / "launcher.py"

    # ------------------------------------------------------------------ paths
    def resolve_path(self, path: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve ``path`` and require it to stay inside the sandbox root."""
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise SandboxError("empty path")
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise SandboxError(f"cannot resolve path {path!r}: {exc}") from exc
        if not _within(resolved, self.root):
            raise SandboxError(f"path escapes sandbox root: {path}")
        if must_exist and not resolved.exists():
            raise SandboxError(f"path does not exist: {path}")
        return resolved

    # ------------------------------------------------------------------- env
    def _build_env(self, extra_env: dict[str, str] | None) -> dict[str, str]:
        env = {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ}
        for key, value in (extra_env or {}).items():
            if key in ENV_ALLOWLIST:
                env[key] = str(value)
        return env

    # ------------------------------------------------------------------- run
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run ``argv`` and always return a typed :class:`ExecResult`."""
        started = time.monotonic()
        payload = tuple(argv) if argv is not None else ()

        def _reject(message: str) -> ExecResult:
            return ExecResult(
                status="error",
                argv=tuple(str(a) for a in payload),
                returncode=None,
                stdout="",
                stderr="",
                duration_s=time.monotonic() - started,
                message=message,
            )

        if not payload:
            return _reject("empty argv")
        if any(not isinstance(a, str) for a in payload):
            return _reject("argv entries must all be strings")

        stem = _executable_stem(payload[0])
        if stem not in self.allowlist:
            return _reject(f"binary not in allowlist: {payload[0]}")

        # resolve the program to an absolute path so PATH cannot surprise us
        program = payload[0]
        if os.path.isabs(program) or os.sep in program or (os.altsep and os.altsep in program):
            if not Path(program).is_file():
                return _reject(f"binary not found: {program}")
            program = str(Path(program).resolve())
        else:
            found = shutil.which(program, path=self._build_env(extra_env).get("PATH"))
            if found is None:
                return _reject(f"binary not found: {program}")
            program = found

        try:
            workdir = self.resolve_path(cwd) if cwd is not None else self.root
        except SandboxError as exc:
            return _reject(str(exc))
        if not workdir.is_dir():
            return _reject(f"cwd is not a directory: {workdir}")

        effective_timeout = float(timeout if timeout is not None else self.limits.wall_seconds)
        if effective_timeout <= 0:
            return _reject("timeout must be positive")

        env = self._build_env(extra_env)

        with tempfile.TemporaryDirectory(prefix="sbox-") as tmp:
            status_path = Path(tmp) / "isolation.json"
            cmd: list[str] = [
                sys.executable,
                str(self.launcher),
                "--status",
                str(status_path),
                "--limits",
                json.dumps(self.limits.as_dict()),
            ]
            if self.allow_degraded:
                cmd.append("--allow-degraded")
            cmd += ["--", program, *payload[1:]]

            proc = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # isolation is completed before exec, so the status file exists
            # long before the payload can finish (or tamper with) anything
            spawn_t = time.monotonic()
            isolation = self._await_status(status_path, proc, effective_timeout)
            remaining = max(0.05, effective_timeout - (time.monotonic() - spawn_t))

            try:
                out, err = proc.communicate(timeout=remaining)
                timed_out = False
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                try:
                    out, err = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    out, err = b"", b""
                timed_out = True

            if isolation is None:
                isolation = self._read_status(status_path)

        duration = time.monotonic() - started
        stdout = (out or b"").decode("utf-8", errors="replace")
        stderr = (err or b"").decode("utf-8", errors="replace")
        isolation = isolation or {}
        warnings = tuple(isolation.get("warnings", []))

        if not isolation:
            status: Status = "error"
            message = "isolation status was never written; payload did not run"
        elif not isolation.get("ok"):
            status = "error"
            message = str(isolation.get("error") or "isolation failed")
        elif timed_out:
            status = "timeout"
            message = f"wall-clock limit of {effective_timeout:g}s exceeded"
        elif proc.returncode == 0:
            status = "pass"
            message = "ok"
        else:
            status = "fail"
            rc = proc.returncode or -1
            message = (
                f"killed by signal {-rc}" if rc < 0 else f"exit code {rc}"
            )

        return ExecResult(
            status=status,
            argv=payload,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            message=message,
            isolation=isolation,
            warnings=warnings,
            enforced_limits=tuple(isolation.get("rlimits", {})),
            unsupported_limits=tuple(isolation.get("unsupported_rlimits", [])),
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _read_status(path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @classmethod
    def _await_status(
        cls, path: Path, proc: subprocess.Popen, budget: float
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + min(budget, 15.0)
        while time.monotonic() < deadline:
            status = cls._read_status(path)
            if status is not None:
                return status
            if proc.poll() is not None:
                return cls._read_status(path)
            time.sleep(0.01)
        return cls._read_status(path)
