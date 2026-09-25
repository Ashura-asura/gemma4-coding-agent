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
import subprocess
import sys
from pathlib import Path

# allow `python .../sandbox/launcher.py` regardless of the payload's cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    limits = Limits()
    it = iter(head)
    for tok in it:
        if tok == "--status":
            status_path = Path(next(it))
        elif tok == "--limits":
            limits = Limits.from_dict(json.loads(next(it)))
        elif tok == "--allow-degraded":
            allow_degraded = True
        else:
            raise SystemExit(f"launcher: unexpected argument {tok!r}")
    if status_path is None:
        raise SystemExit("launcher: --status is required")

    try:
        ns.prepare(status_path, limits, allow_degraded=allow_degraded)
    except ns.IsolationError as exc:
        print(f"launcher: isolation failed: {exc}", file=sys.stderr)
        return FAIL_ISOLATION

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


if __name__ == "__main__":
    raise SystemExit(main())
