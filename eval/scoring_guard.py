"""Anti-reward-hacking guards for scoring (AUDIT.md finding C1).

The submitted patch is applied to a fresh tree *after* the task's test
patch, so without these guards a policy can "solve" a task by editing the
target test, adding a ``conftest.py`` that skips everything, or patching the
test runner. Two layers:

1. :func:`sanitize_patch` drops every file section that touches test /
   harness paths before the patch is applied. Writing tests stays allowed
   (they are simply ignored by the scorer) so honest test-writing is not
   punished, but it can never influence the reward.
2. :func:`verify_outcome` refuses to trust an exit code alone: the run must
   actually have executed and passed at least one test, with nothing skipped.

Residual risk (documented, not solved here): source code that forges the
runner's own output from inside the same process. Filesystem isolation in the
sandbox plus junit parsing make this hard; it is not impossible.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
from typing import Any

from agent.tools.edit_file import split_multifile_patch

#: file basenames that change how tests are collected, selected or reported
PROTECTED_BASENAMES = frozenset(
    {
        "conftest.py",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        "noxfile.py",
        "runtests.py",
        "sitecustomize.py",
        "usercustomize.py",
        "tests.py",  # Django per-app test modules
    }
)
#: exact repo-relative paths that implement the test runner itself
PROTECTED_PATHS = frozenset({"django/test/runner.py", "tests/runtests.py"})
#: directory components that mean "test code"
PROTECTED_DIRS = frozenset({"tests", "testing_data"})


def _norm(path: str) -> str:
    p = PurePosixPath(path.replace("\\", "/"))
    parts = [x for x in p.parts if x not in (".", "")]
    return "/".join(parts)


def paths_in_patch(test_patch: str) -> set[str]:
    if not test_patch:
        return set()
    return {_norm(p) for p in split_multifile_patch(test_patch)}


def is_protected(path: str, extra: set[str] | frozenset[str] = frozenset()) -> bool:
    """True when ``path`` is test code or test-harness configuration."""
    norm = _norm(path)
    if not norm:
        return True
    if norm in extra or norm in PROTECTED_PATHS:
        return True
    pure = PurePosixPath(norm)
    name = pure.name
    if name in PROTECTED_BASENAMES:
        return True
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    if name.endswith((".pth", ".egg-link")):
        return True
    if any(part in PROTECTED_DIRS for part in pure.parts[:-1]):
        return True
    return False


def sanitize_patch(patch: str, test_patch: str = "") -> tuple[str, list[str]]:
    """Return ``(patch_without_protected_sections, dropped_paths)``."""
    extra = paths_in_patch(test_patch)
    sections = split_multifile_patch(patch)
    kept: list[str] = []
    dropped: list[str] = []
    for path, body in sections.items():
        if is_protected(path, extra):
            dropped.append(_norm(path))
        else:
            kept.append(body)
    if not kept:
        return "", dropped
    return "\n".join(part.rstrip("\n") for part in kept) + "\n", dropped


# ------------------------------------------------------------- outcome checks
_DJANGO_RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
_DJANGO_OK = re.compile(r"^OK(?: \(([^)]*)\))?\s*$", re.M)


def parse_junit(xml_text: str) -> dict[str, int]:
    """Aggregate counts from a pytest junit XML document."""
    root = ET.fromstring(xml_text)
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.attrib.get(key, 0) or 0)
    return totals


def verify_outcome(
    runner: str,
    output: str,
    *,
    junit_xml: str | None = None,
) -> tuple[bool, str]:
    """Check that a zero exit status really means "the tests ran and passed"."""
    if runner == "django":
        ran = _DJANGO_RAN.search(output)
        if not ran or int(ran.group(1)) < 1:
            return False, "no tests were executed"
        ok = _DJANGO_OK.search(output)
        if not ok:
            return False, "runner did not report OK"
        if ok.group(1) and "skipped" in ok.group(1):
            return False, "target tests were skipped"
        return True, "tests passed"

    if junit_xml is None:
        return False, "no junit report produced"
    try:
        totals = parse_junit(junit_xml)
    except ET.ParseError:
        return False, "junit report unreadable"
    executed = totals["tests"] - totals["skipped"]
    if totals["tests"] < 1 or executed < 1:
        return False, "no tests were executed"
    if totals["skipped"]:
        return False, f"{totals['skipped']} target test(s) skipped"
    if totals["failures"] or totals["errors"]:
        return False, "junit reports failures"
    return True, "tests passed"


def is_infra_kill(returncode: int | None, status: str, message: str) -> bool:
    """A resource-limit or wall kill says nothing about patch correctness."""
    if status == "timeout":
        return True
    if returncode is not None and returncode < 0 and -returncode in (6, 9, 11, 24, 25):
        return True
    return "killed by signal" in (message or "") and status != "pass"


__all__: list[Any] = [
    "sanitize_patch",
    "is_protected",
    "verify_outcome",
    "parse_junit",
    "is_infra_kill",
    "paths_in_patch",
]
