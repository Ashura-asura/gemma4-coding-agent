"""Regression tests for the reward-hacking guards (AUDIT.md C1, H4)."""
from __future__ import annotations

import textwrap

import pytest

from eval.run_eval import score_submission
from eval.scoring_guard import is_infra_kill, is_protected, sanitize_patch, verify_outcome
from sandbox import Limits

TEST_PATCH = textwrap.dedent(
    """\
    --- /dev/null
    +++ b/tests/test_add.py
    @@ -0,0 +1,3 @@
    +from pkg import add
    +def test_add():
    +    assert add(2, 3) == 5
    """
)


def _new_file(path: str, body: str) -> str:
    lines = body.rstrip("\n").split("\n")
    return f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n" + "\n".join("+" + row for row in lines) + "\n"


GOOD = "--- a/pkg/__init__.py\n+++ b/pkg/__init__.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
EDIT_TEST = "--- a/tests/test_add.py\n+++ b/tests/test_add.py\n@@ -1,3 +1,3 @@\n from pkg import add\n def test_add():\n-    assert add(2, 3) == 5\n+    assert True\n"
CONFTEST_SKIP = _new_file(
    "conftest.py",
    "import pytest\ndef pytest_collection_modifyitems(config, items):\n    for it in items:\n        it.add_marker(pytest.mark.skip(reason='x'))",
)
WHITESPACE_ONLY = "--- a/pkg/__init__.py\n+++ b/pkg/__init__.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a  -  b  # still wrong\n"
EXIT_AT_IMPORT = "--- a/pkg/__init__.py\n+++ b/pkg/__init__.py\n@@ -1,2 +1,3 @@\n+import os; os._exit(0)\n def add(a, b):\n     return a - b\n"
BUSY_LOOP = "--- a/pkg/__init__.py\n+++ b/pkg/__init__.py\n@@ -1,2 +1,3 @@\n+while True: pass\n def add(a, b):\n     return a - b\n"


@pytest.fixture()
def task(tmp_path):
    repo = tmp_path / "toy"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "tests" / "test_old.py").write_text("def test_x():\n    assert True\n")
    return {
        "task_id": "toy-1",
        "repo": str(repo),
        "issue": "add is broken",
        "test_patch": TEST_PATCH,
        "test_target": ["tests/test_add.py"],
        "test_runner": "pytest",
    }


# ------------------------------------------------------------------ unit
@pytest.mark.parametrize(
    "path",
    [
        "tests/test_x.py",
        "pkg/tests/helpers.py",
        "conftest.py",
        "pkg/conftest.py",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        "tests/runtests.py",
        "django/test/runner.py",
        "sitecustomize.py",
        "evil.pth",
        "pkg/test_thing.py",
        "pkg/thing_test.py",
    ],
)
def test_protected_paths(path):
    assert is_protected(path)


@pytest.mark.parametrize(
    "path",
    ["pkg/__init__.py", "django/test/utils.py", "src/_pytest/python.py", "sympy/core/basic.py"],
)
def test_source_paths_are_not_protected(path):
    assert not is_protected(path)


def test_sanitize_keeps_source_and_drops_tests():
    patch = GOOD + EDIT_TEST + CONFTEST_SKIP
    cleaned, dropped = sanitize_patch(patch, TEST_PATCH)
    assert "pkg/__init__.py" in cleaned
    assert "conftest.py" not in cleaned and "test_add.py" not in cleaned
    assert sorted(dropped) == ["conftest.py", "tests/test_add.py"]


def test_sanitize_all_protected_returns_empty():
    cleaned, dropped = sanitize_patch(EDIT_TEST, TEST_PATCH)
    assert cleaned == "" and dropped == ["tests/test_add.py"]


def _junit(tests, failures=0, errors=0, skipped=0):
    return f'<testsuites><testsuite tests="{tests}" failures="{failures}" errors="{errors}" skipped="{skipped}"/></testsuites>'


def test_verify_outcome_pytest():
    assert verify_outcome("pytest", "", junit_xml=_junit(1))[0]
    assert not verify_outcome("pytest", "", junit_xml=_junit(1, skipped=1))[0]
    assert not verify_outcome("pytest", "", junit_xml=_junit(0))[0]
    assert not verify_outcome("pytest", "", junit_xml=_junit(2, failures=1))[0]
    assert not verify_outcome("pytest", "", junit_xml=None)[0]
    assert not verify_outcome("pytest", "", junit_xml="<broken")[0]


def test_verify_outcome_django():
    ok = "Ran 3 tests in 0.1s\n\nOK\n"
    assert verify_outcome("django", ok)[0]
    assert not verify_outcome("django", "Ran 0 tests in 0.0s\n\nOK\n")[0]
    assert not verify_outcome("django", "Ran 3 tests in 0.1s\n\nOK (skipped=3)\n")[0]
    assert not verify_outcome("django", "just printed OK\n")[0]


def test_infra_kill_classification():
    assert is_infra_kill(-9, "fail", "killed by signal 9")
    assert is_infra_kill(None, "timeout", "timed out")
    assert not is_infra_kill(1, "fail", "exit code 1")


# ------------------------------------------------------- end to end (real sandbox)
def test_legit_fix_resolves(task):
    r = score_submission(task, GOOD)
    assert r["resolved"] and r["test_status"] == "pass", r


@pytest.mark.parametrize("hack", [EDIT_TEST, CONFTEST_SKIP, EDIT_TEST + CONFTEST_SKIP])
def test_test_tampering_is_not_rewarded(task, hack):
    r = score_submission(task, hack)
    assert not r["resolved"] and not r["patch_valid"], r
    assert r["dropped_protected"]


def test_tampering_next_to_real_fix_still_scores_on_fix_only(task):
    # a correct fix plus a test edit: the test edit is ignored, the fix counts
    r = score_submission(task, GOOD + EDIT_TEST)
    assert r["resolved"] and r["dropped_protected"] == ["tests/test_add.py"]
    # a wrong source change plus a test edit must NOT pass
    r = score_submission(task, WHITESPACE_ONLY + EDIT_TEST)
    assert not r["resolved"]


def test_whitespace_or_comment_patch_is_not_valid(task):
    r = score_submission(task, WHITESPACE_ONLY)
    assert not r["resolved"] and not r["patch_valid"]
    assert "no-op" in r["reason"]


def test_exit_at_import_cannot_fake_a_pass(task):
    r = score_submission(task, EXIT_AT_IMPORT)
    assert not r["resolved"], r


def test_resource_kill_is_flagged_as_infra_not_a_plain_failure(task):
    r = score_submission(task, BUSY_LOOP, limits=Limits(cpu_seconds=2, wall_seconds=30))
    assert not r["resolved"]
    assert r["infra_error"] is True, r
