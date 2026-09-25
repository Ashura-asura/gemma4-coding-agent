"""Data-pipeline tests — synthetic fixtures only, no network (§2.1, §3.1, §8.3)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from data.scripts import build_holdout_split, build_rl_task_pool, build_sft_trajectories
from data.scripts import common
from data.scripts.common import (
    ensure_verified,
    jsonl_read,
    jsonl_write,
    normalise_instance,
    public_task,
    pythonpath_for,
    split_of,
    test_spec as build_test_spec,
    test_specs as build_test_specs,
    verified_record,
)


# ------------------------------------------------------------------ helpers
def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return result.stdout


@pytest.fixture()
def synthetic_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "upstream"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n", encoding="utf-8")
    (repo / "test_calc.py").write_text("from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n", encoding="utf-8")
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "t@example.com", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "base", cwd=repo)
    return repo


@pytest.fixture()
def instance(synthetic_repo: Path) -> dict:
    return {
        "instance_id": "syn-1",
        "repo": "synthetic/repo",
        "repo_url": str(synthetic_repo),
        "base_commit": _git("rev-parse", "HEAD", cwd=synthetic_repo).strip(),
        "version": "0",
        "issue": "add() subtracts",
        "hints": "",
        "gold_patch": (
            "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n"
            "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
        ),
        "test_patch": (
            "diff --git a/test_calc.py b/test_calc.py\n--- a/test_calc.py\n+++ b/test_calc.py\n"
            "@@ -1,5 +1,10 @@\n from calc import mul\n+from calc import add\n \n \n def test_mul():\n"
            "     assert mul(2, 3) == 6\n+\n+\n+def test_add():\n+    assert add(1, 1) == 2\n"
        ),
        "fail_to_pass": ["test_calc.py::test_add"],
        "pass_to_pass": ["test_calc.py::test_mul"],
        "source": "SWE-bench",
    }


# ------------------------------------------------------------------- splits
def test_split_is_deterministic_and_disjoint() -> None:
    ids = [f"django__django-{i}" for i in range(2000)]
    first = {i: split_of(i) for i in ids}
    assert all(first[i] == split_of(i) for i in ids)
    assert set(first.values()) == {"sft", "rl", "holdout"}
    counts = {name: sum(1 for v in first.values() if v == name) for name in ("sft", "rl", "holdout")}
    assert counts["sft"] > counts["rl"] >= counts["holdout"] > 0
    # 60 / 20 / 20 within a wide tolerance for hashing noise
    assert abs(counts["sft"] / len(ids) - 0.6) < 0.05
    assert abs(counts["holdout"] / len(ids) - 0.2) < 0.05


def test_split_of_single_bucket_boundaries() -> None:
    assert split_of("x", {"a": 100}) == "a"
    assert split_of("x", {"a": 0, "b": 100}) == "b"


# --------------------------------------------------------------- normalise
def test_normalise_instance_parses_both_f2p_formats() -> None:
    pytest_style = normalise_instance(
        {
            "instance_id": "a-1", "repo": "pallets/flask", "base_commit": "abc",
            "problem_statement": "x", "patch": "p", "test_patch": "t",
            "FAIL_TO_PASS": '["tests/test_x.py::test_y"]', "PASS_TO_PASS": ["z"],
        }
    )
    assert pytest_style["fail_to_pass"] == ["tests/test_x.py::test_y"]
    assert pytest_style["pass_to_pass"] == ["z"]
    assert pytest_style["repo_url"] == "https://github.com/pallets/flask.git"

    django_style = normalise_instance(
        {
            "instance_id": "d-1", "repo": "django/django", "base_commit": "abc",
            "problem_statement": "x", "patch": "p", "test_patch": "t",
            "FAIL_TO_PASS": '["test_new_thing (tests.foo.Bar)"]',
        }
    )
    assert django_style["fail_to_pass"] == ["test_new_thing (tests.foo.Bar)"]


# ---------------------------------------------------------------- test spec
def test_test_spec_django_pytest_and_pytest_k() -> None:
    django = build_test_spec(
        {"repo": "django/django", "fail_to_pass": ["test_foo (tests.bar.Baz)", "unparseable"],
         "test_patch": "--- a/tests/bar/tests.py\n+++ b/tests/bar/tests.py\n@@ -1 +1 @@\n-a\n+b\n",
         "pass_to_pass": []},
        python="PY",
    )
    assert django["runner"] == "django"
    assert django["base"][:1] == ["PY"]
    assert any("runtests.py" in arg for arg in django["base"])
    assert django["targets"] == ["tests.bar.Baz.test_foo"]  # method appended
    django_specs = build_test_specs(
        {"repo": "django/django", "fail_to_pass": ["test_foo (tests.bar.Baz)", "unparseable"],
         "test_patch": "--- a/tests/bar/tests.py\n+++ b/tests/bar/tests.py\n@@ -1 +1 @@\n-a\n+b\n",
         "pass_to_pass": []},
        python="PY",
    )
    assert len(django_specs) == 2  # labels first, touched files second
    assert "bar.tests" in django_specs[1]["targets"]

    sympy = build_test_spec(
        {"repo": "sympy/sympy", "fail_to_pass": ["test_solve", "test_limit"],
         "test_patch": "--- a/sympy/solvers/tests/test_solvers.py\n+++ b/sympy/solvers/tests/test_solvers.py\n@@ -1 +1 @@\n-a\n+b\n",
         "pass_to_pass": []},
        python="PY",
    )
    assert sympy["runner"] == "pytest-k"
    assert "-k" in sympy["base"] and "test_limit or test_solve" in sympy["base"]
    assert sympy["targets"] == ["sympy/solvers/tests/test_solvers.py"]

    flask = build_test_spec(
        {"repo": "pallets/flask", "fail_to_pass": ["tests/test_x.py::test_y"],
         "pass_to_pass": ["tests/test_a.py::test_b"], "test_patch": ""},
        python="PY",
    )
    assert flask["runner"] == "pytest"
    assert flask["targets"] == ["tests/test_x.py::test_y", "tests/test_a.py::test_b"]


def test_pythonpath_for_root_and_src(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    roots = pythonpath_for(tmp_path)
    assert roots == [str(tmp_path), str(tmp_path / "src")]
    plain = tmp_path / "plain"
    plain.mkdir()
    assert pythonpath_for(plain) == [str(plain)]


# --------------------------------------------------------------- verification
def test_verify_instance_fail_to_pass(instance: dict, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common, "ensure_venv", lambda repo: Path(sys.executable))
    outcome = common.verify_instance(instance, timeout=180.0, work_root=tmp_path / "verify")
    assert outcome["verified"] is True, outcome
    assert outcome["pre_status"] == "fail"
    assert outcome["post_status"] == "pass"


def test_verify_instance_rejects_when_gold_does_not_fix(instance: dict, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common, "ensure_venv", lambda repo: Path(sys.executable))
    instance = dict(instance, gold_patch=(
        "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n"
        "@@ -5,2 +5,2 @@\n def mul(a, b):\n-    return a * b\n+    return a * b * 1\n"
    ))
    outcome = common.verify_instance(instance, timeout=180.0, work_root=tmp_path / "verify")
    assert outcome["verified"] is False
    assert outcome["post_status"] in {"fail", "error"}


def test_ensure_verified_is_resumable(instance: dict, monkeypatch, tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    monkeypatch.setattr(common, "POOL_PATH", pool)
    calls: list[str] = []

    def fake_verify(inst, **_kwargs):
        calls.append(inst["instance_id"])
        if inst["instance_id"] == "bad":
            return {"verified": False, "reason": "gold does not pass"}
        evidence = {"verified": True, "test_runner": "pytest", "test_target": ["t"],
                    "test_k_expr": "", "pre_status": "fail", "post_status": "pass",
                    "duration_s": 1.0}
        return evidence

    monkeypatch.setattr(common, "verify_instance", fake_verify)
    cands = [dict(instance, instance_id="good"), dict(instance, instance_id="bad")]
    first = common.ensure_verified(cands)
    assert set(first) == {"good"}
    assert calls == ["good", "bad"]

    again = common.ensure_verified(cands)
    assert set(again) == {"good"}
    assert calls == ["good", "bad"]  # nothing re-runs

    retried = common.ensure_verified(cands, retry_failed=True)
    assert set(retried) == {"good"}
    assert calls.count("bad") == 2

    entries = jsonl_read(pool)
    assert {e["task_id"] for e in entries} == {"good", "bad"}
    assert [e for e in entries if e["task_id"] == "bad"][0]["verified"] is False


# ------------------------------------------------------------ task records
def test_public_task_has_no_gold(instance: dict) -> None:
    record = verified_record(instance, {
        "verified": True, "test_runner": "pytest", "test_target": ["test_calc.py::test_add"],
        "test_k_expr": "", "pre_status": "fail", "post_status": "pass", "duration_s": 1.0,
    })
    public = public_task(record)
    assert "gold_patch" not in public
    assert public["test_target"] == ["test_calc.py::test_add"]
    assert record["gold_patch"] == instance["gold_patch"]
    required = {"task_id", "repo", "repo_url", "base_commit", "issue", "test_patch", "venv"}
    assert required <= set(public)


def test_task_records_roundtrip_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    assert jsonl_write(path, [{"a": 1}, {"b": 2}]) == 2
    assert jsonl_read(path) == [{"a": 1}, {"b": 2}]
    assert jsonl_read(tmp_path / "missing.jsonl") == []


# ------------------------------------------------------------------ scripts
def _fake_pool(monkeypatch, verified_ids: set[str]):
    def fake_verify(candidates, **_kwargs):
        out = {}
        for inst in candidates:
            if inst["instance_id"] in verified_ids:
                rec = verified_record(inst, {
                    "verified": True, "test_runner": "pytest", "test_target": ["t"],
                    "test_k_expr": "", "pre_status": "fail", "post_status": "pass",
                    "duration_s": 1.0,
                })
                common.jsonl_append(common.POOL_PATH, rec)
                out[inst["instance_id"]] = rec
        return out

    monkeypatch.setattr(build_holdout_split, "ensure_verified", fake_verify)
    monkeypatch.setattr(build_rl_task_pool, "ensure_verified", fake_verify)
    monkeypatch.setattr(build_sft_trajectories, "ensure_verified", fake_verify)


@pytest.fixture()
def tiny_parquet(tmp_path: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for i in range(40):
        iid = f"pallets__flask-{i}"
        rows.append({
            "instance_id": iid,
            "repo": "pallets/flask",
            "base_commit": f"commit{i}",
            "problem_statement": f"issue {i}",
            "patch": f"gold {i}",
            "test_patch": f"tests {i}",
            "FAIL_TO_PASS": json.dumps([f"tests/test_x.py::test_y{i}"]),
            "PASS_TO_PASS": json.dumps([]),
        })
    path = tmp_path / "swebench_test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_build_holdout_writes_only_holdout_split(
    tiny_parquet: Path, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(common, "POOL_PATH", tmp_path / "pool.jsonl")
    monkeypatch.setattr(build_holdout_split, "POOL_PATH", tmp_path / "pool.jsonl")
    monkeypatch.setattr(build_holdout_split, "HOLDOUT_PATH", tmp_path / "holdout_tasks.jsonl")
    _fake_pool(monkeypatch, verified_ids={f"pallets__flask-{i}" for i in range(40)})

    rc = build_holdout_split.main(["--parquet", str(tiny_parquet)])
    assert rc == 0
    holdout = jsonl_read(tmp_path / "holdout_tasks.jsonl")
    assert holdout
    assert all(split_of(r["task_id"]) == "holdout" for r in holdout)
    assert all("gold_patch" not in r for r in holdout)

    # §8.3: a second run refuses to touch the frozen file
    assert build_holdout_split.main(["--parquet", str(tiny_parquet)]) == 1
    assert build_holdout_split.main(["--parquet", str(tiny_parquet), "--force"]) == 0


def test_build_rl_pool_excludes_holdout_and_gold(
    tiny_parquet: Path, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(common, "POOL_PATH", tmp_path / "pool.jsonl")
    monkeypatch.setattr(build_rl_task_pool, "RL_PATH", tmp_path / "rl_task_pool.jsonl")
    _fake_pool(monkeypatch, verified_ids={f"pallets__flask-{i}" for i in range(40)})

    rc = build_rl_task_pool.main(["--parquet", str(tiny_parquet)])
    assert rc == 0
    pool = jsonl_read(tmp_path / "rl_task_pool.jsonl")
    assert pool
    assert all(split_of(r["task_id"]) == "rl" for r in pool)
    assert all("gold_patch" not in r for r in pool)
    holdout_ids = {r["task_id"] for r in jsonl_read(tmp_path / "holdout_tasks.jsonl")}
    assert not holdout_ids & {r["task_id"] for r in pool}


def test_build_sft_replays_only_verified_sft_split(
    tiny_parquet: Path, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(common, "POOL_PATH", tmp_path / "pool.jsonl")
    monkeypatch.setattr(build_sft_trajectories, "SFT_PATH", tmp_path / "sft_trajectories.jsonl")
    _fake_pool(monkeypatch, verified_ids={f"pallets__flask-{i}" for i in range(40)})

    replayed: list[str] = []

    def fake_replay(record, **_kwargs):
        replayed.append(record["task_id"])
        return {"task_id": record["task_id"], "messages": [], "meta": {"steps": 4}}

    monkeypatch.setattr(build_sft_trajectories, "replay_trajectory", fake_replay)

    rc = build_sft_trajectories.main(["--parquet", str(tiny_parquet)])
    assert rc == 0
    trajectories = jsonl_read(tmp_path / "sft_trajectories.jsonl")
    assert trajectories
    assert replayed == [t["task_id"] for t in trajectories]
    assert all(split_of(t["task_id"]) == "sft" for t in trajectories)

    # resume: nothing replayed twice
    replayed.clear()
    assert build_sft_trajectories.main(["--parquet", str(tiny_parquet)]) == 0
    assert replayed == []
