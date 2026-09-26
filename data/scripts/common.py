"""Shared data-pipeline helpers — ARCHITECTURE.md §2.1, §3.1.

Responsibilities
  * read raw SWE-bench parquet (or any dataset with the same field names)
  * mirror GitHub repos once (``git clone --bare``) and materialise
    per-commit checkouts offline (``git archive``)
  * build one venv per repo with that repo's dependencies
  * verify an instance the way §3.1 requires: the gold patch must make the
    real tests pass in the sandbox, after failing without it
  * replay a verified instance through the tool interface into an SFT
    trajectory (§3.1 ``replay_via_tool_interface``)
  * deterministic, disjoint splits for sft / rl / holdout (§2.1)

Everything that *executes* code goes through :class:`sandbox.Sandbox`.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
# heavy directories can be relocated (e.g. WSL/ext4) while pool + outputs
# stay on the shared tree — overrides are plain env vars
MIRROR_DIR = Path(os.environ.get("SWE_MIRROR_DIR") or RAW_DIR / "mirrors")
WORK_DIR = Path(os.environ.get("SWE_WORK_DIR") or DATA_DIR / "work")
CHECKOUT_DIR = WORK_DIR / "checkouts"
VENV_DIR = Path(os.environ.get("SWE_VENV_DIR") or DATA_DIR / "venvs")
POOL_PATH = RAW_DIR / "verified_pool.jsonl"
SFT_PATH = DATA_DIR / "sft_trajectories.jsonl"
RL_PATH = DATA_DIR / "rl_task_pool.jsonl"
HOLDOUT_PATH = DATA_DIR / "holdout_tasks.jsonl"

#: Splits are disjoint (§2.1) and assigned by hashing the task id, so the
#: assignment never changes as the pool grows.
SPLIT_RATIOS: dict[str, int] = {"sft": 60, "rl": 20, "holdout": 20}

#: Repos whose checkout imports cleanly without compiling C extensions.
#: astropy / matplotlib / scikit-learn are excluded on purpose: their tests
#: import built extensions from the checkout, which we cannot build here.
REPO_SETUP: dict[str, dict[str, Any]] = {
    "django/django": {
        "deps": ["asgiref", "sqlparse", "pytz"],
        "runner": "django",
        "editable": False,
        "shims": ["gettext_codeset", "collections_abc"],
    },
    "sympy/sympy": {"deps": ["mpmath"], "runner": "pytest-k", "editable": False,
                    "shims": ["collections_abc"]},
    "sphinx-doc/sphinx": {
        "deps": [
            "jinja2", "pygments", "docutils", "babel", "imagesize", "alabaster",
            "snowballstemmer", "requests", "setuptools", "packaging",
            "sphinxcontrib-applehelp", "sphinxcontrib-devhelp",
            "sphinxcontrib-jsmath", "sphinxcontrib-htmlhelp",
            # last releases that still support sphinx 2-4 (new ones demand v5+)
            "sphinxcontrib-applehelp==1.0.4", "sphinxcontrib-devhelp==1.0.2",
            "sphinxcontrib-htmlhelp==2.0.0", "sphinxcontrib-qthelp==1.0.3",
            "sphinxcontrib-serializinghtml==1.1.5",
            "roman", "defusedxml", "packaging",
            "jinja2<3.1", "docutils==0.17.1", "alabaster==0.7.13",
        ],
        "runner": "pytest",
        "editable": False,
        "shims": ["types_union"],
    },
    "pytest-dev/pytest": {
        # setuptools-scm writes src/_pytest/_version.py during the -e build
        # six / more-itertools / importlib-metadata / atomicwrites: imported
        # by the old checkouts (4.x-5.x era, Windows) but no longer installed
        "deps": ["iniconfig", "packaging", "pluggy", "exceptiongroup", "tomli", "attrs", "py",
                 "setuptools-scm", "toml", "six", "more-itertools", "importlib-metadata",
                 "atomicwrites"],
        "runner": "pytest",
        "editable": True,  # src/ layout: the checkout must be importable
    },
    # np.unicode_ died in numpy 2.0, pandas 2.0 dropped what old xarray uses
    "pydata/xarray": {"deps": ["numpy<2", "pandas==1.5.3", "pytz", "packaging"], "runner": "pytest", "editable": False},
    "pylint-dev/pylint": {
        # astroid 3.x dropped APIs (TryExcept etc.) old pylint still imports;
        # 2.13.7 is the last 2.x with python 3.11 support
        "deps": ["astroid==2.13.5", "isort", "toml", "tomlkit", "dill", "mccabe",
                 "platformdirs", "appdirs", "pytest==7.4.4"],
        "shims": ["collections_abc"],
        "runner": "pytest",
        "editable": False,
    },
    "psf/requests": {
        # urllib3 2.x dropped SNIMissingWarning etc. old tests import;
        # pre-2.26 requests imports chardet directly
        "deps": ["urllib3==1.26.20", "idna", "certifi", "charset-normalizer", "chardet"],
        "runner": "pytest",
        "editable": False,  # src/ layout is covered by the sandbox pythonpath
        # old requests vendors urllib3 1.x, which still does `from collections import ...`
        "shims": ["collections_abc"],
    },
    "pallets/flask": {
        # 2.3 removed url_quote from werkzeug.urls (needs werkzeug<2.3 for <=2.2);
        # jinja2 3.1 removed escape/Markup-era APIs some flask 2.x still use
        # old conftest uses _pytest.monkeypatch.notset, gone in pytest 8
        "deps": ["click==8.1.7", "jinja2==3.0.3", "werkzeug==2.2.3",
                 "itsdangerous==2.1.2", "blinker", "pytest==7.4.4"],
        "runner": "pytest",
        "editable": False,  # src/ layout is covered by the sandbox pythonpath
    },
    # matplotlib.cm.register_cmap (used by these seaborn versions) died in 3.9
    "mwaskom/seaborn": {
        # seaborn 0.11/0.12 was developed against the 2022 stack — newer
        # matplotlib/scipy changes clustermap colors and mask handling
        "deps": ["numpy<2", "pandas==1.5.3", "matplotlib==3.6.3", "scipy==1.10.1"],
        "runner": "pytest",
        "editable": False,
    },
}

#: sphinx >= 5's ``addnodes`` imports ``docutils.nodes.meta`` (added in
#: docutils 0.18) while sphinx < 5 breaks on docutils >= 0.18 — one venv
#: per docutils era, selected from the instance's sphinx version.
SPHINX_DOCUTILS_TIERS: dict[str, str] = {
    "d18": "docutils==0.18.1",  # sphinx 5.x
    "d19": "docutils==0.19",  # sphinx 6.x+
}

#: ``sitecustomize.py`` bodies installed into a repo venv. They restore
#: APIs the *code under test* was written against but modern Python dropped
#: (``codeset``, ``collections.Mapping``) — plain environment setup: every
#: pre/post run sees the same interpreter, no test behaviour is touched.
VENV_SHIMS: dict[str, str] = {
    "gettext_codeset": '''\
import gettext as _gettext

_orig_translation = _gettext.translation


def _translation(domain, localedir=None, languages=None, class_=None,
                 fallback=False, codeset=None, **kwargs):
    return _orig_translation(domain, localedir, languages, class_, fallback)


_gettext.translation = _translation
if not hasattr(_gettext.GNUTranslations, "set_output_charset"):
    _gettext.GNUTranslations.set_output_charset = lambda self, charset: None
if not hasattr(_gettext.NullTranslations, "set_output_charset"):
    _gettext.NullTranslations.set_output_charset = lambda self, charset: None
''',
    "collections_abc": '''\
import collections as _collections
import collections.abc as _collections_abc

# Python 3.10 dropped the ABC aliases in ``collections``; older sympy still
# says ``from collections import Mapping``.
for _name in ("Mapping", "MutableMapping", "Callable", "Iterable", "Iterator",
              "Sequence", "MutableSequence", "Set", "MutableSet", "Hashable"):
    if not hasattr(_collections, _name):
        setattr(_collections, _name, getattr(_collections_abc, _name))
''',
    "types_union": '''\
import types as _types

# Sphinx 4.1-4.3 guarded ``from types import Union`` behind sys.version_info
# > (3, 10) — a name that never shipped. Alias the real PEP 604 type.
if not hasattr(_types, "Union"):
    _types.Union = getattr(_types, "UnionType", None) or __import__(
        "typing", fromlist=["Union"]).Union
''',
}

DEFAULT_TEST_TIMEOUT_S = 300.0


def _say(text: str, **_ignored: Any) -> None:
    """Print without ever crashing on test output the console can't encode."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe, flush=True)


# ------------------------------------------------------------------ jsonl io
def jsonl_read(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def jsonl_append(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def jsonl_write(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# -------------------------------------------------------------------- splits
def split_of(task_id: str, ratios: dict[str, int] | None = None) -> str:
    """Deterministic split assignment: sha1(task_id) -> bucket in [0, 100)."""
    ratios = ratios or SPLIT_RATIOS
    bucket = int(hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    cursor = 0
    for name, width in ratios.items():
        cursor += width
        if bucket < cursor:
            return name
    return next(iter(ratios))


def slugify(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo.strip("/").removesuffix(".git"))


# ----------------------------------------------------------------- raw input
def load_instances(
    parquet_path: str | Path,
    *,
    repos: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load SWE-bench-style parquet rows with normalised list fields."""
    import pyarrow.parquet as pq

    allowed = set(repos) if repos else None
    instances: list[dict[str, Any]] = []
    for row in pq.read_table(str(parquet_path)).to_pylist():
        if allowed is not None and row.get("repo") not in allowed:
            continue
        instances.append(normalise_instance(row))
        if limit is not None and len(instances) >= limit:
            break
    return instances


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return [value] if value.strip() else []
        return [str(x) for x in parsed] if isinstance(parsed, list) else [str(parsed)]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]


def normalise_instance(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "repo_url": f"https://github.com/{row['repo']}.git",
        "base_commit": row["base_commit"],
        "version": row.get("version"),
        "issue": row.get("problem_statement") or "",
        "hints": row.get("hints_text") or "",
        "gold_patch": row.get("patch") or "",
        "test_patch": row.get("test_patch") or "",
        "fail_to_pass": _as_list(row.get("FAIL_TO_PASS")),
        "pass_to_pass": _as_list(row.get("PASS_TO_PASS")),
        "source": "SWE-bench",
    }


# --------------------------------------------------------------------- git
def _run(cmd: list[str], *, cwd: Path | None = None, timeout: float = 600.0,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _is_remote(repo_url: str) -> bool:
    return repo_url.startswith(("http://", "https://", "git@"))


def ensure_mirror(repo_url: str, *, refresh: bool = False) -> Path:
    """Bare clone cached under data/raw/mirrors (offline afterwards).

    Remote URLs are cached permanently; a local path (synthetic test repo)
    is cloned one-shot and removed again by :func:`checkout`.
    """
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    if not _is_remote(repo_url):
        tmp = Path(tempfile.mkdtemp(prefix="local-", dir=MIRROR_DIR))
        dest = tmp / "mirror.git"
        result = _run(["git", "clone", "--bare", "--quiet", repo_url, str(dest)], timeout=600)
        if result.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError(f"git clone failed for {repo_url}: {result.stderr.decode(errors='replace')[:400]}")
        return dest
    slug = slugify(repo_url.replace("https://", "").replace("http://", ""))
    dest = MIRROR_DIR / f"{slug}.git"
    if dest.exists() and not refresh:
        return dest
    tmp = MIRROR_DIR / f"{slug}.partial.git"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    result = _run(["git", "clone", "--bare", "--quiet", repo_url, str(tmp)], timeout=1800)
    if result.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"git clone failed for {repo_url}: {result.stderr.decode(errors='replace')[:400]}")
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest


def refresh_mirror(repo_url: str) -> Path:
    mirror = ensure_mirror(repo_url)
    _run(["git", "--git-dir", str(mirror), "fetch", "--quiet", "--all", "+refs/*:refs/*"], timeout=1800)
    return mirror


def checkout(repo_url: str, commit: str, dest: str | Path) -> Path:
    """Materialise ``commit`` at ``dest`` from the local mirror (no network)."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    mirror = ensure_mirror(repo_url)
    try:
        # core.autocrlf (true on this machine) otherwise makes `git archive`
        # emit CRLF trees; SWE-bench patches are generated against LF blobs
        result = _run(
            ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
             "--git-dir", str(mirror), "archive", "--format=tar", commit],
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git archive {commit[:10]} failed: {result.stderr.decode(errors='replace')[:300]}"
            )
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
            try:
                tar.extractall(dest, filter="data")  # PEP 706; backported to 3.11.4
            except TypeError:  # pragma: no cover - very old Pythons
                tar.extractall(dest)
    finally:
        if not _is_remote(repo_url):
            shutil.rmtree(mirror.parent, ignore_errors=True)
    return dest


def checkout_instance(repo_url: str, commit: str, dest: str | Path) -> Path:
    return checkout(repo_url, commit, dest)


# --------------------------------------------------------------------- venv
def base_python() -> str:
    """Interpreter used to create repo venvs.

    SWE-bench base commits date from 2015-2024, so prefer a 3.11 over the
    system default: ``cgi`` and friends are gone in 3.13+.
    """
    preferred = os.environ.get("SWE_BASE_PYTHON")
    if preferred and Path(preferred).exists():
        return preferred
    for candidate in ("python3.11", "python3.10", "python3.9"):
        found = shutil.which(candidate)
        if found:
            return found
    return sys.executable


def venv_python(slug: str) -> Path:
    venv = VENV_DIR / slug
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _write_shims(slug: str, setup: dict[str, Any]) -> None:
    names = [n for n in setup.get("shims") or [] if n in VENV_SHIMS]
    if not names:
        return
    venv = VENV_DIR / slug
    if os.name == "nt":
        site = venv / "Lib" / "site-packages"
    else:
        candidates = sorted((venv / "lib").glob("python*/site-packages"))
        if not candidates:
            return
        site = candidates[0]
    site.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(VENV_SHIMS[name] for name in names)
    tmp = site / "sitecustomize.py.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, site / "sitecustomize.py")


def ensure_venv(repo: str, *, tier: str | None = None) -> Path:
    """Create (once) and populate the venv for ``repo``. Returns its python.

    ``tier`` selects an alternate dependency pin within the repo (see
    :data:`SPHINX_DOCUTILS_TIERS`); tiers get their own directory + marker.
    """
    setup = REPO_SETUP.get(repo, {"deps": []})
    deps = list(setup.get("deps", []))
    shims = list(setup.get("shims") or [])
    suffix = ""
    if tier:
        suffix = f"-{tier}"
        deps = [d for d in deps if not d.startswith("docutils")]
        deps.append(SPHINX_DOCUTILS_TIERS[tier])
    slug = slugify(repo) + suffix
    py = venv_python(slug)
    if not py.exists():
        VENV_DIR.mkdir(parents=True, exist_ok=True)
        result = _run([base_python(), "-m", "venv", str(VENV_DIR / slug)], timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"venv creation failed for {repo}: {result.stderr.decode(errors='replace')[:400]}")
    marker = VENV_DIR / f"{slug}.deps.ok"
    if not marker.exists():
        # tzdata: Windows has no system IANA zone database for zoneinfo
        deps = ["pytest", "setuptools", "wheel", "tzdata", *deps]
        result = _run([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", *deps], timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(f"pip install failed for {repo}: {result.stderr.decode(errors='replace')[-600:]}")
        marker.write_text("ok", encoding="utf-8")
    _write_shims(slug, {**setup, "shims": shims})
    return py


def sphinx_tier_for(version: Any) -> str | None:
    """Docutils-era tier for a sphinx instance version, or None for default."""
    match = re.match(r"(\d+(?:\.\d+)?)", str(version or ""))
    if not match:
        return None
    value = float(match.group(1))
    if value >= 6:
        return "d19"
    if value >= 5:
        return "d18"
    return None


def ensure_venv_for_instance(instance: dict[str, Any]) -> tuple[str, Path]:
    """``(venv_slug, python)`` matched to this instance's era, if any."""
    repo = instance["repo"]
    if repo == "sphinx-doc/sphinx":
        tier = sphinx_tier_for(instance.get("version"))
        if tier:
            slug = slugify(repo) + f"-{tier}"
            return slug, ensure_venv(repo, tier=tier)
    return slugify(repo), ensure_venv(repo)


def ensure_venv_by_slug(slug: str) -> Path:
    """Ensure the venv a *task record* refers to (``venv`` field) exists."""
    for repo in REPO_SETUP:
        base = slugify(repo)
        if base == slug:
            return ensure_venv(repo)
        if repo == "sphinx-doc/sphinx" and slug.startswith(base + "-d"):
            tier = slug[len(base) + 1:]
            if tier in SPHINX_DOCUTILS_TIERS:
                return ensure_venv(repo, tier=tier)
    py = venv_python(slug)
    if not py.exists():
        VENV_DIR.mkdir(parents=True, exist_ok=True)
        result = _run([base_python(), "-m", "venv", str(VENV_DIR / slug)], timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"venv creation failed for {slug}: {result.stderr.decode(errors='replace')[:400]}")
    marker = VENV_DIR / f"{slug}.deps.ok"
    if not marker.exists():
        result = _run(
            [str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", "pytest", "setuptools", "wheel"],
            timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pip install failed for {slug}: {result.stderr.decode(errors='replace')[-600:]}")
        marker.write_text("ok", encoding="utf-8")
    return py


def prepare_checkout(repo: str, directory: str | Path) -> None:
    """For src-layout repos, point the venv at *this* checkout (``-e .``)."""
    if not REPO_SETUP.get(repo, {}).get("editable"):
        return
    py = venv_python(slugify(repo))
    # checkouts are git-archive extracts (no .git), so setuptools-scm can't
    # derive a version — hand it a stable one instead of failing the build.
    build_env = dict(os.environ)
    build_env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "6.0.0")
    build_env.setdefault("VCS_VERSIONING_PRETEND_VERSION", "6.0.0")
    result = _run(
        [str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check",
         "--no-deps", "--no-build-isolation", "--no-index", "-e", str(directory)],
        timeout=600,
        env=build_env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"editable install failed: {result.stderr.decode(errors='replace')[-600:]}")


def pythonpath_for(directory: str | Path) -> list[str]:
    """Import roots for a checkout: its root plus ``src/`` when present.

    Passed through the sandbox's sanctioned ``pythonpath`` argument so
    src-layout repos (flask, requests) import from *this* checkout instead of
    site-packages — without ever exposing ``PYTHONPATH`` to ``extra_env``.
    """
    root = Path(directory)
    roots = [root]
    if (root / "src").is_dir():
        roots.append(root / "src")
    return [str(p) for p in roots]


# --------------------------------------------------------------- test spec
_DJANGO_LABEL = re.compile(r"^(\w+) \(([\w.]+)\)$")


def test_files_of(patch: str) -> list[str]:
    from agent.tools.edit_file import split_multifile_patch

    return [path for path in split_multifile_patch(patch) if path.endswith(".py")]


def django_label_for(test_file: str) -> str:
    """tests/migrations/test_writer.py -> migrations.test_writer"""
    parts = Path(test_file).parts
    if parts and parts[0] == "tests":
        parts = parts[1:]
    stem = Path(*parts) if parts else Path(test_file)
    return str(stem.with_suffix("")).replace("\\", ".").replace("/", ".")


def _fit(items: list[str], budget: int = 6000) -> list[str]:
    """Keep argv comfortably below Windows' 32KB command-line limit."""
    kept: list[str] = []
    used = 0
    for item in items:
        if used + len(item) + 1 > budget:
            break
        kept.append(item)
        used += len(item) + 1
    return kept


def _django_labels(f2p: list[str]) -> list[str]:
    """SWE-bench django F2P entries -> ``runtests.py`` labels.

    Classic records ``name (module.Class)`` and modern records
    ``name (module.Class.name)`` both become ``module.Class.name``.
    Docstring-style tests ("Named URLs should be reversible") are not
    addressable on a CLI and are dropped.
    """
    labels = []
    for entry in f2p:
        match = _DJANGO_LABEL.match(entry)
        if not match:
            continue
        name, qualname = match.group(1), match.group(2)
        labels.append(qualname if qualname.endswith("." + name) else f"{qualname}.{name}")
    return labels


def _django_label_to_file(label: str) -> str:
    """``auth_tests.test_validators.Cls.test_x`` -> ``tests/auth_tests/test_validators.py``"""
    parts = label.split(".")
    module = parts[: len(parts) - 2] if len(parts) >= 3 else parts[:1]
    return "tests/" + "/".join(module) + ".py"


def test_specs(instance: dict[str, Any], *, python: Path | str, p2p_cap: int = 0) -> list[dict[str, Any]]:
    """Candidate (runner, targets) specs for one instance, best first.

    Verification escalates: if the precise target set already passes
    without the gold patch, the coarser file-level spec is tried before the
    instance is rejected.
    """
    repo = instance["repo"]
    setup = REPO_SETUP.get(repo, {"runner": "pytest"})
    runner = setup.get("runner", "pytest")
    py = str(python)
    f2p = list(instance.get("fail_to_pass") or [])
    p2p = list(instance.get("pass_to_pass") or [])
    if p2p_cap:
        p2p = sorted(p2p)[:p2p_cap]
    test_files = test_files_of(instance.get("test_patch", ""))

    if runner == "django":
        django_base = [py, "tests/runtests.py", "--verbosity", "1", "--parallel", "1"]
        raw_labels = _django_labels(f2p)
        specs: list[dict[str, Any]] = []
        if raw_labels:
            specs.append({"runner": "django", "base": django_base,
                          "targets": _fit(sorted(set(raw_labels))), "k_expr": ""})
        files = sorted(set(_django_label_to_file(l) for l in raw_labels) | set(test_files))
        file_labels = _fit(sorted(set(django_label_for(f) for f in files)))
        if file_labels and (not specs or file_labels != specs[0]["targets"]):
            specs.append({"runner": "django", "base": django_base,
                          "targets": file_labels, "k_expr": ""})
        if not specs:
            specs.append({"runner": "django", "base": django_base, "targets": [], "k_expr": ""})
        return specs

    if runner == "pytest-k":
        # sympy records bare test function names, not node ids
        expr = " or ".join(_fit(sorted(set(f2p)), budget=5000))
        base = [py, "-m", "pytest", "-q"]
        if expr:
            base += ["-k", expr]
        specs = [{"runner": "pytest-k", "base": base, "targets": test_files, "k_expr": expr}]
        if test_files:
            specs.append({"runner": "pytest-k", "base": [py, "-m", "pytest", "-q"],
                          "targets": test_files, "k_expr": ""})
        return specs

    f2p_set, p2p_set = set(f2p), set(p2p)
    specs = [{"runner": "pytest", "base": [py, "-m", "pytest", "-q"],
              "targets": _fit(sorted(f2p_set) + sorted(p2p_set - f2p_set)), "k_expr": ""}]
    if test_files:
        file_targets = _fit(sorted(set(test_files)))
        if file_targets and file_targets != specs[0]["targets"]:
            specs.append({"runner": "pytest", "base": [py, "-m", "pytest", "-q"],
                          "targets": file_targets, "k_expr": ""})
    return specs


def test_spec(instance: dict[str, Any], *, python: Path | str, p2p_cap: int = 0) -> dict[str, Any]:
    """Primary spec for an instance (see :func:`test_specs`)."""
    return test_specs(instance, python=python, p2p_cap=p2p_cap)[0]


def spec_argv(spec: dict[str, Any]) -> list[str]:
    return list(spec["base"]) + list(spec["targets"])


# --------------------------------------------------------------- verification
_FAILED_LINE = re.compile(r"^(?:FAILED|ERROR) (\S+)")
_VERBOSE_LINE = re.compile(r"^(\S+::\S+) (PASSED|FAILED|ERROR)\b")


def _test_results(output: str) -> tuple[set[str], set[str]]:
    """``(passed, failed)`` test ids from pytest output.

    Reads both the short summary (``FAILED <nodeid>``) and verbose per-test
    lines (``<nodeid> PASSED``) so that skipped/never-run tests are visible
    as neither.
    """
    passed: set[str] = set()
    failed: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        match = _FAILED_LINE.match(line)
        if match:
            failed.add(match.group(1).replace("\\", "/"))
            continue
        match = _VERBOSE_LINE.match(line)
        if match:
            test_id = match.group(1).replace("\\", "/")
            (failed if match[2] in ("FAILED", "ERROR") else passed).add(test_id)
    return passed, failed


def verify_instance(
    instance: dict[str, Any],
    *,
    limits: Any = None,
    timeout: float = DEFAULT_TEST_TIMEOUT_S,
    work_root: str | Path | None = None,
    keep: bool = False,
) -> dict[str, Any]:
    """Fail-to-pass verification of one instance, entirely inside the sandbox.

    Returns a dict with ``verified`` plus the evidence (§0.1: the reward has
    to reduce to "did a real test suite pass").
    """
    from sandbox import Limits, Sandbox

    from agent.tools.edit_file import PatchError, apply_patch_to_tree

    started = time.monotonic()
    repo = instance["repo"]
    workdir = Path(work_root or WORK_DIR / "verify") / instance["instance_id"] / "repo"
    outcome: dict[str, Any] = {
        "instance_id": instance["instance_id"],
        "verified": False,
        "reason": "",
        "pre_status": None,
        "post_status": None,
    }

    def _finish(reason: str) -> dict[str, Any]:
        outcome["reason"] = reason
        outcome["duration_s"] = round(time.monotonic() - started, 1)
        if not keep:
            shutil.rmtree(workdir.parent, ignore_errors=True)
        return outcome

    try:
        checkout(instance["repo_url"], instance["base_commit"], workdir)
        venv_slug, python = ensure_venv_for_instance(instance)
        outcome["venv"] = venv_slug
        prepare_checkout(repo, workdir)
        specs = [s for s in test_specs(instance, python=python) if s["targets"]]
    except Exception as exc:
        return _finish(f"setup failed: {type(exc).__name__}: {exc}")

    if not specs:
        return _finish("no usable test targets")

    sandbox = Sandbox(workdir, limits=limits or Limits(cpu_seconds=60, wall_seconds=timeout))
    pypath = pythonpath_for(workdir)

    try:
        if instance["test_patch"]:
            apply_patch_to_tree(instance["test_patch"], workdir)
    except PatchError as exc:
        return _finish(f"test patch does not apply: {exc}")

    # --- pre-patch: find a spec whose tests actually fail right now
    chosen: dict[str, Any] | None = None
    last_reason = "pre-patch tests passed; nothing to verify"
    collection_broken = False
    for spec in specs:
        try:
            pre = sandbox.run(spec_argv(spec), timeout=timeout, pythonpath=pypath)
        except Exception as exc:  # noqa: BLE001 - the runner itself blew up
            last_reason = f"pre-patch runner crashed: {type(exc).__name__}: {exc}"
            continue
        outcome["pre_status"] = pre.status
        outcome["pre_returncode"] = pre.returncode
        if pre.status == "timeout":
            return _finish("pre-patch tests timed out")
        if pre.status == "error":
            last_reason = f"pre-patch run error: {pre.message}"
            continue
        pre_out = pre.stdout + "\n" + pre.stderr
        # the test patch can exercise APIs the gold patch introduces: the
        # whole module failing to import is a legitimate pre-patch failure
        # (SWE-bench counts collection errors as failures)
        collection_broken = pre.returncode == 4 and any(
            marker in pre_out
            for marker in ("ERROR collecting", "ImportError while importing test module",
                           "ImportError while loading conftest", "errors during collection")
        )
        if spec["runner"] == "pytest" and pre.returncode != 1 and not collection_broken:
            last_reason = f"pre-patch tests did not fail as expected (rc={pre.returncode})"
            continue
        if spec["runner"] != "pytest" and (pre.returncode or 0) == 0:
            last_reason = "pre-patch tests passed; nothing to verify"
            continue
        chosen = spec
        break

    if chosen is None:
        return _finish(last_reason)
    argv = spec_argv(chosen)

    # SWE-bench-style judgement when the spec runs exact node ids: every
    # fail-to-pass test has to flip (§2.1), pass-to-pass tests that are
    # broken in *both* runs count as environment drift and are tolerated —
    # gold/model patches may never break a test that passed pre-patch.
    f2p = {str(t).replace("\\", "/") for t in (instance.get("fail_to_pass") or [])}
    _, failed_pre = _test_results(pre_out)
    if collection_broken:
        failed_pre |= f2p
    node_spec = (
        chosen["runner"] == "pytest"
        and bool(f2p)
        and bool(failed_pre)
        and f2p <= {str(t).replace("\\", "/") for t in chosen["targets"]}
    )
    if node_spec and not (f2p & failed_pre):
        return _finish("fail-to-pass tests already pass pre-patch; cannot verify the flip")

    try:
        apply_patch_to_tree(instance["gold_patch"], workdir)
    except PatchError as exc:
        return _finish(f"gold patch does not apply: {exc}")

    try:
        # -q already counted once in the spec base; two -v's put the run in
        # verbose mode so every test id (including passes) shows up
        post_argv = argv + ["-v", "-v"] if node_spec else argv
        post = sandbox.run(post_argv, timeout=timeout, pythonpath=pypath)
    except Exception as exc:  # noqa: BLE001 - the runner itself blew up
        return _finish(f"post-patch runner crashed: {type(exc).__name__}: {exc}")
    outcome["post_status"] = post.status
    outcome["post_returncode"] = post.returncode

    if node_spec:
        passed_post, failed_post = _test_results(post.stdout + "\n" + post.stderr)
        if post.returncode not in (0, 1):
            tail = (post.stdout + "\n" + post.stderr)[-500:].replace("\n", " ")
            return _finish(f"post-patch tests did not run cleanly (rc={post.returncode}): {tail}")
        not_passing = f2p - passed_post
        if not_passing:
            return _finish("fail-to-pass tests not proven post-patch (missing or failed): "
                           + ", ".join(sorted(not_passing))[:300])
        broken = failed_post - failed_pre
        if broken:
            return _finish("tests broken by the gold patch: "
                           + ", ".join(sorted(broken))[:300])
        outcome["env_drift"] = sorted((failed_post & failed_pre) - f2p)
    elif post.status != "pass":
        tail = (post.stdout + "\n" + post.stderr)[-500:].replace("\n", " ")
        return _finish(f"post-patch tests did not pass ({post.status}, rc={post.returncode}): {tail}")

    outcome["verified"] = True
    outcome["reason"] = "fail-to-pass verified"
    outcome["test_runner"] = chosen["runner"]
    outcome["test_target"] = chosen["targets"]
    outcome["test_k_expr"] = chosen["k_expr"]
    outcome["duration_s"] = round(time.monotonic() - started, 1)
    if not keep:
        shutil.rmtree(workdir.parent, ignore_errors=True)
    return outcome


def verified_record(instance: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Pool record: everything needed to rebuild the task, gold kept aside."""
    return {
        "task_id": instance["instance_id"],
        "verified": True,
        "repo": instance["repo"],
        "repo_url": instance["repo_url"],
        "base_commit": instance["base_commit"],
        "version": instance["version"],
        "issue": instance["issue"],
        "test_patch": instance["test_patch"],
        "gold_patch": instance["gold_patch"],
        "test_runner": evidence["test_runner"],
        "test_target": evidence["test_target"],
        "test_k_expr": evidence.get("test_k_expr"),
        "venv": evidence.get("venv") or slugify(instance["repo"]),
        "source": instance.get("source", "SWE-bench"),
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verify": {
            "pre_status": evidence["pre_status"],
            "post_status": evidence["post_status"],
            "duration_s": evidence["duration_s"],
        },
    }


def ensure_venvs(repos: Iterable[str]) -> None:
    """Create every repo venv up-front (single writer, no pip races)."""
    for repo in sorted(set(repos)):
        _say(f"[venv] {repo} ...", flush=True)
        ensure_venv(repo)
        if repo == "sphinx-doc/sphinx":
            for tier in SPHINX_DOCUTILS_TIERS:
                _say(f"[venv] {repo} ({tier}) ...", flush=True)
                ensure_venv(repo, tier=tier)


def _verify_shard(payload: tuple[list[dict[str, Any]], float, bool]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Worker: verify one shard of candidates (child process, spawned)."""
    instances, timeout, keep = payload
    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for instance in instances:
        evidence = verify_instance(instance, timeout=timeout, keep=keep)
        results.append((instance, evidence))
    return results


def _shard_candidates(candidates: list[dict[str, Any]], jobs: int) -> list[list[dict[str, Any]]]:
    """Split pending work across workers, one *continuous* slice per worker.

    Repos with an editable install (``pytest-dev/pytest``) are never split:
    two concurrent episodes would race over the venv's single ``.pth`` slot.
    """
    if jobs <= 1:
        return [candidates]
    groups: list[list[dict[str, Any]]] = []
    for repo in sorted({c["repo"] for c in candidates}):
        items = [c for c in candidates if c["repo"] == repo]
        shards = 1 if REPO_SETUP.get(repo, {}).get("editable") else min(jobs, len(items))
        if shards <= 1:
            groups.append(items)
        else:
            size = -(-len(items) // shards)
            groups.extend(items[i:i + size] for i in range(0, len(items), size))
    return groups


def ensure_verified(
    candidates: Iterable[dict[str, Any]],
    *,
    limit: int | None = None,
    retry_failed: bool = False,
    timeout: float = DEFAULT_TEST_TIMEOUT_S,
    keep: bool = False,
    jobs: int = 1,
) -> dict[str, dict[str, Any]]:
    """Verify candidates (fail-to-pass in the sandbox), resumably.

    Every attempt — pass or reject — is appended to
    ``data/raw/verified_pool.jsonl`` so a rerun skips what it already did
    (``--retry-failed`` re-checks rejects). Returns the verified pool
    records for these candidates, keyed by task id. ``jobs`` shards the
    pending instances across spawned workers (see :func:`_shard_candidates`).
    """
    done: dict[str, dict[str, Any]] = {}
    rejected: set[str] = set()
    for entry in jsonl_read(POOL_PATH):
        tid = str(entry.get("task_id"))
        if entry.get("verified"):
            done[tid] = entry
        else:
            rejected.add(tid)

    pending: list[dict[str, Any]] = []
    for instance in candidates:
        tid = instance["instance_id"]
        if tid in done:
            continue
        if tid in rejected and not retry_failed:
            continue
        pending.append(instance)
        if limit is not None and len(pending) >= limit:
            break

    if not pending:
        wanted = {i["instance_id"] for i in candidates}
        return {tid: record for tid, record in done.items() if tid in wanted}

    def _record(instance: dict[str, Any], evidence: dict[str, Any]) -> None:
        tid = instance["instance_id"]
        if evidence["verified"]:
            record = verified_record(instance, evidence)
            jsonl_append(POOL_PATH, record)
            done[tid] = record
            _say(f"  ok    {evidence['duration_s']}s", flush=True)
        else:
            jsonl_append(
                POOL_PATH,
                {"task_id": tid, "verified": False, "reason": evidence["reason"]},
            )
            rejected.add(tid)
            _say(f"  skip  {evidence['reason']}", flush=True)

    total = len(pending)
    if jobs <= 1:
        for index, instance in enumerate(pending, start=1):
            _say(f"[verify] {instance['instance_id']} ({index}/{total})", flush=True)
            _record(instance, verify_instance(instance, timeout=timeout, keep=keep))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        shards = _shard_candidates(pending, jobs)
        _say(f"[verify] {total} pending -> {len(shards)} shard(s), {jobs} worker(s)", flush=True)
        finished = 0
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_verify_shard, (shard, timeout, keep)) for shard in shards]
            for future in as_completed(futures):
                try:
                    results = future.result()
                except Exception as exc:  # noqa: BLE001 - keep the other shards alive
                    _say(f"  shard failed: {type(exc).__name__}: {exc}", flush=True)
                    continue
                for instance, evidence in results:
                    finished += 1
                    _say(f"[verify] {instance['instance_id']} ({finished}/{total})", flush=True)
                    _record(instance, evidence)

    wanted = {i["instance_id"] for i in candidates}
    return {tid: record for tid, record in done.items() if tid in wanted}


# --------------------------------------------------------------- trajectories
def replay_trajectory(record: dict[str, Any], *, limits: Any = None, timeout: float = DEFAULT_TEST_TIMEOUT_S) -> dict[str, Any] | None:
    """Replay a verified task through the tool interface (§3.1).

    read -> edit (gold, one call per file) -> run_tests -> submit. The
    trajectory is kept only if the sandboxed test run passes with no tool
    errors.
    """
    from agent.context import ToolCall
    from agent.loop import ScriptedPolicy, run_episode
    from agent.tools import ToolEnv, build_toolset
    from agent.tools.edit_file import split_multifile_patch
    from sandbox import Limits, Sandbox

    from eval.run_eval import describe_repo, task_test_config

    workdir = WORK_DIR / "replay" / record["task_id"] / "repo"
    shutil.rmtree(workdir.parent, ignore_errors=True)
    try:
        checkout(record["repo_url"], record["base_commit"], workdir)
        python = ensure_venv(record["repo"])
        prepare_checkout(record["repo"], workdir)
    except Exception:
        shutil.rmtree(workdir.parent, ignore_errors=True)
        return None

    try:
        if record.get("test_patch"):
            from agent.tools.edit_file import apply_patch_to_tree

            apply_patch_to_tree(record["test_patch"], workdir)
    except Exception:
        shutil.rmtree(workdir.parent, ignore_errors=True)
        return None

    base, targets = task_test_config(record, python=str(python))
    sandbox = Sandbox(workdir, limits=limits or Limits(cpu_seconds=60, wall_seconds=timeout))
    env = ToolEnv(
        sandbox=sandbox,
        test_base=base,
        test_targets=targets,
        test_pythonpath=pythonpath_for(workdir),
    )
    tools = build_toolset(env)

    sections = split_multifile_patch(record["gold_patch"])
    if not sections:
        shutil.rmtree(workdir.parent, ignore_errors=True)
        return None

    calls: list[ToolCall] = []
    for path in list(sections)[:2]:
        if (workdir / path).is_file():
            calls.append(ToolCall.make("read_file", {"path": path}))
    for path, diff in sections.items():
        calls.append(ToolCall.make("edit_file", {"path": path, "diff": diff}))
    calls.append(ToolCall.make("run_tests", {}))
    calls.append(ToolCall.make("submit", {"patch": record["gold_patch"]}))

    episode = run_episode(
        ScriptedPolicy(calls),
        tools,
        task_id=record["task_id"],
        issue=record["issue"],
        repo_state=describe_repo(workdir),
        max_steps=len(calls) + 2,
    )
    keep = episode.status == "submitted" and not any(
        obs.is_error for obs in episode.context.observations
    )
    tests_passed = any(
        obs.name == "run_tests" and obs.content.startswith("status=pass")
        for obs in episode.context.observations
    )
    shutil.rmtree(workdir.parent, ignore_errors=True)
    if not (keep and tests_passed):
        return None

    return {
        "task_id": record["task_id"],
        "repo": record["repo"],
        "repo_url": record["repo_url"],
        "base_commit": record["base_commit"],
        "issue": record["issue"],
        "test_patch": record["test_patch"],
        "test_runner": record["test_runner"],
        "test_target": record["test_target"],
        "test_k_expr": record.get("test_k_expr", ""),
        "venv": record["venv"],
        "source": record.get("source", "SWE-bench"),
        "messages": episode.context.to_dict()["messages"],
        "meta": {
            "steps": episode.steps,
            "replayed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gold_files": list(sections),
        },
    }


# ------------------------------------------------------------------ task out
def public_task(record: dict[str, Any]) -> dict[str, Any]:
    """Task record without the gold patch (RL pool / holdout, §2.1)."""
    keys = (
        "task_id", "repo", "repo_url", "base_commit", "issue", "test_patch",
        "test_runner", "test_target", "test_k_expr", "venv", "source", "version",
    )
    return {key: record[key] for key in keys if key in record}
