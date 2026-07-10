"""Tests for the conftest.py gating logic itself.

The conftest implements a theme-based test filter: it stamps each test
item with a theme marker based on its file path, then reads
``tests/current_step.txt`` (or the ``CURRENT_STEP`` env var) to decide
which themes to run. This file exercises the filter from the outside via
subprocess, with a temporary working directory that pretends to be the
project root.

These tests are marked ``framework`` so they always run regardless of
which theme is active — the gating logic itself is cross-cutting.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


pytestmark = pytest.mark.framework


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_temp_project(tmp_path: Path, *, current_step_contents: str | None) -> Path:
    """Build a minimal project tree with one trivially-passing test per theme.

    Returns the path to the temp project root. The tree has:
      - pytest.ini       (registers ``framework`` and ``strategy`` markers)
      - tests/conftest.py (copied from the real project so the gating
        logic under test matches what production runs)
      - tests/test_strategy_dummy.py    (carries the ``strategy`` theme via the
        conftest's path-based map, plus an explicit ``pytestmark``)
      - tests/test_framework_dummy.py   (carries the ``framework`` marker)
      - optionally tests/current_step.txt with the provided contents
    """
    real_conftest = (
        Path(__file__).resolve().parent / "conftest.py"
    ).read_text(encoding="utf-8")

    proj = tmp_path
    (proj / "tests").mkdir()
    (proj / "tests" / "__init__.py").write_text("", encoding="utf-8")

    # Copy the conftest verbatim — we want to test THE production gating
    # logic, not a mock. The path-based map is keyed on subpaths under
    # ``tests/``, which match the layout we create here.
    (proj / "tests" / "conftest.py").write_text(real_conftest, encoding="utf-8")

    # Carry a 'strategy' theme via the conftest's path map. The map keys
    # are "strategy/test_*.py" — match that layout.
    (proj / "tests" / "strategy").mkdir()
    (proj / "tests" / "strategy" / "__init__.py").write_text("", encoding="utf-8")
    (proj / "tests" / "strategy" / "test_simulator.py").write_text(
        textwrap.dedent(
            """
            import pytest
            pytestmark = pytest.mark.strategy

            def test_dummy_strategy():
                assert True
            """
        ).lstrip(),
        encoding="utf-8",
    )
    # Carry a 'framework' theme explicitly.
    (proj / "tests" / "test_framework.py").write_text(
        textwrap.dedent(
            """
            import pytest
            pytestmark = pytest.mark.framework

            def test_dummy_framework():
                assert True
            """
        ).lstrip(),
        encoding="utf-8",
    )

    (proj / "pytest.ini").write_text(
        textwrap.dedent(
            """
            [pytest]
            pythonpath = .
            testpaths = tests
            filterwarnings =
                ignore::DeprecationWarning
                ignore::pytest.PytestUnknownMarkWarning
            markers =
                framework: cross-cutting invariants (always run)
                strategy: strategy theme
            """
        ).lstrip(),
        encoding="utf-8",
    )

    if current_step_contents is not None:
        (proj / "tests" / "current_step.txt").write_text(
            current_step_contents, encoding="utf-8"
        )

    return proj


def _run_pytest(proj: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Invoke pytest in ``proj`` with explicit args. Returns the completed
    process so callers can inspect returncode + stdout/stderr."""
    cmd = [sys.executable, "-m", "pytest", "-q", *args]
    proc_env = os.environ.copy()
    # Scrub CURRENT_STEP unless the test explicitly sets it
    proc_env.pop("CURRENT_STEP", None)
    if env is not None:
        proc_env.update(env)
    return subprocess.run(
        cmd, cwd=str(proj), capture_output=True, text=True, env=proc_env, timeout=60,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_flag_bypasses_filter(tmp_path: Path) -> None:
    """``pytest --all`` runs every collected test regardless of current_step.txt."""
    proj = _make_temp_project(tmp_path, current_step_contents="framework\n")
    res = _run_pytest(proj, "--all")
    assert res.returncode == 0, f"pytest --all failed: {res.stdout}\n{res.stderr}"
    # Both the strategy and framework dummy tests should have run
    assert "2 passed" in res.stdout, f"expected 2 passed, got: {res.stdout!r}"


def test_filter_with_current_step_runs_only_matching(tmp_path: Path) -> None:
    """A current_step.txt with one theme runs that theme + framework only."""
    proj = _make_temp_project(tmp_path, current_step_contents="strategy\n")
    res = _run_pytest(proj)
    assert res.returncode == 0, f"pytest failed: {res.stdout}\n{res.stderr}"
    # strategy + framework => 2 selected
    assert "2 passed" in res.stdout, f"expected 2 passed, got: {res.stdout!r}"


def test_filter_with_framework_only_runs_only_framework(tmp_path: Path) -> None:
    """When only framework is active, the strategy theme is deselected."""
    proj = _make_temp_project(tmp_path, current_step_contents="framework\n")
    res = _run_pytest(proj)
    assert res.returncode == 0, f"pytest failed: {res.stdout}\n{res.stderr}"
    # Just the framework test runs (strategy deselected)
    assert "1 passed" in res.stdout, f"expected 1 passed, got: {res.stdout!r}"
    assert "1 deselected" in res.stdout, f"expected 1 deselected, got: {res.stdout!r}"


def test_env_var_overrides_file(tmp_path: Path) -> None:
    """CURRENT_STEP=<theme> overrides current_step.txt."""
    proj = _make_temp_project(tmp_path, current_step_contents="framework\n")
    # File says framework, env says strategy => strategy + framework run
    res = _run_pytest(proj, env={"CURRENT_STEP": "strategy"})
    assert res.returncode == 0, f"pytest failed: {res.stdout}\n{res.stderr}"
    assert "2 passed" in res.stdout, f"expected 2 passed, got: {res.stdout!r}"


def test_malformed_comment_only_file_raises_usageerror(tmp_path: Path) -> None:
    """A current_step.txt that's comments + blanks only must raise
    UsageError — silent fall-through to "run everything" would mask a
    broken filter (Agent F's hardening).
    """
    proj = _make_temp_project(
        tmp_path,
        current_step_contents="# only a comment\n  \n  # another comment\n",
    )
    res = _run_pytest(proj)
    assert res.returncode != 0, (
        f"expected nonzero exit for malformed current_step.txt; got rc=0: "
        f"{res.stdout}\n{res.stderr}"
    )
    combined = (res.stdout + res.stderr).lower()
    assert "malformed" in combined or "comment-only" in combined or "empty" in combined, (
        f"expected a 'malformed' / 'empty' message; stdout/stderr:\n"
        f"{res.stdout}\n{res.stderr}"
    )


def test_malformed_empty_file_raises_usageerror(tmp_path: Path) -> None:
    """An entirely empty current_step.txt raises UsageError — the file
    is present so we cannot silently fall back to 'run everything'."""
    proj = _make_temp_project(tmp_path, current_step_contents="")
    res = _run_pytest(proj)
    assert res.returncode != 0, (
        f"expected nonzero exit for empty current_step.txt; got rc=0: "
        f"{res.stdout}\n{res.stderr}"
    )


def test_malformed_commas_only_file_raises_usageerror(tmp_path: Path) -> None:
    """``,,,`` and similar — only separators, no themes — must raise."""
    proj = _make_temp_project(tmp_path, current_step_contents=",,,\n")
    res = _run_pytest(proj)
    assert res.returncode != 0, (
        f"expected nonzero exit for commas-only current_step.txt; got rc=0: "
        f"{res.stdout}\n{res.stderr}"
    )


def test_missing_file_runs_everything(tmp_path: Path) -> None:
    """If current_step.txt does not exist AND CURRENT_STEP env is unset,
    no filter is applied (behaves like --all)."""
    proj = _make_temp_project(tmp_path, current_step_contents=None)
    res = _run_pytest(proj)
    assert res.returncode == 0, f"pytest failed: {res.stdout}\n{res.stderr}"
    # Both tests should run
    assert "2 passed" in res.stdout, f"expected 2 passed, got: {res.stdout!r}"


def test_env_var_empty_raises_usageerror(tmp_path: Path) -> None:
    """A SET-but-empty CURRENT_STEP env var raises UsageError."""
    proj = _make_temp_project(tmp_path, current_step_contents="framework\n")
    res = _run_pytest(proj, env={"CURRENT_STEP": "  # only a comment"})
    assert res.returncode != 0, (
        f"expected nonzero exit for comment-only CURRENT_STEP env; got rc=0: "
        f"{res.stdout}\n{res.stderr}"
    )
