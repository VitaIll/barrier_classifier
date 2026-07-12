"""Environment guard: the frozen numeric stack is enforced, not advisory.

Three layers must agree and stay green together:
  1. requirements.txt pins  ==  VALIDATED_STACK  (single source of truth)
  2. the RUNNING interpreter ==  VALIDATED_STACK  (CI drift tripwire —
     this is the test that fails first, with a precise message, if a
     fresh ``pip install`` ever resolves a different numeric stack)
  3. the live CLI refuses to arm execution on a drifted host
"""

from __future__ import annotations

import logging
import re
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from src.engine.environment import (
    VALIDATED_STACK,
    check_stack,
    enforce_stack,
    installed_version,
    stack_report,
)
from src.engine.errors import EnvironmentDriftError

pytestmark = pytest.mark.engine

_REQUIREMENTS = Path(__file__).parents[2] / "requirements.txt"


def _requirement_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in _REQUIREMENTS.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_requirements_pins_match_validated_stack():
    """requirements.txt and VALIDATED_STACK can never disagree."""
    pins = _requirement_pins()
    for package, validated in VALIDATED_STACK.items():
        assert package in pins, (
            f"{package} is serving-critical but not exact-pinned in "
            f"requirements.txt (VALIDATED_STACK wants =={validated})"
        )
        assert pins[package] == validated, (
            f"requirements.txt pins {package}=={pins[package]} but "
            f"VALIDATED_STACK says {validated}; bump both together"
        )


def test_running_stack_is_validated():
    """The interpreter running this suite reproduces the validated stack.

    If this fails, the environment — not the code — has drifted: every
    bit-exactness guarantee downstream of here is void. Reinstall from
    requirements.txt, or (deliberate upgrade) revalidate and bump
    VALIDATED_STACK + pins together.
    """
    assert check_stack() == [], f"\n{stack_report()}"


def test_drift_detection_reports_mismatch(monkeypatch):
    real = installed_version

    def fake(package: str) -> str:
        return "9.9.9" if package == "pandas" else real(package)

    monkeypatch.setattr("src.engine.environment.installed_version", fake)
    drifts = check_stack()
    assert [d.package for d in drifts] == ["pandas"]
    assert drifts[0].installed == "9.9.9"
    assert drifts[0].validated == VALIDATED_STACK["pandas"]
    assert "pandas" in str(drifts[0])


def test_missing_package_reported_as_missing(monkeypatch):
    def raise_missing(package: str) -> str:
        raise PackageNotFoundError(package)

    monkeypatch.setattr("src.engine.environment._dist_version", raise_missing)
    assert installed_version("numpy") == "missing"
    drifts = check_stack()
    assert {d.installed for d in drifts} == {"missing"}
    assert len(drifts) == len(VALIDATED_STACK)


def test_enforce_strict_raises_on_drift(monkeypatch):
    monkeypatch.setattr(
        "src.engine.environment.installed_version", lambda p: "0.0.1"
    )
    with pytest.raises(EnvironmentDriftError, match="numpy.*0\\.0\\.1"):
        enforce_stack(strict=True, context="test")


def test_enforce_lenient_returns_drifts_without_raising(monkeypatch):
    monkeypatch.setattr(
        "src.engine.environment.installed_version", lambda p: "0.0.1"
    )
    drifts = enforce_stack(strict=False, context="test")
    assert len(drifts) == len(VALIDATED_STACK)


def test_enforce_clean_stack_is_silent():
    assert enforce_stack(strict=True, context="test") == []


def test_live_execute_refuses_on_drifted_stack(tmp_path, monkeypatch, caplog):
    """CLI wiring: --execute on a drifted host exits 1 before touching
    the feed store, broker, or config."""
    from src.engine.__main__ import main

    monkeypatch.setattr(
        "src.engine.environment.installed_version", lambda p: "0.0.1"
    )
    caplog.set_level(logging.ERROR, logger="src.engine.cli")
    rc = main(
        [
            "live", "--execute",
            "--feed", str(tmp_path / "absent-feed.db"),
            "--store", str(tmp_path / "store.db"),
            "--model-dir", str(tmp_path / "models"),
        ]
    )
    assert rc == 1
    assert "EnvironmentDriftError" in caplog.text
    assert not (tmp_path / "store.db").exists()


def test_status_prints_stack_report(tmp_path, capsys):
    from src.engine.__main__ import main

    rc = main(
        [
            "status",
            "--model-dir", str(tmp_path),
            "--store", str(tmp_path / "absent.db"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "numeric stack" in out
    for package in VALIDATED_STACK:
        assert package in out
