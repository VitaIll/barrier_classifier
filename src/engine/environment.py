"""Runtime numeric-stack guard.

The serving contract is bit-exact only on the stack it was validated
under: pandas rolling internals, polars kernels, numpy scalar semantics
and CatBoost scoring all feed the feature values and predictions that
the parity gates pinned. A host with a drifted stack can compute
features the model never saw — silently, because nothing crashes.

``VALIDATED_STACK`` is the single source of truth for the frozen
versions. ``requirements.txt`` must pin the same versions (enforced by
``tests/engine/test_environment.py``), so build-time installs, the CI
matrix, and this runtime check can never disagree with each other.

Upgrading the stack is a deliberate act, not a side effect of
``pip install``: bump the pins and this manifest together, re-run the
full suite and the real-data replay validation, and treat any numeric
drift as a model-retrain event (see docs/PRODUCTION.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from src.engine.errors import EnvironmentDriftError

# Serving-critical packages: these participate in feature computation,
# model scoring, or artifact I/O on the live path. Versions are the ones
# the v0001 parity validation ran under (validate_engine_replay.py:
# 74,613/74,613 predictions bit-exact, 287/287 trades identical).
VALIDATED_STACK: dict[str, str] = {
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "polars": "1.39.3",
    "pyarrow": "19.0.1",
    "catboost": "1.2.8",
}

_MISSING = "missing"


@dataclass(frozen=True)
class StackDrift:
    """One package whose installed version differs from the validated one."""

    package: str
    validated: str
    installed: str  # version string, or "missing" if not importable

    def __str__(self) -> str:
        return f"{self.package}: installed {self.installed}, validated {self.validated}"


def installed_version(package: str) -> str:
    """Installed distribution version, or ``"missing"``. Metadata-only:
    does not import the package (no side effects, sub-millisecond)."""
    try:
        return _dist_version(package)
    except PackageNotFoundError:
        return _MISSING


def check_stack() -> list[StackDrift]:
    """Compare the running environment against ``VALIDATED_STACK``.

    Returns one :class:`StackDrift` per mismatch; empty means the host
    reproduces the validated environment exactly.
    """
    drifts: list[StackDrift] = []
    for package, validated in VALIDATED_STACK.items():
        installed = installed_version(package)
        if installed != validated:
            drifts.append(
                StackDrift(package=package, validated=validated, installed=installed)
            )
    return drifts


def stack_report() -> str:
    """Human-readable table of validated vs installed versions."""
    lines = ["numeric stack (validated -> installed):"]
    for package, validated in VALIDATED_STACK.items():
        installed = installed_version(package)
        mark = "ok" if installed == validated else "DRIFT"
        lines.append(f"  {package:<10} {validated:>10} -> {installed:<10} [{mark}]")
    return "\n".join(lines)


def enforce_stack(*, strict: bool, context: str) -> list[StackDrift]:
    """Gate an entry point on stack alignment.

    ``strict=True`` raises :class:`EnvironmentDriftError` on any drift —
    used before arming live execution, where drifted numerics mean the
    model is being served inputs it was never validated on.
    ``strict=False`` returns the drift list for the caller to log.
    """
    drifts = check_stack()
    if drifts and strict:
        detail = "; ".join(str(d) for d in drifts)
        raise EnvironmentDriftError(
            f"{context}: numeric stack drifted from the validated environment "
            f"({detail}). Reinstall from requirements.txt pins, or pass "
            f"--allow-stack-drift to override deliberately."
        )
    return drifts
