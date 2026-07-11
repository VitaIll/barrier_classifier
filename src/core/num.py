"""Shared numerical guards (docs/TARGET_ARCHITECTURE.md §4).

The standard fixes for the failure classes catalogued in the 2026-07-11
review: unguarded division, overflow in exp/sigmoid, and silent NaN/inf
propagation. Kernel and domain blocks use these instead of re-deriving
ad-hoc guards.

All functions are pure — they never mutate their inputs.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.core.errors import ContractError

# Same value as the legacy ``utils.EPS`` — a single canonical home going
# forward. (Kept numerically identical so migrated formulas stay bit-equal.)
EPS: float = 1e-10


def assert_all_finite(
    arr: np.ndarray,
    *,
    name: str,
    context: str = "",
) -> None:
    """Raise :class:`ContractError` if ``arr`` holds any NaN/inf.

    The error names the array, the violation counts, and the first offending
    flat index — enough to locate the poison without a debugger.
    """
    a = np.asarray(arr)
    if a.dtype.kind not in "fc":  # ints/bools cannot hold NaN/inf
        return
    finite = np.isfinite(a)
    if bool(finite.all()):
        return
    bad = ~finite
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    first = int(np.flatnonzero(bad.ravel())[0])
    where = f" in {context}" if context else ""
    raise ContractError(
        f"{name}{where} contains non-finite values: "
        f"{n_nan} NaN, {n_inf} inf (first at flat index {first}, size {a.size})"
    )


def safe_div(
    num: np.ndarray,
    den: np.ndarray,
    *,
    fill: float = np.nan,
) -> np.ndarray:
    """Elementwise ``num / den`` with ``den == 0`` producing ``fill``.

    Use when a zero denominator is a *legitimate* state that maps to a
    known value (e.g. "no trades -> rate undefined"). For ratios where the
    denominator is nonnegative and a bounded result is wanted, the feature
    layer's ``eps_safe_div`` (den + EPS) remains the right tool.
    """
    num_a = np.asarray(num, dtype=float)
    den_a = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num_a, den_a).shape, fill, dtype=float)
    np.divide(num_a, den_a, out=out, where=den_a != 0)
    return out


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic function (``scipy.special.expit``).

    ``1/(1+exp(-x))`` overflows for large negative ``x``; expit is the
    numerically stable form.
    """
    from scipy.special import expit  # scipy is already a hard dependency

    return np.asarray(expit(np.asarray(x, dtype=float)), dtype=float)


def clip_exp(
    x: np.ndarray,
    *,
    lo: float = -50.0,
    hi: float = 50.0,
) -> np.ndarray:
    """``exp(clip(x, lo, hi))`` — finite by construction.

    ``exp(±50)`` spans ~1e-22..1e21, far beyond any meaningful weight or
    probability ratio in this codebase; the clamp turns overflow into
    saturation.
    """
    return np.exp(np.clip(np.asarray(x, dtype=float), lo, hi))


def require_finite_scalar(
    value: float,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
    context: str = "",
) -> float:
    """Validate one float parameter; return it as ``float``.

    Raises :class:`ContractError` on NaN/inf and on sign violations. The
    canonical guard for prices, barriers, and sizes entering domain objects
    — note ``NaN <= 0`` is False, so plain comparisons silently accept NaN.
    """
    v = float(value)
    where = f" in {context}" if context else ""
    if not math.isfinite(v):
        raise ContractError(f"{name}{where} must be finite; got {v!r}")
    if positive and v <= 0.0:
        raise ContractError(f"{name}{where} must be > 0; got {v!r}")
    if nonnegative and v < 0.0:
        raise ContractError(f"{name}{where} must be >= 0; got {v!r}")
    return v


def shifted_variance(
    cum_sum: np.ndarray | float,
    cum_sq_sum: np.ndarray | float,
    n: np.ndarray | int,
    *,
    shift_used: Optional[float] = None,
) -> np.ndarray:
    """Population variance from cumulative sums of *shifted* data.

    The naive ``E[X²] − E[X]²`` on raw cumulative sums cancels
    catastrophically when ``mean² >> var``. Callers should accumulate
    ``sum(x - c)`` and ``sum((x - c)²)`` for any constant ``c`` near the
    data (e.g. the full-sample mean) and pass those here; variance is
    shift-invariant, so the result is the same quantity computed stably.

    ``shift_used`` is accepted purely as documentation-at-call-site and is
    not used in the computation.
    """
    n_a = np.asarray(n, dtype=float)
    mean_shifted = np.divide(
        np.asarray(cum_sum, dtype=float),
        n_a,
        out=np.full(np.shape(n_a) or (1,), np.nan),
        where=n_a > 0,
    )
    ex2 = np.divide(
        np.asarray(cum_sq_sum, dtype=float),
        n_a,
        out=np.full(np.shape(n_a) or (1,), np.nan),
        where=n_a > 0,
    )
    var = ex2 - mean_shifted**2
    # Clamp the tiny negative values that residual rounding can produce.
    return np.where(var < 0.0, 0.0, var)
