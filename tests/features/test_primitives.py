"""Step 2 tests: foundational primitives.

L1 — synthetic deterministic input → exact expected output (hand-computed)
L2 — adversarial: zero/negative inputs, NaN/null injection, EPS-edge values
L3 — pandas oracle: replicate the legacy logic from utils.py and assert
     bit-equal (or within strict numerical tolerance) on 10k random rows
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.features.config import EPS
from src.features.primitives import (
    clip_pos,
    eps_safe_div,
    log1p_vol,
    log_return,
    safe_log_ratio,
)

pytestmark = pytest.mark.step2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval(expr: pl.Expr, **cols) -> pl.Series:
    """Evaluate one expression against named columns; return the result series."""
    return pl.DataFrame(cols).select(expr.alias("out"))["out"]


def _both_missing(av, ev) -> bool:
    a_miss = av is None or (isinstance(av, float) and math.isnan(av))
    e_miss = ev is None or (isinstance(ev, float) and math.isnan(ev))
    return a_miss and e_miss


def _close(av, ev, *, rtol=1e-12, atol=1e-15) -> bool:
    if _both_missing(av, ev):
        return True
    if av is None or ev is None:
        return False
    if isinstance(av, float) and math.isnan(av):
        return isinstance(ev, float) and math.isnan(ev)
    if av in (float("inf"), float("-inf")):
        return av == ev
    return math.isclose(av, ev, rel_tol=rtol, abs_tol=atol)


def assert_series_close(actual: pl.Series, expected: list, **kw) -> None:
    a = actual.to_list()
    assert len(a) == len(expected), f"length: {len(a)} vs {len(expected)}"
    for i, (av, ev) in enumerate(zip(a, expected)):
        assert _close(av, ev, **kw), f"row {i}: expected {ev!r}, got {av!r}"


def _to_nan_array(s: pl.Series) -> np.ndarray:
    """Turn a polars series into a numpy array, mapping null -> NaN."""
    return np.array([v if v is not None else float("nan") for v in s.to_list()])


# ===========================================================================
# safe_log_ratio
# ===========================================================================


class TestSafeLogRatio:
    """log(num/den) under a configurable guard; null otherwise."""

    # --- L1: hand-computed -------------------------------------------------

    def test_simple_positive(self):
        out = _eval(safe_log_ratio(pl.col("a"), pl.col("b")), a=[2.0], b=[1.0])
        assert_series_close(out, [math.log(2.0)])

    def test_unity_ratio_is_zero(self):
        out = _eval(safe_log_ratio(pl.col("a"), pl.col("b")), a=[1.0], b=[1.0])
        assert_series_close(out, [0.0])

    def test_inverse_ratio_is_negative_log(self):
        out = _eval(safe_log_ratio(pl.col("a"), pl.col("b")), a=[1.0], b=[2.0])
        assert_series_close(out, [-math.log(2.0)])

    def test_explicit_strict_guard_mimics_rho(self):
        # Reproduces rho semantics: guard is `high > low` strictly.
        out = _eval(
            safe_log_ratio(
                pl.col("h"),
                pl.col("l"),
                when=pl.col("h") > pl.col("l"),
            ),
            h=[2.0, 1.0, 1.0],
            l=[1.0, 1.0, 2.0],
        )
        assert _close(out[0], math.log(2.0))
        assert out[1] is None  # equal -> null
        assert out[2] is None  # h < l -> null

    # --- L2: adversarial ---------------------------------------------------

    def test_zero_numerator_yields_null(self):
        out = _eval(safe_log_ratio(pl.col("a"), pl.col("b")), a=[0.0], b=[1.0])
        assert out.to_list() == [None]

    def test_zero_denominator_yields_null(self):
        out = _eval(safe_log_ratio(pl.col("a"), pl.col("b")), a=[1.0], b=[0.0])
        assert out.to_list() == [None]

    def test_negative_inputs_yield_null(self):
        out = _eval(
            safe_log_ratio(pl.col("a"), pl.col("b")),
            a=[-1.0, 1.0, -1.0],
            b=[1.0, -1.0, -1.0],
        )
        assert out.to_list() == [None, None, None]

    def test_nan_input_yields_null(self):
        out = _eval(
            safe_log_ratio(pl.col("a"), pl.col("b")),
            a=[float("nan"), 1.0],
            b=[1.0, float("nan")],
        )
        assert out.to_list() == [None, None]

    def test_polars_null_input_yields_null(self):
        out = _eval(
            safe_log_ratio(pl.col("a"), pl.col("b")),
            a=[None, 1.0],
            b=[1.0, None],
        )
        assert out.to_list() == [None, None]

    # --- L3: pandas oracle parity -----------------------------------------

    def test_parity_default_guard(self):
        rng = np.random.default_rng(42)
        n = 10_000
        a = rng.uniform(0.5, 100.0, size=n)
        b = rng.uniform(0.5, 100.0, size=n)
        # Inject adversarial values
        a[:5] = [0.0, -1.0, float("nan"), 1.0, 1.0]
        b[:5] = [1.0, 1.0, 1.0, 0.0, -1.0]

        with np.errstate(invalid="ignore", divide="ignore"):
            expected = np.where((a > 0) & (b > 0), np.log(a / b), np.nan)

        out = _to_nan_array(_eval(safe_log_ratio(pl.col("a"), pl.col("b")), a=a, b=b))

        np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_array_equal(out[valid], expected[valid])

    def test_parity_strict_guard_rho(self):
        # Reproduces compute_base_series rho: log(h/l) when h > l strictly.
        rng = np.random.default_rng(7)
        n = 10_000
        l = rng.uniform(0.1, 10.0, size=n)
        h = l + rng.uniform(-0.5, 1.0, size=n)
        h[:4] = [1.0, 1.0, 1.0, float("nan")]
        l[:4] = [1.0, 0.5, 2.0, 1.0]

        with np.errstate(invalid="ignore", divide="ignore"):
            expected = np.where(h > l, np.log(h / l), np.nan)

        out = _to_nan_array(
            _eval(
                safe_log_ratio(pl.col("h"), pl.col("l"), when=pl.col("h") > pl.col("l")),
                h=h,
                l=l,
            )
        )
        np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_array_equal(out[valid], expected[valid])


# ===========================================================================
# log_return
# ===========================================================================


class TestLogReturn:
    """log(x_t) - log(x_{t-1}); row 0 null. Unguarded — matches utils.py:1517."""

    # --- L1: hand-computed -------------------------------------------------

    def test_simple(self):
        out = _eval(log_return(pl.col("x")), x=[1.0, 2.0, 4.0])
        assert_series_close(out, [None, math.log(2.0), math.log(2.0)])

    def test_constant_series_is_zero(self):
        out = _eval(log_return(pl.col("x")), x=[5.0, 5.0, 5.0])
        assert out[0] is None
        assert _close(out[1], 0.0)
        assert _close(out[2], 0.0)

    def test_doubling_then_halving(self):
        out = _eval(log_return(pl.col("x")), x=[1.0, 2.0, 1.0])
        assert out[0] is None
        assert _close(out[1], math.log(2.0))
        assert _close(out[2], -math.log(2.0))

    # --- L2: adversarial ---------------------------------------------------

    def test_zero_input_yields_inf(self):
        # No guard: log(0)=-inf propagates through diff.
        out = _eval(log_return(pl.col("x")), x=[1.0, 0.0, 1.0])
        assert out[0] is None
        assert out[1] == float("-inf")
        assert out[2] == float("inf")

    def test_negative_input_yields_nan(self):
        out = _eval(log_return(pl.col("x")), x=[1.0, -1.0, 1.0])
        assert out[0] is None
        assert math.isnan(out[1])
        assert math.isnan(out[2])

    def test_nan_input_propagates(self):
        out = _eval(log_return(pl.col("x")), x=[1.0, float("nan"), 1.0])
        assert out[0] is None
        assert math.isnan(out[1])
        assert math.isnan(out[2])

    def test_single_row_yields_null(self):
        out = _eval(log_return(pl.col("x")), x=[42.0])
        assert out.to_list() == [None]

    # --- L3: pandas oracle parity -----------------------------------------

    def test_parity_random_positive(self):
        rng = np.random.default_rng(42)
        n = 10_000
        x = rng.uniform(0.5, 100.0, size=n)
        x[10] = float("nan")
        x[20] = 0.5

        expected = np.log(pd.Series(x)).diff().to_numpy()
        out = _to_nan_array(_eval(log_return(pl.col("x")), x=x))

        # NaN/null positions match
        np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
        # Non-null values exactly equal
        valid = ~np.isnan(expected) & ~np.isinf(expected)
        np.testing.assert_array_equal(out[valid], expected[valid])


# ===========================================================================
# eps_safe_div
# ===========================================================================


class TestEpsSafeDiv:
    """num / (den + eps). Bounded ratio; does NOT mask."""

    # --- L1 ---------------------------------------------------------------

    def test_simple(self):
        out = _eval(eps_safe_div(pl.col("a"), pl.col("b")), a=[1.0], b=[2.0])
        assert _close(out[0], 1.0 / (2.0 + EPS))

    def test_zero_denominator_yields_one_over_eps(self):
        out = _eval(eps_safe_div(pl.col("a"), pl.col("b")), a=[1.0], b=[0.0])
        assert _close(out[0], 1.0 / EPS)

    def test_zero_over_zero_is_zero(self):
        out = _eval(eps_safe_div(pl.col("a"), pl.col("b")), a=[0.0], b=[0.0])
        assert _close(out[0], 0.0)

    # --- L2 ---------------------------------------------------------------

    def test_eps_dominates_when_den_below_eps(self):
        # 1e-15 / (1e-15 + 1e-10) ≈ 1e-15 / EPS
        out = _eval(eps_safe_div(pl.col("a"), pl.col("b")), a=[1e-15], b=[1e-15])
        assert _close(out[0], 1e-15 / (1e-15 + EPS))

    def test_negative_denominator_unmasked(self):
        # Different from safe_log_ratio: this primitive does NOT mask.
        out = _eval(eps_safe_div(pl.col("a"), pl.col("b")), a=[1.0], b=[-1.0])
        assert _close(out[0], 1.0 / (-1.0 + EPS))

    def test_custom_eps(self):
        out = _eval(
            eps_safe_div(pl.col("a"), pl.col("b"), eps=1e-3),
            a=[1.0], b=[0.0],
        )
        assert _close(out[0], 1.0 / 1e-3)

    def test_nan_propagates(self):
        out = _eval(eps_safe_div(pl.col("a"), pl.col("b")),
                    a=[float("nan"), 1.0], b=[1.0, float("nan")])
        assert math.isnan(out[0])
        assert math.isnan(out[1])

    def test_null_propagates(self):
        out = _eval(eps_safe_div(pl.col("a"), pl.col("b")),
                    a=[None, 1.0], b=[1.0, None])
        assert out.to_list() == [None, None]

    # --- L3 ---------------------------------------------------------------

    def test_parity_random(self):
        rng = np.random.default_rng(2024)
        n = 10_000
        a = rng.uniform(-100.0, 100.0, size=n)
        b = rng.uniform(-100.0, 100.0, size=n)
        b[:3] = [0.0, 1e-15, -1e-15]

        expected = a / (b + EPS)
        out = _eval(eps_safe_div(pl.col("a"), pl.col("b")), a=a, b=b).to_numpy()
        np.testing.assert_array_equal(out, expected)


# ===========================================================================
# log1p_vol
# ===========================================================================


class TestLog1pVol:
    """log(1 + x) for volume-like inputs."""

    # --- L1 ---------------------------------------------------------------

    def test_zero_yields_zero(self):
        out = _eval(log1p_vol(pl.col("x")), x=[0.0])
        assert _close(out[0], 0.0)

    def test_e_minus_one_yields_one(self):
        out = _eval(log1p_vol(pl.col("x")), x=[math.e - 1])
        assert _close(out[0], 1.0)

    def test_typical_volume(self):
        out = _eval(log1p_vol(pl.col("x")), x=[10.0])
        assert _close(out[0], math.log(11.0))

    # --- L2 ---------------------------------------------------------------

    def test_minus_one_yields_neg_inf(self):
        out = _eval(log1p_vol(pl.col("x")), x=[-1.0])
        assert out[0] == float("-inf")

    def test_below_minus_one_yields_nan(self):
        out = _eval(log1p_vol(pl.col("x")), x=[-2.0])
        assert math.isnan(out[0])

    def test_nan_propagates(self):
        out = _eval(log1p_vol(pl.col("x")), x=[float("nan")])
        assert math.isnan(out[0])

    def test_null_propagates(self):
        # Mix with a valid value so polars infers Float64, not Null type.
        out = _eval(log1p_vol(pl.col("x")), x=[None, 1.0])
        assert out[0] is None
        assert _close(out[1], math.log(2.0))

    # --- L3 ---------------------------------------------------------------

    def test_parity_with_np_log1p_for_realistic_volumes(self):
        # For x >= 1e-6 the (x+1).log() identity matches np.log1p exactly.
        rng = np.random.default_rng(99)
        n = 10_000
        x = rng.uniform(0.0, 1e6, size=n)
        x[:3] = [0.0, 1.0, 1e6]

        expected = np.log1p(x)
        out = _eval(log1p_vol(pl.col("x")), x=x).to_numpy()

        # Restrict to inputs where the implementations are mathematically equal.
        mask = x >= 1e-6
        np.testing.assert_allclose(out[mask], expected[mask], rtol=0, atol=0)
        # Where x >= 0 generally, allow sub-ulp differences.
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


# ===========================================================================
# clip_pos
# ===========================================================================


class TestClipPos:
    """max(0, x) — used to clamp variance estimators before sqrt."""

    # --- L1 ---------------------------------------------------------------

    def test_negative_clamped_to_zero(self):
        out = _eval(clip_pos(pl.col("x")), x=[-1.0])
        assert _close(out[0], 0.0)

    def test_zero_unchanged(self):
        out = _eval(clip_pos(pl.col("x")), x=[0.0])
        assert _close(out[0], 0.0)

    def test_positive_unchanged(self):
        out = _eval(clip_pos(pl.col("x")), x=[2.5])
        assert _close(out[0], 2.5)

    def test_large_negative_clamped(self):
        out = _eval(clip_pos(pl.col("x")), x=[-1e9])
        assert _close(out[0], 0.0)

    # --- L2 ---------------------------------------------------------------

    def test_nan_propagates(self):
        # Polars clip preserves NaN (matches np.maximum(0, NaN) -> NaN).
        out = _eval(clip_pos(pl.col("x")), x=[float("nan")])
        assert math.isnan(out[0])

    def test_null_propagates(self):
        # Mix with a valid value so polars infers Float64, not Null type.
        out = _eval(clip_pos(pl.col("x")), x=[None, -1.0, 2.0])
        assert out[0] is None
        assert _close(out[1], 0.0)
        assert _close(out[2], 2.0)

    def test_neg_inf_clamped_to_zero(self):
        out = _eval(clip_pos(pl.col("x")), x=[float("-inf")])
        assert _close(out[0], 0.0)

    def test_pos_inf_unchanged(self):
        out = _eval(clip_pos(pl.col("x")), x=[float("inf")])
        assert out[0] == float("inf")

    # --- L3 ---------------------------------------------------------------

    def test_parity_with_np_maximum(self):
        rng = np.random.default_rng(2025)
        n = 10_000
        x = rng.uniform(-100.0, 100.0, size=n)
        x[:4] = [-1e9, 0.0, 1e9, -1e-15]

        expected = np.maximum(0.0, x)
        out = _eval(clip_pos(pl.col("x")), x=x).to_numpy()
        np.testing.assert_array_equal(out, expected)
