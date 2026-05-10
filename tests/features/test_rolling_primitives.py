"""Step 3 tests: rolling primitives.

Critical parity hazard: pandas ``rolling.std`` defaults to ``ddof=1`` but
the codebase uses ``ddof=0`` throughout. ``rolling_std_pop`` enforces
``ddof=0`` so this divergence cannot leak in by accident.

Inputs are assumed to be NaN-free at primitive entry (the engine calls
``fill_nan(None)`` once on input bars). Tests use polars null directly
to mimic that contract.

L1 — synthetic deterministic input → exact expected output (hand-computed)
L2 — adversarial: zero variance, leading nulls, window > series, monotone
L3 — pandas oracle: identical values on 10k random rows
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.features.primitives import (
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std_pop,
    rolling_sum,
)

pytestmark = pytest.mark.step3


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_primitives.py for self-contained file)
# ---------------------------------------------------------------------------

def _eval(expr: pl.Expr, **cols) -> pl.Series:
    return pl.DataFrame(cols).select(expr.alias("out"))["out"]


def _close(av, ev, *, rtol=1e-12, atol=1e-15) -> bool:
    a_miss = av is None or (isinstance(av, float) and math.isnan(av))
    e_miss = ev is None or (isinstance(ev, float) and math.isnan(ev))
    if a_miss and e_miss:
        return True
    if a_miss or e_miss:
        return False
    return math.isclose(av, ev, rel_tol=rtol, abs_tol=atol)


def assert_series_close(actual: pl.Series, expected: list, **kw) -> None:
    a = actual.to_list()
    assert len(a) == len(expected), f"length: {len(a)} vs {len(expected)}"
    for i, (av, ev) in enumerate(zip(a, expected)):
        assert _close(av, ev, **kw), f"row {i}: expected {ev!r}, got {av!r}"


def _to_nan_array(s: pl.Series) -> np.ndarray:
    return np.array([v if v is not None else float("nan") for v in s.to_list()])


# ===========================================================================
# rolling_mean
# ===========================================================================


class TestRollingMean:

    # --- L1 ---------------------------------------------------------------

    def test_simple(self):
        out = _eval(rolling_mean(pl.col("x"), w=3), x=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert_series_close(out, [None, None, 2.0, 3.0, 4.0])

    def test_constant_series(self):
        out = _eval(rolling_mean(pl.col("x"), w=3), x=[7.0, 7.0, 7.0, 7.0])
        assert_series_close(out, [None, None, 7.0, 7.0])

    def test_window_equals_length(self):
        out = _eval(rolling_mean(pl.col("x"), w=4), x=[1.0, 2.0, 3.0, 4.0])
        assert_series_close(out, [None, None, None, 2.5])

    def test_window_one_is_identity(self):
        out = _eval(rolling_mean(pl.col("x"), w=1), x=[1.0, 2.0, 3.0])
        assert_series_close(out, [1.0, 2.0, 3.0])

    # --- L2 ---------------------------------------------------------------

    def test_window_larger_than_series_yields_all_null(self):
        out = _eval(rolling_mean(pl.col("x"), w=5), x=[1.0, 2.0, 3.0])
        assert out.to_list() == [None, None, None]

    def test_null_in_window_yields_null(self):
        out = _eval(rolling_mean(pl.col("x"), w=3),
                    x=[1.0, None, 3.0, 4.0, 5.0])
        # Windows ending at indices 0..4:
        #   [1, None, 3]  -> contains null -> null
        #   [None, 3, 4]  -> null
        #   [3, 4, 5]     -> 4.0
        assert out.to_list() == [None, None, None, None, 4.0]

    def test_min_samples_override_allows_partial(self):
        out = _eval(rolling_mean(pl.col("x"), w=3, min_samples=1),
                    x=[1.0, 2.0, 3.0, 4.0, 5.0])
        # min_samples=1: first row uses just [1] → 1.0; second [1,2] → 1.5; etc.
        assert_series_close(out, [1.0, 1.5, 2.0, 3.0, 4.0])

    # --- L3 ---------------------------------------------------------------

    def test_parity_with_pandas(self):
        rng = np.random.default_rng(11)
        n = 10_000
        x = rng.normal(size=n)
        x[100:103] = float("nan")  # adversarial NaN cluster

        # Pre-process NaN -> null to mimic engine behavior
        df = pl.DataFrame({"x": x}).with_columns(pl.col("x").fill_nan(None))
        out = df.select(rolling_mean(pl.col("x"), w=20).alias("out"))["out"]

        expected = pd.Series(x).rolling(20, min_periods=20).mean().to_numpy()
        actual = _to_nan_array(out)

        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(actual[valid], expected[valid], rtol=0, atol=1e-15)


# ===========================================================================
# rolling_std_pop
# ===========================================================================


class TestRollingStdPop:
    """ddof=0 is enforced — this is THE parity hazard for the codebase."""

    # --- L1 ---------------------------------------------------------------

    def test_zero_variance_yields_zero(self):
        out = _eval(rolling_std_pop(pl.col("x"), w=3), x=[5.0, 5.0, 5.0, 5.0])
        assert_series_close(out, [None, None, 0.0, 0.0])

    def test_simple_known_std(self):
        # values [1, 2, 3]: mean=2, sumsq_dev = 1+0+1 = 2; var_pop = 2/3
        out = _eval(rolling_std_pop(pl.col("x"), w=3), x=[1.0, 2.0, 3.0])
        assert out[0] is None
        assert out[1] is None
        assert _close(out[2], math.sqrt(2.0 / 3.0))

    def test_first_window_minus_one_rows_null(self):
        out = _eval(rolling_std_pop(pl.col("x"), w=4), x=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert out[0] is None
        assert out[1] is None
        assert out[2] is None
        assert out[3] is not None
        assert out[4] is not None

    # --- L2 ---------------------------------------------------------------

    def test_population_not_sample(self):
        # values [1, 2]: var_pop = 0.25 (mean=1.5, dev² sum = 0.25+0.25, /2);
        #                var_sample = 0.5 (divide by n-1 = 1)
        out = _eval(rolling_std_pop(pl.col("x"), w=2), x=[1.0, 2.0])
        # std_pop = sqrt(0.25) = 0.5; std_sample = sqrt(0.5) ≈ 0.7071
        assert _close(out[1], 0.5)
        assert not _close(out[1], math.sqrt(0.5))  # NOT sample std

    def test_null_in_window_yields_null(self):
        out = _eval(rolling_std_pop(pl.col("x"), w=3),
                    x=[1.0, None, 3.0, 4.0, 5.0])
        # Same null positions as rolling_mean
        assert out[0] is None
        assert out[1] is None
        assert out[2] is None
        assert out[3] is None
        assert out[4] is not None

    def test_window_larger_than_series_all_null(self):
        out = _eval(rolling_std_pop(pl.col("x"), w=10), x=[1.0, 2.0, 3.0])
        assert out.to_list() == [None, None, None]

    # --- L3 ---------------------------------------------------------------

    def test_parity_with_pandas_ddof_zero(self):
        rng = np.random.default_rng(22)
        n = 10_000
        x = rng.normal(size=n)
        x[500:502] = float("nan")

        df = pl.DataFrame({"x": x}).with_columns(pl.col("x").fill_nan(None))
        out = df.select(rolling_std_pop(pl.col("x"), w=30).alias("out"))["out"]

        # Pandas with ddof=0 (population)
        expected = pd.Series(x).rolling(30, min_periods=30).std(ddof=0).to_numpy()
        actual = _to_nan_array(out)

        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-12, atol=1e-12)

    def test_diverges_from_pandas_default_ddof_one(self):
        # Sanity: confirm our ddof=0 result is meaningfully different from
        # pandas' ddof=1 default. If this ever passes, we have a parity bug.
        rng = np.random.default_rng(99)
        x = rng.normal(size=100)
        df = pl.DataFrame({"x": x})
        out_pop = df.select(rolling_std_pop(pl.col("x"), w=20).alias("out"))["out"].to_numpy()
        sample_default = pd.Series(x).rolling(20, min_periods=20).std().to_numpy()  # ddof=1 default
        # Where both are valid, pop and sample must differ.
        valid = ~np.isnan(sample_default)
        assert not np.allclose(out_pop[valid], sample_default[valid])


# ===========================================================================
# rolling_sum
# ===========================================================================


class TestRollingSum:

    def test_simple(self):
        out = _eval(rolling_sum(pl.col("x"), w=3), x=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert_series_close(out, [None, None, 6.0, 9.0, 12.0])

    def test_zeros(self):
        out = _eval(rolling_sum(pl.col("x"), w=3), x=[0.0, 0.0, 0.0, 0.0])
        assert_series_close(out, [None, None, 0.0, 0.0])

    def test_negative_values(self):
        out = _eval(rolling_sum(pl.col("x"), w=2), x=[-1.0, -2.0, -3.0])
        assert_series_close(out, [None, -3.0, -5.0])

    def test_null_in_window_yields_null(self):
        out = _eval(rolling_sum(pl.col("x"), w=3), x=[1.0, None, 3.0, 4.0, 5.0])
        assert out.to_list() == [None, None, None, None, 12.0]

    def test_parity_with_pandas(self):
        rng = np.random.default_rng(33)
        n = 10_000
        x = rng.uniform(0, 100, size=n)

        df = pl.DataFrame({"x": x})
        out = df.select(rolling_sum(pl.col("x"), w=15).alias("out"))["out"]

        expected = pd.Series(x).rolling(15, min_periods=15).sum().to_numpy()
        actual = _to_nan_array(out)

        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-12, atol=1e-12)


# ===========================================================================
# rolling_min / rolling_max
# ===========================================================================


class TestRollingMinMax:

    # --- L1 ---------------------------------------------------------------

    def test_min_simple(self):
        out = _eval(rolling_min(pl.col("x"), w=3), x=[3.0, 1.0, 2.0, 5.0, 4.0])
        assert_series_close(out, [None, None, 1.0, 1.0, 2.0])

    def test_max_simple(self):
        out = _eval(rolling_max(pl.col("x"), w=3), x=[3.0, 1.0, 2.0, 5.0, 4.0])
        assert_series_close(out, [None, None, 3.0, 5.0, 5.0])

    def test_min_max_constant(self):
        out_min = _eval(rolling_min(pl.col("x"), w=3), x=[7.0, 7.0, 7.0])
        out_max = _eval(rolling_max(pl.col("x"), w=3), x=[7.0, 7.0, 7.0])
        assert_series_close(out_min, [None, None, 7.0])
        assert_series_close(out_max, [None, None, 7.0])

    def test_min_monotone_increasing(self):
        out = _eval(rolling_min(pl.col("x"), w=3), x=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert_series_close(out, [None, None, 1.0, 2.0, 3.0])

    def test_max_monotone_decreasing(self):
        out = _eval(rolling_max(pl.col("x"), w=3), x=[5.0, 4.0, 3.0, 2.0, 1.0])
        assert_series_close(out, [None, None, 5.0, 4.0, 3.0])

    # --- L2 ---------------------------------------------------------------

    def test_null_in_window_yields_null(self):
        out_min = _eval(rolling_min(pl.col("x"), w=3),
                        x=[1.0, None, 3.0, 4.0, 5.0])
        assert out_min.to_list() == [None, None, None, None, 3.0]

    def test_negative_extremes(self):
        out_min = _eval(rolling_min(pl.col("x"), w=2), x=[-1.0, -1e9, 5.0])
        assert _close(out_min[1], -1e9)
        out_max = _eval(rolling_max(pl.col("x"), w=2), x=[1e9, 1.0, -5.0])
        assert _close(out_max[1], 1e9)

    # --- L3 ---------------------------------------------------------------

    def test_min_parity_with_pandas(self):
        rng = np.random.default_rng(44)
        n = 10_000
        x = rng.normal(size=n)

        df = pl.DataFrame({"x": x})
        out = df.select(rolling_min(pl.col("x"), w=25).alias("out"))["out"]

        expected = pd.Series(x).rolling(25, min_periods=25).min().to_numpy()
        actual = _to_nan_array(out)

        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_array_equal(actual[valid], expected[valid])

    def test_max_parity_with_pandas(self):
        rng = np.random.default_rng(55)
        n = 10_000
        x = rng.normal(size=n)

        df = pl.DataFrame({"x": x})
        out = df.select(rolling_max(pl.col("x"), w=25).alias("out"))["out"]

        expected = pd.Series(x).rolling(25, min_periods=25).max().to_numpy()
        actual = _to_nan_array(out)

        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_array_equal(actual[valid], expected[valid])
