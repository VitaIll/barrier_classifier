"""Step 5 tests: custom primitives (no polars native equivalent).

population_corr — rolling Pearson with ddof=0 variance, mask on var==0.
                  Legacy oracle: utils._rolling_corr_population.

wilder_smooth  — alpha=1/W EW seeded with SMA of first W values. Legacy
                 _wilder_rsi has an off-by-one offset (seed at index W
                 from mean(x[1:W+1])); the RSI feature compensates with
                 shift(-1) / shift(1). For the standalone primitive we
                 match a numpy reference.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.features.primitives import (
    _wilder_smooth_np,
    population_corr,
    wilder_smooth,
)
from src.utils import _rolling_corr_population, _wilder_rsi  # legacy oracles

pytestmark = pytest.mark.step5


# ---------------------------------------------------------------------------
# Helpers
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


def _to_nan_array(s: pl.Series) -> np.ndarray:
    return np.array([v if v is not None else float("nan") for v in s.to_list()])


# ===========================================================================
# population_corr
# ===========================================================================


class TestPopulationCorr:
    """Rolling population (ddof=0) Pearson correlation."""

    # --- L1: hand-computed ------------------------------------------------

    def test_perfect_positive_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # y = 2x; perfect correlation
        out = _eval(population_corr(pl.col("x"), pl.col("y"), w=5),
                    x=x, y=y)
        assert _close(out[4], 1.0)

    def test_perfect_negative_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [-1.0, -2.0, -3.0, -4.0, -5.0]
        out = _eval(population_corr(pl.col("x"), pl.col("y"), w=5),
                    x=x, y=y)
        assert _close(out[4], -1.0)

    def test_zero_correlation_orthogonal(self):
        # Centred orthogonal vectors: x = [-1, 0, 1, 0], y = [0, 1, 0, -1]
        x = [-1.0, 0.0, 1.0, 0.0]
        y = [0.0, 1.0, 0.0, -1.0]
        out = _eval(population_corr(pl.col("x"), pl.col("y"), w=4),
                    x=x, y=y)
        assert _close(out[3], 0.0, atol=1e-12)

    def test_warmup_rows_null(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        out = _eval(population_corr(pl.col("x"), pl.col("y"), w=4),
                    x=x, y=y)
        for i in range(3):
            assert out[i] is None
        assert out[3] is not None
        assert out[4] is not None

    # --- L2: adversarial --------------------------------------------------

    def test_constant_x_yields_null(self):
        out = _eval(population_corr(pl.col("x"), pl.col("y"), w=4),
                    x=[5.0, 5.0, 5.0, 5.0],
                    y=[1.0, 2.0, 3.0, 4.0])
        # var_x = 0 -> guard fails -> null
        assert out[3] is None

    def test_constant_y_yields_null(self):
        out = _eval(population_corr(pl.col("x"), pl.col("y"), w=4),
                    x=[1.0, 2.0, 3.0, 4.0],
                    y=[5.0, 5.0, 5.0, 5.0])
        assert out[3] is None

    def test_both_constant_yields_null(self):
        out = _eval(population_corr(pl.col("x"), pl.col("y"), w=3),
                    x=[7.0, 7.0, 7.0],
                    y=[3.0, 3.0, 3.0])
        assert out[2] is None

    def test_null_in_window_yields_null(self):
        out = _eval(population_corr(pl.col("x"), pl.col("y"), w=3),
                    x=[1.0, None, 3.0, 4.0, 5.0],
                    y=[2.0, 4.0, 6.0, 8.0, 10.0])
        assert out[3] is None  # x[3] window contains x[1]=null
        assert out[4] is not None  # window x[2..4] is fully valid

    def test_window_larger_than_series(self):
        out = _eval(population_corr(pl.col("x"), pl.col("y"), w=10),
                    x=[1.0, 2.0, 3.0],
                    y=[1.0, 2.0, 3.0])
        assert out.to_list() == [None, None, None]

    # --- L3: legacy oracle parity -----------------------------------------

    def test_parity_with_legacy_oracle(self):
        rng = np.random.default_rng(303)
        n = 10_000
        x = rng.normal(size=n)
        y = 0.6 * x + 0.4 * rng.normal(size=n)  # ~0.6 correlation
        # Inject adversarial patches
        y[100:115] = y[100]  # constant patch in y
        x[200:215] = x[200]  # constant patch in x

        # Legacy oracle (pandas)
        expected = _rolling_corr_population(pd.Series(x), pd.Series(y), 30).to_numpy()

        # Polars
        df = pl.DataFrame({"x": x, "y": y})
        out = df.select(
            population_corr(pl.col("x"), pl.col("y"), w=30).alias("out")
        )["out"]
        actual = _to_nan_array(out)

        # Same null/NaN positions
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        # Same values where defined
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-10, atol=1e-12)

    def test_parity_acf1_pattern(self):
        # Mimics utils.py:1953: _rolling_corr_population(r, r.shift(1), W)
        rng = np.random.default_rng(404)
        n = 10_000
        r = rng.normal(size=n)

        expected = _rolling_corr_population(
            pd.Series(r), pd.Series(r).shift(1), 60
        ).to_numpy()

        df = pl.DataFrame({"r": r})
        out = df.select(
            population_corr(pl.col("r"), pl.col("r").shift(1), w=60).alias("out")
        )["out"]
        actual = _to_nan_array(out)

        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-10, atol=1e-12)


# ===========================================================================
# wilder_smooth (standard form; RSI offset handled at the feature layer)
# ===========================================================================


class TestWilderSmooth:
    """Standard Wilder smoothing: seed at index W-1 = mean(x[0:W])."""

    # --- L1: hand-computed ------------------------------------------------

    def test_constant_series_passes_through(self):
        out = _eval(wilder_smooth(pl.col("x"), w=3),
                    x=[5.0, 5.0, 5.0, 5.0, 5.0])
        # Seed at index 2: mean(5,5,5) = 5; recursion preserves 5.
        assert out[0] is None or math.isnan(out[0])
        assert out[1] is None or math.isnan(out[1])
        for i in range(2, 5):
            assert _close(out[i], 5.0)

    def test_seed_is_simple_mean_of_first_w(self):
        x = [1.0, 2.0, 3.0, 4.0]
        out = _eval(wilder_smooth(pl.col("x"), w=3), x=x)
        # Seed at index 2 = mean(1, 2, 3) = 2.0
        assert _close(out[2], 2.0)
        # Index 3: ((3-1)*2.0 + 4) / 3 = (4 + 4)/3 = 8/3
        assert _close(out[3], 8.0 / 3.0)

    def test_recursive_step(self):
        # x = [10, 20, 30], w=2
        # Seed at index 1: mean(10, 20) = 15
        # Index 2: (1*15 + 30) / 2 = 22.5
        x = [10.0, 20.0, 30.0]
        out = _eval(wilder_smooth(pl.col("x"), w=2), x=x)
        assert _close(out[1], 15.0)
        assert _close(out[2], 22.5)

    def test_warmup_rows_nan(self):
        out = _eval(wilder_smooth(pl.col("x"), w=4),
                    x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        for i in range(3):
            v = out[i]
            assert v is None or math.isnan(v)
        assert out[3] is not None and not math.isnan(out[3])

    # --- L2: adversarial --------------------------------------------------

    def test_window_larger_than_series(self):
        out = _eval(wilder_smooth(pl.col("x"), w=10),
                    x=[1.0, 2.0, 3.0])
        for v in out.to_list():
            assert v is None or math.isnan(v)

    def test_window_equals_series_length(self):
        x = [1.0, 2.0, 3.0]
        out = _eval(wilder_smooth(pl.col("x"), w=3), x=x)
        # Only index 2 has a value: mean(1, 2, 3) = 2.0
        assert out[0] is None or math.isnan(out[0])
        assert out[1] is None or math.isnan(out[1])
        assert _close(out[2], 2.0)

    def test_nan_in_seed_propagates(self):
        # If any of the first W values is NaN, seed becomes NaN, and
        # recursion propagates NaN forever after.
        out = _eval(wilder_smooth(pl.col("x"), w=3),
                    x=[1.0, float("nan"), 3.0, 4.0, 5.0])
        # Seed = mean(1, NaN, 3) = NaN
        for i in range(2, 5):
            v = out[i]
            assert v is None or math.isnan(v)

    def test_zero_input_yields_zero(self):
        out = _eval(wilder_smooth(pl.col("x"), w=3),
                    x=[0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(2, 5):
            assert _close(out[i], 0.0)

    def test_negative_input(self):
        # Wilder is linear; negative values handled normally.
        x = [-1.0, -2.0, -3.0, -4.0]
        out = _eval(wilder_smooth(pl.col("x"), w=3), x=x)
        assert _close(out[2], -2.0)  # mean(-1, -2, -3)
        assert _close(out[3], (2 * -2.0 + -4.0) / 3.0)

    # --- L3: numpy reference parity ---------------------------------------

    def test_parity_with_numpy_reference(self):
        rng = np.random.default_rng(505)
        n = 10_000
        x = rng.normal(size=n)

        expected = _wilder_smooth_np(x, 14)

        df = pl.DataFrame({"x": x})
        out = df.select(wilder_smooth(pl.col("x"), w=14).alias("out"))["out"]
        actual = _to_nan_array(out)

        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-12, atol=1e-12)

    def test_rsi_offset_pattern_matches_legacy(self):
        """Document and verify the shift trick that bridges standard
        wilder_smooth to the RSI-specific seed offset (utils.py:1835).

        Legacy seeds at index W using ``mean(gain[1:W+1])``; we achieve
        the same result by applying standard wilder_smooth to ``gain.shift(-1)``
        and then shifting the output back by +1.
        """
        rng = np.random.default_rng(606)
        n = 1000
        # Mimic the gain channel: r > 0 picks gain, else 0; r[0] = NaN.
        r = rng.normal(size=n)
        r[0] = float("nan")
        gain = np.where(r > 0, r, 0.0)
        # In the legacy, gain[0] = 0 (because NaN > 0 is False).

        # Legacy avg_gain via _wilder_rsi internals (we replicate inline).
        W = 14
        legacy = np.full(n, np.nan, dtype=float)
        legacy[W] = float(np.mean(gain[1: W + 1]))
        for i in range(W + 1, n):
            legacy[i] = ((W - 1) * legacy[i - 1] + gain[i]) / W

        # Polars: shift-trick around standard wilder_smooth.
        df = pl.DataFrame({"gain": gain})
        out = df.select(
            wilder_smooth(pl.col("gain").shift(-1), w=W)
            .shift(1)
            .alias("out")
        )["out"]
        actual = _to_nan_array(out)

        # Compare from index W onwards (both are NaN before W).
        np.testing.assert_allclose(actual[W:], legacy[W:], rtol=1e-12, atol=1e-12)
