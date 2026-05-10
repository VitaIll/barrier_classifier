"""Step 6 tests: specialized primitives.

rolling_quantile  — enforce linear interpolation (parity hazard).
rolling_mad       — two-pass median per window; not the same as
                    rolling_median(|x - rolling_median(x)|).
perm_entropy_m3   — m=3, tau=1; legacy parity oracle reproduces the
                    cumulative-sum approach used in
                    compute_permutation_entropy.
signed_run_*      — three primitives sharing a state machine; legacy
                    oracle is compute_event_features's loop.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.features.primitives import (
    _perm_entropy_m3_np,
    _rolling_mad_np,
    _signed_run_np,
    perm_entropy_m3,
    rolling_mad,
    rolling_quantile,
    signed_run_cumret,
    signed_run_dir,
    signed_run_length,
)
from src.utils import M as BOUNDARY_M

pytestmark = pytest.mark.step6


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
    out: list[float] = []
    for v in s.to_list():
        if v is None:
            out.append(float("nan"))
        else:
            out.append(float(v))
    return np.array(out)


# ===========================================================================
# rolling_quantile
# ===========================================================================


class TestRollingQuantile:
    """Enforces linear interpolation; matches numpy default."""

    # --- L1 ---------------------------------------------------------------

    def test_median_simple(self):
        out = _eval(
            rolling_quantile(pl.col("x"), w=3, q=0.5),
            x=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        # Window ending at row 2: [1,2,3] median=2; row 3: [2,3,4] median=3; etc.
        assert out[0] is None
        assert out[1] is None
        assert _close(out[2], 2.0)
        assert _close(out[3], 3.0)
        assert _close(out[4], 4.0)

    def test_q10_q50_q90_on_known_window(self):
        # Window [0,1,2,...,9] of length 10
        x = list(range(10))
        x = [float(v) for v in x]
        q10 = _eval(rolling_quantile(pl.col("x"), w=10, q=0.10), x=x)
        q50 = _eval(rolling_quantile(pl.col("x"), w=10, q=0.50), x=x)
        q90 = _eval(rolling_quantile(pl.col("x"), w=10, q=0.90), x=x)
        # Linear interpolation on [0..9] of length 10:
        # q10 = 0.9, q50 = 4.5, q90 = 8.1
        assert _close(q10[9], 0.9)
        assert _close(q50[9], 4.5)
        assert _close(q90[9], 8.1)

    # --- L2 ---------------------------------------------------------------

    def test_diverges_from_nearest_default(self):
        # Even-window median: linear gives midpoint, nearest gives one of the two.
        x = [1.0, 2.0]
        out_linear = _eval(rolling_quantile(pl.col("x"), w=2, q=0.5), x=x)
        out_nearest = _eval(
            pl.col("x").rolling_quantile(
                quantile=0.5, interpolation="nearest", window_size=2, min_samples=2,
            ),
            x=x,
        )
        # linear: 1.5; nearest: 1.0 or 2.0 (whichever polars picks)
        assert _close(out_linear[1], 1.5)
        assert not _close(out_nearest[1], 1.5)

    def test_null_in_window_yields_null(self):
        out = _eval(
            rolling_quantile(pl.col("x"), w=3, q=0.5),
            x=[1.0, None, 3.0, 4.0, 5.0],
        )
        assert out[0] is None
        assert out[1] is None
        assert out[2] is None  # window has null
        assert out[3] is None  # window has null
        assert out[4] is not None

    def test_constant_window(self):
        out = _eval(
            rolling_quantile(pl.col("x"), w=3, q=0.5),
            x=[5.0, 5.0, 5.0, 5.0],
        )
        assert _close(out[2], 5.0)
        assert _close(out[3], 5.0)

    # --- L3: numpy linear-interpolation parity ----------------------------

    def test_parity_with_numpy_linear(self):
        rng = np.random.default_rng(701)
        n = 5_000
        x = rng.normal(size=n)

        # Numpy oracle on every valid window
        w = 30
        expected = np.full(n, np.nan)
        for i in range(w - 1, n):
            expected[i] = np.quantile(x[i - w + 1 : i + 1], 0.25, method="linear")

        df = pl.DataFrame({"x": x})
        out = _to_nan_array(
            df.select(rolling_quantile(pl.col("x"), w=w, q=0.25).alias("out"))["out"]
        )

        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


# ===========================================================================
# rolling_mad
# ===========================================================================


class TestRollingMad:
    """Two-pass median per window; matches utils.compute_quantile_features."""

    # --- L1 ---------------------------------------------------------------

    def test_constant_window_yields_zero(self):
        out = _eval(rolling_mad(pl.col("x"), w=3),
                    x=[5.0, 5.0, 5.0, 5.0])
        assert math.isnan(out[0])
        assert math.isnan(out[1])
        assert _close(out[2], 0.0)
        assert _close(out[3], 0.0)

    def test_known_mad(self):
        # Window [1, 2, 3]: median=2; abs_devs=[1, 0, 1]; median(devs)=1
        out = _eval(rolling_mad(pl.col("x"), w=3), x=[1.0, 2.0, 3.0])
        assert _close(out[2], 1.0)

    def test_known_mad_even_window(self):
        # Window [1, 2, 3, 4]: median=2.5 (linear); devs=[1.5, 0.5, 0.5, 1.5];
        # MAD = median(devs) = 1.0 (linear interp on [0.5, 0.5, 1.5, 1.5] → 1.0)
        out = _eval(rolling_mad(pl.col("x"), w=4), x=[1.0, 2.0, 3.0, 4.0])
        assert _close(out[3], 1.0)

    # --- L2 ---------------------------------------------------------------

    def test_nan_in_window_yields_nan(self):
        out = _eval(rolling_mad(pl.col("x"), w=3),
                    x=[1.0, float("nan"), 3.0, 4.0, 5.0])
        assert math.isnan(out[2])
        assert math.isnan(out[3])
        assert not math.isnan(out[4])

    def test_window_larger_than_series(self):
        out = _eval(rolling_mad(pl.col("x"), w=10),
                    x=[1.0, 2.0, 3.0])
        for v in out.to_list():
            assert v is None or math.isnan(v)

    def test_differs_from_rolling_median_of_rolling_median(self):
        # Confirm rolling_mad ≠ rolling_median(|x - rolling_median(x)|).
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        w = 4
        ours = _to_nan_array(_eval(rolling_mad(pl.col("x"), w=w), x=x))
        # The "wrong" formula: outer median of |x - inner_rolling_median|
        x_arr = np.array(x)
        inner_med = np.full(len(x_arr), np.nan)
        for i in range(w - 1, len(x_arr)):
            inner_med[i] = np.quantile(x_arr[i - w + 1 : i + 1], 0.5, method="linear")
        abs_dev = np.abs(x_arr - inner_med)
        wrong = np.full(len(x_arr), np.nan)
        for i in range(w - 1, len(x_arr)):
            window = abs_dev[i - w + 1 : i + 1]
            if not np.isnan(window).any():
                wrong[i] = np.quantile(window, 0.5, method="linear")
        # The two arrays should differ at some valid row.
        valid = ~np.isnan(ours)
        assert not np.allclose(ours[valid], wrong[valid])

    # --- L3: legacy parity at boundary indices ----------------------------

    def test_parity_with_legacy_at_boundary_indices(self):
        """Legacy compute_quantile_features computes MAD only at boundary
        indices (every M=20 rows). At those indices our output must match."""
        rng = np.random.default_rng(702)
        n = 5_000
        x = rng.normal(size=n)
        w = 60

        # Polars: row-by-row
        df = pl.DataFrame({"x": x})
        ours = _to_nan_array(
            df.select(rolling_mad(pl.col("x"), w=w).alias("out"))["out"]
        )

        # Legacy boundary-sparse oracle (literal copy of utils.py:1654-1687 logic).
        bidx = np.arange(0, n, BOUNDARY_M, dtype=np.int64)
        eligible = bidx[bidx >= w - 1]
        offsets = np.arange(w - 1, -1, -1, dtype=np.int64)
        rows = eligible[:, None] - offsets[None, :]
        window_vals = x[rows]
        invalid = np.isnan(window_vals).any(axis=1)
        medians = np.quantile(window_vals, 0.5, axis=1, method="linear")
        abs_devs = np.abs(window_vals - medians[:, None])
        mads = np.quantile(abs_devs, 0.5, axis=1, method="linear")
        mads[invalid] = np.nan

        np.testing.assert_allclose(ours[eligible], mads, rtol=1e-12, atol=1e-12)


# ===========================================================================
# perm_entropy_m3
# ===========================================================================


class TestPermEntropyM3:
    """Rolling normalized permutation entropy, m=3, tau=1."""

    # --- L1 ---------------------------------------------------------------

    def test_monotone_increasing_yields_zero_entropy(self):
        # All ordinal patterns are (0,1,2): only one unique pattern → H=0.
        x = [float(i) for i in range(50)]
        out = _eval(perm_entropy_m3(pl.col("x"), w=32), x=x)
        # Output starts being non-null at index w-1 = 31
        assert _close(out[31], 0.0)

    def test_monotone_decreasing_yields_zero_entropy(self):
        x = [float(i) for i in reversed(range(50))]
        out = _eval(perm_entropy_m3(pl.col("x"), w=32), x=x)
        assert _close(out[31], 0.0)

    def test_max_entropy_uniform_random(self):
        # With enough data and a balanced distribution of patterns, the
        # normalized entropy should approach 1.0.
        rng = np.random.default_rng(801)
        n = 2_000
        x = rng.normal(size=n)
        out = _eval(perm_entropy_m3(pl.col("x"), w=600), x=list(x))
        # At the last row we have a 600-bar window → ~598 patterns →
        # all 6 codes appear → near-max entropy.
        assert out[n - 1] > 0.95

    # --- L2 ---------------------------------------------------------------

    def test_window_too_small_returns_all_null(self):
        # n_patterns = w - 2 must be >= 30.
        x = [float(i) for i in range(40)]
        out = _eval(perm_entropy_m3(pl.col("x"), w=10), x=x)
        for v in out.to_list():
            assert v is None or math.isnan(v)

    def test_nan_in_window_yields_nan(self):
        x = [float(i) for i in range(50)]
        x[20] = float("nan")
        out = _eval(perm_entropy_m3(pl.col("x"), w=32), x=x)
        # NaN at position 20 corrupts every window that contains row 20.
        # Pattern indices i where window includes 20: i in [max(0, 18), min(46, 49)]
        # In bar-position terms, end_bar = pattern_idx + 2.
        # Affected bar positions: 20..47ish. Sample some:
        assert math.isnan(out[31])  # window 0..31 includes idx 20
        assert math.isnan(out[40])

    # --- L3: legacy parity ------------------------------------------------

    def test_parity_with_legacy(self):
        from src.utils import compute_permutation_entropy as legacy_fn
        rng = np.random.default_rng(802)
        n = 2_000
        r = rng.normal(size=n)

        w = 240
        df_in = pd.DataFrame({"r": r})
        legacy_df = legacy_fn(df_in, [w])
        col = f"pentropy_norm__inst__f__w{w}__m3__tau1"
        expected = legacy_df[col].to_numpy()

        df = pl.DataFrame({"x": r})
        out = _to_nan_array(
            df.select(perm_entropy_m3(pl.col("x"), w=w).alias("out"))["out"]
        )

        np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(out[valid], expected[valid], rtol=1e-12, atol=1e-12)


# ===========================================================================
# signed_run_dir / length / cumret
# ===========================================================================


class TestSignedRun:
    """State machine: same-sign extends, opposite flips, NaN/zero resets."""

    # --- L1 ---------------------------------------------------------------

    def test_simple_uptrend(self):
        x = [0.1, 0.2, 0.3]
        d = _eval(signed_run_dir(pl.col("x")), x=x).to_list()
        l = _eval(signed_run_length(pl.col("x")), x=x).to_list()
        c = _eval(signed_run_cumret(pl.col("x")), x=x).to_list()
        assert d == [1, 1, 1]
        assert l == [1, 2, 3]
        assert all(_close(a, e) for a, e in zip(c, [0.1, 0.3, 0.6]))

    def test_flip_direction(self):
        x = [0.1, 0.2, -0.1]
        d = _eval(signed_run_dir(pl.col("x")), x=x).to_list()
        l = _eval(signed_run_length(pl.col("x")), x=x).to_list()
        c = _eval(signed_run_cumret(pl.col("x")), x=x).to_list()
        assert d == [1, 1, -1]
        assert l == [1, 2, 1]
        assert all(_close(a, e) for a, e in zip(c, [0.1, 0.3, -0.1]))

    def test_zero_resets(self):
        x = [0.1, 0.0, 0.2]
        d = _eval(signed_run_dir(pl.col("x")), x=x).to_list()
        l = _eval(signed_run_length(pl.col("x")), x=x).to_list()
        c = _eval(signed_run_cumret(pl.col("x")), x=x).to_list()
        assert d == [1, 0, 1]
        assert l == [1, 0, 1]
        assert all(_close(a, e) for a, e in zip(c, [0.1, 0.0, 0.2]))

    def test_nan_resets(self):
        x = [0.1, float("nan"), 0.2]
        d = _eval(signed_run_dir(pl.col("x")), x=x).to_list()
        l = _eval(signed_run_length(pl.col("x")), x=x).to_list()
        c = _eval(signed_run_cumret(pl.col("x")), x=x).to_list()
        assert d == [1, 0, 1]
        assert l == [1, 0, 1]
        assert all(_close(a, e) for a, e in zip(c, [0.1, 0.0, 0.2]))

    # --- L2 ---------------------------------------------------------------

    def test_all_zero_yields_zero_state(self):
        x = [0.0, 0.0, 0.0]
        d = _eval(signed_run_dir(pl.col("x")), x=x).to_list()
        l = _eval(signed_run_length(pl.col("x")), x=x).to_list()
        c = _eval(signed_run_cumret(pl.col("x")), x=x).to_list()
        assert d == [0, 0, 0]
        assert l == [0, 0, 0]
        assert all(_close(a, 0.0) for a in c)

    def test_all_nan_yields_zero_state(self):
        x = [float("nan"), float("nan"), float("nan")]
        d = _eval(signed_run_dir(pl.col("x")), x=x).to_list()
        l = _eval(signed_run_length(pl.col("x")), x=x).to_list()
        assert d == [0, 0, 0]
        assert l == [0, 0, 0]

    def test_inf_resets(self):
        x = [0.1, float("inf"), 0.2]
        d = _eval(signed_run_dir(pl.col("x")), x=x).to_list()
        # inf is non-finite → reset
        assert d == [1, 0, 1]

    def test_long_run_does_not_overflow_int32(self):
        # int32 max ~2.1B; we only ever see ~1.5M bars.
        x = [0.1] * 1000
        l = _eval(signed_run_length(pl.col("x")), x=x).to_list()
        assert l[-1] == 1000

    # --- L3: legacy parity ------------------------------------------------

    def test_parity_with_legacy_compute_event_features(self):
        from src.utils import compute_event_features as legacy_fn
        rng = np.random.default_rng(902)
        n = 5_000
        r = rng.normal(size=n) * 0.001
        # Inject zeros and NaNs to exercise resets
        r[100] = 0.0
        r[200] = float("nan")
        r[1000:1010] = 0.0

        legacy_df = legacy_fn(pd.DataFrame({"r": r}))
        d_exp = legacy_df["event__run_dir__f__w0"].to_numpy()
        l_exp = legacy_df["event__run_len__f__w0"].to_numpy()
        c_exp = legacy_df["event__run_cumret__f__w0"].to_numpy()

        df = pl.DataFrame({"r": r})
        d_out = df.select(signed_run_dir(pl.col("r")).alias("out"))["out"].to_numpy()
        l_out = df.select(signed_run_length(pl.col("r")).alias("out"))["out"].to_numpy()
        c_out = df.select(signed_run_cumret(pl.col("r")).alias("out"))["out"].to_numpy()

        np.testing.assert_array_equal(d_out, d_exp)
        np.testing.assert_array_equal(l_out, l_exp)
        np.testing.assert_allclose(c_out, c_exp, rtol=1e-12, atol=1e-15)
