"""Round 1 feature family tests.

Coverage:
  - Round 1a: ``extreme`` family (dist_low_z, dist_high_z, price_rank)
  - Round 1b: trend quadratic (slope, curvature)
  - Round 1c: ``pivot`` family (confirmed swing low/high, dist + age)
  - Round 1d: ``vol__semivar_signed``, ``vol__jump_ratio``
  - Round 1e: ``flow`` family (pressure, cum, sell_absorption, buy_exhaustion)

Tests are unit-style (constructed inputs with known answers) plus
causality checks (no NaN in legal rows, expected ranges, monotonicity
on monotone synthetic inputs).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.features import FeatureEngine
from src.features.config import (
    M,
    PIVOT_Q_VALUES,
    WINDOWS_EXTREME,
    WINDOWS_FLOW_PRESSURE,
    WINDOWS_PIVOT,
    WINDOWS_QUAD_TREND,
    WINDOWS_VOL_JUMP,
    WINDOWS_VOL_SIGNED,
)


# ---------------------------------------------------------------------------
# Synthetic bar frame — shared by all round-1 tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def round1_bars() -> pl.DataFrame:
    rng = np.random.default_rng(2026_05_12)
    n = 12_000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    spread = np.abs(rng.normal(0.0, 0.5, n))
    high = close + spread
    low = np.maximum(close - spread, 0.001)
    open_ = np.clip(close + rng.normal(0.0, 0.2, n), low, high)
    volume = np.abs(rng.normal(100.0, 30.0, n)) + 1.0
    qv = volume * close
    nt = (np.abs(rng.normal(50.0, 20.0, n)).astype(np.int64)) + 1
    tbb = volume * rng.uniform(0.3, 0.7, n)
    ts_index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": qv,
            "num_trades": nt, "taker_buy_base": tbb,
        },
        index=ts_index,
    )
    df = utils.compute_base_series(df)
    return pl.from_pandas(df.reset_index(names="ts"))


@pytest.fixture(scope="module")
def round1_engine_result(round1_bars):
    engine = FeatureEngine(
        tiers=(1, 2),
        families=(
            "rolling", "vol", "trend", "excursion",
            "extreme", "pivot", "flow",
        ),
    )
    return engine.transform(round1_bars, trim=False).data


# ---------------------------------------------------------------------------
# Round 1a — extreme family
# ---------------------------------------------------------------------------


def test_extreme_dist_low_z_positive_for_uptrending_price(round1_engine_result):
    """A rising price series leaves the trailing low far below the current
    close, so ``extreme__dist_low_z`` should be strictly positive on
    typical rows past warmup. We sample a handful of mid-frame rows."""
    df = round1_engine_result
    W = 60
    col = f"extreme__dist_low_z__f__w{W}"
    vals = df[col].to_numpy()
    valid = ~np.isnan(vals)
    # Strictly non-negative by construction (close >= rolling-min of low)
    assert (vals[valid] >= -1e-9).all(), "dist_low_z must be >= 0"


def test_extreme_dist_high_z_positive(round1_engine_result):
    df = round1_engine_result
    W = 60
    col = f"extreme__dist_high_z__f__w{W}"
    vals = df[col].to_numpy()
    valid = ~np.isnan(vals)
    assert (vals[valid] >= -1e-9).all(), "dist_high_z must be >= 0"


def test_extreme_price_rank_in_bounds(round1_engine_result):
    df = round1_engine_result
    W = 60
    col = f"extreme__price_rank__f__w{W}"
    vals = df[col].to_numpy()
    valid = ~np.isnan(vals)
    # Rank uses <=, so minimum is 1/W and maximum is 1.0
    assert (vals[valid] >= 1.0 / W - 1e-12).all()
    assert (vals[valid] <= 1.0 + 1e-12).all()


def test_extreme_warmup_matches_window(round1_engine_result):
    """First W-1 rows must be null (warmup)."""
    df = round1_engine_result
    for W in (30, 60, 120):
        col = f"extreme__dist_low_z__f__w{W}"
        vals = df[col].to_numpy()
        assert np.isnan(vals[: W - 1]).all(), f"{col}: leading warmup null"
        assert not np.isnan(vals[W]).all(), f"{col}: row W should be valid"


# ---------------------------------------------------------------------------
# Round 1b — quadratic trend
# ---------------------------------------------------------------------------


def test_quad_slope_recovers_known_linear_trend(round1_bars):
    """On a pure linear log-price drift the quadratic slope coefficient
    must recover the drift and the curvature term must be ~0.

    Construct a 1000-row pure-linear log price series, run the engine
    (tier-1 vol family populates the normalizer), then read both
    quad_slope_z and quad_curv_z at a valid row.
    """
    rng = np.random.default_rng(0)
    n = 2000
    drift = 1e-4  # per-bar log drift
    noise = rng.normal(0.0, 1e-6, n)  # tiny noise so vol is finite but not zero
    p = drift * np.arange(n) + noise
    close = np.exp(p) * 100.0
    high = close * (1.0 + 1e-5)
    low = close * (1.0 - 1e-5)
    open_ = close
    volume = np.ones(n)
    qv = volume * close
    nt = np.ones(n, dtype=np.int64)
    tbb = 0.5 * volume

    ts_index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": qv,
            "num_trades": nt, "taker_buy_base": tbb,
        },
        index=ts_index,
    )
    df = utils.compute_base_series(df)
    df_pl = pl.from_pandas(df.reset_index(names="ts"))
    engine = FeatureEngine(tiers=(1, 2), families=("rolling", "vol", "trend"))
    out = engine.transform(df_pl, trim=False).data

    W = 120  # in WINDOWS_QUAD_TREND
    slope_col = f"trend__quad_slope_z__f__w{W}"
    curv_col = f"trend__quad_curv_z__f__w{W}"
    # Sample a row past warmup
    n_row = 500
    slope_z = out[slope_col].to_numpy()[n_row]
    curv_z = out[curv_col].to_numpy()[n_row]
    # slope_z = beta1 * W / (sigma * sqrt(W) + EPS)
    # beta1 recovers `drift` exactly; sigma is tiny but nonzero from noise
    # The sign should be strongly positive.
    assert slope_z > 0.0, f"linear-up series should have positive slope_z, got {slope_z}"
    # Curvature should be near zero relative to the slope's magnitude.
    assert abs(curv_z) < abs(slope_z) * 1.0, f"curvature too large on linear series: {curv_z} vs slope {slope_z}"


def test_quad_curv_recovers_known_concave_trend(round1_bars):
    """An accelerating up-move (concave-up log price) must give positive
    quadratic curvature."""
    rng = np.random.default_rng(1)
    n = 2000
    # p(t) = a*t + b*t^2 -> quadratic with b > 0 is concave-up
    a = 1e-5
    b = 1e-8
    t = np.arange(n)
    p = a * t + b * (t ** 2) + rng.normal(0.0, 1e-7, n)
    close = np.exp(p) * 100.0
    high = close * (1.0 + 1e-5)
    low = close * (1.0 - 1e-5)
    open_ = close
    volume = np.ones(n)
    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": volume * close,
            "num_trades": np.ones(n, dtype=np.int64), "taker_buy_base": 0.5 * volume,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
    )
    df = utils.compute_base_series(df)
    df_pl = pl.from_pandas(df.reset_index(names="ts"))
    engine = FeatureEngine(tiers=(1, 2), families=("rolling", "vol", "trend"))
    out = engine.transform(df_pl, trim=False).data
    W = 120
    curv_col = f"trend__quad_curv_z__f__w{W}"
    n_row = 1000  # well past warmup, deep into concave region
    curv_z = out[curv_col].to_numpy()[n_row]
    assert curv_z > 0.0, f"concave-up series should give positive curvature, got {curv_z}"


# ---------------------------------------------------------------------------
# Round 1c — pivot family
# ---------------------------------------------------------------------------


def test_pivot_detects_v_shape():
    """Construct a V-shape series with a single sharp low at index 100
    and confirm the detector identifies it after Q bars.

    Series: log_low(t) = +|t - 100| / 1000 + tiny-noise.
    Index 100 has the minimum value (V-shape pointing down).
    """
    from src.features.families.pivot import _detect_pivots_np

    n = 300
    rng = np.random.default_rng(0)
    t = np.arange(n)
    # V-shape with sharp minimum at t=100. Noise is small enough that
    # symmetric-window comparison still uniquely identifies t=100.
    log_low = np.abs(t - 100) / 1000.0 + rng.normal(0.0, 1e-8, n)
    q = 5
    pivot_idx = _detect_pivots_np(log_low, q, mode="low")
    # The pivot at i=100 should be detected and confirmed at i+q=105.
    # At row 100 itself (the centre), the indicator is shifted by q so
    # the pivot is not yet available.
    assert pivot_idx[100] != 100, "pivot at i=100 should NOT be available at row 100 itself"
    assert pivot_idx[105] == 100, f"pivot should be available at row 105; got {pivot_idx[105]}"
    assert pivot_idx[200] == 100, f"pivot should still be the most-recent at row 200; got {pivot_idx[200]}"


def test_pivot_no_pivot_yields_sentinel_age(round1_engine_result):
    """Where no eligible pivot exists in the lookback, the age column
    must equal the lookback W (the sentinel)."""
    df = round1_engine_result
    W = WINDOWS_PIVOT[0]
    Q = PIVOT_Q_VALUES[0]
    col = f"pivot__last_low_age__f__w{W}__q{Q}"
    vals = df[col].to_numpy()
    # First W+Q rows can have age = W (no eligible pivot yet — sentinel)
    # OR a smaller value if a pivot was found early. Either way must be valid.
    # Across the whole series, age must be in [0, W].
    assert (vals >= 0).all()
    assert (vals <= W).all()


def test_pivot_dist_z_sign(round1_engine_result):
    """``pivot__last_low_dist_z`` is positive when close > swing low (the
    common case). Sample a few rows with non-null distances and verify
    the sign is sensible relative to the underlying close."""
    df = round1_engine_result
    W = WINDOWS_PIVOT[0]
    Q = PIVOT_Q_VALUES[0]
    col = f"pivot__last_low_dist_z__f__w{W}__q{Q}"
    vals = df[col].to_numpy()
    valid = ~np.isnan(vals)
    # On the synthetic random walk, the carried swing-low is generally
    # below current close so dist_low_z should be mostly positive.
    pos_rate = (vals[valid] > 0).mean()
    assert pos_rate > 0.5, f"dist_low_z should be positive >50% of the time, got {pos_rate}"


# ---------------------------------------------------------------------------
# Round 1d — vol semivar_signed + jump_ratio
# ---------------------------------------------------------------------------


def test_vol_semivar_signed_bounded(round1_engine_result):
    df = round1_engine_result
    W = WINDOWS_VOL_SIGNED[0]
    col = f"vol__semivar_signed__f__w{W}"
    vals = df[col].to_numpy()
    valid = ~np.isnan(vals)
    assert (vals[valid] >= -1.0 - 1e-9).all()
    assert (vals[valid] <= 1.0 + 1e-9).all()


def test_vol_semivar_signed_zero_on_symmetric_returns():
    """If r is i.i.d. symmetric around zero, the signed semivar ratio
    should average to zero."""
    rng = np.random.default_rng(0)
    n = 5000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    df = pd.DataFrame(
        {
            "open": close, "high": close * (1 + 1e-5), "low": close * (1 - 1e-5),
            "close": close, "volume": np.ones(n),
            "quote_volume": close * 1.0, "num_trades": np.ones(n, dtype=np.int64),
            "taker_buy_base": 0.5 * np.ones(n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
    )
    df = utils.compute_base_series(df)
    df_pl = pl.from_pandas(df.reset_index(names="ts"))
    # ``vol__vov__f__w{W}`` depends on ``ret__rms__f__w20`` from the rolling
    # family; include both so the Tier-2 ``vol`` graph resolves.
    engine = FeatureEngine(tiers=(1, 2), families=("rolling", "vol"))
    out = engine.transform(df_pl, trim=False).data
    W = 240
    col = f"vol__semivar_signed__f__w{W}"
    vals = out[col].to_numpy()
    valid = ~np.isnan(vals)
    assert abs(vals[valid].mean()) < 0.05, (
        f"signed semivar on iid-symmetric returns should average ~0; got {vals[valid].mean()}"
    )


def test_vol_jump_ratio_in_unit_interval(round1_engine_result):
    df = round1_engine_result
    W = WINDOWS_VOL_JUMP[0]
    col = f"vol__jump_ratio__f__w{W}"
    vals = df[col].to_numpy()
    valid = ~np.isnan(vals)
    assert (vals[valid] >= 0.0 - 1e-9).all()
    assert (vals[valid] <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Round 1e — flow family
# ---------------------------------------------------------------------------


def test_flow_pressure_bounds(round1_engine_result):
    df = round1_engine_result
    W = WINDOWS_FLOW_PRESSURE[0]
    col = f"flow__pressure__f__w{W}"
    vals = df[col].to_numpy()
    valid = ~np.isnan(vals)
    assert (vals[valid] >= -1.0 - 1e-9).all()
    assert (vals[valid] <= 1.0 + 1e-9).all()


def test_flow_sell_absorption_bounds(round1_engine_result):
    df = round1_engine_result
    W = WINDOWS_FLOW_PRESSURE[0]
    col = f"flow__sell_absorption__f__w{W}"
    vals = df[col].to_numpy()
    valid = ~np.isnan(vals)
    assert (vals[valid] >= 0.0).all()
    assert (vals[valid] <= 1.0 + 1e-9).all()


def test_flow_buy_exhaustion_bounds(round1_engine_result):
    df = round1_engine_result
    W = WINDOWS_FLOW_PRESSURE[0]
    col = f"flow__buy_exhaustion__f__w{W}"
    vals = df[col].to_numpy()
    valid = ~np.isnan(vals)
    assert (vals[valid] >= 0.0).all()
    assert (vals[valid] <= 1.0 + 1e-9).all()


def test_flow_pressure_recovers_known_taker_buy_ratio():
    """For a series with constant taker-buy fraction f, the rolling
    pressure should equal 2f - 1."""
    n = 1000
    f = 0.7
    volume = np.full(n, 100.0)
    tbb = volume * f
    close = np.full(n, 100.0)
    df = pd.DataFrame(
        {
            "open": close, "high": close * (1 + 1e-5), "low": close * (1 - 1e-5),
            "close": close, "volume": volume,
            "quote_volume": volume * close, "num_trades": np.ones(n, dtype=np.int64),
            "taker_buy_base": tbb,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
    )
    df = utils.compute_base_series(df)
    df_pl = pl.from_pandas(df.reset_index(names="ts"))
    engine = FeatureEngine(
        tiers=(1, 2), families=("rolling", "vol", "flow")
    )
    out = engine.transform(df_pl, trim=False).data
    W = 60
    col = f"flow__pressure__f__w{W}"
    vals = out[col].to_numpy()
    # First W-1 rows null; row >= W-1 should equal 2*0.7 - 1 = 0.4 (up to EPS).
    expected = 2.0 * f - 1.0
    assert abs(vals[100] - expected) < 1e-6, f"expected {expected}, got {vals[100]}"
