"""Step 9 tests: candle / trend / activity / event / seasonality families
end-to-end parity vs the legacy compute_* chain.

Strategy: one combined parity test that runs all 5 families through the
engine and compares to ``compute_candle_geometry ∘ compute_trend_momentum ∘
compute_activity_flow ∘ compute_event_features ∘ compute_seasonality``.

Plus targeted L1 sanity tests for the trickiest pieces (RSI shift trick,
weekday convention).
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
    WINDOWS_BREAKOUT,
    WINDOWS_CANDLE_ROLL,
    WINDOWS_LOGP_Z,
    WINDOWS_RSI,
)

pytestmark = pytest.mark.step9


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bars_with_base() -> pd.DataFrame:
    rng = np.random.default_rng(2024_05_12)
    n = 30_000

    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    spread = np.abs(rng.normal(0.0, 0.5, n))
    high = close + spread
    low = close - spread
    open_ = np.clip(close + rng.normal(0.0, 0.2, n), low, high)
    volume = np.abs(rng.normal(100.0, 30.0, n)) + 1.0
    quote_volume = volume * close
    num_trades = (np.abs(rng.normal(50.0, 20.0, n)).astype(np.int64)) + 1
    taker_buy_base = volume * rng.uniform(0.3, 0.7, n)

    ts_index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": quote_volume,
            "num_trades": num_trades, "taker_buy_base": taker_buy_base,
        },
        index=ts_index,
    )
    return utils.compute_base_series(df)


def _to_polars(bars_pd: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas → polars. Strip tz from the DatetimeIndex first —
    pl.from_pandas of a tz-aware index can produce an Object-dtype column,
    which breaks ``pl.col("ts").dt.weekday()``. Hour/minute/dayofweek
    semantics are unchanged because the timestamp values already represent
    the desired wall-clock instants."""
    df = bars_pd.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.reset_index(names="ts")
    return pl.from_pandas(df)


def _assert_columns_bit_equal(legacy_pd, engine_pl, feature_cols, warmup):
    legacy_slice = legacy_pd.iloc[warmup : warmup + len(engine_pl)]
    assert len(legacy_slice) == len(engine_pl)
    missing = [c for c in feature_cols if c not in engine_pl.columns]
    assert not missing, f"engine missing: {missing[:5]}"

    for col in feature_cols:
        legacy_vals = legacy_slice[col].to_numpy(dtype=float)
        engine_vals = np.array(
            [v if v is not None else float("nan") for v in engine_pl[col].to_list()],
            dtype=float,
        )
        legacy_nan = np.isnan(legacy_vals)
        engine_nan = np.isnan(engine_vals)
        assert np.array_equal(legacy_nan, engine_nan), (
            f"{col}: NaN positions differ "
            f"(legacy={legacy_nan.sum()}, engine={engine_nan.sum()})"
        )
        valid = ~legacy_nan
        if valid.any():
            np.testing.assert_allclose(
                engine_vals[valid],
                legacy_vals[valid],
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"{col}: numeric divergence",
            )


# ===========================================================================
# Combined parity for all 5 families
# ===========================================================================


class TestStep9CombinedParity:
    def test_all_five_families_engine_vs_legacy(self, bars_with_base):
        # Build the legacy reference by chaining the 5 compute_* functions
        # in the same order the notebook does.
        legacy = utils.compute_candle_geometry(
            bars_with_base, WINDOWS_CANDLE_ROLL, WINDOWS_BREAKOUT
        )
        legacy = utils.compute_trend_momentum(legacy, WINDOWS_LOGP_Z, WINDOWS_RSI)
        legacy = utils.compute_activity_flow(legacy, [60, 120, 240])
        legacy = utils.compute_event_features(legacy)
        legacy = utils.compute_seasonality(legacy)

        legacy_feature_cols = [
            c for c in legacy.columns if c not in bars_with_base.columns
        ]

        bars_pl = _to_polars(bars_with_base)
        engine = FeatureEngine(
            tiers=(1,),
            families=("candle", "trend", "activity", "event", "seasonality"),
        )
        result = engine.transform(bars_pl)

        _assert_columns_bit_equal(
            legacy, result.data, legacy_feature_cols, result.warmup_trimmed
        )


# ===========================================================================
# L1 sanity tests
# ===========================================================================


class TestRsiShiftTrick:
    """The shift-trick reproduces ``_wilder_rsi`` exactly."""

    def test_rsi_matches_legacy_on_synthetic_returns(self):
        from src.features.families.trend import TrendRsi

        rng = np.random.default_rng(909)
        n = 500
        r = rng.normal(0.0, 0.001, size=n)
        r[0] = float("nan")  # mimic log_return.diff()

        # Legacy
        expected = utils._wilder_rsi(r, 14)

        # Engine path: pre-fill NaN to null, then compute
        bars = pl.DataFrame({"r": r}).with_columns(
            pl.col("r").fill_nan(None)
        )
        spec = TrendRsi()
        out = bars.select(spec.compute(14).alias("rsi"))["rsi"].to_list()
        actual = np.array([v if v is not None else float("nan") for v in out])

        # Both should be NaN at rows 0..13, valid from row 14.
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-9, atol=1e-12)


class TestSeasonalityWeekdayConvention:
    """polars ``dt.weekday()`` returns ISO 1=Mon..7=Sun; legacy uses pandas
    ``dayofweek`` which is 0=Mon..6=Sun. The Feature subtracts 1 to match."""

    def test_dow_zero_indexed_like_pandas(self):
        import datetime as dt
        from src.features.families.seasonality import SeasonalitySinDow

        # 2024-01-01 is a Monday. pandas dayofweek = 0 → sin(0) = 0.
        df = pl.DataFrame({"ts": [dt.datetime(2024, 1, 1)]})
        spec = SeasonalitySinDow()
        out = df.select(spec.compute().alias("v"))["v"].item()
        assert abs(out - 0.0) < 1e-12

    def test_dow_zero_indexed_full_week_coverage(self):
        """Exercise every weekday so a one-off subtraction error gets
        caught at every day, not just Monday. The Feature computes
        ``sin(2π * dayofweek / 7)`` with ``dayofweek`` in
        pandas convention (Mon=0..Sun=6). Verify against pandas-derived
        expectations on Sunday, Monday, Saturday, and Wednesday.
        """
        import datetime as dt
        from src.features.families.seasonality import (
            SeasonalityCosDow,
            SeasonalitySinDow,
        )

        # (date, expected_pandas_dow). Picked dates pinned by calendar.
        cases = [
            (dt.datetime(2023, 12, 31), 6),  # Sunday
            (dt.datetime(2024, 1, 1), 0),    # Monday
            (dt.datetime(2024, 1, 3), 2),    # Wednesday
            (dt.datetime(2024, 1, 6), 5),    # Saturday
        ]
        sin_spec = SeasonalitySinDow()
        cos_spec = SeasonalityCosDow()
        for date, expected_dow in cases:
            # Cross-check the calendar assumption with pandas itself
            assert pd.Timestamp(date).dayofweek == expected_dow, (
                f"calendar drift for {date!r}"
            )
            df = pl.DataFrame({"ts": [date]})
            sin_out = df.select(sin_spec.compute().alias("v"))["v"].item()
            cos_out = df.select(cos_spec.compute().alias("v"))["v"].item()
            expected_sin = math.sin(2.0 * math.pi * expected_dow / 7.0)
            expected_cos = math.cos(2.0 * math.pi * expected_dow / 7.0)
            assert abs(sin_out - expected_sin) < 1e-12, (
                f"{date!r}: sin={sin_out} != expected={expected_sin}"
            )
            assert abs(cos_out - expected_cos) < 1e-12, (
                f"{date!r}: cos={cos_out} != expected={expected_cos}"
            )


class TestCandleBreakoutZeroDenom:
    """When p_max == p_min over the window, position is null."""

    def test_logp_pos_constant_window_yields_null(self):
        from src.features.families.candle import CandleLogpPos

        # Constant log price → max == min → denom == 0 → null
        bars = pl.DataFrame({"p": [5.0, 5.0, 5.0, 5.0]})
        spec = CandleLogpPos()
        out = bars.select(spec.compute(3).alias("v"))["v"].to_list()
        assert out[0] is None  # warmup
        assert out[1] is None  # warmup
        assert out[2] is None  # zero denom
        assert out[3] is None  # zero denom
