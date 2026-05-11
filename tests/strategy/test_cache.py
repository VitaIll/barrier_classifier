"""Tests for cache augmentation helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.strategy.cache import (
    augment_cache_with_boundary_ohlc,
    augment_cache_with_r_realized,
)

pytestmark = pytest.mark.strategy_v1


def _make_synthetic_raw_and_cache(M: int = 20, n_bars: int = 200):
    """Synthetic OHLC + boundary cache aligned to it."""
    ts = pd.date_range("2025-06-01", periods=n_bars, freq="1min")
    closes = 100.0 * np.exp(np.cumsum(np.full(n_bars, 0.0001)))
    raw = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes}, index=ts)
    boundary = raw.iloc[::M].copy().reset_index(names="ts")
    boundary["k"] = np.arange(len(boundary))
    boundary["close"] = boundary["close"].astype(float)
    boundary["split"] = "test"
    return raw, boundary


def test_augment_cache_with_r_realized_appends_column():
    raw, cache = _make_synthetic_raw_and_cache(M=20, n_bars=200)
    out = augment_cache_with_r_realized(cache, raw, M=20)
    assert "r_realized" in out.columns
    # First boundary maps to bar 0; bar 0+20 should have r ~ 20*0.0001 = 0.002
    assert out["r_realized"].iloc[0] == pytest.approx(20 * 0.0001, abs=1e-9)
    # Last boundary should be NaN (no future M bars available)
    assert math.isnan(out["r_realized"].iloc[-1])


def test_augment_cache_with_r_realized_idempotent():
    raw, cache = _make_synthetic_raw_and_cache(M=20, n_bars=200)
    cache_with = augment_cache_with_r_realized(cache, raw, M=20)
    cache_twice = augment_cache_with_r_realized(cache_with, raw, M=20, skip_if_present=True)
    pd.testing.assert_frame_equal(cache_with, cache_twice)


def test_augment_cache_with_r_realized_handles_tz_aware_raw():
    raw, cache = _make_synthetic_raw_and_cache(M=20, n_bars=200)
    raw = raw.tz_localize("UTC")  # tz-aware
    out = augment_cache_with_r_realized(cache, raw, M=20)
    assert "r_realized" in out.columns
    assert out["r_realized"].iloc[0] == pytest.approx(20 * 0.0001, abs=1e-9)


def test_augment_cache_with_boundary_ohlc_joins_correctly():
    raw, cache = _make_synthetic_raw_and_cache(M=20, n_bars=200)
    cache_thin = cache.drop(columns=["open", "high", "low", "close"], errors="ignore")
    out = augment_cache_with_boundary_ohlc(cache_thin, raw)
    for col in ("open", "high", "low", "close"):
        assert col in out.columns
    # First boundary close should equal raw close at index 0
    assert out["close"].iloc[0] == pytest.approx(raw.iloc[0]["close"])
    assert out["close"].iloc[-1] == pytest.approx(raw.iloc[-20]["close"])


def test_augment_cache_with_boundary_ohlc_idempotent():
    raw, cache = _make_synthetic_raw_and_cache(M=20, n_bars=200)
    once = augment_cache_with_boundary_ohlc(cache, raw, skip_if_present=True)
    twice = augment_cache_with_boundary_ohlc(once, raw, skip_if_present=True)
    pd.testing.assert_frame_equal(once, twice)


def test_augment_cache_with_r_realized_with_real_data():
    """Light smoke test against the real on-disk dataset if it exists."""
    import os
    raw_path = "data/raw_data/klines_1m.parquet"
    cache_path = "data/model_dataset/research_predictions.parquet"
    if not (os.path.exists(raw_path) and os.path.exists(cache_path)):
        pytest.skip("real data files not present")
    raw = pd.read_parquet(raw_path, columns=["close"])
    cache = pd.read_parquet(cache_path)
    out = augment_cache_with_r_realized(cache, raw, M=20)
    assert "r_realized" in out.columns
    # At least 95% non-NaN (we tolerate boundary effects)
    valid = out["r_realized"].notna()
    assert valid.mean() > 0.95
