"""Tests for streaming online primitives (RollingQuantileRank, FastVolEWMA,
DriftADWIN, RollingRegimeBaseRate)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.strategy.online import (
    DriftADWIN,
    FastVolEWMA,
    RollingQuantileRank,
    RollingRegimeBaseRate,
)

pytestmark = pytest.mark.strategy_v1


# ---------------------------------------------------------------------------
# RollingQuantileRank
# ---------------------------------------------------------------------------


def test_rolling_quantile_rank_warmup_returns_nan():
    r = RollingQuantileRank(window=100, min_warmup=30)
    for x in range(10):
        r.update(float(x))
    assert math.isnan(r.rank(5.0))


def test_rolling_quantile_rank_uniform_distribution_recovers_q():
    r = RollingQuantileRank(window=1000, min_warmup=100)
    rng = np.random.default_rng(0)
    for x in rng.uniform(0, 1, size=2000):
        r.update(float(x))
    # Median of buffer should rank ~ 0.5
    assert abs(r.rank(0.5) - 0.5) < 0.05
    assert r.rank(1.0) == pytest.approx(1.0, abs=0.01)
    assert r.rank(-1.0) == pytest.approx(0.0, abs=0.01)


def test_rolling_quantile_rank_window_size_caps_buffer():
    r = RollingQuantileRank(window=10, min_warmup=5)
    for x in range(100):
        r.update(float(x))
    assert r.n == 10  # only last 10 retained


def test_rolling_quantile_rank_ignores_nan_inputs():
    r = RollingQuantileRank(window=10, min_warmup=3)
    r.update(1.0)
    r.update(float("nan"))
    r.update(2.0)
    assert r.n == 2
    assert math.isnan(r.rank(float("nan")))


def test_rolling_quantile_rank_and_update_uses_pre_update_history():
    r = RollingQuantileRank(window=1000, min_warmup=2)
    for x in [1.0, 2.0, 3.0]:
        r.update(x)
    # Rank 4 against [1,2,3]; should be 1.0 (4 is greater than all)
    assert r.rank_and_update(4.0) == pytest.approx(1.0)
    # Now buffer is [1,2,3,4]; rank 0.5 against it should be 0.0
    assert r.rank(0.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FastVolEWMA
# ---------------------------------------------------------------------------


def test_fast_vol_warmup_returns_nan():
    v = FastVolEWMA(halflife_bars=30, min_warmup=10)
    for r in [0.001, -0.001, 0.0005]:
        v.update(r)
    assert math.isnan(v.value())


def test_fast_vol_constant_input_converges_to_value():
    v = FastVolEWMA(halflife_bars=10, min_warmup=5)
    # All log-returns of magnitude exactly 0.001 → σ̂ → 0.001
    for _ in range(200):
        v.update(0.001)
    assert v.value() == pytest.approx(0.001, rel=1e-4)


def test_fast_vol_responds_to_regime_change():
    v = FastVolEWMA(halflife_bars=10, min_warmup=5)
    # Low-vol prefix
    for _ in range(50):
        v.update(0.0001)
    low = v.value()
    # High-vol suffix
    for _ in range(50):
        v.update(0.005)
    high = v.value()
    assert high > 5 * low


def test_fast_vol_ignores_nan_inputs():
    v = FastVolEWMA(halflife_bars=10, min_warmup=3)
    v.update(0.001)
    v.update(float("nan"))
    v.update(0.001)
    assert v.n == 2


# ---------------------------------------------------------------------------
# DriftADWIN
# ---------------------------------------------------------------------------


def test_drift_adwin_no_drift_on_constant_stream():
    d = DriftADWIN(delta=0.01, grace_period=10)
    for _ in range(200):
        fired = d.update(0.5)
    assert d.n_detections == 0


def test_drift_adwin_fires_after_regime_change():
    """Big mean shift after a long stable prefix → ADWIN should fire at least once."""
    d = DriftADWIN(delta=0.01, grace_period=10)
    rng = np.random.default_rng(0)
    fired_any = False
    for _ in range(200):
        d.update(float(rng.normal(0.0, 0.1)))
    for _ in range(200):
        if d.update(float(rng.normal(2.0, 0.1))):
            fired_any = True
    assert fired_any, f"expected drift after mean shift; n_detections={d.n_detections}"


def test_drift_adwin_ignores_nan():
    d = DriftADWIN()
    assert d.update(float("nan")) is False
    assert d.n == 0


# ---------------------------------------------------------------------------
# RollingRegimeBaseRate
# ---------------------------------------------------------------------------


def test_regime_base_rate_empty_returns_nan():
    rb = RollingRegimeBaseRate(window=100, n_bins=5)
    assert math.isnan(rb.base_rate_at(0.5))


def test_regime_base_rate_recovers_per_bin_means():
    """Synthetic: (q, y) where p(y=1 | q in [0, 0.5)) = 0.1 and p(y=1 | q in
    [0.5, 1]) = 0.6. Verify per-bin estimates after 1000 samples."""
    rng = np.random.default_rng(0)
    rb = RollingRegimeBaseRate(window=2000, n_bins=2)
    for _ in range(2000):
        q = float(rng.uniform(0.0, 1.0))
        if q < 0.5:
            y = float(rng.binomial(1, 0.1))
        else:
            y = float(rng.binomial(1, 0.6))
        rb.update(q, y)
    assert abs(rb.base_rate_at(0.25) - 0.1) < 0.04
    assert abs(rb.base_rate_at(0.75) - 0.6) < 0.04


def test_regime_base_rate_nan_q_returns_global_mean():
    rb = RollingRegimeBaseRate(window=100, n_bins=5)
    rb.update(0.1, 0.0)
    rb.update(0.9, 1.0)
    rb.update(0.5, 0.5)
    # Global mean = (0+1+0.5)/3 = 0.5
    assert rb.base_rate_at(float("nan")) == pytest.approx(0.5)
