"""Step 8 tests: volatility families (Tier-1 OHLC + Tier-2 decomposition).

Tier-1: VolParkinson, VolGk, VolRs vs ``compute_volatility_ohlc``.
Tier-2: VolBpvRatio, VolSemivarDown/Up/Ratio, VolVov vs
        ``compute_volatility_decomposition`` — exercises the engine's
        cross-tier execution (Tier-2 reads ``ret__rms__f__w20`` emitted
        by the rolling family in Tier-1).

Reuses the parity assertion helper pattern from step 7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.features import FeatureEngine
from src.features.config import (
    WINDOWS_F,
    WINDOWS_VOL_DECOMP,
    WINDOWS_VOL_OHLC,
)

pytestmark = pytest.mark.step8


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bars_with_base() -> pd.DataFrame:
    """Synthetic 30k-row bar frame with all base-series columns populated."""
    rng = np.random.default_rng(2024_05_11)
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
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
            "num_trades": num_trades,
            "taker_buy_base": taker_buy_base,
        },
        index=ts_index,
    )
    return utils.compute_base_series(df)


def _to_polars(bars_pd: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(bars_pd.reset_index(names="ts"))


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
# Engine tier-ordering smoke test (framework integration)
# ===========================================================================


class TestEngineTierOrdering:
    """Tier-2 features must run AFTER Tier-1 features in a single transform."""

    def test_tier_2_can_reference_tier_1_output(self, bars_with_base):
        """If the engine ran all expressions in one with_columns, VolVov
        would fail to find ret__rms__f__w20. Success here proves the
        engine groups by tier."""
        bars_pl = _to_polars(bars_with_base)
        engine = FeatureEngine(tiers=(1, 2), families=("vol", "rolling"))
        result = engine.transform(bars_pl)
        # The vol__vov column should exist for every WINDOWS_VOL_DECOMP entry.
        for w in WINDOWS_VOL_DECOMP:
            assert f"vol__vov__f__w{w}" in result.data.columns


# ===========================================================================
# Tier-1: OHLC volatility parity
# ===========================================================================


class TestVolatilityOhlcParity:
    """Parkinson, GK, RS estimators vs compute_volatility_ohlc."""

    def test_vol_ohlc_engine_vs_legacy(self, bars_with_base):
        legacy = utils.compute_volatility_ohlc(bars_with_base, WINDOWS_VOL_OHLC)
        legacy_feature_cols = [
            c
            for c in legacy.columns
            if c.startswith("vol__") and c not in bars_with_base.columns
        ]
        # 3 estimators × len(WINDOWS_VOL_OHLC)
        assert len(legacy_feature_cols) == 3 * len(WINDOWS_VOL_OHLC)

        bars_pl = _to_polars(bars_with_base)
        engine = FeatureEngine(tiers=(1,), families=("vol",))
        result = engine.transform(bars_pl)

        # Tier-1 vol warmup = max(WINDOWS_VOL_OHLC) - 1 = 4319
        assert result.warmup_trimmed == max(WINDOWS_VOL_OHLC) - 1

        _assert_columns_bit_equal(
            legacy, result.data, legacy_feature_cols, result.warmup_trimmed
        )


# ===========================================================================
# Tier-2: volatility decomposition parity
# ===========================================================================


class TestVolatilityDecompositionParity:
    """All five Tier-2 vol-decomp columns match
    compute_volatility_decomposition. Requires Tier-1 outputs to exist."""

    def test_vol_decomp_engine_vs_legacy(self, bars_with_base):
        # Legacy chain: rolling_stats (for ret__rms__f__w20) -> vol_decomp.
        # vol_decomp.requires ret__rms__f__w20 in df, so we feed the
        # rolling-stats-augmented frame in.
        with_rolling = utils.compute_rolling_stats(bars_with_base, WINDOWS_F)
        legacy = utils.compute_volatility_decomposition(with_rolling, WINDOWS_VOL_DECOMP)
        legacy_feature_cols = [
            c
            for c in legacy.columns
            if c not in with_rolling.columns and c.startswith("vol__")
        ]
        # 5 outputs × len(WINDOWS_VOL_DECOMP)
        assert len(legacy_feature_cols) == 5 * len(WINDOWS_VOL_DECOMP)

        bars_pl = _to_polars(bars_with_base)
        engine = FeatureEngine(tiers=(1, 2), families=("vol", "rolling"))
        result = engine.transform(bars_pl)

        # The combined warmup is dominated by max(WINDOWS_F) - 1 from the
        # rolling family.
        assert result.warmup_trimmed == max(WINDOWS_F) - 1

        _assert_columns_bit_equal(
            legacy, result.data, legacy_feature_cols, result.warmup_trimmed
        )


# ===========================================================================
# Targeted L1: hand-checked values to catch silly bugs
# ===========================================================================


class TestVolL1Sanity:
    """Tiny synthetic data with hand-computed expected outputs."""

    def test_parkinson_flat_candle_zero(self):
        from src.features.families.volatility import VolParkinson
        # Flat candles: high == low → log_hl null → var_p null → mean null.
        bars = pl.DataFrame(
            {"open": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5,
             "close": [1.0] * 5}
        )
        spec = VolParkinson()
        out = bars.select(spec.compute(3).alias("out"))["out"].to_list()
        # All output null (every input row has h == l → null).
        for v in out:
            assert v is None

    def test_rs_flat_candle_zero(self):
        from src.features.families.volatility import VolRs
        # Flat candle: log(h/o)=log(h/c)=log(l/o)=log(l/c)=0 → rs_variance=0
        # rolling_mean(0, w) = 0; clip_pos(0) = 0; sqrt(0) = 0.
        bars = pl.DataFrame(
            {"open": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5,
             "close": [1.0] * 5}
        )
        spec = VolRs()
        out = bars.select(spec.compute(3).alias("out"))["out"].to_list()
        # First two rows null (warmup), then 0.0.
        assert out[0] is None
        assert out[1] is None
        assert out[2] == 0.0
        assert out[3] == 0.0
        assert out[4] == 0.0

    def test_bpv_ratio_warmup_is_w(self):
        """First valid row is at index w, not w-1, because r[0]=null
        plus prod[0..1]=null from shift."""
        from src.features.families.volatility import VolBpvRatio
        bars = pl.DataFrame({"r": [None, 0.01, -0.02, 0.03, -0.01, 0.02, 0.01]})
        spec = VolBpvRatio()
        # w=3: warmup=3, so first valid at row 3
        out = bars.select(spec.compute(3).alias("out"))["out"].to_list()
        for i in range(3):
            assert out[i] is None, f"row {i}: expected null, got {out[i]}"
        assert out[3] is not None
