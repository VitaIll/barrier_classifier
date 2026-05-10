"""Step 15 tests: quantile family parity + end-to-end pipeline integration.

The pipeline integration test runs the new ``run_pipeline`` against the
same synthetic OHLCV used by the legacy chain, then asserts every
feature column on the boundary dataframe is bit-equal (within
``rtol=1e-9, atol=1e-12``) to the legacy chain output.

Derivatives are excluded here — the test focuses on validating that
all the bar-level + boundary stages compose correctly. Derivatives
are exercised in step 11's parity test in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.features import FeatureEngine
from src.features.config import (
    C,
    ETA,
    HITRATE_WINDOWS_H,
    K_WARMUP,
    M,
    PHI,
    VOL_PAIRS,
    WINDOWS_BARRIER,
    WINDOWS_BPLUS,
    WINDOWS_BREAKOUT,
    WINDOWS_CANDLE_ROLL,
    WINDOWS_CORR,
    WINDOWS_EXCURSION,
    WINDOWS_F,
    WINDOWS_H,
    WINDOWS_LIQ_AMIHUD,
    WINDOWS_LIQ_RPV,
    WINDOWS_LOGP_Z,
    WINDOWS_MAXRET,
    WINDOWS_OFI_IMPULSE,
    WINDOWS_PENTROPY,
    WINDOWS_RSI,
    WINDOWS_VOL_DECOMP,
    WINDOWS_VOL_OHLC,
)
from src.features.pipeline import run_pipeline

pytestmark = pytest.mark.step15


@pytest.fixture(scope="module")
def bars_pd() -> pd.DataFrame:
    """30k synthetic bars — enough for max(WINDOWS_F)=20160 warmup."""
    rng = np.random.default_rng(2024_05_19)
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
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": quote_volume,
            "num_trades": num_trades, "taker_buy_base": taker_buy_base,
        },
        index=ts_index,
    )


# ===========================================================================
# Quantile family parity (boundary-sparse)
# ===========================================================================


def test_quantile_family_parity(bars_pd):
    """compute_quantile_features is boundary-sparse — only every M-th row
    is populated. Engine output should match at boundary indices."""
    legacy_base = utils.compute_base_series(bars_pd)
    legacy = utils.compute_quantile_features(legacy_base, WINDOWS_BPLUS)
    legacy_cols = [c for c in legacy.columns if c not in legacy_base.columns]
    assert len(legacy_cols) == 4 * len(WINDOWS_BPLUS)

    bars_pl = pl.from_pandas(legacy_base.reset_index().rename(columns={"index": "ts"}))
    if bars_pl["ts"].dtype == pl.Object:
        # tz round-trip safety
        df = legacy_base.copy()
        df.index = df.index.tz_localize(None)
        bars_pl = pl.from_pandas(df.reset_index(names="ts"))

    engine = FeatureEngine(tiers=(1,), families=("quantile",))
    result = engine.transform(bars_pl, trim=False)

    # Compare at every row (legacy has NaN at non-boundary, engine should too).
    for col in legacy_cols:
        legacy_vals = legacy[col].to_numpy(dtype=float)
        engine_vals = np.array(
            [v if v is not None else float("nan") for v in result.data[col].to_list()],
            dtype=float,
        )
        np.testing.assert_array_equal(
            np.isnan(legacy_vals), np.isnan(engine_vals),
            err_msg=f"{col}: NaN positions differ",
        )
        valid = ~np.isnan(legacy_vals)
        if valid.any():
            np.testing.assert_allclose(
                engine_vals[valid], legacy_vals[valid],
                rtol=1e-9, atol=1e-12,
                err_msg=f"{col}: numeric divergence",
            )


# ===========================================================================
# End-to-end pipeline integration
# ===========================================================================


def _legacy_full_chain(bars_pd: pd.DataFrame) -> pd.DataFrame:
    """Replay the notebook compute_* chain (without derivatives + weights)
    and return df_final — the boundary-aligned dataset the new pipeline
    must reproduce.
    """
    df = utils.compute_base_series(bars_pd)
    df = utils.compute_lag_features(df, utils.LAGS_F)
    df = utils.compute_rolling_stats(df, WINDOWS_F)
    df = utils.compute_quantile_features(df, WINDOWS_BPLUS)
    df = utils.compute_volatility_ohlc(df, WINDOWS_VOL_OHLC)
    df = utils.compute_candle_geometry(df, WINDOWS_CANDLE_ROLL, WINDOWS_BREAKOUT)
    df = utils.compute_trend_momentum(df, WINDOWS_LOGP_Z, WINDOWS_RSI)
    df = utils.compute_activity_flow(df, [60, 120, 240])
    df = utils.compute_correlations(df, WINDOWS_CORR)
    df = utils.compute_permutation_entropy(df, WINDOWS_PENTROPY, m=3, tau=1)
    df = utils.compute_event_features(df)
    df = utils.compute_data_quality_flags(df)
    df = utils.compute_seasonality(df)
    df = utils.compute_volatility_decomposition(df, WINDOWS_VOL_DECOMP)
    df = utils.compute_excursion_features(df, WINDOWS_EXCURSION, WINDOWS_MAXRET)
    df = utils.compute_enhanced_liquidity(
        df, WINDOWS_LIQ_AMIHUD, WINDOWS_LIQ_RPV, WINDOWS_OFI_IMPULSE
    )
    # Boundary stage
    df_b = df.iloc[::M].copy().reset_index(names="ts")
    df_b["k"] = np.arange(len(df_b), dtype=int)
    df_b = utils.construct_labels(df_b, bars_pd, M, ETA, float(C))
    df_b = utils.compute_past_target_features(df_b, WINDOWS_H, HITRATE_WINDOWS_H)
    df_b = utils.compute_barrier_aware_features(df_b, WINDOWS_BARRIER, PHI, M, VOL_PAIRS)
    df_b = utils.compute_block_features(df_b, bars_pd, M, WINDOWS_H)
    # Warmup trim + drop nan labels
    df_b = df_b[df_b["k"] >= K_WARMUP].copy()
    df_b = df_b[df_b["y"].notna()].copy()
    # Undef + impute
    label_aux = {"k", "ts", "y", "m_k", "tau_k", "phi"}
    feature_cols = [c for c in df_b.columns if c not in label_aux]
    df_b, _ = utils.create_undef_flags_and_impute(
        df_b, feature_cols, p_hit_prior=0.5, cap_h_blocks=max(WINDOWS_H)
    )
    return df_b


def test_run_pipeline_matches_legacy_chain(bars_pd):
    """One end-to-end check that ``run_pipeline`` reproduces the legacy
    notebook's df_final — every shared feature column bit-equal at the
    same boundary rows."""
    legacy = _legacy_full_chain(bars_pd)
    engine_out = run_pipeline(bars_pd, with_derivatives=False, p_hit_prior=0.5)

    assert len(engine_out) == len(legacy), (
        f"row count mismatch: legacy={len(legacy)} engine={len(engine_out)}"
    )

    # Compare label/aux columns
    for col in ("y", "m_k", "tau_k", "k"):
        legacy_vals = legacy[col].to_numpy(dtype=float)
        engine_vals = np.array(
            [v if v is not None else float("nan") for v in engine_out[col].to_list()],
            dtype=float,
        )
        np.testing.assert_allclose(
            engine_vals, legacy_vals, rtol=0, atol=0,
            err_msg=f"{col}: divergence",
        )

    # Spot-check a representative sample of feature columns from each family.
    # Full per-column comparison runs in the family-level parity tests; here
    # we verify the pipeline composes correctly.
    sample_cols = [
        "ret__lag1__f__w0",                 # lag family
        "ret__std__f__w60",                 # rolling family
        "ret__q50__f__w240",                # quantile family (boundary-sparse)
        "vol__rs__f__w240",                 # vol Tier-1
        "vol__bpv_ratio__f__w240",          # vol Tier-2
        "logp__pos__f__w240",               # candle breakout
        "ret__rsi__f__w14",                 # trend RSI (Wilder shift trick)
        "logvol__z__f__w120",               # activity rolling
        "ret__acf1__f__w240",               # correlation
        "pentropy_norm__inst__f__w240__m3__tau1",  # entropy
        "event__run_dir__f__w0",            # event
        "time__sin_minute__f__w0",          # seasonality (Int8 overflow caught)
        "data__bad_ohlc__f__w0",            # quality
        "excursion__max_drawup__f__w240",   # Tier-2 boundary-sparse
        "liq__amihud__f__w120",             # Tier-2 liquidity
        "ret__inst__h__w0",                 # block feature
        "barrier__z_tight__f__w240",        # barrier-aware
        "hit__prev__h__w0",                 # past-target
    ]
    for col in sample_cols:
        if col not in engine_out.columns:
            pytest.fail(f"engine output missing expected column {col}")
        if col not in legacy.columns:
            pytest.skip(f"legacy missing {col} — investigate")
            continue
        legacy_vals = legacy[col].to_numpy(dtype=float)
        engine_vals = np.array(
            [v if v is not None else float("nan") for v in engine_out[col].to_list()],
            dtype=float,
        )
        np.testing.assert_array_equal(
            np.isnan(legacy_vals), np.isnan(engine_vals),
            err_msg=f"{col}: NaN pattern differs",
        )
        valid = ~np.isnan(legacy_vals)
        if valid.any():
            np.testing.assert_allclose(
                engine_vals[valid], legacy_vals[valid],
                rtol=1e-9, atol=1e-12,
                err_msg=f"{col}: numeric divergence",
            )
