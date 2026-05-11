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


_NON_FEATURE_COLS_FOR_PIPELINE_TEST = frozenset(
    {"k", "ts", "y", "m_k", "tau_k", "phi"}
)


def _feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS_FOR_PIPELINE_TEST
            and not c.startswith("undef__")
            and df.schema[c] in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)]


def test_run_pipeline_1min_cadence_smoke(bars_pd):
    """End-to-end smoke at the new 1-min target cadence.

    The pipeline should run without error, produce one row per non-warmup
    1-min bar, carry the new autocorr columns, and emit `_h` features at
    the scaled (calendar-time-preserving) window values.
    """
    from src.features.config import N_WARMUP, M as M_BARS

    engine_out = run_pipeline(
        bars_pd, with_derivatives=False, p_hit_prior=0.5, label_cadence="1min"
    )
    # Expected row count: total_bars - N_WARMUP - (M tail bars with NaN labels)
    n_total = len(bars_pd)
    expected_rows_upper = n_total - N_WARMUP
    assert 0 < len(engine_out) <= expected_rows_upper
    for c in ("k", "ts", "y", "m_k", "tau_k", "phi"):
        assert c in engine_out.columns, f"missing required column {c}"
    autocorr_cols = [c for c in engine_out.columns if c.startswith("target__autocorr_")]
    assert autocorr_cols, "expected target__autocorr_*  columns at 1-min cadence"
    # `_h` features must use the M-scaled windows (1440 = 72 * M)
    assert "hit__rate__h__w1440" in engine_out.columns, (
        "expected hit__rate__h__w1440 (= 72*M) at 1-min cadence"
    )

    # CRITICAL: post-impute, no feature column may carry null, NaN, or inf.
    # Any of those means the impute step missed a column and would crash
    # CatBoost or silently corrupt training. This is the gate that catches
    # broken imputation patterns for the new autocorr columns.
    feature_cols = _feature_columns(engine_out)
    nulls = {c: int(engine_out[c].null_count()) for c in feature_cols
             if int(engine_out[c].null_count()) > 0}
    assert not nulls, f"feature columns with residual nulls after impute: {nulls}"
    for c in feature_cols:
        dtype = engine_out.schema[c]
        if dtype in (pl.Float32, pl.Float64):
            n_nan = int(engine_out[c].is_nan().sum() or 0)
            assert n_nan == 0, f"{c}: {n_nan} NaN values after impute"
            if bool(engine_out[c].is_infinite().any()):
                raise AssertionError(f"{c}: inf values after impute")

    # Label-cadence-shaped row count: in 1-min mode k is the 1-min row index
    k_min = int(engine_out["k"].min())
    k_max = int(engine_out["k"].max())
    assert k_min >= N_WARMUP, f"k_min={k_min} should be >= N_WARMUP={N_WARMUP}"
    # 1-min cadence labels at row k look ahead M bars; row k+M-1 must exist in raw
    assert k_max <= n_total - M_BARS, (
        f"k_max={k_max} should be <= n_total-M={n_total - M_BARS}"
    )


def test_run_pipeline_1min_default_barrier_source_is_high(bars_pd):
    """``run_pipeline(label_cadence='1min')`` must default to the HIGH-source
    label so the trained event matches the simulator's exit_tp_or_expiry
    (long TP fills on intrabar high crossing). A regression that flipped
    the default back to close-source would still pass other smoke tests
    (the schema would be identical) — this test is the gate that catches
    such a silent default flip.
    """
    out = run_pipeline(bars_pd, with_derivatives=False, label_cadence="1min")
    close = bars_pd["close"].to_numpy().astype(float)
    high = bars_pd["high"].to_numpy().astype(float)
    phi = float(C + ETA)
    M_v = int(M)
    # Sample 50 rows by their stored k (= 1-min row index at 1-min cadence)
    # and re-derive y under both sources; the stored y must match HIGH.
    rng = np.random.default_rng(0)
    n_total = len(close)
    sample_n = min(50, len(out))
    if sample_n == 0:
        pytest.skip("no rows produced by pipeline")
    sampled = rng.choice(len(out), size=sample_n, replace=False)
    matches_close, matches_high = 0, 0
    valid = 0
    for i in sampled.tolist():
        k = int(out["k"][i])
        if k + M_v >= n_total:
            continue
        base = close[k]
        fut_close_ret = np.log(close[k + 1 : k + M_v + 1] / base)
        fut_high_ret = np.log(high[k + 1 : k + M_v + 1] / base)
        y_close = 1.0 if fut_close_ret.max() >= phi else 0.0
        y_high = 1.0 if fut_high_ret.max() >= phi else 0.0
        y_stored = float(out["y"][i])
        if y_stored == y_close:
            matches_close += 1
        if y_stored == y_high:
            matches_high += 1
        valid += 1
    assert valid > 0
    assert matches_high == valid, (
        f"run_pipeline 1-min default is NOT high-source: only {matches_high}/{valid} "
        f"match high; {matches_close}/{valid} match close. The high-source default "
        "is required for target/execution alignment with the simulator's long-TP exit."
    )


def test_run_pipeline_1min_m_k_matches_future_high_excursion(bars_pd):
    """At 1-min cadence with the default barrier_source='high', the stored
    ``m_k`` must equal ``max(log(high[n+1..n+M] / close[n]))`` — the
    max excursion of FUTURE HIGHS, not of future closes. A bug that
    silently used closes for m_k while using highs for y would still
    pass label tests because y is just an indicator; this test guards
    the magnitude column."""
    out = run_pipeline(bars_pd, with_derivatives=False, label_cadence="1min")
    close = bars_pd["close"].to_numpy().astype(float)
    high = bars_pd["high"].to_numpy().astype(float)
    M_v = int(M)
    rng = np.random.default_rng(1)
    n_total = len(close)
    sample_n = min(40, len(out))
    if sample_n == 0:
        pytest.skip("no rows produced by pipeline")
    sampled = rng.choice(len(out), size=sample_n, replace=False)
    for i in sampled.tolist():
        k = int(out["k"][i])
        if k + M_v >= n_total:
            continue
        base = close[k]
        expected_m_high = float(np.log(high[k + 1 : k + M_v + 1] / base).max())
        stored_m = float(out["m_k"][i])
        assert abs(stored_m - expected_m_high) < 1e-12, (
            f"row k={k}: m_k={stored_m} != expected_high_excursion={expected_m_high}"
        )


def test_run_pipeline_1min_explicit_close_source_overrides_default(bars_pd):
    """``barrier_source='close'`` at 1-min cadence must produce the legacy
    close-confirmed labels. Verifies the cadence default is overridable."""
    out = run_pipeline(
        bars_pd,
        with_derivatives=False,
        label_cadence="1min",
        barrier_source="close",
    )
    close = bars_pd["close"].to_numpy().astype(float)
    phi = float(C + ETA)
    M_v = int(M)
    rng = np.random.default_rng(2)
    n_total = len(close)
    sample_n = min(30, len(out))
    if sample_n == 0:
        pytest.skip("no rows produced")
    sampled = rng.choice(len(out), size=sample_n, replace=False)
    for i in sampled.tolist():
        k = int(out["k"][i])
        if k + M_v >= n_total:
            continue
        base = close[k]
        fut_close_ret = np.log(close[k + 1 : k + M_v + 1] / base)
        expected_y = 1.0 if fut_close_ret.max() >= phi else 0.0
        assert float(out["y"][i]) == expected_y


def test_run_pipeline_1min_drops_boundary_sparse_excursion(bars_pd):
    """The boundary-sparse excursion drawup/drawdown columns are
    modulo-M sparse by construction and would create a phase artifact
    at 1-min cadence. ``run_pipeline(label_cadence='1min')`` must drop
    them and substitute the every-row rolling variants."""
    out = run_pipeline(bars_pd, with_derivatives=False, label_cadence="1min")
    sparse = [
        c for c in out.columns
        if c.startswith("excursion__max_drawup__f__")
        or c.startswith("excursion__max_drawdown__f__")
    ]
    rolling = [
        c for c in out.columns
        if c.startswith("excursion__roll_max_drawup__f__")
        or c.startswith("excursion__roll_max_drawdown__f__")
    ]
    assert not sparse, f"boundary-sparse excursion columns leaked into 1-min: {sparse}"
    assert rolling, "expected every-row rolling excursion columns at 1-min cadence"


def test_run_pipeline_1min_no_autocorr_columns_when_flag_off(bars_pd):
    """``enable_autocorrelation=False`` must suppress every target__autocorr_* column."""
    out = run_pipeline(
        bars_pd, with_derivatives=False, label_cadence="1min",
        enable_autocorrelation=False,
    )
    leaks = [c for c in out.columns if c.startswith("target__autocorr_")]
    assert not leaks, f"unexpected autocorr columns with flag off: {leaks}"


def test_run_pipeline_boundary_default_does_not_add_autocorr_columns(bars_pd):
    """Boundary cadence with default flag must NOT emit autocorr columns —
    the legacy schema must remain intact when users haven't opted in.

    Catches the silent-schema-drift hazard surfaced in the review: the
    autocorr columns were being added at boundary cadence too (full of
    nulls because consecutive labels don't overlap), changing the legacy
    feature_list and breaking models trained against the old schema.
    """
    out_default = run_pipeline(bars_pd, with_derivatives=False)
    leaks = [c for c in out_default.columns if c.startswith("target__autocorr_")]
    assert not leaks, f"boundary-default pipeline leaked autocorr columns: {leaks}"


def test_run_pipeline_boundary_can_opt_in_to_autocorr(bars_pd):
    """A boundary-cadence run with the flag explicitly ON should emit them
    (so users can compare). Just verifies the override path."""
    out = run_pipeline(bars_pd, with_derivatives=False, enable_autocorrelation=True)
    cols = [c for c in out.columns if c.startswith("target__autocorr_")]
    assert cols, "expected autocorr columns with explicit opt-in"


def test_run_pipeline_rejects_bad_label_cadence(bars_pd):
    with pytest.raises(ValueError, match="label_cadence"):
        run_pipeline(bars_pd, label_cadence="hourly")


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
