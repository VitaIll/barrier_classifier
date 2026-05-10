"""Step 13 tests: boundary-stage transformations parity.

Test approach: build a realistic synthetic bars df, run the legacy
compute_* chain (base series + vol_ohlc + boundary functions) to get
the legacy reference. Then run the polars boundary functions against
the same boundary df and assert the outputs match.

Each polars boundary function is independent — we test them isolated
rather than as one chain so failures point at the offending function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.features.boundary import (
    compute_barrier_aware_features_pl,
    compute_block_features_pl,
    compute_past_target_features_pl,
    construct_labels_pl,
)
from src.features.config import (
    ETA,
    HITRATE_WINDOWS_H,
    M,
    PHI,
    VOL_PAIRS,
    WINDOWS_BARRIER,
    WINDOWS_H,
    WINDOWS_VOL_OHLC,
)
from src.features.config import C as COST_C

pytestmark = pytest.mark.step13


@pytest.fixture(scope="module")
def fixtures():
    """Build raw bars + base series + vol_ohlc + boundary frame.

    40k rows so we have enough bars even for long block windows.
    """
    rng = np.random.default_rng(2024_05_16)
    n = 40_000
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
    df_raw = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": quote_volume,
            "num_trades": num_trades, "taker_buy_base": taker_buy_base,
        },
        index=ts_index,
    )
    df = utils.compute_base_series(df_raw)
    # vol_ohlc gives us vol__rs__f__w{W} which barrier-aware needs
    df = utils.compute_volatility_ohlc(df, WINDOWS_VOL_OHLC)

    # Sample boundaries every M rows (matches notebook STAGE 4).
    df_boundaries_pd = df.iloc[::M].copy().reset_index(names="ts")
    df_boundaries_pd["k"] = np.arange(len(df_boundaries_pd), dtype=int)

    # polars conversions (strip tz first to avoid Object dtype on Windows)
    df_raw_naive = df.copy()
    if df_raw_naive.index.tz is not None:
        df_raw_naive.index = df_raw_naive.index.tz_localize(None)
    df_raw_pl = pl.from_pandas(df_raw_naive.reset_index(names="ts"))

    boundaries_pd_for_pl = df_boundaries_pd.copy()
    if boundaries_pd_for_pl["ts"].dt.tz is not None:
        boundaries_pd_for_pl["ts"] = boundaries_pd_for_pl["ts"].dt.tz_localize(None)
    df_boundaries_pl = pl.from_pandas(boundaries_pd_for_pl)

    return {
        "df_raw_pd": df,
        "df_raw_pl": df_raw_pl,
        "df_boundaries_pd": df_boundaries_pd,
        "df_boundaries_pl": df_boundaries_pl,
    }


def _compare_columns(legacy_pd, engine_pl, cols, *, rtol=1e-9, atol=1e-12):
    for col in cols:
        legacy_vals = legacy_pd[col].to_numpy(dtype=float)
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
                engine_vals[valid], legacy_vals[valid],
                rtol=rtol, atol=atol,
                err_msg=f"{col}: numeric divergence",
            )


def test_construct_labels_parity(fixtures):
    df_boundaries_pd = fixtures["df_boundaries_pd"]
    df_raw_pd = fixtures["df_raw_pd"]
    df_boundaries_pl = fixtures["df_boundaries_pl"]
    df_raw_pl = fixtures["df_raw_pl"]

    legacy = utils.construct_labels(df_boundaries_pd, df_raw_pd, M, ETA, COST_C)
    engine = construct_labels_pl(df_boundaries_pl, df_raw_pl, M, ETA, COST_C)

    _compare_columns(legacy, engine, ["y", "m_k", "tau_k", "phi"])


def test_past_target_features_parity(fixtures):
    df_boundaries_pd = fixtures["df_boundaries_pd"]
    df_raw_pd = fixtures["df_raw_pd"]
    df_boundaries_pl = fixtures["df_boundaries_pl"]
    df_raw_pl = fixtures["df_raw_pl"]

    legacy = utils.construct_labels(df_boundaries_pd, df_raw_pd, M, ETA, COST_C)
    legacy = utils.compute_past_target_features(legacy, WINDOWS_H, HITRATE_WINDOWS_H)

    engine = construct_labels_pl(df_boundaries_pl, df_raw_pl, M, ETA, COST_C)
    engine = compute_past_target_features_pl(engine, WINDOWS_H, HITRATE_WINDOWS_H)

    cols = ["hit__prev__h__w0"]
    cols += [f"hit__rate__h__w{W}" for W in HITRATE_WINDOWS_H]
    cols += ["hit__since__h__w0"]
    _compare_columns(legacy, engine, cols)


def test_barrier_aware_features_parity(fixtures):
    df_boundaries_pd = fixtures["df_boundaries_pd"]
    df_boundaries_pl = fixtures["df_boundaries_pl"]

    legacy = utils.compute_barrier_aware_features(
        df_boundaries_pd, WINDOWS_BARRIER, PHI, M, VOL_PAIRS
    )
    engine = compute_barrier_aware_features_pl(
        df_boundaries_pl, WINDOWS_BARRIER, PHI, M, VOL_PAIRS, c=float(COST_C)
    )

    cols = []
    for W in WINDOWS_BARRIER:
        cols.append(f"barrier__z_tight__f__w{W}")
        cols.append(f"barrier__emax_ratio__f__w{W}")
    for ws, wl in VOL_PAIRS:
        cols.append(f"vol__ratio__f__ws{ws}__wl{wl}")
    cols += ["cost__c__h__w0", "barrier__phi__h__w0"]
    _compare_columns(legacy, engine, cols)


def test_block_features_parity(fixtures):
    df_boundaries_pd = fixtures["df_boundaries_pd"]
    df_raw_pd = fixtures["df_raw_pd"]
    df_boundaries_pl = fixtures["df_boundaries_pl"]
    df_raw_pl = fixtures["df_raw_pl"]

    legacy = utils.compute_block_features(df_boundaries_pd, df_raw_pd, M, WINDOWS_H)
    engine = compute_block_features_pl(df_boundaries_pl, df_raw_pl, M, WINDOWS_H)

    cols = [
        "ret__inst__h__w0",
        "range__inst__h__w0",
        "logvol__inst__h__w0",
        "ofi__inst__h__w0",
        "block__maxret__h__w0",
        "block__minret__h__w0",
        "block__close_to_high__h__w0",
    ]
    cols += [f"ret__std__h__w{W}" for W in WINDOWS_H]
    _compare_columns(legacy, engine, cols)
