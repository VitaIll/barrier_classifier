"""Step 11 tests: derivatives families parity.

Six sub-families: deriv_basis, deriv_flow, deriv_oi, deriv_funding,
deriv_options, deriv_volidx. All Tier-1; depend on derivatives base
columns produced by ``utils.compute_derivatives_base_series``.

The 30-day vol_realized window (43200 min) requires a long synthetic
fixture; we use 50k rows so at least the last 6800 rows have a fully
populated 30-day window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.features import FeatureEngine
from src.features.config import (
    WINDOWS_BASIS,
    WINDOWS_FLOW_CSUM,
    WINDOWS_FUNDING,
    WINDOWS_OI_CHG,
    WINDOWS_OPTIONS,
    WINDOWS_VOL_IDX,
)

pytestmark = pytest.mark.step11


@pytest.fixture(scope="module")
def bars_with_derivatives() -> pd.DataFrame:
    rng = np.random.default_rng(2024_05_14)
    n = 50_000

    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    spread = np.abs(rng.normal(0.0, 0.5, n))
    high = close + spread
    low = close - spread
    open_ = np.clip(close + rng.normal(0.0, 0.2, n), low, high)
    volume = np.abs(rng.normal(100.0, 30.0, n)) + 1.0
    quote_volume = volume * close
    num_trades = (np.abs(rng.normal(50.0, 20.0, n)).astype(np.int64)) + 1
    taker_buy_base = volume * rng.uniform(0.3, 0.7, n)

    # Synthetic derivatives data
    close_fut = close * (1.0 + rng.normal(0.0, 0.0005, n))
    volume_fut = np.abs(rng.normal(200.0, 60.0, n)) + 1.0
    quote_volume_fut = volume_fut * close_fut
    taker_buy_base_fut = volume_fut * rng.uniform(0.3, 0.7, n)
    num_trades_fut = (np.abs(rng.normal(100.0, 30.0, n)).astype(np.int64)) + 1
    funding_rate = rng.normal(0.0, 0.0001, n)
    oi_usd = 15e9 * (1.0 + np.cumsum(rng.normal(0.0, 0.0001, n)))
    opt_oi = 1e9 * np.abs(rng.normal(1.0, 0.1, n))
    call_open_interest = opt_oi * rng.uniform(0.4, 0.6, n)
    put_open_interest = opt_oi - call_open_interest
    opt_volume = 1e8 * np.abs(rng.normal(1.0, 0.2, n))
    call_volume = opt_volume * rng.uniform(0.4, 0.6, n)
    put_volume = opt_volume - call_volume
    bvol = 60.0 + rng.normal(0.0, 5.0, n)

    ts_index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": quote_volume,
            "num_trades": num_trades, "taker_buy_base": taker_buy_base,
            "close_fut": close_fut, "volume_fut": volume_fut,
            "quote_volume_fut": quote_volume_fut,
            "taker_buy_base_fut": taker_buy_base_fut,
            "num_trades_fut": num_trades_fut,
            "funding_rate": funding_rate, "oi_usd": oi_usd,
            "opt_oi": opt_oi, "put_open_interest": put_open_interest,
            "call_open_interest": call_open_interest,
            "opt_volume": opt_volume, "put_volume": put_volume,
            "call_volume": call_volume, "bvol": bvol,
        },
        index=ts_index,
    )
    df = utils.compute_base_series(df)
    df = utils.compute_derivatives_base_series(df)
    return df


def _to_polars(bars_pd):
    df = bars_pd.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return pl.from_pandas(df.reset_index(names="ts"))


def _assert_parity(legacy_pd, engine_pl, feature_cols, warmup):
    legacy_slice = legacy_pd.iloc[warmup : warmup + len(engine_pl)]
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
                engine_vals[valid], legacy_vals[valid],
                rtol=1e-9, atol=1e-12,
                err_msg=f"{col}: numeric divergence",
            )


def test_all_derivatives_families_parity(bars_with_derivatives):
    legacy = utils.compute_basis_features(bars_with_derivatives, WINDOWS_BASIS)
    legacy = utils.compute_flow_features(legacy, WINDOWS_FLOW_CSUM)
    legacy = utils.compute_oi_features(legacy, WINDOWS_OI_CHG)
    legacy = utils.compute_funding_features(legacy, WINDOWS_FUNDING)
    legacy = utils.compute_options_features(legacy, WINDOWS_OPTIONS)
    legacy = utils.compute_vol_index_features(legacy, WINDOWS_VOL_IDX)

    legacy_cols = [c for c in legacy.columns if c not in bars_with_derivatives.columns]

    bars_pl = _to_polars(bars_with_derivatives)
    engine = FeatureEngine(
        tiers=(1,),
        families=(
            "deriv_basis", "deriv_flow", "deriv_oi", "deriv_funding",
            "deriv_options", "deriv_volidx",
        ),
    )
    result = engine.transform(bars_pl)
    _assert_parity(legacy, result.data, legacy_cols, result.warmup_trimmed)
