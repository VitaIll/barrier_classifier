"""Step 12 tests: Tier-2 excursion + liquidity families parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.features import FeatureEngine
from src.features.config import (
    WINDOWS_EXCURSION,
    WINDOWS_LIQ_AMIHUD,
    WINDOWS_LIQ_RPV,
    WINDOWS_MAXRET,
    WINDOWS_OFI_IMPULSE,
)

pytestmark = pytest.mark.step12


@pytest.fixture(scope="module")
def bars_with_base() -> pd.DataFrame:
    rng = np.random.default_rng(2024_05_15)
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


def test_excursion_and_liquidity_parity(bars_with_base):
    legacy = utils.compute_excursion_features(bars_with_base, WINDOWS_EXCURSION, WINDOWS_MAXRET)
    legacy = utils.compute_enhanced_liquidity(
        legacy, WINDOWS_LIQ_AMIHUD, WINDOWS_LIQ_RPV, WINDOWS_OFI_IMPULSE
    )
    legacy_cols = [c for c in legacy.columns if c not in bars_with_base.columns]

    bars_pl = _to_polars(bars_with_base)
    engine = FeatureEngine(tiers=(1, 2), families=("excursion", "liquidity"))
    result = engine.transform(bars_pl)
    _assert_parity(legacy, result.data, legacy_cols, result.warmup_trimmed)
