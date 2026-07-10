"""Step 14 tests: data quality flags + undef/imputation pipeline parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.features.quality import (
    compute_data_quality_flags_pl,
    create_undef_flags_and_impute_pl,
)

pytestmark = pytest.mark.step14


def _to_polars(bars_pd: pd.DataFrame) -> pl.DataFrame:
    df = bars_pd.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return pl.from_pandas(df.reset_index(names="ts"))


@pytest.fixture
def bars_with_ohlc_anomalies() -> pd.DataFrame:
    """OHLCV with a few crafted bad-OHLC rows and a gap."""
    rng = np.random.default_rng(2024_05_17)
    n = 1000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    spread = np.abs(rng.normal(0.0, 0.5, n))
    high = close + spread
    low = close - spread
    open_ = np.clip(close + rng.normal(0.0, 0.2, n), low, high)
    # Inject 3 bad-OHLC rows
    high[10] = low[10] - 0.1   # high < low
    high[20] = open_[20] - 0.1  # high < open
    low[30] = close[30] + 0.1   # low > close

    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    # Inject a gap between idx 500 and 501 (skip a minute)
    ts_list = list(ts)
    for i in range(501, n):
        ts_list[i] = ts_list[i] + pd.Timedelta(minutes=1)
    idx = pd.DatetimeIndex(ts_list)

    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": np.full(n, 100.0),
            "quote_volume": np.full(n, 10000.0),
            "num_trades": np.full(n, 50, dtype=np.int64),
            "taker_buy_base": np.full(n, 50.0),
        },
        index=idx,
    )


def test_data_quality_flags_parity(bars_with_ohlc_anomalies):
    legacy = utils.compute_data_quality_flags(bars_with_ohlc_anomalies)
    bars_pl = _to_polars(bars_with_ohlc_anomalies)
    engine = compute_data_quality_flags_pl(bars_pl)

    # Bad-OHLC must match exactly (5 conditions).
    legacy_bad = legacy["data__bad_ohlc__f__w0"].to_numpy().astype(np.int64)
    engine_bad = np.array(engine["data__bad_ohlc__f__w0"].to_list(), dtype=np.int64)
    np.testing.assert_array_equal(engine_bad, legacy_bad)
    # The 3 injected anomalies should fire
    assert legacy_bad[10] == 1
    assert legacy_bad[20] == 1
    assert legacy_bad[30] == 1

    # Gap must match: row 0 = 0; row 501 should fire (1-minute jump injected)
    legacy_gap = legacy["data__gap__f__w0"].to_numpy().astype(np.int64)
    engine_gap = np.array(engine["data__gap__f__w0"].to_list(), dtype=np.int64)
    np.testing.assert_array_equal(engine_gap, legacy_gap)


def test_impute_coerces_float_nan_to_null_then_fills():
    """Engine map_batches kernels can emit float NaN. Impute must catch
    them just like null — coerce NaN -> null upfront, raise undef flag,
    fill with the registry value, and verify zero remaining NaN/null/inf
    on exit. Without the coercion, the silent NaN bug shipped to disk
    once on real data."""
    df = pl.DataFrame(
        {
            "ret__rsi__f__w14": pl.Series(
                [50.0, float("nan"), 60.0, None, 55.0],
                dtype=pl.Float64,
            ),
        }
    )
    out, undef_cols = create_undef_flags_and_impute_pl(
        df, ["ret__rsi__f__w14"], p_hit_prior=0.5
    )

    assert undef_cols == ["undef__ret__rsi__f__w14"]
    # Undef flag fires for both the NaN and the null row
    flags = out["undef__ret__rsi__f__w14"].to_list()
    assert flags == [0, 1, 0, 1, 0]
    # The two missing cells get filled with 50.0 (rsi prior)
    vals = out["ret__rsi__f__w14"].to_list()
    assert vals == [50.0, 50.0, 60.0, 50.0, 55.0]
    # Post-impute frame has zero null and zero NaN
    assert int(out["ret__rsi__f__w14"].null_count()) == 0
    assert int(out["ret__rsi__f__w14"].is_nan().sum()) == 0


def test_impute_raises_on_residual_nan_after_fill(monkeypatch):
    """If the imputation registry returns a non-finite value (NaN/inf),
    the impute step must raise rather than silently fill the column
    with NaN and ship a poisoned dataset to training.

    Patches ``utils.get_imputation_value`` to return ``float('nan')`` so
    we exercise the safety check at the registry boundary."""
    from src.features import quality as quality_mod

    df = pl.DataFrame(
        {
            "ret__rsi__f__w14": pl.Series(
                [50.0, None, 60.0, 55.0], dtype=pl.Float64
            )
        }
    )

    def _nan_registry(col, *, p_hit_prior, cap_h_blocks):  # noqa: ARG001
        return float("nan")

    monkeypatch.setattr(
        quality_mod._legacy, "get_imputation_value", _nan_registry
    )
    with pytest.raises(ValueError, match="non-finite"):
        quality_mod.create_undef_flags_and_impute_pl(
            df, ["ret__rsi__f__w14"], p_hit_prior=0.5
        )


def test_impute_runs_on_mixed_float_int_dtypes():
    """``is_infinite`` is float-only — older code paths that called it on
    every numeric dtype broke on Int8 / Int32 columns under newer polars.
    Verify the impute pipeline handles a mixed-dtype frame without
    raising."""
    df = pl.DataFrame(
        {
            "ret__rsi__f__w14": pl.Series(
                [50.0, None, 60.0, 55.0], dtype=pl.Float64
            ),
            "event__run_dir__f__w0": pl.Series(
                [1, -1, 0, 1], dtype=pl.Int8
            ),
            "event__run_len__f__w0": pl.Series(
                [1, 1, 2, 3], dtype=pl.Int32
            ),
        }
    )
    out, undef_cols = create_undef_flags_and_impute_pl(
        df,
        [
            "ret__rsi__f__w14",
            "event__run_dir__f__w0",
            "event__run_len__f__w0",
        ],
        p_hit_prior=0.5,
    )
    # Only the float column had a null; integer columns must pass through
    # unchanged with no undef flag.
    assert undef_cols == ["undef__ret__rsi__f__w14"]
    assert out["event__run_dir__f__w0"].dtype == pl.Int8
    assert out["event__run_len__f__w0"].dtype == pl.Int32


def test_undef_and_impute_parity():
    """Synthetic feature df with NaN scattered across known patterns:
    imputation values come from the legacy regex registry, so equality
    means we are reading the right impute value per column.
    """
    n = 200
    rng = np.random.default_rng(2024_05_18)

    # A handful of feature columns with crafted NaN patterns
    base = rng.normal(0.0, 1.0, n)
    df_pd = pd.DataFrame(
        {
            "ret__rsi__f__w14": np.where(rng.random(n) < 0.1, np.nan, 50 + 10 * base),
            "vol__bpv_ratio__f__w60": np.where(rng.random(n) < 0.1, np.nan, 1 + 0.1 * base),
            "logp__pos__f__w20": np.where(rng.random(n) < 0.1, np.nan, 0.5 + 0.1 * base),
            "hit__rate__h__w12": np.where(rng.random(n) < 0.1, np.nan, 0.3 + 0.05 * base),
            "logvol__mean__f__w60": base,  # no NaN — should produce no flag
        }
    )
    feature_cols = list(df_pd.columns)

    legacy_imputed, legacy_undef_cols = utils.create_undef_flags_and_impute(
        df_pd, feature_cols, p_hit_prior=0.5, cap_h_blocks=144
    )

    df_pl = pl.from_pandas(df_pd)
    engine_imputed, engine_undef_cols = create_undef_flags_and_impute_pl(
        df_pl, feature_cols, p_hit_prior=0.5, cap_h_blocks=144
    )

    assert sorted(engine_undef_cols) == sorted(legacy_undef_cols)

    for col in feature_cols + legacy_undef_cols:
        legacy_vals = legacy_imputed[col].to_numpy()
        engine_vals = np.array(engine_imputed[col].to_list())
        np.testing.assert_allclose(
            engine_vals.astype(float),
            legacy_vals.astype(float),
            rtol=1e-12, atol=1e-15,
            err_msg=f"{col}: divergence",
        )
