"""Step 7 tests: lag + rolling families end-to-end parity.

Strategy:
  1. Generate synthetic OHLCV (~30k rows; enough that warmup leaves
     a healthy validation window even for the 20160-bar feature).
  2. Run ``utils.compute_base_series`` to materialize r, rho, clv, etc.
  3. Run the legacy ``utils.compute_lag_features`` / ``compute_rolling_stats``.
  4. Run the new ``FeatureEngine`` with the same input.
  5. Assert each emitted feature column is bit-equal across the engine's
     post-warmup row range.

This is the first end-to-end engine test — it exercises the registry,
window expansion, single-pass with_columns execution, and warmup trim
all at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.features import FeatureEngine
from src.features.config import LAGS_F, WINDOWS_F

pytestmark = pytest.mark.step7


# ---------------------------------------------------------------------------
# Test fixture: realistic synthetic bar dataframe with full base series.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bars_with_base() -> pd.DataFrame:
    """Synthetic 30k-row bar frame with all base-series columns populated.

    Generated once per module — module scope amortizes the OHLCV synthesis
    + base-series compute across all parity tests in this file.
    """
    rng = np.random.default_rng(2024_05_10)
    n = 30_000

    # Synthetic geometric Brownian close
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    spread = np.abs(rng.normal(0.0, 0.5, n))
    high = close + spread
    low = close - spread
    open_ = close + rng.normal(0.0, 0.2, n)
    open_ = np.clip(open_, low, high)
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
    """Convert pandas → polars while preserving the timestamp as a column."""
    df = pl.from_pandas(bars_pd.reset_index(names="ts"))
    return df


def _assert_columns_bit_equal(
    legacy_pd: pd.DataFrame,
    engine_pl: pl.DataFrame,
    feature_cols: list[str],
    warmup: int,
) -> None:
    """Assert every column in ``feature_cols`` matches between legacy
    (pandas) and engine (polars), aligned by row position after the
    engine's warmup trim.

    Tolerance: ``rtol=1e-9, atol=1e-12``. Polars and pandas use slightly
    different summation orders for rolling std, producing last-ulp noise
    on small-window/small-value cases (relative diffs ~1e-10). Real
    algorithmic bugs would diverge orders of magnitude beyond this.
    Null/NaN positions must match exactly.
    """
    legacy_slice = legacy_pd.iloc[warmup : warmup + len(engine_pl)]
    assert len(legacy_slice) == len(engine_pl), (
        f"row alignment mismatch: legacy={len(legacy_slice)} engine={len(engine_pl)}"
    )

    missing = [c for c in feature_cols if c not in engine_pl.columns]
    assert not missing, f"engine missing columns: {missing[:5]} (total {len(missing)})"

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
            f"(legacy_nan_count={legacy_nan.sum()}, engine_nan_count={engine_nan.sum()})"
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
# Lag family parity
# ===========================================================================


class TestLagFamilyParity:
    """All ret/absret/range/clv/logvol/logtrades/ofi lag columns match legacy."""

    def test_lag_engine_vs_legacy_compute_lag_features(self, bars_with_base):
        legacy = utils.compute_lag_features(bars_with_base, LAGS_F)
        legacy_feature_cols = [c for c in legacy.columns if "__lag" in c]

        bars_pl = _to_polars(bars_with_base)
        engine = FeatureEngine(tiers=(1,), families=("lag",))
        result = engine.transform(bars_pl)

        assert result.warmup_trimmed == max(LAGS_F)
        assert result.tail_trimmed == 0

        # 7 outputs per lag × 53 lags = 371 columns
        expected_count = 7 * len(LAGS_F)
        actual_count = sum(1 for c in result.data.columns if "__lag" in c)
        assert actual_count == expected_count, (
            f"engine emitted {actual_count} lag columns; expected {expected_count}"
        )
        assert len(legacy_feature_cols) == expected_count

        _assert_columns_bit_equal(
            legacy_pd=legacy,
            engine_pl=result.data,
            feature_cols=legacy_feature_cols,
            warmup=result.warmup_trimmed,
        )


# ===========================================================================
# Rolling family parity
# ===========================================================================


class TestRollingFamilyParity:
    """All 9 rolling output columns match legacy compute_rolling_stats."""

    def test_rolling_engine_vs_legacy_compute_rolling_stats(self, bars_with_base):
        legacy = utils.compute_rolling_stats(bars_with_base, WINDOWS_F)
        # Rolling features end with __f__w{W}; everything in compute_rolling_stats
        # follows that pattern, but other features in compute_base_series do not.
        legacy_feature_cols = [
            c for c in legacy.columns if c not in bars_with_base.columns
        ]

        bars_pl = _to_polars(bars_with_base)
        engine = FeatureEngine(tiers=(1,), families=("rolling",))
        result = engine.transform(bars_pl)

        # 9 outputs per window × 32 windows = 288 columns
        expected_count = 9 * len(WINDOWS_F)
        assert len(legacy_feature_cols) == expected_count
        engine_feature_cols = [
            c for c in result.data.columns if c.endswith("__f__w" + str(WINDOWS_F[0]))
            or any(c.endswith(f"__f__w{w}") for w in WINDOWS_F)
        ]
        # Lighter assertion: all expected columns are present
        for col in legacy_feature_cols:
            assert col in result.data.columns, f"engine missing {col}"

        # Warmup = max(WINDOWS_F) - 1 (rolling default)
        assert result.warmup_trimmed == max(WINDOWS_F) - 1
        assert result.tail_trimmed == 0

        _assert_columns_bit_equal(
            legacy_pd=legacy,
            engine_pl=result.data,
            feature_cols=legacy_feature_cols,
            warmup=result.warmup_trimmed,
        )


# ===========================================================================
# Combined parity (lag + rolling in one engine pass)
# ===========================================================================


class TestCombinedParity:
    """Run both families in a single engine call; verify bit-equal output.

    The engine trims warmup to the maximum across all features, so the
    rolling family's larger warmup dominates. Lag features still produce
    valid values for every row in the post-trim window.
    """

    def test_lag_and_rolling_in_one_pass(self, bars_with_base):
        legacy = utils.compute_rolling_stats(
            utils.compute_lag_features(bars_with_base, LAGS_F),
            WINDOWS_F,
        )
        legacy_feature_cols = [
            c for c in legacy.columns if c not in bars_with_base.columns
        ]
        assert len(legacy_feature_cols) == 7 * len(LAGS_F) + 9 * len(WINDOWS_F)

        bars_pl = _to_polars(bars_with_base)
        engine = FeatureEngine(tiers=(1,), families=("lag", "rolling"))
        result = engine.transform(bars_pl)

        # Rolling warmup dominates
        assert result.warmup_trimmed == max(WINDOWS_F) - 1

        _assert_columns_bit_equal(
            legacy_pd=legacy,
            engine_pl=result.data,
            feature_cols=legacy_feature_cols,
            warmup=result.warmup_trimmed,
        )


# ===========================================================================
# Targeted L1: a few hand-checked values to catch silly bugs
# ===========================================================================


class TestLagL1Sanity:
    """Tiny synthetic data with hand-computed expected outputs."""

    def test_lag_ret_one_row(self):
        bars = pl.DataFrame(
            {"r": [0.1, 0.2, 0.3, 0.4, 0.5], "rho": [0.0] * 5,
             "clv": [0.0] * 5, "logvol": [0.0] * 5, "logtrades": [0.0] * 5,
             "ofi": [0.0] * 5}
        )
        # Manually instantiate one feature outside the registry
        from src.features.families.lag import LagRet
        spec = LagRet()
        out = bars.select(spec.compute(2).alias("out"))["out"].to_list()
        # shift(2): [null, null, 0.1, 0.2, 0.3]
        assert out[0] is None
        assert out[1] is None
        assert out[2] == 0.1
        assert out[3] == 0.2
        assert out[4] == 0.3


class TestRollingL1Sanity:
    """Tiny synthetic data with hand-computed expected outputs."""

    def test_rolling_ret_mean_one_window(self):
        from src.features.families.rolling import RollingRetMean
        bars = pl.DataFrame({"r": [1.0, 2.0, 3.0, 4.0, 5.0]})
        spec = RollingRetMean()
        out = bars.select(spec.compute(3).alias("out"))["out"].to_list()
        assert out[0] is None
        assert out[1] is None
        assert out[2] == 2.0
        assert out[3] == 3.0
        assert out[4] == 4.0

    def test_posfrac_handles_null_as_zero(self):
        from src.features.families.rolling import RollingRetPosfrac
        # Mimics engine behavior: null at index 0 (fill_nan(None) of NaN diff).
        bars = pl.DataFrame({"r": [None, 0.1, -0.2, 0.3, 0.4]})
        spec = RollingRetPosfrac()
        out = bars.select(spec.compute(3).alias("out"))["out"].to_list()
        # After fill_null(0.0): [0.0, 0.1, -0.2, 0.3, 0.4]
        # > 0: [F, T, F, T, T] → [0, 1, 0, 1, 1]
        # rolling mean w=3: [null, null, (0+1+0)/3, (1+0+1)/3, (0+1+1)/3]
        assert out[0] is None
        assert out[1] is None
        assert abs(out[2] - 1 / 3) < 1e-12
        assert abs(out[3] - 2 / 3) < 1e-12
        assert abs(out[4] - 2 / 3) < 1e-12
