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


def test_excursion_rolling_equals_sparse_at_boundary_rows(bars_with_base):
    """At rows that are multiples of M, the new every-row trailing
    drawup/drawdown values must equal the legacy boundary-sparse values.

    Proves the rolling variant is a strict superset (same value at the
    legacy-populated rows, plus values at every other row). This is the
    contract that makes the rolling variant a safe drop-in replacement
    at 1-min cadence without losing boundary-cadence comparability."""
    from src.features.config import M as M_BARS

    bars_pl = _to_polars(bars_with_base)
    engine = FeatureEngine(tiers=(2,), families=("excursion",))
    out = engine.transform(bars_pl, trim=False).data

    for W in WINDOWS_EXCURSION:
        sparse_up = f"excursion__max_drawup__f__w{W}"
        sparse_dn = f"excursion__max_drawdown__f__w{W}"
        roll_up = f"excursion__roll_max_drawup__f__w{W}"
        roll_dn = f"excursion__roll_max_drawdown__f__w{W}"
        for sparse_col, roll_col in [(sparse_up, roll_up), (sparse_dn, roll_dn)]:
            sparse = np.array(
                [v if v is not None else np.nan for v in out[sparse_col].to_list()],
                dtype=float,
            )
            roll = np.array(
                [v if v is not None else np.nan for v in out[roll_col].to_list()],
                dtype=float,
            )
            # On rows that are multiples of M and have a populated sparse
            # value, the rolling value must coincide. Sparse rows beyond
            # warmup have a value; non-multiples-of-M have NaN.
            n = len(sparse)
            for i in range(W - 1, n, M_BARS):
                if np.isnan(sparse[i]):
                    continue
                assert not np.isnan(roll[i]), (
                    f"{sparse_col}: sparse is populated at row {i} but roll is NaN"
                )
                assert abs(sparse[i] - roll[i]) < 1e-12, (
                    f"{sparse_col} vs {roll_col} at row {i}: {sparse[i]} != {roll[i]}"
                )
            # Rolling values must also be populated AFTER warmup at rows
            # that are NOT multiples of M (the whole point of the refactor).
            non_M_rows_after_warmup = [
                i for i in range(W - 1, min(W + 200, n)) if i % M_BARS != 0
            ]
            n_dense = sum(1 for i in non_M_rows_after_warmup if not np.isnan(roll[i]))
            assert n_dense == len(non_M_rows_after_warmup), (
                f"{roll_col}: expected all non-M rows after warmup to be dense; "
                f"got {n_dense}/{len(non_M_rows_after_warmup)}"
            )


def test_excursion_rolling_is_strictly_causal_under_masking(bars_with_base):
    """Mask price-history rows AFTER a chosen probe row and verify the
    rolling drawup/drawdown values at and before that row are unchanged.

    A non-causal computation (e.g. one that peeked at a centered window)
    would see different values when the future is masked. The trailing-
    window formula must not."""
    bars_pl = _to_polars(bars_with_base)
    engine = FeatureEngine(tiers=(2,), families=("excursion",))
    out_full = engine.transform(bars_pl, trim=False).data

    # Mask: replace p (the log-price column the family reads) after t_probe
    # with null and recompute the engine.
    t_probe = 5000
    p_full = out_full["p"].to_list()
    p_masked = [v if i <= t_probe else None for i, v in enumerate(p_full)]
    bars_masked = bars_pl.with_columns(pl.Series("p", p_masked))
    out_masked = engine.transform(bars_masked, trim=False).data

    cols_to_check = [
        c for c in out_full.columns
        if c.startswith("excursion__roll_max_drawup__f__")
        or c.startswith("excursion__roll_max_drawdown__f__")
    ]
    for col in cols_to_check:
        v_full = out_full[col][t_probe]
        v_masked = out_masked[col][t_probe]
        if v_full is None and v_masked is None:
            continue
        assert v_full == v_masked, (
            f"causality violation in {col} at probe row {t_probe}: "
            f"full={v_full} masked={v_masked}"
        )
