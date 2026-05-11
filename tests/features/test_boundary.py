"""Step 13 tests: boundary-stage transformations parity.

Test approach: build a realistic synthetic bars df, run the legacy
compute_* chain (base series + vol_ohlc + boundary functions) to get
the legacy reference. Then run the polars boundary functions against
the same boundary df and assert the outputs match.

Each polars boundary function is independent — we test them isolated
rather than as one chain so failures point at the offending function.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.features.boundary import (
    compute_barrier_aware_features_pl,
    compute_block_features_pl,
    compute_past_target_autocorrelation_pl,
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


# ===========================================================================
# 1-min cadence (bar_stride=1) — new target definition
# ===========================================================================


@pytest.fixture(scope="module")
def fixtures_1min(fixtures):
    """1-min-cadence boundaries frame: every raw row is a decision row."""
    df_raw_pl = fixtures["df_raw_pl"]
    df_boundaries_1min_pl = df_raw_pl.with_columns(pl.int_range(pl.len()).alias("k"))
    return {"df_raw_pl": df_raw_pl, "df_boundaries_1min_pl": df_boundaries_1min_pl}


def test_construct_labels_1min_matches_boundary_at_every_M(fixtures, fixtures_1min):
    """1-min labels at row k*M must equal boundary labels at boundary row k.

    Both look M bars forward from bar n = k*M, so the resulting y/m_k/tau_k
    are identical at those positions."""
    df_raw_pl = fixtures["df_raw_pl"]
    df_boundaries_pl = fixtures["df_boundaries_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]

    labels_boundary = construct_labels_pl(df_boundaries_pl, df_raw_pl, M, ETA, COST_C)
    labels_1min = construct_labels_pl(
        df_b1min_pl, df_raw_pl, M, ETA, COST_C, bar_stride=1
    )
    assert len(labels_1min) == len(df_raw_pl)
    # At rows i*M of the 1-min frame, the label should match the boundary frame's row i
    for i in range(min(50, len(df_boundaries_pl))):
        bnd_y = labels_boundary["y"][i]
        min_y = labels_1min["y"][i * M]
        # Both null OR equal
        if bnd_y is None:
            assert min_y is None, f"row {i}: boundary y=null but 1min y={min_y}"
        else:
            assert min_y == bnd_y, f"row {i}: y mismatch {bnd_y} vs {min_y}"


def test_construct_labels_1min_each_row_uses_its_own_lookahead(fixtures_1min):
    """At 1-min cadence, y_n must reflect max(future_ret over [n+1, n+M]).
    A shift of 1 bar in the entry reference produces a different label."""
    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]

    labels_1min = construct_labels_pl(
        df_b1min_pl, df_raw_pl, M, ETA, COST_C, bar_stride=1
    )
    # Verify by recomputing y at row n directly from raw closes
    close = df_raw_pl["close"].to_numpy().astype(float)
    phi = float(COST_C + ETA)
    n_total = len(close)
    # Pick a few random rows in the safe range
    rng = np.random.default_rng(0)
    rows_to_check = rng.choice(np.arange(M, n_total - M - 1), size=20, replace=False)
    for n in rows_to_check:
        base = close[n]
        m = float(np.log(close[n + 1 : n + M + 1] / base).max())
        expected_y = 1.0 if m >= phi else 0.0
        got_y = labels_1min["y"][int(n)]
        assert got_y == expected_y, f"row {n}: expected y={expected_y}, got {got_y}"


def test_construct_labels_1min_rejects_bad_bar_stride(fixtures_1min):
    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]
    with pytest.raises(ValueError, match="bar_stride"):
        construct_labels_pl(
            df_b1min_pl, df_raw_pl, M, ETA, COST_C, bar_stride=0
        )


def test_construct_labels_high_source_uses_future_highs(fixtures_1min):
    """barrier_source='high' must compute y from future highs, not closes.

    Aligns the label with the simulator's TP execution semantics: a long
    TP-limit order at entry * exp(+phi) fills whenever an intrabar high
    crosses the barrier. The label hits whenever any of the next M highs
    crosses, and is generally >= the close-based label (close[i] <= high[i]).
    """
    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]
    labels_high = construct_labels_pl(
        df_b1min_pl, df_raw_pl, M, ETA, COST_C, bar_stride=1, barrier_source="high"
    )
    close = df_raw_pl["close"].to_numpy().astype(float)
    high = df_raw_pl["high"].to_numpy().astype(float)
    phi = float(COST_C + ETA)
    n_total = len(close)
    rng = np.random.default_rng(7)
    rows_to_check = rng.choice(np.arange(M, n_total - M - 1), size=30, replace=False)
    for n in rows_to_check:
        base = close[n]
        future_high_ret = np.log(high[n + 1 : n + M + 1] / base)
        expected_m = float(future_high_ret.max())
        expected_y = 1.0 if expected_m >= phi else 0.0
        got_y = labels_high["y"][int(n)]
        got_m = labels_high["m_k"][int(n)]
        assert got_y == expected_y, f"row {n}: high-source y mismatch"
        assert got_m is not None and math.isclose(
            float(got_m), expected_m, rel_tol=1e-12, abs_tol=1e-15
        ), f"row {n}: high-source m_k mismatch"


def test_construct_labels_high_source_implies_close_source(fixtures_1min):
    """For every row, the high-source label dominates the close-source label:
    if a future close crosses +phi the corresponding high must also cross it.
    So y_high >= y_close pointwise on all matured rows."""
    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]
    labels_close = construct_labels_pl(
        df_b1min_pl, df_raw_pl, M, ETA, COST_C, bar_stride=1, barrier_source="close"
    )
    labels_high = construct_labels_pl(
        df_b1min_pl, df_raw_pl, M, ETA, COST_C, bar_stride=1, barrier_source="high"
    )
    y_close = labels_close["y"].to_numpy()
    y_high = labels_high["y"].to_numpy()
    # On rows where both labels are matured (not None / NaN), high-source y must
    # be >= close-source y. Convert nulls to NaN and skip them in the check.
    y_close_arr = np.array([float(v) if v is not None else np.nan for v in y_close])
    y_high_arr = np.array([float(v) if v is not None else np.nan for v in y_high])
    both_matured = ~np.isnan(y_close_arr) & ~np.isnan(y_high_arr)
    assert np.all(y_high_arr[both_matured] >= y_close_arr[both_matured])
    # And the high-source variant has at least as many positives as the close-
    # source variant — this is the whole point of the refactor.
    assert y_high_arr[both_matured].sum() >= y_close_arr[both_matured].sum()


def test_construct_labels_triple_barrier_aux_emits_downside(fixtures_1min):
    """add_triple_barrier_aux=True emits m_dn / tau_dn from future lows."""
    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]
    labels = construct_labels_pl(
        df_b1min_pl,
        df_raw_pl,
        M,
        ETA,
        COST_C,
        bar_stride=1,
        barrier_source="high",
        add_triple_barrier_aux=True,
    )
    assert "m_dn" in labels.columns
    assert "tau_dn" in labels.columns
    close = df_raw_pl["close"].to_numpy().astype(float)
    low = df_raw_pl["low"].to_numpy().astype(float)
    phi = float(COST_C + ETA)
    n_total = len(close)
    rng = np.random.default_rng(11)
    rows_to_check = rng.choice(np.arange(M, n_total - M - 1), size=15, replace=False)
    for n in rows_to_check:
        base = close[n]
        future_low_ret = np.log(low[n + 1 : n + M + 1] / base)
        expected_m_dn = float(-future_low_ret.min())
        got_m_dn = labels["m_dn"][int(n)]
        assert got_m_dn is not None and math.isclose(
            float(got_m_dn), expected_m_dn, rel_tol=1e-12, abs_tol=1e-15
        ), f"row {n}: m_dn mismatch"


def test_construct_labels_high_source_requires_high_column():
    """barrier_source='high' on a df_raw without 'high' raises."""
    df_close_only = pl.DataFrame({"close": [100.0, 101.0, 102.0]})
    df_b = pl.DataFrame({"k": [0]})
    with pytest.raises(ValueError, match="requires a 'high' column"):
        construct_labels_pl(
            df_b, df_close_only, 2, ETA, COST_C, bar_stride=1, barrier_source="high"
        )


def test_construct_labels_rejects_bad_barrier_source(fixtures_1min):
    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]
    with pytest.raises(ValueError, match="barrier_source"):
        construct_labels_pl(
            df_b1min_pl, df_raw_pl, M, ETA, COST_C, bar_stride=1, barrier_source="bogus"
        )


def test_past_target_features_1min_strictly_causal(fixtures_1min):
    """At 1-min cadence, hit__prev should use y shifted by M rows so that
    the label is mature (its full future window has elapsed)."""
    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]

    labels_1min = construct_labels_pl(
        df_b1min_pl, df_raw_pl, M, ETA, COST_C, bar_stride=1
    )
    past = compute_past_target_features_pl(
        labels_1min, [3, 6, 12], [3, 6, 12], bar_stride=1, M=M
    )
    # hit__prev should equal y shifted by M rows
    hit_prev = past["hit__prev__h__w0"].to_numpy()
    y = past["y"].to_numpy()
    # Convert None -> NaN for comparison
    y_arr = np.array([float(v) if v is not None else np.nan for v in y], dtype=float)
    hp_arr = np.array(
        [float(v) if v is not None else np.nan for v in hit_prev], dtype=float
    )
    # First M rows of hit_prev must be NaN (no mature label yet)
    assert np.all(np.isnan(hp_arr[:M]))
    # For rows n >= M with non-null y_{n-M}, hit_prev[n] should equal y[n-M]
    n_check = min(2000, len(y_arr) - M)
    for n in range(M, M + n_check):
        if np.isnan(y_arr[n - M]):
            continue
        assert hp_arr[n] == y_arr[n - M], (
            f"row {n}: hit_prev={hp_arr[n]} != y[{n-M}]={y_arr[n-M]}"
        )


def test_past_target_autocorrelation_columns_emitted(fixtures_1min):
    """The autocorrelation feature emits target__autocorr_lag{L}__h__w{W}."""
    from src.features.boundary import compute_past_target_autocorrelation_pl

    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]

    labels = construct_labels_pl(
        df_b1min_pl, df_raw_pl, M, ETA, COST_C, bar_stride=1
    )
    out = compute_past_target_autocorrelation_pl(
        labels, windows=(60, 240), bar_stride=1, M=M, lags=(1, 2, 5)
    )
    for lag in (1, 2, 5):
        for W in (60, 240):
            assert f"target__autocorr_lag{lag}__h__w{W}" in out.columns


def test_past_target_autocorrelation_no_future_leakage(fixtures_1min):
    """Causality probe: mask all labels at rows > t and verify the
    autocorr feature at row t is unchanged.

    If the feature peeked at any y_k for k > t (in row-index sense), the
    masking would change its value. The label-maturity shift of M means
    even the most recent input used at row t is y_{t-M}, so masking
    rows after t cannot affect the row-t output.
    """
    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]

    labels = construct_labels_pl(
        df_b1min_pl, df_raw_pl, M, ETA, COST_C, bar_stride=1
    )
    out_full = compute_past_target_autocorrelation_pl(
        labels, windows=(60, 240), bar_stride=1, M=M, lags=(1, 5)
    )

    # Mask labels after some chosen t to None, then recompute and compare.
    t_probe = 5000  # well past warmup
    n = len(labels)
    y_full = labels["y"].to_list()
    y_masked = [v if i <= t_probe else None for i, v in enumerate(y_full)]
    labels_masked = labels.with_columns(pl.Series("y", y_masked))
    out_masked = compute_past_target_autocorrelation_pl(
        labels_masked, windows=(60, 240), bar_stride=1, M=M, lags=(1, 5)
    )

    # At t_probe the feature must be identical (no future leakage).
    # Check all autocorr columns.
    for col in out_full.columns:
        if not col.startswith("target__autocorr_"):
            continue
        v_full = out_full[col][t_probe]
        v_masked = out_masked[col][t_probe]
        # Handle null comparison
        if v_full is None and v_masked is None:
            continue
        assert v_full == v_masked, (
            f"causality violation in {col} at row {t_probe}: "
            f"full={v_full} != masked={v_masked}"
        )


def test_block_features_1min_each_row_uses_trailing_M_bars(fixtures_1min):
    """At 1-min cadence, block features must be rolling over the trailing M
    bars (not non-overlapping blocks). Verify ret__inst[n] = log(close[n]/close[n-M])."""
    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]

    out = compute_block_features_pl(
        df_b1min_pl, df_raw_pl, M, WINDOWS_H, bar_stride=1
    )
    close = df_raw_pl["close"].to_numpy().astype(float)
    ret_inst = out["ret__inst__h__w0"].to_numpy()
    # First M rows of ret_inst must be NaN
    assert np.all(np.isnan(ret_inst[:M].astype(float)))
    # For n >= M: ret_inst[n] = log(close[n] / close[n-M])
    n_check = min(2000, len(close) - M)
    for n in range(M, M + n_check):
        expected = float(np.log(close[n] / close[n - M]))
        got = float(ret_inst[n])
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-15)


def test_block_features_1min_block_maxret_strictly_causal(fixtures_1min):
    """block__maxret at row n must be the max excursion in [n-M+1, n] relative
    to close[n-M] — never peeks past n."""
    df_raw_pl = fixtures_1min["df_raw_pl"]
    df_b1min_pl = fixtures_1min["df_boundaries_1min_pl"]

    out = compute_block_features_pl(
        df_b1min_pl, df_raw_pl, M, WINDOWS_H, bar_stride=1
    )
    close = df_raw_pl["close"].to_numpy().astype(float)
    p = np.log(close)
    bmax = out["block__maxret__h__w0"].to_numpy()
    n_check = min(500, len(close) - M)
    rng = np.random.default_rng(42)
    rows_to_check = rng.choice(np.arange(M, M + n_check), size=20, replace=False)
    for n in rows_to_check:
        baseline = p[n - M]
        excursions = p[n - M + 1 : n + 1] - baseline
        expected = float(excursions.max())
        got = float(bmax[int(n)])
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12), (
            f"row {n}: block_maxret={got} != expected={expected}"
        )


def test_block_features_boundary_vs_1min_align_at_boundary_rows(fixtures):
    """At boundary positions, the 1-min-cadence block features should match
    the boundary-cadence values (they look at the same M-bar window)."""
    df_raw_pl = fixtures["df_raw_pl"]
    df_b_pl = fixtures["df_boundaries_pl"]
    df_b1min_pl = df_raw_pl.with_columns(pl.int_range(pl.len()).alias("k"))

    out_boundary = compute_block_features_pl(df_b_pl, df_raw_pl, M, WINDOWS_H)
    out_1min = compute_block_features_pl(
        df_b1min_pl, df_raw_pl, M, WINDOWS_H, bar_stride=1
    )
    # ret__inst at boundary row i corresponds to 1-min row i*M
    rb = out_boundary["ret__inst__h__w0"].to_numpy()
    r1 = out_1min["ret__inst__h__w0"].to_numpy()
    for i in range(1, min(50, len(rb))):
        a = float(rb[i]) if rb[i] is not None and not np.isnan(rb[i]) else None
        b = float(r1[i * M]) if r1[i * M] is not None and not np.isnan(r1[i * M]) else None
        if a is None and b is None:
            continue
        assert a is not None and b is not None
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (
            f"boundary i={i} (n={i*M}): boundary={a} vs 1min={b}"
        )
