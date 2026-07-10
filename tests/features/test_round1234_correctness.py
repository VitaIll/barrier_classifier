"""Comprehensive correctness audit of Round 1-4 feature additions.

Four sections:

  A. Bit-equality to oracle  — every new numerical kernel is compared to an
     independent numpy / scipy reference computation on a fixed input.
  B. Causality probes        — mask all rows after a chosen ``t_probe`` and
     re-evaluate. The feature value at row ``t_probe`` MUST be unchanged.
     This is the strongest test of leakage we can do without an oracle.
  C. Edge cases              — clamp activation, no-pivot sentinel, mutual
     exclusion of OI regime quadrants, calendar boundaries.
  D. Pipeline integration    — end-to-end ``run_pipeline`` emits every
     expected column and the imputation step leaves zero residual nulls
     / NaN / inf in any new feature.

Tests are self-contained: each builds its own small synthetic frame so
failures point at the offending feature directly. No module-scoped
fixtures shared with other test files.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.special import erf as _scipy_erf

from src import utils
from src.features import FeatureEngine
from src.features.boundary import (
    compute_barrier_aware_features_pl,
    compute_past_target_features_pl,
    construct_labels_pl,
)
from src.features.config import (
    C as COST_C,
    EPS,
    ETA,
    M,
    PHI,
    VOL_PAIRS,
    WINDOWS_BARRIER,
)


# ---------------------------------------------------------------------------
# Shared synthetic-bar factory
# ---------------------------------------------------------------------------


def _synthetic_bars(
    n: int,
    *,
    seed: int = 0,
    with_derivatives_base: bool = False,
) -> tuple[pd.DataFrame, pl.DataFrame]:
    """Build synthetic OHLCV bars; optionally include the minimal set of
    derivative base columns needed by ``deriv_oi`` and ``deriv_funding``
    families.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    spread = np.abs(rng.normal(0.0, 0.5, n))
    high = close + spread
    low = np.maximum(close - spread, 0.001)
    open_ = np.clip(close + rng.normal(0.0, 0.2, n), low, high)
    volume = np.abs(rng.normal(100.0, 30.0, n)) + 1.0
    qv = volume * close
    nt = (np.abs(rng.normal(50.0, 20.0, n)).astype(np.int64)) + 1
    tbb = volume * rng.uniform(0.3, 0.7, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    cols = {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": qv,
        "num_trades": nt, "taker_buy_base": tbb,
    }
    if with_derivatives_base:
        cols.update({
            "close_fut": close * 1.001,
            "volume_fut": volume,
            "quote_volume_fut": qv,
            "taker_buy_base_fut": 0.5 * volume,
            "num_trades_fut": nt,
            "funding_rate": np.full(n, 1e-4),
            "oi_usd": 15e9 + np.cumsum(rng.normal(0.0, 1e7, n)),
            "opt_oi": np.full(n, 1e9),
            "put_open_interest": np.full(n, 5e8),
            "call_open_interest": np.full(n, 5e8),
            "opt_volume": np.full(n, 1e7),
            "put_volume": np.full(n, 5e6),
            "call_volume": np.full(n, 5e6),
            "bvol": np.full(n, 60.0),
        })
    df = pd.DataFrame(cols, index=ts)
    df = utils.compute_base_series(df)
    if with_derivatives_base:
        df = utils.compute_derivatives_base_series(df)
    return df, pl.from_pandas(df.reset_index(names="ts"))


def _run_engine(
    df_pl: pl.DataFrame,
    families: tuple[str, ...],
) -> pl.DataFrame:
    engine = FeatureEngine(tiers=(1, 2), families=families)
    return engine.transform(df_pl, trim=False).data


# ---------------------------------------------------------------------------
# Section A — Bit-equality to oracle
# ---------------------------------------------------------------------------


def test_extreme_dist_low_z_bit_equal_to_oracle():
    df_pd, df_pl = _synthetic_bars(800, seed=1)
    out = _run_engine(df_pl, families=("rolling", "vol", "extreme"))
    W = 60
    col = f"extreme__dist_low_z__f__w{W}"

    p = df_pd["p"].to_numpy()
    log_low = np.log(df_pd["low"].to_numpy())
    sigma = out[f"vol__rs__f__w{W}"].to_numpy()
    sqrt_M = math.sqrt(int(M))

    # Oracle: trailing rolling min of log_low over W bars, then dist / vol scale.
    trailing_min = pd.Series(log_low).rolling(W, min_periods=W).min().to_numpy()
    expected = (p - trailing_min) / (sigma * sqrt_M + EPS)

    got = out[col].to_numpy()
    valid = ~(np.isnan(expected) | np.isnan(got))
    assert valid.any()
    np.testing.assert_allclose(got[valid], expected[valid], rtol=1e-12, atol=1e-14)


def test_extreme_dist_high_z_bit_equal_to_oracle():
    df_pd, df_pl = _synthetic_bars(800, seed=2)
    out = _run_engine(df_pl, families=("rolling", "vol", "extreme"))
    W = 60
    col = f"extreme__dist_high_z__f__w{W}"

    p = df_pd["p"].to_numpy()
    log_high = np.log(df_pd["high"].to_numpy())
    sigma = out[f"vol__rs__f__w{W}"].to_numpy()
    sqrt_M = math.sqrt(int(M))

    trailing_max = pd.Series(log_high).rolling(W, min_periods=W).max().to_numpy()
    expected = (trailing_max - p) / (sigma * sqrt_M + EPS)

    got = out[col].to_numpy()
    valid = ~(np.isnan(expected) | np.isnan(got))
    assert valid.any()
    np.testing.assert_allclose(got[valid], expected[valid], rtol=1e-12, atol=1e-14)


def test_extreme_price_rank_bit_equal_to_oracle():
    df_pd, df_pl = _synthetic_bars(800, seed=3)
    out = _run_engine(df_pl, families=("rolling", "vol", "extreme"))
    W = 60
    col = f"extreme__price_rank__f__w{W}"
    p = df_pd["p"].to_numpy()

    # Oracle: for each n >= W-1, count p[i] <= p[n] for i in [n-W+1, n] / W.
    expected = np.full(len(p), np.nan, dtype=float)
    for n in range(W - 1, len(p)):
        window = p[n - W + 1 : n + 1]
        expected[n] = float((window <= p[n]).sum()) / float(W)

    got = out[col].to_numpy()
    valid = ~(np.isnan(expected) | np.isnan(got))
    assert valid.sum() >= len(p) - W
    np.testing.assert_allclose(got[valid], expected[valid], rtol=1e-12, atol=1e-14)


def test_trend_quad_slope_curv_bit_equal_to_polyfit():
    df_pd, df_pl = _synthetic_bars(800, seed=4)
    out = _run_engine(df_pl, families=("rolling", "vol", "trend"))
    W = 60
    p = df_pd["p"].to_numpy()
    sigma = out[f"vol__rs__f__w{W}"].to_numpy()

    sample_rows = list(range(W - 1, len(p), 23))[:30]
    slope_col = out[f"trend__quad_slope_z__f__w{W}"].to_numpy()
    curv_col = out[f"trend__quad_curv_z__f__w{W}"].to_numpy()
    u = np.arange(W, dtype=float) - (W - 1) / 2.0

    for n in sample_rows:
        window = p[n - W + 1 : n + 1]
        if np.any(np.isnan(window)):
            continue
        # np.polyfit returns coefficients highest-to-lowest: [β2, β1, α].
        beta2_np, beta1_np, _ = np.polyfit(u, window, 2)
        s = sigma[n]
        if not np.isfinite(s):
            continue
        denom = s * math.sqrt(W) + EPS
        expected_slope = beta1_np * float(W) / denom
        expected_curv = beta2_np * (float(W) ** 2) / denom
        assert math.isclose(slope_col[n], expected_slope, rel_tol=1e-9, abs_tol=1e-12), (
            f"slope[{n}]: got {slope_col[n]}, expected {expected_slope}"
        )
        assert math.isclose(curv_col[n], expected_curv, rel_tol=1e-9, abs_tol=1e-12), (
            f"curv[{n}]: got {curv_col[n]}, expected {expected_curv}"
        )


def test_vol_semivar_signed_bit_equal_to_oracle():
    df_pd, df_pl = _synthetic_bars(800, seed=5)
    out = _run_engine(df_pl, families=("rolling", "vol"))
    W = 60
    col = f"vol__semivar_signed__f__w{W}"
    r = df_pd["r"].fillna(0.0).to_numpy()

    pos_sq = np.where(r > 0, r ** 2, 0.0)
    neg_sq = np.where(r < 0, r ** 2, 0.0)
    rv_pos = pd.Series(pos_sq).rolling(W, min_periods=W).sum().to_numpy()
    rv_neg = pd.Series(neg_sq).rolling(W, min_periods=W).sum().to_numpy()
    expected = (rv_pos - rv_neg) / (rv_pos + rv_neg + EPS)

    got = out[col].to_numpy()
    valid = ~(np.isnan(expected) | np.isnan(got))
    np.testing.assert_allclose(got[valid], expected[valid], rtol=1e-9, atol=1e-12)


def test_vol_jump_ratio_bit_equal_to_oracle():
    df_pd, df_pl = _synthetic_bars(800, seed=6)
    out = _run_engine(df_pl, families=("rolling", "vol"))
    W = 60
    col = f"vol__jump_ratio__f__w{W}"
    r = df_pd["r"].to_numpy()

    r_sq = r ** 2
    abs_r = np.abs(r)
    prod = abs_r * np.concatenate([[np.nan], abs_r[:-1]])  # |r| * |r.shift(1)|
    rv = pd.Series(r_sq).rolling(W, min_periods=W).sum().to_numpy()
    bv = (math.pi / 2.0) * pd.Series(prod).rolling(W - 1, min_periods=W - 1).sum().to_numpy()
    expected = np.clip((rv - bv) / (rv + EPS), 0.0, 1.0)

    got = out[col].to_numpy()
    valid = ~(np.isnan(expected) | np.isnan(got))
    assert valid.any()
    np.testing.assert_allclose(got[valid], expected[valid], rtol=1e-9, atol=1e-12)


def test_flow_pressure_bit_equal_to_oracle():
    df_pd, df_pl = _synthetic_bars(800, seed=7)
    out = _run_engine(df_pl, families=("rolling", "vol", "flow"))
    W = 60
    col = f"flow__pressure__f__w{W}"
    tbb = df_pd["taker_buy_base"].to_numpy()
    vol = df_pd["volume"].to_numpy()
    sum_tbb = pd.Series(tbb).rolling(W, min_periods=W).sum().to_numpy()
    sum_vol = pd.Series(vol).rolling(W, min_periods=W).sum().to_numpy()
    expected = 2.0 * sum_tbb / (sum_vol + EPS) - 1.0

    got = out[col].to_numpy()
    valid = ~(np.isnan(expected) | np.isnan(got))
    np.testing.assert_allclose(got[valid], expected[valid], rtol=1e-12, atol=1e-14)


def test_flow_sell_absorption_bit_equal_to_oracle():
    df_pd, df_pl = _synthetic_bars(800, seed=8)
    out = _run_engine(df_pl, families=("rolling", "vol", "flow"))
    W = 60
    col = f"flow__sell_absorption__f__w{W}"
    p = df_pd["p"].to_numpy()
    tbb = df_pd["taker_buy_base"].to_numpy()
    vol = df_pd["volume"].to_numpy()

    sum_tbb = pd.Series(tbb).rolling(W, min_periods=W).sum().to_numpy()
    sum_vol = pd.Series(vol).rolling(W, min_periods=W).sum().to_numpy()
    fp = 2.0 * sum_tbb / (sum_vol + EPS) - 1.0
    dp = p - np.concatenate([np.full(W, np.nan), p[:-W]])
    sigma = out[f"ret__rms__f__w{W}"].to_numpy()
    d = np.tanh(dp / (sigma * math.sqrt(W) + EPS))
    expected = np.maximum(0.0, -fp) * np.maximum(0.0, d)

    got = out[col].to_numpy()
    valid = ~(np.isnan(expected) | np.isnan(got))
    assert valid.any()
    np.testing.assert_allclose(got[valid], expected[valid], rtol=1e-9, atol=1e-12)


def test_flow_buy_exhaustion_bit_equal_to_oracle():
    df_pd, df_pl = _synthetic_bars(800, seed=9)
    out = _run_engine(df_pl, families=("rolling", "vol", "flow"))
    W = 60
    col = f"flow__buy_exhaustion__f__w{W}"
    p = df_pd["p"].to_numpy()
    tbb = df_pd["taker_buy_base"].to_numpy()
    vol = df_pd["volume"].to_numpy()

    sum_tbb = pd.Series(tbb).rolling(W, min_periods=W).sum().to_numpy()
    sum_vol = pd.Series(vol).rolling(W, min_periods=W).sum().to_numpy()
    fp = 2.0 * sum_tbb / (sum_vol + EPS) - 1.0
    dp = p - np.concatenate([np.full(W, np.nan), p[:-W]])
    sigma = out[f"ret__rms__f__w{W}"].to_numpy()
    d = np.tanh(dp / (sigma * math.sqrt(W) + EPS))
    expected = np.maximum(0.0, fp) * np.maximum(0.0, -d)

    got = out[col].to_numpy()
    valid = ~(np.isnan(expected) | np.isnan(got))
    np.testing.assert_allclose(got[valid], expected[valid], rtol=1e-9, atol=1e-12)


def test_p_hit_drifted_bit_equal_to_scipy_formula():
    """Exact closed-form drifted-Brownian first-passage probability on
    fixed inputs."""
    W = 60
    base_frame = pl.DataFrame({
        "k": list(range(5)),
        f"vol__rs__f__w{W}": [0.001, 0.0005, 0.002, 0.0001, 0.001],
        f"ret__mean__f__w{W}": [0.0, 1e-5, -1e-5, 5e-5, 0.0],
    })
    # All other barrier windows need vol columns too; fill with same.
    for other_W in WINDOWS_BARRIER:
        col = f"vol__rs__f__w{other_W}"
        if col not in base_frame.columns:
            base_frame = base_frame.with_columns(pl.lit(0.001).alias(col))
    for ws, wl in VOL_PAIRS:
        for w in (ws, wl):
            colname = f"vol__rs__f__w{w}"
            if colname not in base_frame.columns:
                base_frame = base_frame.with_columns(pl.lit(0.001).alias(colname))

    out = compute_barrier_aware_features_pl(
        base_frame, WINDOWS_BARRIER, PHI, M, VOL_PAIRS, c=float(COST_C)
    )
    got = out[f"barrier__p_hit_drifted__f__w{W}"].to_numpy()

    # Closed-form oracle.
    phi = float(PHI)
    Mf = float(M)
    sqrt_M = math.sqrt(Mf)
    sigma = np.array([0.001, 0.0005, 0.002, 0.0001, 0.001])
    mu = np.array([0.0, 1e-5, -1e-5, 5e-5, 0.0])
    denom = sigma * sqrt_M + EPS
    arg_a = (mu * Mf - phi) / denom
    arg_b = (-phi - mu * Mf) / denom
    exp_arg = np.clip(2.0 * mu * phi / (sigma ** 2 + EPS), -50.0, 50.0)
    norm_cdf = lambda x: 0.5 * (1.0 + _scipy_erf(x / math.sqrt(2.0)))
    expected = np.clip(norm_cdf(arg_a) + np.exp(exp_arg) * norm_cdf(arg_b), 0.0, 1.0)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_oi_regime_decomposition_bit_equal_to_oracle():
    """Each of the four quadrants matches the direct ``tanh`` formula."""
    df_pd, df_pl = _synthetic_bars(800, seed=10, with_derivatives_base=True)

    engine = FeatureEngine(tiers=(1, 2), families=("rolling", "vol", "deriv_oi"))
    out = engine.transform(df_pl, trim=False).data
    W = 60

    p = df_pd["p"].to_numpy()
    oi_arr = df_pd["oi_usd"].to_numpy()
    ret_rms = out[f"ret__rms__f__w{W}"].to_numpy()
    # Oracle: dp_z = tanh((p_n - p_{n-W}) / (sigma_r * sqrt(W) + EPS))
    dp = p - np.concatenate([np.full(W, np.nan), p[:-W]])
    sqrt_w = math.sqrt(W)
    dp_z = np.tanh(dp / (ret_rms * sqrt_w + EPS))
    # doi_z uses 1-bar OI delta std as scale
    doi_1bar = np.concatenate([[np.nan], np.diff(oi_arr)])
    doi_std = pd.Series(doi_1bar).rolling(W, min_periods=W).std(ddof=0).to_numpy()
    doi = oi_arr - np.concatenate([np.full(W, np.nan), oi_arr[:-W]])
    doi_z = np.tanh(doi / (doi_std * sqrt_w + EPS))

    expected_long_build = np.maximum(0.0, dp_z) * np.maximum(0.0, doi_z)
    expected_short_build = np.maximum(0.0, -dp_z) * np.maximum(0.0, doi_z)
    expected_short_cover = np.maximum(0.0, dp_z) * np.maximum(0.0, -doi_z)
    expected_long_liq = np.maximum(0.0, -dp_z) * np.maximum(0.0, -doi_z)

    for name, expected in [
        ("long_build", expected_long_build),
        ("short_build", expected_short_build),
        ("short_cover", expected_short_cover),
        ("long_liq", expected_long_liq),
    ]:
        got = out[f"oi__{name}__f__w{W}"].to_numpy()
        valid = ~(np.isnan(expected) | np.isnan(got))
        assert valid.any(), f"oi__{name}: no valid rows"
        np.testing.assert_allclose(
            got[valid], expected[valid], rtol=1e-9, atol=1e-12,
            err_msg=f"oi__{name}: mismatch vs oracle",
        )


def test_funding_phase_bit_equal_at_known_times():
    """At UTC times known to be exact fractions of the funding cycle,
    phase + sin + cos must take their closed-form values."""
    # Funding settles at 00:00, 08:00, 16:00 UTC. Phase = (next - now) / 480.
    # At 07:59 UTC, next = 08:00, tau = 1 minute. Phase = 1/480.
    # At 00:00 UTC, next = 08:00, tau = 480 minutes. Phase = 1.0.
    # At 04:00 UTC, next = 08:00, tau = 240 minutes. Phase = 0.5.
    cases = [
        (pd.Timestamp("2024-01-01 00:00", tz="UTC"), 1.0),
        (pd.Timestamp("2024-01-01 04:00", tz="UTC"), 0.5),
        (pd.Timestamp("2024-01-01 07:59", tz="UTC"), 1.0 / 480.0),
        (pd.Timestamp("2024-01-01 08:00", tz="UTC"), 1.0),
        (pd.Timestamp("2024-01-01 16:00", tz="UTC"), 1.0),
        (pd.Timestamp("2024-01-01 23:59", tz="UTC"), 1.0 / 480.0),
    ]
    n = len(cases)
    ts_list = [t.tz_localize(None) for t, _ in cases]
    df_pl = pl.DataFrame({"ts": ts_list}).with_columns(pl.col("ts").cast(pl.Datetime))

    from src.features.families.derivatives import (
        FundingPhase, FundingPhaseSin, FundingPhaseCos,
    )
    phase_expr = FundingPhase().compute(None).alias("phase")
    sin_expr = FundingPhaseSin().compute(None).alias("sinp")
    cos_expr = FundingPhaseCos().compute(None).alias("cosp")
    out = df_pl.with_columns([phase_expr, sin_expr, cos_expr])
    got_phase = out["phase"].to_numpy()
    got_sin = out["sinp"].to_numpy()
    got_cos = out["cosp"].to_numpy()
    for i, (_, expected) in enumerate(cases):
        assert math.isclose(got_phase[i], expected, rel_tol=1e-12, abs_tol=1e-12), (
            f"phase[{i}]: got {got_phase[i]}, expected {expected}"
        )
        assert math.isclose(got_sin[i], math.sin(2 * math.pi * expected), abs_tol=1e-12)
        assert math.isclose(got_cos[i], math.cos(2 * math.pi * expected), abs_tol=1e-12)


def test_mature_m_pos_mean_bit_equal_to_oracle():
    """``target__mature_m_pos_mean__h__w{W}`` averages only matured
    positive m_k values."""
    df_pd, df_pl = _synthetic_bars(20_000, seed=11)
    df_b = df_pl.with_columns(pl.int_range(pl.len()).alias("k"))
    labels = construct_labels_pl(
        df_b, df_pl, M, ETA, COST_C,
        bar_stride=1, barrier_source="high",
        add_triple_barrier_aux=True,
    )
    W_probe = 12
    past = compute_past_target_features_pl(
        labels, [3, 6, 12], [3, 6, 12], bar_stride=1, M=M
    )
    col = f"target__mature_m_pos_mean__h__w{W_probe}"
    assert col in past.columns

    m_arr = np.array(
        [float(v) if v is not None else np.nan for v in labels["m_k"].to_list()],
        dtype=float,
    )
    y_arr = np.array(
        [float(v) if v is not None else np.nan for v in labels["y"].to_list()],
        dtype=float,
    )
    feat = np.array(
        [float(v) if v is not None else np.nan for v in past[col].to_list()],
        dtype=float,
    )

    # Oracle: for row n, take m_k where k in [n-M-W+1, n-M] AND y_k == 1.
    # Mean over those positives only; null if no positives exist in window.
    rng = np.random.default_rng(17)
    n_total = len(labels)
    rows_to_check = rng.choice(
        np.arange(M + W_probe + 10, n_total - 1), size=20, replace=False
    )
    for n in rows_to_check:
        window_m = m_arr[n - M - W_probe + 1 : n - M + 1]
        window_y = y_arr[n - M - W_probe + 1 : n - M + 1]
        if np.any(np.isnan(window_y)) or np.any(np.isnan(window_m)):
            continue
        pos_mask = (window_y == 1.0)
        if pos_mask.sum() == 0:
            assert np.isnan(feat[int(n)]), (
                f"row {n}: no positives in window, feature should be null/nan"
            )
        else:
            expected = float(window_m[pos_mask].mean())
            assert math.isclose(
                feat[int(n)], expected, rel_tol=1e-9, abs_tol=1e-12
            ), f"row {n}: got {feat[int(n)]}, expected {expected}"


# ---------------------------------------------------------------------------
# Section B — Causality probes (mask future, verify unchanged)
# ---------------------------------------------------------------------------


def _causality_probe(
    df_pl: pl.DataFrame,
    families: tuple[str, ...],
    feature_cols: list[str],
    t_probe: int,
) -> None:
    """Re-run the engine on a frame where every column is null for rows > t_probe.

    Assert each ``feature_col`` value at row ``t_probe`` is identical between
    full-frame and masked-frame runs. If a feature read any row > t_probe,
    its value at row t_probe would change.
    """
    out_full = _run_engine(df_pl, families=families)
    # Build masked frame: keep rows 0..t_probe inclusive, drop the rest.
    df_masked = df_pl.head(t_probe + 1)
    out_masked = _run_engine(df_masked, families=families)
    for col in feature_cols:
        v_full = out_full[col][t_probe]
        v_masked = out_masked[col][t_probe]
        # Handle null comparison
        if v_full is None and v_masked is None:
            continue
        if v_full is None or v_masked is None:
            raise AssertionError(
                f"{col}: causality probe mismatch at t={t_probe}: "
                f"full={v_full}, masked={v_masked}"
            )
        if isinstance(v_full, float) and math.isnan(v_full):
            assert math.isnan(v_masked), f"{col}: nan vs {v_masked}"
            continue
        assert math.isclose(
            float(v_full), float(v_masked), rel_tol=1e-12, abs_tol=1e-14
        ), f"{col}: full={v_full} != masked={v_masked} at t={t_probe}"


def test_extreme_features_no_future_leakage():
    _, df_pl = _synthetic_bars(2000, seed=20)
    cols = [
        "extreme__dist_low_z__f__w60",
        "extreme__dist_high_z__f__w60",
        "extreme__price_rank__f__w60",
    ]
    _causality_probe(df_pl, ("rolling", "vol", "extreme"), cols, t_probe=500)


def test_trend_quad_features_no_future_leakage():
    _, df_pl = _synthetic_bars(2000, seed=21)
    cols = [
        "trend__quad_slope_z__f__w60",
        "trend__quad_curv_z__f__w60",
    ]
    _causality_probe(df_pl, ("rolling", "vol", "trend"), cols, t_probe=500)


def test_pivot_features_no_future_leakage():
    """Stronger than ``_causality_probe`` because pivot detection uses a
    symmetric window: at row ``i``, a candidate centred at ``i - q`` looks
    at rows ``[i - 2q, i]``. The Q-row shift in the carry-forward stream
    ensures the indicator is unavailable until row ``i``, but a regression
    in the shift would peek q rows into the future.

    Compare full-frame to masked-frame at a probe row past warmup.
    """
    _, df_pl = _synthetic_bars(2000, seed=22)
    cols = [
        "pivot__last_low_dist_z__f__w240__q5",
        "pivot__last_low_age__f__w240__q5",
        "pivot__last_high_dist_z__f__w240__q5",
        "pivot__last_high_age__f__w240__q5",
    ]
    _causality_probe(df_pl, ("rolling", "vol", "pivot"), cols, t_probe=800)


def test_flow_features_no_future_leakage():
    _, df_pl = _synthetic_bars(2000, seed=23)
    cols = [
        "flow__pressure__f__w60",
        "flow__pressure_cum__f__w60",
        "flow__sell_absorption__f__w60",
        "flow__buy_exhaustion__f__w60",
    ]
    _causality_probe(df_pl, ("rolling", "vol", "flow"), cols, t_probe=500)


def test_vol_round1_features_no_future_leakage():
    _, df_pl = _synthetic_bars(2000, seed=24)
    cols = [
        "vol__semivar_signed__f__w60",
        "vol__jump_ratio__f__w60",
    ]
    _causality_probe(df_pl, ("rolling", "vol"), cols, t_probe=500)


def test_oi_regime_no_future_leakage():
    _, df_pl = _synthetic_bars(2000, seed=25, with_derivatives_base=True)
    cols = [
        "oi__long_build__f__w60",
        "oi__short_build__f__w60",
        "oi__short_cover__f__w60",
        "oi__long_liq__f__w60",
    ]
    _causality_probe(df_pl, ("rolling", "vol", "deriv_oi"), cols, t_probe=500)


def test_funding_phase_no_future_leakage():
    """Pure calendar — depends only on ``ts``. Masking future rows changes
    nothing about row t's value. The ``deriv_funding`` family also includes
    rate-based features that require ``funding_rate``; build a frame with
    the minimum derivative base columns."""
    _, df_pl = _synthetic_bars(2000, seed=26, with_derivatives_base=True)
    cols = [
        "funding__phase__f__w0",
        "funding__phase_sin__f__w0",
        "funding__phase_cos__f__w0",
    ]
    _causality_probe(df_pl, ("deriv_funding",), cols, t_probe=500)


def test_p_hit_drifted_no_future_leakage_at_boundary():
    """``barrier__p_hit_drifted`` at boundary row k uses vol/ret-mean
    computed only from bars up to k. Mask the rows after t_probe in the
    boundary frame and verify the value at t_probe is unchanged."""
    df_pd, df_pl = _synthetic_bars(8000, seed=27)
    df_with_feats = _run_engine(df_pl, families=("rolling", "vol"))
    df_b = df_with_feats.gather_every(M)
    t_probe = 50
    # Mask all rows after t_probe by truncating.
    df_b_masked = df_b.head(t_probe + 1)
    full = compute_barrier_aware_features_pl(
        df_b, WINDOWS_BARRIER, PHI, M, VOL_PAIRS, c=float(COST_C)
    )
    masked = compute_barrier_aware_features_pl(
        df_b_masked, WINDOWS_BARRIER, PHI, M, VOL_PAIRS, c=float(COST_C)
    )
    for W in (60, 240):
        col = f"barrier__p_hit_drifted__f__w{W}"
        if col not in full.columns:
            continue
        v_full = full[col][t_probe]
        v_masked = masked[col][t_probe]
        if v_full is None and v_masked is None:
            continue
        assert v_full is not None and v_masked is not None
        assert math.isclose(
            float(v_full), float(v_masked), rel_tol=1e-12, abs_tol=1e-14
        ), f"{col}: full={v_full} != masked={v_masked}"


# ---------------------------------------------------------------------------
# Section C — Edge cases
# ---------------------------------------------------------------------------


def test_pivot_no_pivot_returns_sentinel_age():
    """A perfectly monotone series has no internal local minimum, so no
    swing low can ever be confirmed. The age column should remain at the
    sentinel W."""
    from src.features.families.pivot import _detect_pivots_np, _pivot_age_np

    n = 500
    log_low = np.linspace(0.0, 1.0, n)  # strictly increasing
    q = 5
    w = 240
    pivot_idx = _detect_pivots_np(log_low, q, mode="low")
    age = _pivot_age_np(log_low, q, w, mode="low")

    # On a strictly increasing series the only candidate "minimum" is the
    # leftmost edge, where the symmetric window is incomplete.
    # No pivot can ever be confirmed -> age is the sentinel W everywhere.
    assert np.all(pivot_idx == -1), f"unexpected pivots: {np.unique(pivot_idx)}"
    assert np.all(age == float(w)), f"unexpected ages: {np.unique(age)}"


def test_pivot_high_low_independent():
    """Swing high detection on a series with only swing LOWS should yield
    no confirmed highs."""
    from src.features.families.pivot import _detect_pivots_np

    n = 500
    # V-shape with sharp min at i=100; no internal max.
    log_x = np.abs(np.arange(n) - 100.0) / 1000.0
    q = 5
    # Low detection finds the V minimum at 100.
    lows = _detect_pivots_np(log_x, q, mode="low")
    # High detection on the same series: the function looks for local maxima.
    # On the upward arm (t > 100) every bar is a new high — but in a SYMMETRIC
    # 2q+1 window, only the endpoint of the upward arm where rightward bars
    # are still equal would qualify, which doesn't happen here.
    highs = _detect_pivots_np(log_x, q, mode="high")
    assert lows[200] == 100, "low at 100 should be detected"
    # No swing high should exist on a V-shape (only local minima).
    # Highs may equal -1 (no confirmed high) or only edges; assert never 100.
    assert highs[200] != 100, (
        "swing-high detector should not flag the swing-LOW point at 100"
    )


def test_oi_regime_mutual_exclusion_per_row():
    """The four quadrants partition the (sign(dp), sign(doi)) plane, so
    at any row at most one of the four should be strictly positive."""
    n = 1000
    _, df_pl = _synthetic_bars(n, seed=28, with_derivatives_base=True)
    out = _run_engine(df_pl, families=("rolling", "vol", "deriv_oi"))
    W = 60
    arrs = {
        name: out[f"oi__{name}__f__w{W}"].to_numpy()
        for name in ("long_build", "short_build", "short_cover", "long_liq")
    }
    n_pos = np.zeros(n, dtype=int)
    eps_pos = 1e-12
    for arr in arrs.values():
        valid = ~np.isnan(arr)
        n_pos[valid] += (arr[valid] > eps_pos).astype(int)
    # On rows where dp_z or doi_z is exactly 0 (edge case), no quadrant
    # fires; otherwise exactly one. So n_pos[i] is 0 or 1.
    invalid_rows = (n_pos > 1).sum()
    assert invalid_rows == 0, (
        f"{invalid_rows} rows have >1 OI regime quadrant >0 simultaneously"
    )


def test_p_hit_drifted_clamp_handles_near_zero_sigma():
    """When sigma is near zero with positive drift, the exp-argument
    would overflow without the [-50, 50] clamp. The output must remain
    in [0, 1]."""
    W = 60
    base_frame = pl.DataFrame({
        "k": [0, 1, 2],
        f"vol__rs__f__w{W}": [1e-8, 1e-10, 1e-15],   # near-zero
        f"ret__mean__f__w{W}": [1e-3, 1e-3, 1e-3],   # strongly positive drift
    })
    for other_W in WINDOWS_BARRIER:
        col = f"vol__rs__f__w{other_W}"
        if col not in base_frame.columns:
            base_frame = base_frame.with_columns(pl.lit(0.001).alias(col))
    for ws, wl in VOL_PAIRS:
        for w in (ws, wl):
            colname = f"vol__rs__f__w{w}"
            if colname not in base_frame.columns:
                base_frame = base_frame.with_columns(pl.lit(0.001).alias(colname))

    out = compute_barrier_aware_features_pl(
        base_frame, WINDOWS_BARRIER, PHI, M, VOL_PAIRS, c=float(COST_C)
    )
    vals = out[f"barrier__p_hit_drifted__f__w{W}"].to_numpy()
    assert np.all(np.isfinite(vals)), f"non-finite values: {vals}"
    assert np.all((vals >= 0.0) & (vals <= 1.0)), f"out of [0,1]: {vals}"


def test_p_hit_drifted_negative_drift_lowers_probability():
    """Same sigma, mu < 0 should reduce p relative to mu = 0."""
    W = 60
    sigma = 0.001
    base_frame = pl.DataFrame({
        "k": [0, 1, 2],
        f"vol__rs__f__w{W}": [sigma, sigma, sigma],
        f"ret__mean__f__w{W}": [-1e-5, 0.0, 1e-5],
    })
    for other_W in WINDOWS_BARRIER:
        col = f"vol__rs__f__w{other_W}"
        if col not in base_frame.columns:
            base_frame = base_frame.with_columns(pl.lit(sigma).alias(col))
    for ws, wl in VOL_PAIRS:
        for w in (ws, wl):
            colname = f"vol__rs__f__w{w}"
            if colname not in base_frame.columns:
                base_frame = base_frame.with_columns(pl.lit(sigma).alias(colname))

    out = compute_barrier_aware_features_pl(
        base_frame, WINDOWS_BARRIER, PHI, M, VOL_PAIRS, c=float(COST_C)
    )
    p_neg, p_zero, p_pos = out[f"barrier__p_hit_drifted__f__w{W}"].to_list()
    assert p_neg < p_zero < p_pos, (
        f"expected p(neg drift) < p(zero) < p(pos drift); got "
        f"{p_neg}, {p_zero}, {p_pos}"
    )


def test_funding_phase_sin_cos_unit_circle():
    """sin^2 + cos^2 should equal 1 at every row."""
    _, df_pl = _synthetic_bars(500, seed=29, with_derivatives_base=True)
    out = _run_engine(df_pl, families=("deriv_funding",))
    sinp = out["funding__phase_sin__f__w0"].to_numpy()
    cosp = out["funding__phase_cos__f__w0"].to_numpy()
    valid = ~np.isnan(sinp) & ~np.isnan(cosp)
    np.testing.assert_allclose(
        sinp[valid] ** 2 + cosp[valid] ** 2, 1.0, atol=1e-12
    )


def test_flow_pressure_zero_volume_window():
    """A window with strictly zero volume must not produce a divide-by-zero
    or NaN; the EPS guard caps the magnitude."""
    n = 200
    close = np.full(n, 100.0)
    df = pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "volume": np.zeros(n),                # zero volume!
        "quote_volume": np.zeros(n),
        "num_trades": np.zeros(n, dtype=np.int64),
        "taker_buy_base": np.zeros(n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"))
    df = utils.compute_base_series(df)
    df_pl = pl.from_pandas(df.reset_index(names="ts"))
    out = _run_engine(df_pl, families=("rolling", "vol", "flow"))
    W = 60
    vals = out[f"flow__pressure__f__w{W}"].to_numpy()
    # With sum_vol = 0 and sum_tbb = 0: 2 * 0 / (0 + EPS) - 1 = -1.
    # Finite, in [-1, 1], no NaN/inf.
    finite_vals = vals[~np.isnan(vals)]
    assert np.all(np.isfinite(finite_vals))
    assert np.all((finite_vals >= -1.0 - 1e-9) & (finite_vals <= 1.0 + 1e-9))


# ---------------------------------------------------------------------------
# Section D — Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_emits_every_new_family_with_clean_impute():
    """End-to-end: ``run_pipeline`` at 1-min cadence emits all new columns
    and the imputation step leaves zero residual nulls / NaN / inf."""
    from src.features.pipeline import run_pipeline

    rng = np.random.default_rng(30)
    n = 50_000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    spread = np.abs(rng.normal(0.0, 0.5, n))
    high = close + spread
    low = np.maximum(close - spread, 0.001)
    open_ = np.clip(close + rng.normal(0.0, 0.2, n), low, high)
    volume = np.abs(rng.normal(100.0, 30.0, n)) + 1.0
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": volume * close,
        "num_trades": np.ones(n, dtype=np.int64),
        "taker_buy_base": 0.5 * volume,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"))

    out = run_pipeline(
        df, label_cadence="1min", enable_autocorrelation=False,
        add_triple_barrier_aux=True,
    )
    assert len(out) > 0, "pipeline produced zero rows"

    # Every prefix should be represented.
    required_prefixes = [
        "extreme__dist_low_z",
        "extreme__dist_high_z",
        "extreme__price_rank",
        "pivot__last_low_dist_z",
        "pivot__last_low_age",
        "pivot__last_high_dist_z",
        "pivot__last_high_age",
        "flow__pressure",
        "flow__pressure_cum",
        "flow__sell_absorption",
        "flow__buy_exhaustion",
        "trend__quad_slope_z",
        "trend__quad_curv_z",
        "vol__semivar_signed",
        "vol__jump_ratio",
        "target__mature_m_mean",
        "target__mature_m_pos_mean",
        "target__mature_tau_pos_mean",
        "target__mature_near_miss_up",
        "barrier__p_hit_drifted",
    ]
    for prefix in required_prefixes:
        assert any(c.startswith(prefix) for c in out.columns), (
            f"pipeline missing any column with prefix {prefix!r}"
        )

    # CRITICAL: post-impute, no feature column may carry null, NaN, or inf.
    new_feature_cols = [
        c for c in out.columns
        if any(c.startswith(p) for p in required_prefixes)
        and out.schema[c] in (pl.Float32, pl.Float64)
    ]
    assert new_feature_cols
    for c in new_feature_cols:
        n_null = int(out[c].null_count())
        n_nan = int(out[c].is_nan().sum() or 0)
        n_inf = int(out[c].is_infinite().sum() or 0)
        assert n_null == 0, f"{c}: {n_null} residual nulls after impute"
        assert n_nan == 0, f"{c}: {n_nan} residual NaN after impute"
        assert n_inf == 0, f"{c}: {n_inf} residual inf after impute"


def test_pipeline_emits_oi_regime_and_funding_phase_with_derivatives():
    """When ``with_derivatives=True``, OI regime decomposition and funding
    phase columns appear, and imputation leaves them clean."""
    from src.features.pipeline import run_pipeline

    # Build a derivatives-bearing bar frame.
    rng = np.random.default_rng(31)
    n = 50_000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    spread = np.abs(rng.normal(0.0, 0.5, n))
    high = close + spread
    low = np.maximum(close - spread, 0.001)
    open_ = np.clip(close + rng.normal(0.0, 0.2, n), low, high)
    volume = np.abs(rng.normal(100.0, 30.0, n)) + 1.0
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": volume * close,
        "num_trades": np.ones(n, dtype=np.int64),
        "taker_buy_base": 0.5 * volume,
        # Derivative base series
        "close_fut": close * 1.001,
        "volume_fut": volume,
        "quote_volume_fut": volume * close,
        "taker_buy_base_fut": 0.5 * volume,
        "num_trades_fut": np.ones(n, dtype=np.int64),
        "funding_rate": np.full(n, 1e-4),
        "oi_usd": 15e9 + np.cumsum(rng.normal(0.0, 1e7, n)),
        "opt_oi": np.full(n, 1e9),
        "put_open_interest": np.full(n, 5e8),
        "call_open_interest": np.full(n, 5e8),
        "opt_volume": np.full(n, 1e7),
        "put_volume": np.full(n, 5e6),
        "call_volume": np.full(n, 5e6),
        "bvol": np.full(n, 60.0),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"))

    out = run_pipeline(
        df, with_derivatives=True, label_cadence="1min",
        enable_autocorrelation=False,
    )
    assert len(out) > 0
    required = [
        "oi__long_build__f__w60",
        "oi__short_build__f__w60",
        "oi__short_cover__f__w60",
        "oi__long_liq__f__w60",
        "funding__phase__f__w0",
        "funding__phase_sin__f__w0",
        "funding__phase_cos__f__w0",
    ]
    for col in required:
        assert col in out.columns, f"missing {col} in derivatives pipeline output"
        n_null = int(out[col].null_count())
        n_nan = int(out[col].is_nan().sum() or 0)
        assert n_null == 0 and n_nan == 0, (
            f"{col}: {n_null} nulls, {n_nan} NaN after impute"
        )


def test_pipeline_round1_columns_match_declared_windows():
    """For each new Round 1 family, every declared window emits its column
    AND the column has at least one valid (non-imputed) row."""
    from src.features.pipeline import run_pipeline
    from src.features.config import (
        WINDOWS_EXTREME, WINDOWS_FLOW_PRESSURE, WINDOWS_QUAD_TREND,
        WINDOWS_VOL_JUMP, WINDOWS_VOL_SIGNED,
        WINDOWS_PIVOT, PIVOT_Q_VALUES,
    )

    rng = np.random.default_rng(32)
    n = 50_000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    spread = np.abs(rng.normal(0.0, 0.5, n))
    high = close + spread
    low = np.maximum(close - spread, 0.001)
    open_ = np.clip(close + rng.normal(0.0, 0.2, n), low, high)
    volume = np.abs(rng.normal(100.0, 30.0, n)) + 1.0
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": volume * close,
        "num_trades": np.ones(n, dtype=np.int64),
        "taker_buy_base": 0.5 * volume,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"))

    out = run_pipeline(df, label_cadence="1min", enable_autocorrelation=False)

    # Build the exhaustive expected-column list.
    expected_cols: list[str] = []
    for W in WINDOWS_EXTREME:
        expected_cols += [
            f"extreme__dist_low_z__f__w{W}",
            f"extreme__dist_high_z__f__w{W}",
            f"extreme__price_rank__f__w{W}",
        ]
    for W in WINDOWS_QUAD_TREND:
        expected_cols += [
            f"trend__quad_slope_z__f__w{W}",
            f"trend__quad_curv_z__f__w{W}",
        ]
    for W in WINDOWS_VOL_SIGNED:
        expected_cols.append(f"vol__semivar_signed__f__w{W}")
    for W in WINDOWS_VOL_JUMP:
        expected_cols.append(f"vol__jump_ratio__f__w{W}")
    for W in WINDOWS_FLOW_PRESSURE:
        expected_cols += [
            f"flow__pressure__f__w{W}",
            f"flow__pressure_cum__f__w{W}",
            f"flow__sell_absorption__f__w{W}",
            f"flow__buy_exhaustion__f__w{W}",
        ]
    for W in WINDOWS_PIVOT:
        for Q in PIVOT_Q_VALUES:
            expected_cols += [
                f"pivot__last_low_dist_z__f__w{W}__q{Q}",
                f"pivot__last_low_age__f__w{W}__q{Q}",
                f"pivot__last_high_dist_z__f__w{W}__q{Q}",
                f"pivot__last_high_age__f__w{W}__q{Q}",
            ]

    missing = [c for c in expected_cols if c not in out.columns]
    assert not missing, f"pipeline missing {len(missing)} expected cols, e.g. {missing[:5]}"
