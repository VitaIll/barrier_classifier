"""Boundary-stage transformations.

Operate on ``df_boundaries`` (the every-M sample of the bar dataframe)
rather than on the full bars. Plain functions, not Feature classes,
because the boundary-stage shape (different df, sometimes needing both
boundary and raw) does not fit the row-aligned with_columns model.

Mirrors:
  - utils.construct_labels                    -> construct_labels_pl
  - utils.compute_past_target_features        -> compute_past_target_features_pl
  - utils.compute_barrier_aware_features      -> compute_barrier_aware_features_pl
  - utils.compute_block_features              -> compute_block_features_pl
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl

from src.features.config import C, EPS
from src.features.primitives import rolling_mean


# =============================================================================
# Labels
# =============================================================================


def construct_labels_pl(
    df_boundaries: pl.DataFrame,
    df_raw: pl.DataFrame,
    M: int,
    eta: float,
    c: float,
) -> pl.DataFrame:
    """Add ``y, m_k, tau_k, phi`` columns to df_boundaries.

    Mirrors utils.construct_labels (utils.py:2416-2457).
    """
    close = df_raw["close"].to_numpy().astype(float)
    n_total = len(close)
    K = len(df_boundaries)
    phi = float(c + eta)

    y = np.full(K, np.nan)
    m_k = np.full(K, np.nan)
    tau_k = np.full(K, np.nan)

    for k in range(K):
        n_k = k * M
        if n_k + M >= n_total:
            continue
        base = close[n_k]
        future = close[n_k + 1 : n_k + M + 1]
        future_ret = np.log(future / base)
        m_val = float(np.max(future_ret))
        m_k[k] = m_val
        hit = m_val >= phi
        y[k] = 1.0 if hit else 0.0
        if hit:
            tau_k[k] = float(np.argmax(future_ret >= phi) + 1)

    # Convert NaN -> null so downstream is_not_null()/notna() filters work
    # uniformly. Pandas notna catches both; polars is_not_null only catches
    # null. Coerce here so the boundary df has a consistent "missing" type.
    return df_boundaries.with_columns(
        [
            pl.Series("y", y).fill_nan(None),
            pl.Series("m_k", m_k).fill_nan(None),
            pl.Series("tau_k", tau_k).fill_nan(None),
            pl.lit(phi).alias("phi"),
        ]
    )


# =============================================================================
# Past-target features (post-label)
# =============================================================================


def compute_past_target_features_pl(
    df_boundaries: pl.DataFrame,
    windows_h: Iterable[int],
    hitrate_windows_h: Iterable[int],
) -> pl.DataFrame:
    """Add hit__prev, hit__rate, hit__since columns.

    Mirrors utils.compute_past_target_features (utils.py:2460-2482).
    Note: ``windows_h`` parameter is accepted for legacy signature parity
    but only ``hitrate_windows_h`` is actually used.
    """
    new_cols = [pl.col("y").shift(1).alias("hit__prev__h__w0")]

    y_shift = pl.col("y").shift(1)
    for W in hitrate_windows_h:
        new_cols.append(rolling_mean(y_shift, W).alias(f"hit__rate__h__w{W}"))

    # hit_k = k where y == 1 else null; ffill last hit forward
    hit_k = pl.when(pl.col("y") == 1).then(pl.col("k")).otherwise(None)
    last_hit_before = hit_k.shift(1).forward_fill()
    new_cols.append((pl.col("k") - last_hit_before).alias("hit__since__h__w0"))

    return df_boundaries.with_columns(new_cols)


# =============================================================================
# Barrier-aware features
# =============================================================================


def compute_barrier_aware_features_pl(
    df_boundaries: pl.DataFrame,
    windows_barrier: Iterable[int],
    phi: float,
    M: int,
    vol_pairs: Iterable[tuple[int, int]],
    c: float | None = None,
) -> pl.DataFrame:
    """Add barrier-aware features. Mirrors utils.compute_barrier_aware_features."""
    if c is None:
        c = float(C)

    new_cols: list[pl.Expr] = []
    sqrt_M = math.sqrt(M)
    sqrt_2logM = math.sqrt(2.0 * math.log(M))

    for W in windows_barrier:
        col = f"vol__rs__f__w{W}"
        if col not in df_boundaries.columns:
            raise ValueError(
                f"compute_barrier_aware_features_pl requires '{col}' at boundaries"
            )
        vol = pl.col(col)
        new_cols.append((phi / (vol * sqrt_M + EPS)).alias(f"barrier__z_tight__f__w{W}"))
        new_cols.append(
            ((vol * sqrt_2logM) / (phi + EPS)).alias(f"barrier__emax_ratio__f__w{W}")
        )

    for ws, wl in vol_pairs:
        col_s = f"vol__rs__f__w{ws}"
        col_l = f"vol__rs__f__w{wl}"
        if col_s not in df_boundaries.columns or col_l not in df_boundaries.columns:
            raise ValueError(
                f"compute_barrier_aware_features_pl requires {col_s} and {col_l}"
            )
        new_cols.append(
            (pl.col(col_s) / (pl.col(col_l) + EPS)).alias(
                f"vol__ratio__f__ws{ws}__wl{wl}"
            )
        )

    new_cols.append(pl.lit(float(c)).alias("cost__c__h__w0"))
    new_cols.append(pl.lit(float(phi)).alias("barrier__phi__h__w0"))

    return df_boundaries.with_columns(new_cols)


# =============================================================================
# Block features (look back into raw bars per boundary)
# =============================================================================


def compute_block_features_pl(
    df_boundaries: pl.DataFrame,
    df_raw: pl.DataFrame,
    M: int,
    windows_h: Iterable[int],
) -> pl.DataFrame:
    """Add block-aggregated features. Mirrors utils.compute_block_features.

    Block 0 = the single bar at index 0; block k (k>=1) = bars
    [(k-1)·M + 1 .. k·M]. Block aggregates feed downstream features.
    """
    K = int(len(df_boundaries))
    n_max = (K - 1) * M
    if n_max >= len(df_raw):
        raise ValueError(
            "compute_block_features_pl: boundary count implies n_max beyond raw data length"
        )

    close = df_raw["close"].to_numpy().astype(float)
    p = np.log(close)
    high = df_raw["high"].to_numpy().astype(float)
    low = df_raw["low"].to_numpy().astype(float)
    volume = df_raw["volume"].to_numpy().astype(float)
    quote_volume = df_raw["quote_volume"].to_numpy().astype(float)
    num_trades = df_raw["num_trades"].to_numpy().astype(float)
    taker_buy_base = df_raw["taker_buy_base"].to_numpy().astype(float)

    H = np.full(K, np.nan, dtype=float)
    L = np.full(K, np.nan, dtype=float)
    V = np.full(K, np.nan, dtype=float)
    Q = np.full(K, np.nan, dtype=float)
    Ntr = np.full(K, np.nan, dtype=float)
    VTB = np.full(K, np.nan, dtype=float)

    H[0] = high[0]
    L[0] = low[0]
    V[0] = volume[0]
    Q[0] = quote_volume[0]
    Ntr[0] = num_trades[0]
    VTB[0] = taker_buy_base[0]

    if K > 1:
        sl = slice(1, n_max + 1)
        H[1:] = np.max(high[sl].reshape(K - 1, M), axis=1)
        L[1:] = np.min(low[sl].reshape(K - 1, M), axis=1)
        V[1:] = np.sum(volume[sl].reshape(K - 1, M), axis=1)
        Q[1:] = np.sum(quote_volume[sl].reshape(K - 1, M), axis=1)
        Ntr[1:] = np.sum(num_trades[sl].reshape(K - 1, M), axis=1)
        VTB[1:] = np.sum(taker_buy_base[sl].reshape(K - 1, M), axis=1)

    close_boundary = close[::M][:K]
    p_boundary = p[::M][:K]

    ret_inst = np.full(K, np.nan, dtype=float)
    ret_inst[1:] = np.log(close_boundary[1:] / close_boundary[:-1])

    range_inst = np.where(H > L, np.log(H / L), np.nan)
    logvol_inst = np.log1p(V)
    ofi_inst = np.where(V > 0, 2.0 * (VTB / V) - 1.0, np.nan)

    # ret__std__h__w{W} via pandas rolling (ddof=0) for legacy parity
    ret_inst_s = pd.Series(ret_inst)
    ret_std_cols: dict[str, np.ndarray] = {}
    for W in windows_h:
        ret_std_cols[f"ret__std__h__w{W}"] = (
            ret_inst_s.rolling(W, min_periods=W).std(ddof=0).to_numpy()
        )

    block_maxret = np.full(K, np.nan, dtype=float)
    block_minret = np.full(K, np.nan, dtype=float)
    if K > 1:
        p_blocks = p[1 : n_max + 1].reshape(K - 1, M)
        p_prev = p_boundary[:-1].reshape(K - 1, 1)
        diffs = p_blocks - p_prev
        block_maxret[1:] = np.max(diffs, axis=1)
        block_minret[1:] = np.min(diffs, axis=1)

    denom_hl = np.log(H) - np.log(L)
    close_to_high = (p_boundary - np.log(L)) / (denom_hl + EPS)
    close_to_high = np.where(denom_hl != 0.0, close_to_high, np.nan)

    new_cols = [
        pl.Series("ret__inst__h__w0", ret_inst),
        pl.Series("range__inst__h__w0", range_inst),
        pl.Series("logvol__inst__h__w0", logvol_inst),
        pl.Series("ofi__inst__h__w0", ofi_inst),
        pl.Series("block__maxret__h__w0", block_maxret),
        pl.Series("block__minret__h__w0", block_minret),
        pl.Series("block__close_to_high__h__w0", close_to_high),
    ]
    for col_name, vals in ret_std_cols.items():
        new_cols.append(pl.Series(col_name, vals))

    return df_boundaries.with_columns(new_cols)
