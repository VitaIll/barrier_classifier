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

**Label cadence (``bar_stride``)**

All four functions accept a ``bar_stride`` parameter that defaults to ``M``
(the legacy boundary cadence, one label every M bars). Setting
``bar_stride=1`` switches to **base-frequency cadence**: a label is
generated at every 1-min bar, using that bar as the entry reference and
looking M bars forward for the +φ barrier hit. This makes adjacent
labels heavily autocorrelated (their prediction windows overlap by
M-1 bars). The companion ``compute_past_target_autocorrelation_pl``
surfaces that autocorrelation as a feature.

Causality contract: when ``bar_stride=1``, the most recent *mature*
label at row n is ``y_{n-M}`` (its full prediction window ends at bar n).
The past-target features shift by ``M // bar_stride`` rows to ensure
features never use a label whose horizon is still open.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl

from src.features.config import C, EPS
from src.features.primitives import population_corr, rolling_mean


# =============================================================================
# Labels
# =============================================================================


def construct_labels_pl(
    df_boundaries: pl.DataFrame,
    df_raw: pl.DataFrame,
    M: int,
    eta: float,
    c: float,
    *,
    bar_stride: int | None = None,
    barrier_source: str = "close",
    add_triple_barrier_aux: bool = False,
) -> pl.DataFrame:
    """Add ``y, m_k, tau_k, phi`` columns to df_boundaries.

    Mirrors utils.construct_labels (utils.py:2416-2457).

    Parameters
    ----------
    df_boundaries : pl.DataFrame
        Each row is an entry-reference row; the function assigns a label
        to each row that looks ``M`` bars forward in ``df_raw`` from the
        bar at index ``k * bar_stride``.
    bar_stride : int, optional
        Distance in raw bars between consecutive rows of ``df_boundaries``.
        Defaults to ``M`` (legacy boundary cadence). Set to ``1`` to
        generate a label at every 1-min bar.
    barrier_source : {"close", "high"}, default ``"close"``
        Which future price series to test against the upper barrier.
        ``"close"`` is the legacy close-confirmed definition (parity with
        ``utils.construct_labels``). ``"high"`` aligns the label with a
        TP-limit-order strategy that fills when the intrabar high crosses
        ``close[n_k] * exp(+phi)`` — this is what the simulator's
        ``exit_tp_or_expiry`` checks for a long position, so high-based
        labels train the same event the strategy trades. When
        ``barrier_source="high"``, ``m_k`` is the max log return of future
        highs over ``close[n_k]`` and ``tau_k`` is the first future bar
        whose high crosses the barrier. df_raw must carry a ``"high"``
        column for this mode.
    add_triple_barrier_aux : bool, default False
        If True, also emits ``m_dn`` (max negative excursion, i.e.
        ``-min(log(low[n+1..n+M] / close[n_k]))``) and ``tau_dn`` (first j
        where ``log(low[n+j] / close[n_k]) <= -phi``). These are
        downside-barrier diagnostics for the López de Prado triple-barrier
        family; they are not used by the long-only TP label but are useful
        for SL / first-touch analytics. Requires a ``"low"`` column when
        ``barrier_source="high"`` (highs already imply OHLC presence).
    """
    if bar_stride is None:
        bar_stride = int(M)
    if bar_stride <= 0:
        raise ValueError(f"bar_stride must be > 0, got {bar_stride}")
    if barrier_source not in ("close", "high"):
        raise ValueError(
            f"barrier_source must be 'close' or 'high', got {barrier_source!r}"
        )

    close = df_raw["close"].to_numpy().astype(float)
    n_total = len(close)
    K = len(df_boundaries)
    phi = float(c + eta)

    if barrier_source == "high":
        if "high" not in df_raw.columns:
            raise ValueError("barrier_source='high' requires a 'high' column on df_raw")
        high = df_raw["high"].to_numpy().astype(float)
        upper_series = high
    else:
        upper_series = close

    add_dn = bool(add_triple_barrier_aux)
    if add_dn:
        low_col = df_raw["low"] if "low" in df_raw.columns else None
        if low_col is None:
            raise ValueError(
                "add_triple_barrier_aux=True requires a 'low' column on df_raw"
            )
        low = low_col.to_numpy().astype(float)

    y = np.full(K, np.nan)
    m_k = np.full(K, np.nan)
    tau_k = np.full(K, np.nan)
    m_dn = np.full(K, np.nan) if add_dn else None
    tau_dn = np.full(K, np.nan) if add_dn else None

    for k in range(K):
        n_k = k * bar_stride
        if n_k + M >= n_total:
            continue
        base = close[n_k]
        future_up = upper_series[n_k + 1 : n_k + M + 1]
        future_up_ret = np.log(future_up / base)
        m_val = float(np.max(future_up_ret))
        m_k[k] = m_val
        hit = m_val >= phi
        y[k] = 1.0 if hit else 0.0
        if hit:
            tau_k[k] = float(np.argmax(future_up_ret >= phi) + 1)
        if add_dn:
            future_dn = low[n_k + 1 : n_k + M + 1]
            future_dn_ret = np.log(future_dn / base)
            m_dn[k] = float(-np.min(future_dn_ret))
            dn_hit = future_dn_ret <= -phi
            if dn_hit.any():
                tau_dn[k] = float(np.argmax(dn_hit) + 1)

    # Convert NaN -> null so downstream is_not_null()/notna() filters work
    # uniformly. Pandas notna catches both; polars is_not_null only catches
    # null. Coerce here so the boundary df has a consistent "missing" type.
    new_cols = [
        pl.Series("y", y).fill_nan(None),
        pl.Series("m_k", m_k).fill_nan(None),
        pl.Series("tau_k", tau_k).fill_nan(None),
        pl.lit(phi).alias("phi"),
    ]
    if add_dn:
        new_cols.append(pl.Series("m_dn", m_dn).fill_nan(None))
        new_cols.append(pl.Series("tau_dn", tau_dn).fill_nan(None))
    return df_boundaries.with_columns(new_cols)


# =============================================================================
# Past-target features (post-label)
# =============================================================================


def _label_maturity_shift(bar_stride: int | None, M: int) -> int:
    """Number of df rows to shift to land on a mature label.

    At time ``t`` (row index in the ``df``), label ``y_t`` has its
    prediction window over bars ``[n_t+1, n_t+M]`` where ``n_t = t*bar_stride``.
    Label maturity requires the full future window to have elapsed —
    i.e. ``n_t + M <= n_now``. At row ``t_now``, that gives
    ``t <= t_now − M//bar_stride``. So a shift of ``M // bar_stride``
    rows is the smallest causal shift.

    For legacy boundary cadence (``bar_stride = M``), this is 1 (one
    boundary back). For 1-min cadence (``bar_stride = 1``), this is M.
    """
    if bar_stride is None or bar_stride <= 0:
        return 1
    return max(1, int(M) // int(bar_stride))


def compute_past_target_features_pl(
    df_boundaries: pl.DataFrame,
    windows_h: Iterable[int],
    hitrate_windows_h: Iterable[int],
    *,
    bar_stride: int | None = None,
    M: int | None = None,
) -> pl.DataFrame:
    """Add hit__prev, hit__rate, hit__since columns.

    Mirrors utils.compute_past_target_features (utils.py:2460-2482).

    The label-maturity shift is parameterized so the function is causal
    under both boundary cadence (default; shift = 1 row) and 1-min cadence
    (shift = M rows). ``windows_h`` is accepted for legacy signature parity
    but only ``hitrate_windows_h`` is actually used.
    """
    shift_units = _label_maturity_shift(bar_stride, M if M is not None else 1)

    new_cols = [pl.col("y").shift(shift_units).alias("hit__prev__h__w0")]

    y_shift = pl.col("y").shift(shift_units)
    for W in hitrate_windows_h:
        new_cols.append(rolling_mean(y_shift, W).alias(f"hit__rate__h__w{W}"))

    # hit_k = k where y == 1 else null; ffill last hit forward, but only
    # consider hits whose label is mature (shifted by shift_units).
    hit_k = pl.when(pl.col("y") == 1).then(pl.col("k")).otherwise(None)
    last_hit_before = hit_k.shift(shift_units).forward_fill()
    new_cols.append((pl.col("k") - last_hit_before).alias("hit__since__h__w0"))

    return df_boundaries.with_columns(new_cols)


def compute_past_target_autocorrelation_pl(
    df_boundaries: pl.DataFrame,
    windows: Iterable[int],
    *,
    bar_stride: int | None = None,
    M: int | None = None,
    lags: Iterable[int] = (1, 2, 5, 10),
) -> pl.DataFrame:
    """Rolling autocorrelation of past mature labels — strictly causal.

    When the label-generation cadence is 1-min (``bar_stride=1``), adjacent
    labels share M-1 of their M future bars, so they're strongly correlated.
    This function surfaces that autocorrelation as a feature:

        ``target__autocorr_lag{L}__h__w{W}`` for each L in ``lags`` and W
        in ``windows`` is the rolling population correlation of
        ``y_mature_t`` with ``y_mature_{t-L}`` over the most recent W
        mature labels.

    Causality is enforced via ``_label_maturity_shift`` — both the base
    ``y_mature`` and its ``lag``-step lag use only labels whose prediction
    windows have already closed before the current row.

    Parameters
    ----------
    df_boundaries : pl.DataFrame
        Must contain ``y`` (the label column produced by ``construct_labels_pl``).
    windows : iterable of int
        Rolling window sizes for the correlation estimator (in *rows of df*).
    bar_stride, M : int, optional
        See ``compute_past_target_features_pl``. Determines the causal shift.
    lags : iterable of int
        Lag offsets at which to compute the autocorrelation.
    """
    shift_units = _label_maturity_shift(bar_stride, M if M is not None else 1)
    y_mature = pl.col("y").shift(shift_units)

    new_cols: list[pl.Expr] = []
    for lag in lags:
        if lag <= 0:
            raise ValueError(f"lag must be > 0, got {lag}")
        y_lag = y_mature.shift(int(lag))
        for W in windows:
            if W <= int(lag) + 1:
                # Below this floor population_corr lacks enough non-null
                # pairs to produce a meaningful estimate; emit null.
                new_cols.append(
                    pl.lit(None, dtype=pl.Float64).alias(
                        f"target__autocorr_lag{int(lag)}__h__w{int(W)}"
                    )
                )
                continue
            new_cols.append(
                population_corr(y_mature, y_lag, w=int(W)).alias(
                    f"target__autocorr_lag{int(lag)}__h__w{int(W)}"
                )
            )
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
    """Add barrier-aware features. Mirrors utils.compute_barrier_aware_features.

    Cadence-independent: every operation is row-wise on the boundary frame,
    referencing ``vol__rs__f__w{W}`` columns that are already produced at
    whatever the engine's row cadence is.
    """
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
    *,
    bar_stride: int | None = None,
) -> pl.DataFrame:
    """Add block-aggregated features. Mirrors utils.compute_block_features
    at legacy cadence (``bar_stride=None`` or ``bar_stride=M``).

    At 1-min cadence (``bar_stride=1``), the M-bar block for row n is the
    trailing M bars ``[n-M+1, n]`` (rolling), and the "previous boundary
    close" used for ret__inst becomes ``close[n-M]``. This gives one
    block-feature value per 1-min row, with strictly causal lookback.
    """
    if bar_stride is None:
        bar_stride = int(M)
    if bar_stride == int(M):
        return _compute_block_features_boundary(df_boundaries, df_raw, M, windows_h)
    if bar_stride == 1:
        return _compute_block_features_1min(df_boundaries, df_raw, M, windows_h)
    raise ValueError(
        f"compute_block_features_pl: bar_stride must be 1 or M={M}, got {bar_stride}"
    )


def _compute_block_features_boundary(
    df_boundaries: pl.DataFrame,
    df_raw: pl.DataFrame,
    M: int,
    windows_h: Iterable[int],
) -> pl.DataFrame:
    """Legacy boundary-cadence block features (non-overlapping M-bar blocks)."""
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


def _compute_block_features_1min(
    df_boundaries: pl.DataFrame,
    df_raw: pl.DataFrame,
    M: int,
    windows_h: Iterable[int],
) -> pl.DataFrame:
    """1-min-cadence block features (rolling M-bar windows ending at each row).

    Strictly causal: each row's block aggregates only bars at indices
    ``[n-M+1, n]``. The reference price for ret__inst is ``close[n-M]``
    (M bars ago, fully observed).
    """
    K = int(len(df_boundaries))
    if K > len(df_raw):
        raise ValueError(
            "compute_block_features_pl (1min): df_boundaries longer than df_raw"
        )

    close = df_raw["close"].to_numpy().astype(float)[:K]
    p = np.log(close)
    high = df_raw["high"].to_numpy().astype(float)[:K]
    low = df_raw["low"].to_numpy().astype(float)[:K]
    volume = df_raw["volume"].to_numpy().astype(float)[:K]
    quote_volume = df_raw["quote_volume"].to_numpy().astype(float)[:K]
    num_trades = df_raw["num_trades"].to_numpy().astype(float)[:K]
    taker_buy_base = df_raw["taker_buy_base"].to_numpy().astype(float)[:K]

    # Pandas rolling (ddof/min_periods semantics match legacy)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    qvol_s = pd.Series(quote_volume)
    ntr_s = pd.Series(num_trades)
    tbb_s = pd.Series(taker_buy_base)

    H = high_s.rolling(M, min_periods=M).max().to_numpy()
    L = low_s.rolling(M, min_periods=M).min().to_numpy()
    V = volume_s.rolling(M, min_periods=M).sum().to_numpy()
    Q = qvol_s.rolling(M, min_periods=M).sum().to_numpy()
    Ntr = ntr_s.rolling(M, min_periods=M).sum().to_numpy()
    VTB = tbb_s.rolling(M, min_periods=M).sum().to_numpy()

    # ret_inst[n] = log(close[n] / close[n-M])
    ret_inst = np.full(K, np.nan, dtype=float)
    if K > M:
        ret_inst[M:] = np.log(close[M:] / close[:-M])

    range_inst = np.where((H > L) & np.isfinite(H) & np.isfinite(L), np.log(H / L), np.nan)
    logvol_inst = np.where(np.isfinite(V), np.log1p(V), np.nan)
    ofi_inst = np.where((V > 0) & np.isfinite(V), 2.0 * (VTB / np.where(V > 0, V, np.nan)) - 1.0, np.nan)

    # block__maxret[n] = max over k in [n-M+1, n] of log(close[k] / close[n-M])
    # block__minret[n] = same with min
    # Compute with rolling apply over the log-price series, baselined to p[n-M]
    block_maxret = np.full(K, np.nan, dtype=float)
    block_minret = np.full(K, np.nan, dtype=float)
    if K > M:
        # For each n>=M: max over i in [n-M+1..n] of (p[i] - p[n-M])
        # = max(p[n-M+1..n]) - p[n-M]
        p_s = pd.Series(p)
        roll_pmax = p_s.rolling(M, min_periods=M).max().to_numpy()
        roll_pmin = p_s.rolling(M, min_periods=M).min().to_numpy()
        # p[n-M] = p shifted by M; valid from index M onward.
        p_lag = np.full(K, np.nan, dtype=float)
        p_lag[M:] = p[:-M]
        block_maxret[M:] = roll_pmax[M:] - p_lag[M:]
        block_minret[M:] = roll_pmin[M:] - p_lag[M:]

    denom_hl = np.log(H) - np.log(L)
    close_to_high = (p - np.log(L)) / (denom_hl + EPS)
    close_to_high = np.where(np.isfinite(denom_hl) & (denom_hl != 0.0), close_to_high, np.nan)

    # ret__std__h__w{W}: rolling std of ret_inst over W rows
    ret_inst_s = pd.Series(ret_inst)
    ret_std_cols: dict[str, np.ndarray] = {}
    for W in windows_h:
        ret_std_cols[f"ret__std__h__w{W}"] = (
            ret_inst_s.rolling(int(W), min_periods=int(W)).std(ddof=0).to_numpy()
        )

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
