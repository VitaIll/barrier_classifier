"""End-to-end feature-engineering pipeline.

Orchestrates the pieces built in steps 1-14 to produce the final dataset
that the legacy notebook produces. The legacy ``compute_base_series`` and
``compute_derivatives_base_series`` are still used as base-series
generators (out of scope for this refactor); everything downstream is
the new polars stack.

Stages (mirrors the notebook):
  1. ``compute_base_series`` — adds r, rho, clv, etc. (legacy)
  2. ``compute_derivatives_base_series`` — adds basis_abs, pcr_oi, etc. (legacy, optional)
  3. ``FeatureEngine.transform`` — Tier-1 + Tier-2 features (no trim)
  4. ``compute_data_quality_flags_pl`` — bad_ohlc + gap
  5. Boundary sample (every M-th row)
  6. ``construct_labels_pl``
  7. ``compute_past_target_features_pl``
  8. ``compute_barrier_aware_features_pl``
  9. ``compute_block_features_pl``
 10. Warmup trim (k >= K_WARMUP) + drop NaN-label rows
 11. ``create_undef_flags_and_impute_pl``

Returns the final boundary-aligned dataframe ready for training.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from src import utils as _legacy
from src.features import FeatureEngine
from src.features.boundary import (
    compute_barrier_aware_features_pl,
    compute_block_features_pl,
    compute_past_target_autocorrelation_pl,
    compute_past_target_features_pl,
    construct_labels_pl,
)
from src.features.config import (
    C,
    ETA,
    HITRATE_WINDOWS_H,
    K_WARMUP,
    M,
    N_WARMUP,
    PHI,
    VOL_PAIRS,
    WINDOWS_BARRIER,
    WINDOWS_H,
)
from src.features.quality import (
    compute_data_quality_flags_pl,
    create_undef_flags_and_impute_pl,
)


# Families included by default (tier 1 + tier 2 of the bar-level engine).
_DEFAULT_FAMILIES_NO_DERIV: tuple[str, ...] = (
    "lag", "rolling", "quantile", "vol", "candle", "trend", "activity",
    "correlation", "entropy", "event", "seasonality",
    "excursion", "liquidity",
)
_DERIVATIVES_FAMILIES: tuple[str, ...] = (
    "deriv_basis", "deriv_flow", "deriv_oi", "deriv_funding",
    "deriv_options", "deriv_volidx",
)


# Columns that are NOT model features even though they sit on the boundary
# frame. Mirrors the notebook's NON_FEATURE_COLS contract — labels, weights,
# raw OHLCV, base series, and derivatives base series. ``m_dn`` and
# ``tau_dn`` carry the optional downside-barrier diagnostics emitted by
# ``construct_labels_pl(..., add_triple_barrier_aux=True)``; they sit next
# to the long-only label and are never model features.
_LABEL_AUX_COLS: tuple[str, ...] = (
    "k", "ts", "y", "m_k", "tau_k", "phi", "m_dn", "tau_dn",
)
_RAW_COLS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "quote_volume",
    "num_trades", "taker_buy_base", "taker_buy_quote",
)
_BASE_COLS: tuple[str, ...] = (
    "p", "r", "rho", "r_oc", "g", "logvol", "logtrades", "logquotevol",
    "b", "ofi", "clv", "bodyfrac", "wickup", "wickdn", "vwap", "vwapdev",
    "qpertrade",
)
_DERIV_BASE_COLS: tuple[str, ...] = (
    "close_fut", "volume_fut", "quote_volume_fut", "taker_buy_base_fut",
    "num_trades_fut", "funding_rate", "oi_usd", "opt_oi",
    "put_open_interest", "call_open_interest", "opt_volume", "put_volume",
    "call_volume", "bvol", "basis_abs", "basis_pct", "tb_ratio_fut",
    "net_vol_fut", "pcr_oi", "pcr_vol",
)


def _to_polars(df_pd: pd.DataFrame) -> pl.DataFrame:
    """pandas → polars with tz stripped (keeps Datetime dtype on Windows)."""
    df = df_pd.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return pl.from_pandas(df.reset_index(names="ts"))


def _coerce_expected_numeric(df_pl: pl.DataFrame) -> pl.DataFrame:
    """Force any expected-numeric column to ``Float64`` with null preservation.

    A left-join onto a date range outside the EOH coverage window (e.g.
    2025 bars joined against 2023-05..2023-10 EOH option data) produces
    an all-null column. ``pl.from_pandas`` of such a column can land as
    ``Object``/``str`` dtype, which then breaks downstream feature compute
    (``pl.col(...).is_infinite()`` rejects ``str``). Cast defensively here.
    """
    expected_numeric = set(_RAW_COLS) | set(_BASE_COLS) | set(_DERIV_BASE_COLS)
    casts: list[pl.Expr] = []
    for col in df_pl.columns:
        if col in expected_numeric and df_pl.schema[col] != pl.Float64:
            casts.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    if casts:
        df_pl = df_pl.with_columns(casts)
    return df_pl


_AUTOCORR_WINDOWS_DEFAULT_BOUNDARY: tuple[int, ...] = (12, 24, 72, 144)
_AUTOCORR_WINDOWS_DEFAULT_1MIN: tuple[int, ...] = (60, 240, 1440, 2880)
_AUTOCORR_LAGS_DEFAULT: tuple[int, ...] = (1, 2, 5, 10)


def run_pipeline(
    df_raw_pd: pd.DataFrame,
    *,
    with_derivatives: bool = False,
    p_hit_prior: float = 0.5,
    cap_h_blocks: int | None = None,
    label_cadence: str = "boundary",
    enable_autocorrelation: bool | None = None,
    autocorr_windows: tuple[int, ...] | None = None,
    autocorr_lags: tuple[int, ...] = _AUTOCORR_LAGS_DEFAULT,
    barrier_source: str | None = None,
    add_triple_barrier_aux: bool = False,
) -> pl.DataFrame:
    """Run the full feature pipeline end-to-end on a pandas bar frame.

    ``df_raw_pd`` must have OHLCV columns and a tz-aware DatetimeIndex.
    For ``with_derivatives=True`` the frame must also carry the
    derivatives raw columns (close_fut, funding_rate, oi_usd, etc.) —
    the legacy ``compute_derivatives_base_series`` runs first to derive
    basis_abs, pcr_oi, and friends.

    ``label_cadence`` is the toggle for the new target definition:

    - ``"boundary"`` (default): legacy mode. Sample every M-th bar; one
      label per boundary; M-bar non-overlapping prediction windows. The
      output is row-aligned to the legacy schema and preserves bit-equal
      parity with ``utils.compute_*`` for the no-derivatives path.
    - ``"1min"``: every 1-min bar gets its own label (look M bars forward
      from that bar). Adjacent labels share M−1 of their M future bars,
      so they are strongly autocorrelated by construction. The new
      ``target__autocorr_lag{L}__h__w{W}`` features surface that
      autocorrelation in a strictly causal way (label-maturity shift of
      ``M // bar_stride`` rows).

    ``barrier_source`` controls the upper-barrier crossing test. The
    default is cadence-dependent:

    - ``"close"`` at boundary cadence — preserves the legacy close-
      confirmed label and bit-equal parity with ``utils.construct_labels``.
    - ``"high"`` at 1-min cadence — aligns the label with a long TP-limit
      order that fills when an intrabar high crosses the barrier (matches
      the simulator's ``exit_tp_or_expiry`` for a long position). Mixing
      close-based labels with high-based exits trains one event and
      trades a different one, so the high-based default is the correct
      target/execution alignment for the production strategy.

    Passing ``barrier_source`` explicitly overrides the cadence-based
    default in either direction.

    Reverting from ``"1min"`` to ``"boundary"`` is a one-line flip of
    this argument; the resulting frame is the legacy schema.
    """
    if label_cadence not in ("boundary", "1min"):
        raise ValueError(
            f"label_cadence must be 'boundary' or '1min', got {label_cadence!r}"
        )
    bar_stride = int(M) if label_cadence == "boundary" else 1
    if barrier_source is None:
        barrier_source = "close" if label_cadence == "boundary" else "high"
    if barrier_source not in ("close", "high"):
        raise ValueError(
            f"barrier_source must be 'close' or 'high', got {barrier_source!r}"
        )
    # Autocorrelation columns default on at 1-min (where labels overlap by M-1
    # bars and the signal is informative) and off at boundary cadence (where
    # the signal is weak and adding the columns silently changes the legacy
    # schema). Explicit True/False overrides this.
    if enable_autocorrelation is None:
        enable_autocorrelation = label_cadence == "1min"
    # Calendar-time semantics: at boundary cadence, "_h" windows are counted
    # in boundary rows (1 row = M bars). At 1-min cadence, they're counted in
    # 1-min rows. Scale by M so the *calendar* time of each window is preserved
    # across cadences — w=72 at boundary cadence (24h) becomes w=1440 at 1-min
    # cadence (also 24h). Column names follow the value used, so a model
    # trained at one cadence has a distinct feature vector from a model
    # trained at the other.
    scale = 1 if label_cadence == "boundary" else int(M)
    hitrate_windows = [int(w) * scale for w in HITRATE_WINDOWS_H]
    block_windows = [int(w) * scale for w in WINDOWS_H]
    warmup_rows = int(K_WARMUP) if label_cadence == "boundary" else int(N_WARMUP)
    if cap_h_blocks is None:
        # cap_h_blocks is used by the impute step for hit__since at the
        # right calendar scale; max of the scaled hitrate windows is the
        # natural ceiling.
        cap_h_blocks = max(hitrate_windows)
    if autocorr_windows is None:
        autocorr_windows = (
            _AUTOCORR_WINDOWS_DEFAULT_BOUNDARY
            if label_cadence == "boundary"
            else _AUTOCORR_WINDOWS_DEFAULT_1MIN
        )

    # --- Stages 1-2: base series (legacy) ----------------------------------
    df_pd = _legacy.compute_base_series(df_raw_pd)
    if with_derivatives:
        df_pd = _legacy.compute_derivatives_base_series(df_pd)

    df_pl = _coerce_expected_numeric(_to_polars(df_pd))
    df_raw_pl = _coerce_expected_numeric(_to_polars(df_raw_pd))

    # --- Stage 3: bar-level features (Tier 1 + Tier 2) ---------------------
    families = list(_DEFAULT_FAMILIES_NO_DERIV)
    if with_derivatives:
        families.extend(_DERIVATIVES_FAMILIES)
    engine = FeatureEngine(tiers=(1, 2), families=tuple(families))
    df_pl = engine.transform(df_pl, trim=False).data

    # At 1-min cadence the boundary-sparse excursion drawup/drawdown columns
    # are NaN at every non-multiple-of-M row by construction, which would
    # leak a phase artifact (modulo-M missingness) through the imputation
    # step and create a synthetic "regime label" of zero information. The
    # every-row trailing variants ``excursion__roll_max_drawup__f__w*`` and
    # ``excursion__roll_max_drawdown__f__w*`` cover the same statistic at
    # every row, so the sparse pair is redundant at 1-min cadence — drop it.
    if label_cadence == "1min":
        sparse_excursion = [
            c for c in df_pl.columns
            if c.startswith("excursion__max_drawup__f__")
            or c.startswith("excursion__max_drawdown__f__")
        ]
        if sparse_excursion:
            df_pl = df_pl.drop(sparse_excursion)

    # --- Stage 4: data quality flags --------------------------------------
    df_pl = compute_data_quality_flags_pl(df_pl)

    # --- Stage 5: sample decision rows -------------------------------------
    # Boundary cadence keeps every M-th row; 1-min cadence keeps every row.
    if label_cadence == "boundary":
        df_boundaries = df_pl.gather_every(M).with_columns(
            pl.int_range(pl.len()).alias("k")
        )
    else:
        df_boundaries = df_pl.with_columns(pl.int_range(pl.len()).alias("k"))

    # --- Stages 6-9: boundary stages ---------------------------------------
    df_boundaries = construct_labels_pl(
        df_boundaries,
        df_raw_pl,
        M,
        ETA,
        C,
        bar_stride=bar_stride,
        barrier_source=barrier_source,
        add_triple_barrier_aux=add_triple_barrier_aux,
    )
    df_boundaries = compute_past_target_features_pl(
        df_boundaries, block_windows, hitrate_windows, bar_stride=bar_stride, M=int(M)
    )
    if enable_autocorrelation:
        df_boundaries = compute_past_target_autocorrelation_pl(
            df_boundaries,
            autocorr_windows,
            bar_stride=bar_stride,
            M=int(M),
            lags=tuple(autocorr_lags),
        )
    df_boundaries = compute_barrier_aware_features_pl(
        df_boundaries, WINDOWS_BARRIER, PHI, M, VOL_PAIRS, c=float(C)
    )
    df_boundaries = compute_block_features_pl(
        df_boundaries, df_raw_pl, M, block_windows, bar_stride=bar_stride
    )

    # --- Stage 10: warmup trim + drop NaN labels ---------------------------
    # Warmup is in the row units of the current cadence:
    #   boundary cadence -> K_WARMUP boundary rows = N_WARMUP / M
    #   1-min cadence    -> N_WARMUP 1-min rows
    df_boundaries = (
        df_boundaries
        .filter(pl.col("k") >= warmup_rows)
        .filter(pl.col("y").is_not_null())
    )

    # --- Stage 11: undef flags + imputation -------------------------------
    # Feature columns = everything on the boundary frame that is NOT a
    # label, weight, raw bar, base series, or derivatives base series.
    # Without this exclusion the impute step tries to is_infinite() check
    # an all-null Object-dtyped column (e.g. opt_oi outside EOH coverage),
    # which polars rejects.
    non_feature = set(_LABEL_AUX_COLS + _RAW_COLS + _BASE_COLS + _DERIV_BASE_COLS)
    feature_cols = [c for c in df_boundaries.columns if c not in non_feature]
    df_final, _ = create_undef_flags_and_impute_pl(
        df_boundaries,
        feature_cols,
        p_hit_prior=p_hit_prior,
        cap_h_blocks=cap_h_blocks,
    )

    return df_final
