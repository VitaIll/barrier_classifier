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
    compute_past_target_features_pl,
    construct_labels_pl,
)
from src.features.config import (
    C,
    ETA,
    HITRATE_WINDOWS_H,
    K_WARMUP,
    M,
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


def _to_polars(df_pd: pd.DataFrame) -> pl.DataFrame:
    """pandas → polars with tz stripped (keeps Datetime dtype on Windows)."""
    df = df_pd.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return pl.from_pandas(df.reset_index(names="ts"))


def run_pipeline(
    df_raw_pd: pd.DataFrame,
    *,
    with_derivatives: bool = False,
    p_hit_prior: float = 0.5,
    cap_h_blocks: int | None = None,
) -> pl.DataFrame:
    """Run the full feature pipeline end-to-end on a pandas bar frame.

    ``df_raw_pd`` must have OHLCV columns and a tz-aware DatetimeIndex.
    For ``with_derivatives=True`` the frame must also carry the
    derivatives raw columns (close_fut, funding_rate, oi_usd, etc.) —
    the legacy ``compute_derivatives_base_series`` runs first to derive
    basis_abs, pcr_oi, and friends.
    """
    if cap_h_blocks is None:
        cap_h_blocks = max(WINDOWS_H)

    # --- Stages 1-2: base series (legacy) ----------------------------------
    df_pd = _legacy.compute_base_series(df_raw_pd)
    if with_derivatives:
        df_pd = _legacy.compute_derivatives_base_series(df_pd)

    df_pl = _to_polars(df_pd)
    df_raw_pl = _to_polars(df_raw_pd)

    # --- Stage 3: bar-level features (Tier 1 + Tier 2) ---------------------
    families = list(_DEFAULT_FAMILIES_NO_DERIV)
    if with_derivatives:
        families.extend(_DERIVATIVES_FAMILIES)
    engine = FeatureEngine(tiers=(1, 2), families=tuple(families))
    df_pl = engine.transform(df_pl, trim=False).data

    # --- Stage 4: data quality flags --------------------------------------
    df_pl = compute_data_quality_flags_pl(df_pl)

    # --- Stage 5: sample decision boundaries every M rows ------------------
    df_boundaries = df_pl.gather_every(M).with_columns(
        pl.int_range(pl.len()).alias("k")
    )

    # --- Stages 6-9: boundary stages ---------------------------------------
    df_boundaries = construct_labels_pl(df_boundaries, df_raw_pl, M, ETA, C)
    df_boundaries = compute_past_target_features_pl(
        df_boundaries, WINDOWS_H, HITRATE_WINDOWS_H
    )
    df_boundaries = compute_barrier_aware_features_pl(
        df_boundaries, WINDOWS_BARRIER, PHI, M, VOL_PAIRS, c=float(C)
    )
    df_boundaries = compute_block_features_pl(df_boundaries, df_raw_pl, M, WINDOWS_H)

    # --- Stage 10: warmup trim + drop NaN labels ---------------------------
    df_boundaries = (
        df_boundaries
        .filter(pl.col("k") >= K_WARMUP)
        .filter(pl.col("y").is_not_null())
    )

    # --- Stage 11: undef flags + imputation -------------------------------
    label_aux = ("k", "ts", "y", "m_k", "tau_k", "phi")
    feature_cols = [c for c in df_boundaries.columns if c not in label_aux]
    df_final, _ = create_undef_flags_and_impute_pl(
        df_boundaries,
        feature_cols,
        p_hit_prior=p_hit_prior,
        cap_h_blocks=cap_h_blocks,
    )

    return df_final
