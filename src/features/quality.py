"""Data quality flags + undef-flag/imputation pipeline.

Mirrors utils.compute_data_quality_flags + utils.create_undef_flags_and_impute.
Imputation values come from utils.get_imputation_value (legacy registry).

These run AFTER feature computation but BEFORE training; they belong to
neither bars-tier nor boundary-stage and are kept as plain functions.
"""

from __future__ import annotations

from typing import Iterable

import polars as pl

from src import utils as _legacy
from src.features.config import EPS  # noqa: F401  (re-exported for callers)


def compute_data_quality_flags_pl(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``data__bad_ohlc__f__w0`` and ``data__gap__f__w0`` columns.

    Mirrors utils.compute_data_quality_flags (utils.py:2003-2026).
    Requires a ``ts`` column of dtype Datetime and OHLC columns.
    """
    bad_ohlc = (
        (pl.col("high") < pl.col("low"))
        | (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
    )

    diffs = pl.col("ts").diff()
    # Row 0 has null diff; treat as no-gap (matches legacy ``gap.iloc[0] = 0``).
    gap = (
        (diffs != pl.duration(minutes=1)).cast(pl.Int8).fill_null(0)
    )

    return df.with_columns(
        [
            bad_ohlc.cast(pl.Int8).alias("data__bad_ohlc__f__w0"),
            gap.alias("data__gap__f__w0"),
        ]
    )


def create_undef_flags_and_impute_pl(
    df: pl.DataFrame,
    feature_cols: Iterable[str],
    *,
    p_hit_prior: float,
    cap_h_blocks: int = 144,
) -> tuple[pl.DataFrame, list[str]]:
    """Add ``undef__<col>`` Int8 flags for any feature column with nulls,
    then fill the original column with the per-feature imputation value.

    Mirrors utils.create_undef_flags_and_impute. Imputation values resolved
    by ``utils.get_imputation_value`` (legacy regex registry); we reuse it
    directly to avoid duplicating the long pattern table.
    """
    flag_exprs: list[pl.Expr] = []
    fill_exprs: list[pl.Expr] = []
    undef_cols: list[str] = []

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(
                f"create_undef_flags_and_impute_pl: feature column missing: {col}"
            )
        if int(df[col].null_count()) == 0:
            continue

        undef_col = f"undef__{col}"
        flag_exprs.append(pl.col(col).is_null().cast(pl.Int8).alias(undef_col))
        undef_cols.append(undef_col)

        impute_value = _legacy.get_imputation_value(
            col, p_hit_prior=p_hit_prior, cap_h_blocks=cap_h_blocks
        )
        fill_exprs.append(pl.col(col).fill_null(impute_value).alias(col))

    if flag_exprs or fill_exprs:
        out = df.with_columns(flag_exprs + fill_exprs)
    else:
        out = df

    remaining = sum(int(out[c].null_count()) for c in feature_cols)
    if remaining != 0:
        raise ValueError(f"NaNs remain after imputation: {remaining}")

    for col in feature_cols:
        if bool(out[col].is_infinite().any()):
            raise ValueError(f"Infs remain after imputation in column: {col}")

    return out, undef_cols
