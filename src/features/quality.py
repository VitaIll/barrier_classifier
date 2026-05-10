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
    """Add ``undef__<col>`` Int8 flags for any feature column with missing
    values, then fill the original column with the per-feature imputation
    value.

    Mirrors utils.create_undef_flags_and_impute. Imputation values resolved
    by ``utils.get_imputation_value`` (legacy regex registry); we reuse it
    directly to avoid duplicating the long pattern table.

    **Nullability discipline**: polars distinguishes ``null`` (typed
    missing) from float ``NaN``. ``null_count`` / ``fill_null`` only see
    null. Some primitives (e.g. ``map_batches`` numpy kernels) can emit
    float NaN. Without coercion the impute step silently leaves NaN in
    place — that bit us once on the live BTCUSDT run. We now coerce
    NaN -> null on every Float feature column up front, so the flag and
    fill logic catches both kinds of missingness, and the post-impute
    assertions verify zero null AND zero NaN AND zero inf.
    """
    feature_cols = list(feature_cols)
    available = set(df.columns)
    for col in feature_cols:
        if col not in available:
            raise ValueError(
                f"create_undef_flags_and_impute_pl: feature column missing: {col}"
            )

    # Coerce float NaN -> null on every numeric feature column so that
    # null_count / is_null / fill_null all report the unified "missing"
    # status. This is the parity contract with pandas notna().
    coercion_exprs = [
        pl.col(c).fill_nan(None).alias(c)
        for c in feature_cols
        if df.schema[c] in (pl.Float32, pl.Float64)
    ]
    if coercion_exprs:
        df = df.with_columns(coercion_exprs)

    flag_exprs: list[pl.Expr] = []
    fill_exprs: list[pl.Expr] = []
    undef_cols: list[str] = []

    for col in feature_cols:
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

    # Three independent post-impute assertions. Any one firing means the
    # dataset is not safe to hand to a model.
    remaining_null = sum(int(out[c].null_count()) for c in feature_cols)
    if remaining_null:
        raise ValueError(f"Nulls remain after imputation: {remaining_null}")

    for col in feature_cols:
        dtype = out.schema[col]
        if dtype in (pl.Float32, pl.Float64):
            n_nan = int(out[col].is_nan().sum() or 0)
            if n_nan:
                raise ValueError(
                    f"Float NaN remains after imputation in column {col!r}: {n_nan}"
                )
        if dtype.is_numeric():
            if bool(out[col].is_infinite().any()):
                raise ValueError(
                    f"Infs remain after imputation in column {col!r}"
                )

    return out, undef_cols
