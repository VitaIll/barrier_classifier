"""Data quality flags + undef-flag/imputation pipeline.

Imputation values are declared WITH their features: registry features on
the Feature class (``Feature.impute_default`` / ``impute_value``),
boundary-stage columns in ``boundary.BOUNDARY_IMPUTE_PREFIXES`` next to
their constructors. This replaced the legacy order-sensitive regex table
(``utils.get_imputation_value``) whose ``.*`` catch-all silently imputed
unregistered columns to 0.0 — an UNRESOLVED column is now a hard
:class:`~src.core.errors.ContractError` (the bridge suite pinned the two
resolutions equal across every produced column before the switch).

These run AFTER feature computation but BEFORE training; they belong to
neither bars-tier nor boundary-stage and are kept as plain functions.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional

import polars as pl

from src.core.errors import ContractError
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


def default_imputation_map() -> dict[str, float]:
    """Registry-declared fills for every registered feature column.

    Convenience for direct callers that did not build a
    :class:`~src.features.engine.FeatureEngine` themselves (the pipeline
    threads its engine's map through instead, so custom configs resolve
    with their own window grids).
    """
    from src.features.engine import FeatureEngine

    return FeatureEngine(tiers=(1, 2)).imputation_map()


def resolve_imputation_value(
    col: str,
    *,
    impute_map: Mapping[str, float],
    boundary_entries: list[tuple[str, Optional[float]]],
) -> float:
    """Fill value for ``col``: exact registry name, then boundary prefix.

    Raises :class:`ContractError` for a column neither source declares —
    the replacement for the legacy silent 0.0 catch-all — and for columns
    declared "never missing" (fill ``None``) that nevertheless carried
    nulls.
    """
    if col in impute_map:
        return float(impute_map[col])
    for prefix, value in boundary_entries:
        if col.startswith(prefix):
            if value is None:
                raise ContractError(
                    f"column {col!r} is declared never-missing (constant) "
                    "but carries nulls — upstream pipeline bug"
                )
            return float(value)
    raise ContractError(
        f"no imputation declared for column {col!r} — declare "
        "Feature.impute_default on its class (registry features) or add a "
        "prefix to boundary.BOUNDARY_IMPUTE_PREFIXES (boundary-stage "
        "columns). The silent 0.0 catch-all is gone by design."
    )


def create_undef_flags_and_impute_pl(
    df: pl.DataFrame,
    feature_cols: Iterable[str],
    *,
    p_hit_prior: float,
    cap_h_blocks: int = 144,
    impute_map: Mapping[str, float] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Add ``undef__<col>`` Int8 flags for any feature column with missing
    values, then fill the original column with the per-feature imputation
    value.

    ``impute_map`` is the registry half of the contract (from
    ``FeatureEngine.imputation_map()``); ``None`` builds the
    default-config map. Boundary-stage columns resolve through
    ``boundary.BOUNDARY_IMPUTE_PREFIXES`` with the supplied
    ``p_hit_prior`` / ``cap_h_blocks`` context.

    **Nullability discipline**: polars distinguishes ``null`` (typed
    missing) from float ``NaN``. ``null_count`` / ``fill_null`` only see
    null. Some primitives (e.g. ``map_batches`` numpy kernels) can emit
    float NaN. Without coercion the impute step silently leaves NaN in
    place — that bit us once on the live BTCUSDT run. We coerce
    NaN -> null on every Float feature column up front, so the flag and
    fill logic catches both kinds of missingness, and the post-impute
    assertions verify zero null AND zero NaN AND zero inf.
    """
    from src.features.boundary import boundary_imputation_entries

    feature_cols = list(feature_cols)
    available = set(df.columns)
    for col in feature_cols:
        if col not in available:
            raise ValueError(
                f"create_undef_flags_and_impute_pl: feature column missing: {col}"
            )

    if impute_map is None:
        impute_map = default_imputation_map()
    boundary_entries = boundary_imputation_entries(
        p_hit_prior=float(p_hit_prior), cap_h_blocks=int(cap_h_blocks)
    )

    # Snapshot the schema ONCE — ``df.schema`` rebuilds a Schema object per
    # access, and per-column access in a ~1,600-column loop was ~3.5s of a
    # 22s live rolling call (profiled 2026-07-11).
    schema = dict(df.schema)

    # Coerce float NaN -> null on every numeric feature column so that
    # null_count / is_null / fill_null all report the unified "missing"
    # status. This is the parity contract with pandas notna().
    coercion_exprs = [
        pl.col(c).fill_nan(None).alias(c)
        for c in feature_cols
        if schema[c] in (pl.Float32, pl.Float64)
    ]
    if coercion_exprs:
        df = df.with_columns(coercion_exprs)

    # All null counts in one engine pass instead of one query per column.
    null_counts = df.select(pl.col(feature_cols).null_count()).row(0)
    null_by_col = dict(zip(feature_cols, null_counts))

    flag_exprs: list[pl.Expr] = []
    fill_exprs: list[pl.Expr] = []
    undef_cols: list[str] = []

    for col in feature_cols:
        if int(null_by_col[col]) == 0:
            continue

        undef_col = f"undef__{col}"
        flag_exprs.append(pl.col(col).is_null().cast(pl.Int8).alias(undef_col))
        undef_cols.append(undef_col)

        impute_value = resolve_imputation_value(
            col, impute_map=impute_map, boundary_entries=boundary_entries
        )
        # A non-finite impute value (NaN / inf) would silently re-poison
        # the column we just flagged. Catch it at the declaration boundary
        # so the error names the column responsible.
        if not math.isfinite(impute_value):
            raise ValueError(
                f"Imputation declaration is non-finite for "
                f"column {col!r}: {impute_value}"
            )
        fill_exprs.append(pl.col(col).fill_null(impute_value).alias(col))

    if flag_exprs or fill_exprs:
        out = df.with_columns(flag_exprs + fill_exprs)
    else:
        out = df

    # Three independent post-impute assertions. Any one firing means the
    # dataset is not safe to hand to a model. Each scan is one engine pass
    # over all columns (not one query per column).
    remaining_null = sum(
        int(v) for v in out.select(pl.col(feature_cols).null_count()).row(0)
    )
    if remaining_null:
        raise ValueError(f"Nulls remain after imputation: {remaining_null}")

    # ``is_nan``/``is_infinite`` are float-only — newer polars raises on
    # integer dtypes. Integer columns cannot carry NaN/inf so the check is
    # vacuous for them; narrow to floats.
    out_schema = dict(out.schema)
    float_cols = [
        c for c in feature_cols if out_schema[c] in (pl.Float32, pl.Float64)
    ]
    if float_cols:
        nan_counts = out.select(pl.col(float_cols).is_nan().sum()).row(0)
        for col, n_nan in zip(float_cols, nan_counts):
            if int(n_nan or 0):
                raise ValueError(
                    f"Float NaN remains after imputation in column {col!r}: {n_nan}"
                )
        inf_counts = out.select(pl.col(float_cols).is_infinite().sum()).row(0)
        for col, n_inf in zip(float_cols, inf_counts):
            if int(n_inf or 0):
                raise ValueError(
                    f"Infs remain after imputation in column {col!r}"
                )

    return out, undef_cols
