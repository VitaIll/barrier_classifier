"""Feature observability: per-column validation, reports, and feature-set
quality inspection helpers used by the notebook for post-build analysis.

Two layers:

1. ``Validator`` + ``FeatureReport`` — a per-Feature structural check that
   runs at engine compute time (currently stub-level; will fill in once
   the engine wires up validation hooks).

2. Inspection helpers (``compute_feature_health``, ``summarize_by_family``,
   ``flag_issues``, ``monthly_target_balance``) — operate on the FINAL
   wide dataframe produced by the pipeline and surface quality issues
   (high imputation rate, constant features, infinities, heavy skew),
   plus target-stability snapshots for plotting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import polars as pl

from src.features.base import Feature


@dataclass(frozen=True)
class FeatureReport:
    name: str
    n: int
    nan_rate: float
    mean: float
    std: float
    min: float
    max: float
    range_ok: bool
    finite_ok: bool
    warmup_ok: bool
    tail_ok: bool
    parity_ok: bool | None = None


class Validator:
    """Runs Feature.expected_* checks against a computed column."""

    def check(self, spec: Feature, name: str, col: pl.Series) -> FeatureReport:
        n = len(col)
        n_null = int(col.null_count())
        nan_rate = (n_null / n) if n else 0.0
        return FeatureReport(
            name=name,
            n=n,
            nan_rate=nan_rate,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            range_ok=True,
            finite_ok=True,
            warmup_ok=True,
            tail_ok=True,
            parity_ok=None,
        )


# ===========================================================================
# Feature-set quality inspection (post-pipeline)
# ===========================================================================


def _column_family(col: str) -> str:
    """Family group = prefix before the first ``__``. e.g.
    ``ret__lag1__f__w0`` -> ``ret``. Used for grouping in summaries and
    plots; differs from ``Feature.family`` (which is a per-class attr)
    because two families like ``deriv_basis`` and ``deriv_flow`` both
    emit columns starting with ``basis``/``flow``/etc.
    """
    return col.split("__", 1)[0]


def _classify_missing_pattern(
    s: pl.Series, n: int, is_float: bool
) -> tuple[str, int, int, int, int]:
    """Classify the missing-value pattern of a series.

    Returns ``(pattern, n_null, n_nan, n_leading, n_trailing)``:

    - ``pattern`` ∈ {"clean", "all_missing", "leading", "trailing",
       "edge", "scattered"}
    - ``n_null`` polars-null count, ``n_nan`` float-NaN count (0 if not
      a float column), and the contiguous leading/trailing missing
      blocks.

    "Scattered" means at least one missing cell sits between two valid
    cells — an unexpected pattern for any feature post-impute, since
    rolling-window warmup produces only "leading" and forward-looking
    label features produce only "trailing". Scattered triggers an
    issue flag in :func:`flag_issues`.
    """
    n_null = int(s.null_count())
    n_nan = int(s.is_nan().sum() or 0) if is_float else 0

    # Combined "missing" mask
    if is_float:
        missing = s.is_null() | s.is_nan()
    else:
        missing = s.is_null()
    n_missing = n_null + n_nan

    if n == 0 or n_missing == 0:
        return "clean", n_null, n_nan, 0, 0
    if n_missing == n:
        return "all_missing", n_null, n_nan, n, n

    valid_idx = (~missing).arg_true()
    n_valid = len(valid_idx)
    if n_valid == 0:
        return "all_missing", n_null, n_nan, n, n
    first_valid = int(valid_idx[0])
    last_valid = int(valid_idx[-1])

    n_leading = first_valid
    n_trailing = (n - 1) - last_valid
    n_inner_missing = n_missing - n_leading - n_trailing

    if n_inner_missing > 0:
        return "scattered", n_null, n_nan, n_leading, n_trailing
    if n_leading > 0 and n_trailing > 0:
        return "edge", n_null, n_nan, n_leading, n_trailing
    if n_leading > 0:
        return "leading", n_null, n_nan, n_leading, 0
    return "trailing", n_null, n_nan, 0, n_trailing


def compute_feature_health(
    df: pl.DataFrame,
    feature_cols: Iterable[str],
    *,
    constant_threshold: float = 1e-12,
) -> pl.DataFrame:
    """Per-feature health summary.

    Returns one row per feature with:

    - ``name``, ``family``: identifiers
    - ``n``, ``n_valid``: total rows / non-missing rows
    - ``n_null``, ``n_nan``: separate counts of polars-null and float-NaN
      (the impute step coerces NaN→null upfront, so post-impute both
      should be 0; surfacing them separately catches engine bugs that
      emit float NaN where null is expected)
    - ``missing_rate``: ``(n_null + n_nan) / n``
    - ``null_pattern``: clean / all_missing / leading / trailing / edge /
      scattered (see :func:`_classify_missing_pattern`)
    - ``n_leading_missing``, ``n_trailing_missing``: contiguous missing
      blocks at the start and end of the series
    - ``undef_rate``: rate from the matching ``undef__<feature>`` flag
      column if present (pre-impute missingness signal)
    - ``mean`` / ``std`` / ``min`` / ``max`` / ``skew`` / ``kurt``: stats
      over non-missing values; ``std`` is population (``ddof=0``)
    - ``n_inf``, ``has_inf``: count and bool for ±inf
    - ``is_constant``: ``std < constant_threshold``
    """
    rows: list[dict[str, object]] = []
    available = set(df.columns)

    for col in feature_cols:
        if col not in available:
            continue
        s = df[col]
        n = int(s.len())
        is_float = s.dtype in (pl.Float32, pl.Float64)
        is_numeric = s.dtype.is_numeric()

        pattern, n_null, n_nan, n_leading, n_trailing = _classify_missing_pattern(
            s, n, is_float
        )
        n_missing = n_null + n_nan
        missing_rate = (n_missing / n) if n else 0.0

        if is_float:
            s_clean = s.drop_nulls().drop_nans()
        else:
            s_clean = s.drop_nulls()
        n_valid = len(s_clean)

        if n_valid > 0 and is_numeric:
            mean_v = s_clean.mean()
            mean = float(mean_v) if mean_v is not None else float("nan")
            std = float(s_clean.std(ddof=0)) if n_valid > 1 else 0.0
            mn_v = s_clean.min()
            mx_v = s_clean.max()
            mn = float(mn_v) if mn_v is not None else float("nan")
            mx = float(mx_v) if mx_v is not None else float("nan")
            try:
                skew = float(s_clean.skew()) if n_valid > 2 else float("nan")
            except Exception:
                skew = float("nan")
            try:
                kurt = float(s_clean.kurtosis()) if n_valid > 3 else float("nan")
            except Exception:
                kurt = float("nan")
            try:
                n_inf = int(s_clean.is_infinite().sum() or 0)
            except Exception:
                n_inf = 0
        else:
            mean = std = mn = mx = skew = kurt = float("nan")
            n_inf = 0

        undef_col = f"undef__{col}"
        if undef_col in available:
            uv = df[undef_col].mean()
            undef_rate = float(uv) if uv is not None else 0.0
        else:
            undef_rate = 0.0

        is_const = (
            (n_valid > 1)
            and (not math.isnan(std))
            and (std < constant_threshold)
        )

        rows.append(
            {
                "name": col,
                "family": _column_family(col),
                "n": n,
                "n_valid": n_valid,
                "n_null": n_null,
                "n_nan": n_nan,
                "missing_rate": missing_rate,
                "null_pattern": pattern,
                "n_leading_missing": n_leading,
                "n_trailing_missing": n_trailing,
                "undef_rate": undef_rate,
                "mean": mean,
                "std": std,
                "min": mn,
                "max": mx,
                "skew": skew,
                "kurt": kurt,
                "n_inf": n_inf,
                "has_inf": n_inf > 0,
                "is_constant": is_const,
            }
        )

    return pl.DataFrame(rows)


def summarize_by_family(health: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-family stats from a ``compute_feature_health`` frame.

    Sorted by feature count (descending).
    """
    return (
        health.group_by("family")
        .agg(
            [
                pl.len().alias("n_features"),
                pl.col("undef_rate").mean().alias("avg_undef_rate"),
                pl.col("undef_rate").max().alias("max_undef_rate"),
                pl.col("missing_rate").max().alias("max_missing_rate"),
                pl.col("n_nan").sum().alias("n_features_with_nan"),
                pl.col("null_pattern")
                .eq("scattered")
                .sum()
                .alias("n_scattered"),
                pl.col("is_constant").sum().alias("n_constant"),
                pl.col("has_inf").sum().alias("n_with_inf"),
                pl.col("std").mean().alias("avg_std"),
                pl.col("skew").abs().mean().alias("avg_abs_skew"),
            ]
        )
        .sort("n_features", descending=True)
    )


def flag_issues(
    health: pl.DataFrame,
    *,
    max_undef_rate: float = 0.5,
    max_abs_skew: float = 20.0,
) -> pl.DataFrame:
    """Return rows from a ``compute_feature_health`` frame that look
    problematic. Each issue row gets an ``issue`` label so callers can
    group / summarise reasons.

    Issue priorities (highest first; first match wins):
      1. ``residual_nan``  — float NaN survived the impute step
      2. ``inf``           — ±inf in the column
      3. ``scattered_missing`` — pre-impute missing cells are not at
         the edges (warmup / coverage tail); flags either a real data
         anomaly or an engine bug. Should never go unnoticed.
      4. ``all_nan``       — column is fully missing
      5. ``constant``      — std < threshold
      6. ``high_undef``    — pre-impute undef rate > threshold
      7. ``heavy_skew``    — |skew| > threshold
    """
    return (
        health.filter(
            (pl.col("n_nan") > 0)
            | pl.col("has_inf")
            | (pl.col("null_pattern") == "scattered")
            | pl.col("mean").is_nan()
            | pl.col("is_constant")
            | (pl.col("undef_rate") > max_undef_rate)
            | (pl.col("skew").abs() > max_abs_skew)
        )
        .with_columns(
            pl.when(pl.col("n_nan") > 0)
            .then(pl.lit("residual_nan"))
            .when(pl.col("has_inf"))
            .then(pl.lit("inf"))
            .when(pl.col("null_pattern") == "scattered")
            .then(pl.lit("scattered_missing"))
            .when(pl.col("mean").is_nan())
            .then(pl.lit("all_nan"))
            .when(pl.col("is_constant"))
            .then(pl.lit("constant"))
            .when(pl.col("undef_rate") > max_undef_rate)
            .then(pl.lit("high_undef"))
            .when(pl.col("skew").abs() > max_abs_skew)
            .then(pl.lit("heavy_skew"))
            .otherwise(pl.lit("other"))
            .alias("issue")
        )
        .sort(
            ["issue", "missing_rate", "undef_rate"],
            descending=[False, True, True],
        )
    )


def monthly_target_balance(
    df: pl.DataFrame,
    *,
    ts_col: str = "ts",
    target_col: str = "y",
) -> pl.DataFrame:
    """Per-month total / hits / base rate of the binary target.

    Useful for a stability line chart — base rate drift across months
    indicates regime shift in the underlying barrier-crossing dynamics.
    """
    return (
        df.with_columns(pl.col(ts_col).dt.truncate("1mo").alias("_month"))
        .group_by("_month")
        .agg(
            [
                pl.len().alias("total"),
                pl.col(target_col).sum().alias("hits"),
                pl.col(target_col).mean().alias("base_rate"),
            ]
        )
        .sort("_month")
    )
