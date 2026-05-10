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


def compute_feature_health(
    df: pl.DataFrame,
    feature_cols: Iterable[str],
    *,
    constant_threshold: float = 1e-12,
) -> pl.DataFrame:
    """Per-feature health summary.

    Returns one row per feature with:

    - ``name``, ``family``: identifiers
    - ``n``, ``n_valid``: total rows / non-null rows
    - ``nan_rate``: post-impute null rate (should be 0; >0 indicates a
      regression in the impute stage)
    - ``undef_rate``: pre-impute null rate read from the matching
      ``undef__<feature>`` flag column if present (mean of the Int8 flag)
    - ``mean``, ``std``, ``min``, ``max``, ``skew``, ``kurt``: distribution
      stats over non-null values; ``std`` is population (``ddof=0``)
    - ``has_inf``: whether any value is ±inf
    - ``is_constant``: ``std < constant_threshold``
    """
    rows: list[dict[str, object]] = []
    available = set(df.columns)

    for col in feature_cols:
        if col not in available:
            continue
        s = df[col]
        n = int(s.len())
        n_null = int(s.null_count())
        nan_rate = (n_null / n) if n else 0.0
        s_clean = s.drop_nulls()
        n_valid = len(s_clean)
        is_numeric = s.dtype.is_numeric()

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
                has_inf = bool(s_clean.is_infinite().any())
            except Exception:
                has_inf = False
        else:
            mean = std = mn = mx = skew = kurt = float("nan")
            has_inf = False

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
                "nan_rate": nan_rate,
                "undef_rate": undef_rate,
                "mean": mean,
                "std": std,
                "min": mn,
                "max": mx,
                "skew": skew,
                "kurt": kurt,
                "has_inf": has_inf,
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
                pl.col("nan_rate").max().alias("max_nan_rate"),
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
    """
    return (
        health.filter(
            (pl.col("undef_rate") > max_undef_rate)
            | pl.col("is_constant")
            | pl.col("has_inf")
            | (pl.col("skew").abs() > max_abs_skew)
            | pl.col("mean").is_nan()
        )
        .with_columns(
            pl.when(pl.col("is_constant"))
            .then(pl.lit("constant"))
            .when(pl.col("has_inf"))
            .then(pl.lit("inf"))
            .when(pl.col("mean").is_nan())
            .then(pl.lit("all_nan"))
            .when(pl.col("undef_rate") > max_undef_rate)
            .then(pl.lit("high_undef"))
            .when(pl.col("skew").abs() > max_abs_skew)
            .then(pl.lit("heavy_skew"))
            .otherwise(pl.lit("other"))
            .alias("issue")
        )
        .sort("undef_rate", descending=True)
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
