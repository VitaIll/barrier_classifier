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
    include_stationarity: bool = True,
    n_stationarity_chunks: int = 3,
    outlier_ratio_threshold: float = 1.0e4,
    mean_drift_threshold: float = 1.0,
    std_drift_ratio_threshold: float = 100.0,
) -> pl.DataFrame:
    """Per-feature health summary.

    Returns one row per feature with:

    Identity / nullability:
      ``name``, ``family``, ``n``, ``n_valid``, ``n_null``, ``n_nan``,
      ``missing_rate``, ``null_pattern`` ∈ {clean, all_missing, leading,
      trailing, edge, scattered}, ``n_leading_missing``,
      ``n_trailing_missing``, ``undef_rate``.

    Distribution stats over non-missing values (ddof=0):
      ``mean``, ``std``, ``min``, ``max``, ``skew``, ``kurt``,
      ``n_inf``, ``has_inf``.

    Constancy (separates organic from imputation-driven):
      ``is_constant`` (``std < constant_threshold``),
      ``is_imputed_constant`` (constant + ``undef_rate > 0.5``), and
      ``is_organic_constant`` (constant + low undef_rate, i.e.
      hand-crafted scalars like ``cost__c__h__w0``).

    Outlier signal:
      ``outlier_ratio`` = ``max(|x|) / median(|x| | x ≠ 0)`` over
      non-missing values. ``is_extreme_outlier`` =
      ``outlier_ratio > outlier_ratio_threshold``. Catches single-row
      blowups (e.g. division by ``EPS`` when a denominator is zero) that
      heavy-skew flags but doesn't quantify.

    Stationarity drift (only when ``include_stationarity``):
      ``mean_drift_chunks`` = ``(max - min) of per-chunk means / overall
      std``; ``std_drift_ratio_chunks`` = ``max / min`` of per-chunk
      std. ``is_non_stationary`` = either exceeds threshold. Defaults
      flag drift > 1σ between chunks or std-ratio > 100x. The data are
      split into ``n_stationarity_chunks`` chronological slices.
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
        is_imputed_constant = is_const and undef_rate > 0.5
        is_organic_constant = is_const and not is_imputed_constant

        # Outlier ratio: max|x| / median|x≠0| on non-missing values.
        if n_valid >= 2 and is_numeric:
            try:
                abs_s = s_clean.abs()
                max_abs = float(abs_s.max() or 0.0)
                nonzero = abs_s.filter(abs_s > 0)
                if len(nonzero) > 0:
                    med_abs = float(nonzero.median() or 0.0)
                else:
                    med_abs = 0.0
                if med_abs > 0:
                    outlier_ratio = max_abs / med_abs
                elif max_abs > 0:
                    outlier_ratio = float("inf")
                else:
                    outlier_ratio = 0.0
            except Exception:
                outlier_ratio = 0.0
        else:
            outlier_ratio = 0.0
        is_extreme_outlier = (
            (not math.isinf(outlier_ratio))
            and outlier_ratio > outlier_ratio_threshold
        ) or math.isinf(outlier_ratio)

        # Stationarity drift across chronological chunks. Skip for
        # constant or low-signal features (the drift metric is undefined).
        mean_drift = 0.0
        std_drift = 1.0
        is_non_stationary = False
        if (
            include_stationarity
            and is_numeric
            and not is_const
            and n >= n_stationarity_chunks * 10
            and not math.isnan(std)
            and std > 0
        ):
            chunk_size = n // n_stationarity_chunks
            chunk_means: list[float] = []
            chunk_stds: list[float] = []
            ok = True
            for i in range(n_stationarity_chunks):
                start = i * chunk_size
                size = (
                    chunk_size
                    if i < n_stationarity_chunks - 1
                    else n - start
                )
                chunk = s.slice(start, size)
                chunk_clean = (
                    chunk.drop_nulls().drop_nans()
                    if is_float
                    else chunk.drop_nulls()
                )
                if len(chunk_clean) < 2:
                    ok = False
                    break
                cm = chunk_clean.mean()
                cs = chunk_clean.std(ddof=0)
                if cm is None or cs is None:
                    ok = False
                    break
                chunk_means.append(float(cm))
                chunk_stds.append(float(cs))
            if ok and chunk_means:
                mean_drift = (max(chunk_means) - min(chunk_means)) / std
                if min(chunk_stds) > 0:
                    std_drift = max(chunk_stds) / min(chunk_stds)
                elif max(chunk_stds) > 0:
                    std_drift = float("inf")
                else:
                    std_drift = 1.0
            is_non_stationary = (
                mean_drift > mean_drift_threshold
                or (
                    not math.isinf(std_drift)
                    and std_drift > std_drift_ratio_threshold
                )
                or math.isinf(std_drift)
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
                "is_imputed_constant": is_imputed_constant,
                "is_organic_constant": is_organic_constant,
                "outlier_ratio": outlier_ratio,
                "is_extreme_outlier": is_extreme_outlier,
                "mean_drift_chunks": mean_drift,
                "std_drift_ratio_chunks": std_drift,
                "is_non_stationary": is_non_stationary,
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
                pl.col("is_imputed_constant").sum().alias("n_imputed_constant"),
                pl.col("is_organic_constant").sum().alias("n_organic_constant"),
                pl.col("is_extreme_outlier").sum().alias("n_extreme_outlier"),
                pl.col("is_non_stationary").sum().alias("n_non_stationary"),
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
      1. ``residual_nan``      — float NaN survived the impute step
      2. ``inf``               — ±inf in the column
      3. ``scattered_missing`` — pre-impute missing cells are not at
         the edges (warmup / coverage tail); flags either a real data
         anomaly or an engine bug
      4. ``all_nan``           — column is fully missing
      5. ``extreme_outlier``   — max|x| / median|x≠0| > threshold
         (catches single-row blowups like denominator-near-zero)
      6. ``imputed_constant``  — constant value but driven by impute
         (vs. organic constants like cost__c which are by design)
      7. ``non_stationary``    — mean drifts > 1σ across chronological
         thirds OR std-ratio > 100×; alerts on regime-change features
      8. ``organic_constant``  — constant by design (low priority)
      9. ``high_undef``        — pre-impute undef rate > threshold
     10. ``heavy_skew``        — |skew| > threshold
    """
    return (
        health.filter(
            (pl.col("n_nan") > 0)
            | pl.col("has_inf")
            | (pl.col("null_pattern") == "scattered")
            | pl.col("mean").is_nan()
            | pl.col("is_extreme_outlier")
            | pl.col("is_imputed_constant")
            | pl.col("is_non_stationary")
            | pl.col("is_organic_constant")
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
            .when(pl.col("is_extreme_outlier"))
            .then(pl.lit("extreme_outlier"))
            .when(pl.col("is_imputed_constant"))
            .then(pl.lit("imputed_constant"))
            .when(pl.col("is_non_stationary"))
            .then(pl.lit("non_stationary"))
            .when(pl.col("is_organic_constant"))
            .then(pl.lit("organic_constant"))
            .when(pl.col("undef_rate") > max_undef_rate)
            .then(pl.lit("high_undef"))
            .when(pl.col("skew").abs() > max_abs_skew)
            .then(pl.lit("heavy_skew"))
            .otherwise(pl.lit("other"))
            .alias("issue")
        )
        .sort(
            ["issue", "outlier_ratio", "missing_rate", "undef_rate"],
            descending=[False, True, True, True],
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
