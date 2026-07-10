"""Tests for the post-pipeline inspection helpers in observability.py.

Marked ``framework`` so they always run regardless of the active step
filter — these are general-purpose utilities, not tied to a step port.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from src.features.observability import (
    compute_feature_health,
    flag_issues,
    monthly_target_balance,
    summarize_by_family,
)

pytestmark = pytest.mark.framework


@pytest.fixture
def synthetic_health_frame() -> pl.DataFrame:
    """Frame with one feature per intentional issue type, plus matching
    ``undef__`` flags so undef_rate has something to read."""
    rng = np.random.default_rng(42)
    n = 100

    return pl.DataFrame(
        {
            # Healthy: normal feature with stable distribution.
            "ret__mean__f__w60": rng.normal(0.0, 0.001, n),
            # Constant — std should be 0, flagged as constant.
            "vol__broken__f__w20": np.full(n, 7.0),
            # Has inf — should fire has_inf.
            "logp__bad__f__w0": np.concatenate([rng.normal(0, 1, n - 1), [np.inf]]),
            # Heavy skew (lognormal-ish).
            "liq__skewed__f__w30": np.exp(rng.normal(0.0, 5.0, n)),
            # 60% imputed (high undef_rate).
            "ofi__sparse__f__w0": rng.normal(0, 1, n),
            # Matching undef flags
            "undef__ofi__sparse__f__w0": np.where(rng.random(n) < 0.6, 1, 0).astype(np.int8),
            "undef__ret__mean__f__w60": np.zeros(n, dtype=np.int8),
        }
    )


def test_compute_feature_health_shape(synthetic_health_frame):
    feature_cols = [
        "ret__mean__f__w60",
        "vol__broken__f__w20",
        "logp__bad__f__w0",
        "liq__skewed__f__w30",
        "ofi__sparse__f__w0",
    ]
    health = compute_feature_health(synthetic_health_frame, feature_cols)
    assert len(health) == 5
    assert set(health.columns) >= {
        "name", "family", "n", "n_valid",
        "n_null", "n_nan", "missing_rate",
        "null_pattern", "n_leading_missing", "n_trailing_missing",
        "undef_rate",
        "mean", "std", "min", "max", "skew", "kurt",
        "n_inf", "has_inf",
        "is_constant", "is_imputed_constant", "is_organic_constant",
        "outlier_ratio", "is_extreme_outlier",
        "mean_drift_chunks", "std_drift_ratio_chunks", "is_non_stationary",
    }


def test_imputed_constant_distinguished_from_organic_constant():
    """Constant-by-imputation must be flagged separately from a hand-
    crafted scalar like ``cost__c__h__w0`` so the ML team can drop the
    former and accept the latter."""
    n = 100
    df = pl.DataFrame(
        {
            # Imputed constant: undef flag fired everywhere.
            "ret__dead__f__w0": pl.Series([0.0] * n, dtype=pl.Float64),
            "undef__ret__dead__f__w0": pl.Series([1] * n, dtype=pl.Int8),
            # Organic constant: no undef flag (column was always 1.0).
            "cost__c__h__w0": pl.Series([1.0] * n, dtype=pl.Float64),
        }
    )
    health = compute_feature_health(df, ["ret__dead__f__w0", "cost__c__h__w0"])
    rows = {r["name"]: r for r in health.iter_rows(named=True)}

    assert rows["ret__dead__f__w0"]["is_imputed_constant"] is True
    assert rows["ret__dead__f__w0"]["is_organic_constant"] is False
    assert rows["cost__c__h__w0"]["is_imputed_constant"] is False
    assert rows["cost__c__h__w0"]["is_organic_constant"] is True


def test_extreme_outlier_detection():
    """A single-row blowup (e.g. division by EPS) sits inside an
    otherwise tame distribution; max/median ratio surfaces it even
    when skew alone wouldn't."""
    n = 1000
    rng = np.random.default_rng(42)
    vals = rng.normal(0.0, 1.0, n)
    vals[7] = 1.0e22  # the kind of blowup we saw with oi__chg_pct
    df = pl.DataFrame({"oi__chg_pct__f__w0": pl.Series(vals, dtype=pl.Float64)})
    health = compute_feature_health(
        df, ["oi__chg_pct__f__w0"], outlier_ratio_threshold=1.0e6
    )
    row = health.row(0, named=True)
    assert row["is_extreme_outlier"] is True
    assert row["outlier_ratio"] > 1.0e6


def test_non_stationarity_detection():
    """Step change in mean across thirds should fire non_stationary."""
    n = 600
    rng = np.random.default_rng(7)
    chunk_size = n // 3
    vals = np.concatenate(
        [
            rng.normal(0.0, 1.0, chunk_size),
            rng.normal(0.0, 1.0, chunk_size),
            rng.normal(5.0, 1.0, n - 2 * chunk_size),  # step shift in last third
        ]
    )
    df = pl.DataFrame({"ret__regime__f__w0": pl.Series(vals, dtype=pl.Float64)})
    health = compute_feature_health(
        df,
        ["ret__regime__f__w0"],
        mean_drift_threshold=1.0,
        std_drift_ratio_threshold=100.0,
    )
    row = health.row(0, named=True)
    assert row["is_non_stationary"] is True
    assert row["mean_drift_chunks"] > 1.0


def test_stationarity_skipped_for_constant_features():
    """Constant features have zero std → drift is undefined; we should
    not flag them as non_stationary (they get the constant flag instead)."""
    df = pl.DataFrame(
        {"x__const__f__w0": pl.Series([3.14] * 100, dtype=pl.Float64)}
    )
    health = compute_feature_health(df, ["x__const__f__w0"])
    row = health.row(0, named=True)
    assert row["is_constant"] is True
    assert row["is_non_stationary"] is False


def test_compute_feature_health_distinguishes_null_from_nan():
    """polars null and float NaN are distinct kinds of missingness;
    the health frame must report them separately so an engine bug that
    emits NaN where null is expected does not silently slip through."""
    df = pl.DataFrame(
        {
            "ret__mixed__f__w0": pl.Series(
                [1.0, None, float("nan"), 4.0, 5.0],
                dtype=pl.Float64,
            ),
        }
    )
    health = compute_feature_health(df, ["ret__mixed__f__w0"])
    row = health.row(0, named=True)
    assert row["n_null"] == 1
    assert row["n_nan"] == 1
    assert row["missing_rate"] == pytest.approx(0.4)
    # n_valid is non-null AND non-NaN
    assert row["n_valid"] == 3


def test_null_pattern_classification():
    """Each pattern bucket lights up on the right input."""
    n = 10
    cases = {
        "clean": [1.0] * n,
        "all_missing": [None] * n,
        "leading": [None, None, None] + [1.0] * (n - 3),
        "trailing": [1.0] * (n - 2) + [None, None],
        "edge": [None] + [1.0] * (n - 2) + [None],
        "scattered": [1.0, 1.0, None, 1.0, None, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    for expected, vals in cases.items():
        df = pl.DataFrame({"ret__x__f__w0": pl.Series(vals, dtype=pl.Float64)})
        health = compute_feature_health(df, ["ret__x__f__w0"])
        assert health.row(0, named=True)["null_pattern"] == expected, (
            f"expected {expected} for input {vals}"
        )


def test_null_pattern_counts_leading_trailing():
    df = pl.DataFrame(
        {
            "ret__x__f__w0": pl.Series(
                [None, None, 1.0, 2.0, 3.0, 4.0, None],
                dtype=pl.Float64,
            ),
        }
    )
    row = compute_feature_health(df, ["ret__x__f__w0"]).row(0, named=True)
    assert row["null_pattern"] == "edge"
    assert row["n_leading_missing"] == 2
    assert row["n_trailing_missing"] == 1


def test_flag_issues_priorities_residual_nan_first():
    """If a feature has both NaN and high undef, the NaN issue wins."""
    df = pl.DataFrame(
        {
            "ret__bad__f__w0": pl.Series(
                [float("nan"), 1.0, 2.0, 3.0],
                dtype=pl.Float64,
            ),
            "undef__ret__bad__f__w0": pl.Series([1, 0, 0, 0], dtype=pl.Int8),
        }
    )
    health = compute_feature_health(df, ["ret__bad__f__w0"])
    issues = flag_issues(health, max_undef_rate=0.0)
    assert len(issues) == 1
    assert issues.row(0, named=True)["issue"] == "residual_nan"


def test_flag_issues_full_priority_order():
    """Enumerate the full priority ladder of :func:`flag_issues`. One
    column per issue class so the first-match-wins logic shows up.

    Order under test (highest first):
      residual_nan > inf > scattered_missing > all_missing >
      extreme_outlier > imputed_constant > non_stationary >
      organic_constant > high_undef > heavy_skew

    For each column we provide just-enough inputs so that exactly that
    priority bucket fires (and lower-priority buckets may also be
    triggered by side effect, but the test asserts the higher-priority
    label is the one emitted).
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n = 600  # large enough for stationarity chunks
    chunk = n // 3

    # 1. residual_nan: contains float NaN, plus high undef rate
    nan_col = np.concatenate([[float("nan")], rng.normal(0.0, 1.0, n - 1)])
    nan_undef = np.zeros(n, dtype=np.int8)
    nan_undef[0] = 1

    # 2. inf: contains +inf
    inf_col = rng.normal(0.0, 1.0, n)
    inf_col[10] = float("inf")

    # 3. scattered_missing: a single hole in the middle
    scattered_col = rng.normal(0.0, 1.0, n).tolist()
    scattered_col[100] = None

    # 4. all_missing: every cell null
    all_missing_col = [None] * n

    # 5. extreme_outlier: one row blow-up
    extreme_col = rng.normal(0.0, 1.0, n)
    extreme_col[20] = 1.0e12

    # 6. imputed_constant: constant col + matching undef flag everywhere
    imp_const_col = np.full(n, 7.5)
    imp_const_undef = np.ones(n, dtype=np.int8)

    # 7. non_stationary: step shift across thirds (no inf / no NaN)
    ns_col = np.concatenate(
        [
            rng.normal(0.0, 1.0, chunk),
            rng.normal(0.0, 1.0, chunk),
            rng.normal(10.0, 1.0, n - 2 * chunk),
        ]
    )

    # 8. organic_constant: constant col with NO undef flag
    organic_col = np.full(n, 3.14)

    # 9. high_undef: high undef rate, but column itself is NOT constant
    # and has no nan / inf / scattered
    high_undef_col = rng.normal(0.0, 1.0, n)
    high_undef_flags = np.where(rng.random(n) < 0.8, 1, 0).astype(np.int8)

    # 10. heavy_skew: lognormal tail, no other issues
    skew_col = np.exp(rng.normal(0.0, 3.0, n))

    df = pl.DataFrame(
        {
            "ret__nan__f__w0": pl.Series(nan_col, dtype=pl.Float64),
            "undef__ret__nan__f__w0": pl.Series(nan_undef, dtype=pl.Int8),
            "ret__inf__f__w0": pl.Series(inf_col, dtype=pl.Float64),
            "ret__scattered__f__w0": pl.Series(scattered_col, dtype=pl.Float64),
            "ret__allmiss__f__w0": pl.Series(all_missing_col, dtype=pl.Float64),
            "ret__outlier__f__w0": pl.Series(extreme_col, dtype=pl.Float64),
            "ret__impconst__f__w0": pl.Series(imp_const_col, dtype=pl.Float64),
            "undef__ret__impconst__f__w0": pl.Series(imp_const_undef, dtype=pl.Int8),
            "ret__nonstat__f__w0": pl.Series(ns_col, dtype=pl.Float64),
            "ret__orgconst__f__w0": pl.Series(organic_col, dtype=pl.Float64),
            "ret__highundef__f__w0": pl.Series(high_undef_col, dtype=pl.Float64),
            "undef__ret__highundef__f__w0": pl.Series(high_undef_flags, dtype=pl.Int8),
            "ret__heavyskew__f__w0": pl.Series(skew_col, dtype=pl.Float64),
        }
    )
    feature_cols = [
        "ret__nan__f__w0",
        "ret__inf__f__w0",
        "ret__scattered__f__w0",
        "ret__allmiss__f__w0",
        "ret__outlier__f__w0",
        "ret__impconst__f__w0",
        "ret__nonstat__f__w0",
        "ret__orgconst__f__w0",
        "ret__highundef__f__w0",
        "ret__heavyskew__f__w0",
    ]
    health = compute_feature_health(
        df,
        feature_cols,
        outlier_ratio_threshold=1.0e6,
        mean_drift_threshold=1.0,
        std_drift_ratio_threshold=100.0,
    )
    issues = flag_issues(health, max_undef_rate=0.5, max_abs_skew=2.0)
    label_by_name = dict(zip(issues["name"].to_list(), issues["issue"].to_list()))

    expected = {
        "ret__nan__f__w0": "residual_nan",
        "ret__inf__f__w0": "inf",
        "ret__scattered__f__w0": "scattered_missing",
        "ret__allmiss__f__w0": "all_missing",
        "ret__outlier__f__w0": "extreme_outlier",
        "ret__impconst__f__w0": "imputed_constant",
        "ret__nonstat__f__w0": "non_stationary",
        "ret__orgconst__f__w0": "organic_constant",
        "ret__highundef__f__w0": "high_undef",
        "ret__heavyskew__f__w0": "heavy_skew",
    }
    for col, want in expected.items():
        got = label_by_name.get(col)
        assert got == want, (
            f"{col}: expected priority {want!r} but flag_issues emitted {got!r}"
        )


def test_flag_issues_picks_up_scattered_missing():
    df = pl.DataFrame(
        {
            "ret__x__f__w0": pl.Series(
                [1.0, 1.0, None, 1.0, 1.0, None, 1.0, 1.0],
                dtype=pl.Float64,
            ),
        }
    )
    health = compute_feature_health(df, ["ret__x__f__w0"])
    issues = flag_issues(health)
    issue_labels = set(issues["issue"].to_list())
    assert "scattered_missing" in issue_labels


def test_compute_feature_health_detects_constant(synthetic_health_frame):
    health = compute_feature_health(synthetic_health_frame, ["vol__broken__f__w20"])
    row = health.row(0, named=True)
    assert row["is_constant"] is True
    assert row["std"] == 0.0


def test_compute_feature_health_detects_inf(synthetic_health_frame):
    health = compute_feature_health(synthetic_health_frame, ["logp__bad__f__w0"])
    row = health.row(0, named=True)
    assert row["has_inf"] is True


def test_compute_feature_health_reads_undef_flag(synthetic_health_frame):
    health = compute_feature_health(synthetic_health_frame, ["ofi__sparse__f__w0"])
    row = health.row(0, named=True)
    # Mean of the Int8 flag is the rate; we injected ~60% nonzero.
    assert 0.4 < row["undef_rate"] < 0.8


def test_family_extracted_from_first_double_underscore(synthetic_health_frame):
    health = compute_feature_health(
        synthetic_health_frame,
        ["ret__mean__f__w60", "vol__broken__f__w20", "ofi__sparse__f__w0"],
    )
    fams = health["family"].to_list()
    assert fams == ["ret", "vol", "ofi"]


def test_summarize_by_family_aggregates(synthetic_health_frame):
    health = compute_feature_health(
        synthetic_health_frame,
        [
            "ret__mean__f__w60",
            "vol__broken__f__w20",
            "logp__bad__f__w0",
            "liq__skewed__f__w30",
            "ofi__sparse__f__w0",
        ],
    )
    summary = summarize_by_family(health)
    # Five distinct families
    assert len(summary) == 5
    assert set(summary.columns) >= {
        "family", "n_features", "avg_undef_rate", "max_undef_rate",
        "max_missing_rate", "n_features_with_nan", "n_scattered",
        "n_imputed_constant", "n_organic_constant", "n_extreme_outlier",
        "n_non_stationary", "n_with_inf", "avg_abs_skew",
    }
    # vol family has the constant feature (organic — no undef flag in fixture)
    vol_row = summary.filter(pl.col("family") == "vol").row(0, named=True)
    assert vol_row["n_organic_constant"] == 1


def test_flag_issues_picks_up_all_categories(synthetic_health_frame):
    health = compute_feature_health(
        synthetic_health_frame,
        [
            "ret__mean__f__w60",
            "vol__broken__f__w20",
            "logp__bad__f__w0",
            "liq__skewed__f__w30",
            "ofi__sparse__f__w0",
        ],
    )
    issues = flag_issues(health, max_undef_rate=0.5, max_abs_skew=2.0)
    issue_names = set(issues["name"].to_list())
    issue_labels = set(issues["issue"].to_list())
    # The healthy feature must NOT appear
    assert "ret__mean__f__w60" not in issue_names
    # The four bad features must all be flagged
    assert "vol__broken__f__w20" in issue_names
    assert "logp__bad__f__w0" in issue_names
    assert "liq__skewed__f__w30" in issue_names
    assert "ofi__sparse__f__w0" in issue_names
    # Issue labels cover the categories we triggered. The constant
    # vol__broken feature has no undef flag in the fixture so it lands
    # under organic_constant; inf and high_undef are direct triggers.
    # liq__skewed is classified as extreme_outlier (lognormal max/median
    # ratio is huge), which has higher priority than heavy_skew.
    assert "organic_constant" in issue_labels
    assert "inf" in issue_labels
    assert "high_undef" in issue_labels


def test_monthly_target_balance_groups_correctly():
    rng = np.random.default_rng(7)
    n = 90
    base = datetime(2024, 1, 1)
    df = pl.DataFrame(
        {
            "ts": [base.replace(day=1, month=((i // 30) + 1)) for i in range(n)],
            "y": rng.binomial(1, 0.2, n).astype(float),
        }
    )
    monthly = monthly_target_balance(df)
    # 3 months of data
    assert len(monthly) == 3
    # Each row reports total = 30
    assert monthly["total"].to_list() == [30, 30, 30]
    # base_rate is hits / total
    for row in monthly.iter_rows(named=True):
        assert abs(row["base_rate"] - row["hits"] / row["total"]) < 1e-12
