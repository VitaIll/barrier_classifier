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
        "n_inf", "has_inf", "is_constant",
    }


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
        "n_constant", "n_with_inf", "avg_abs_skew",
    }
    # vol family has the constant feature
    vol_row = summary.filter(pl.col("family") == "vol").row(0, named=True)
    assert vol_row["n_constant"] == 1


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
    # Issue labels cover the categories we triggered
    assert "constant" in issue_labels
    assert "inf" in issue_labels
    assert "heavy_skew" in issue_labels
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
