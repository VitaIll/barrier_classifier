"""Tests for src.analytics.thresholds — threshold derivation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.thresholds import (
    derive_conditional_unc_cap,
    derive_top_q_threshold,
    summarize_gate_overlap,
)


def test_derive_top_q_threshold_returns_quantile():
    p = np.linspace(0.0, 1.0, 101)  # 0, 0.01, ..., 1.0
    # 85th percentile of evenly spaced [0, 1] is 0.85 exactly
    assert derive_top_q_threshold(p, q=0.85) == pytest.approx(0.85, abs=1e-9)
    # Default arg
    assert derive_top_q_threshold(p) == pytest.approx(0.85, abs=1e-9)
    # Median
    assert derive_top_q_threshold(p, q=0.5) == pytest.approx(0.5, abs=1e-9)


def test_derive_top_q_threshold_rejects_empty_and_out_of_range_q():
    with pytest.raises(ValueError, match="empty"):
        derive_top_q_threshold(np.array([]))
    with pytest.raises(ValueError, match="q must be in"):
        derive_top_q_threshold(np.array([0.1, 0.2]), q=0.0)
    with pytest.raises(ValueError, match="q must be in"):
        derive_top_q_threshold(np.array([0.1, 0.2]), q=1.0)


def test_derive_conditional_unc_cap_uses_only_passing_rows():
    """Cap should be the q-quantile of unc among rows where p >= threshold.

    Setup: 10 rows. p increases linearly with row index. unc is *high* for
    low-p rows and *low* for high-p rows — so the conditional median (on
    high-p rows) is much smaller than the unconditional median.
    """
    p = np.linspace(0.0, 1.0, 10)              # 0.0, 0.111, ..., 1.0
    unc = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.05, 0.04, 0.03, 0.02, 0.01])

    # Conditional on top-30% (q_threshold=0.7), passing rows have unc in {0.05, 0.04, 0.03, 0.02, 0.01}
    # Wait — p >= 0.7 is rows 7, 8, 9 (p = 7/9, 8/9, 9/9 = 0.777..., 0.888..., 1.0)
    # so unc among passing = {0.03, 0.02, 0.01}; median = 0.02
    cap = derive_conditional_unc_cap(p, unc, p_threshold=0.7, q=0.5)
    assert cap == pytest.approx(0.02, abs=1e-9)


def test_derive_conditional_unc_cap_quartile_smaller_than_median():
    """Lower q gives stricter (smaller) cap — monotone."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 10_000)
    unc = rng.uniform(0, 0.1, 10_000)

    median_cap = derive_conditional_unc_cap(p, unc, p_threshold=0.5, q=0.5)
    quartile_cap = derive_conditional_unc_cap(p, unc, p_threshold=0.5, q=0.25)
    assert quartile_cap < median_cap


def test_derive_conditional_unc_cap_raises_when_no_passing_rows():
    p = np.array([0.1, 0.2, 0.3])
    unc = np.array([0.01, 0.02, 0.03])
    with pytest.raises(ValueError, match="no training rows pass"):
        derive_conditional_unc_cap(p, unc, p_threshold=0.99, q=0.5)


def test_derive_conditional_unc_cap_skips_nan_unc():
    """NaN MI rows must be excluded from the cohort — otherwise the
    quantile would be polluted with NaN and the result could itself be NaN."""
    p = np.array([0.9, 0.9, 0.9, 0.9])
    unc = np.array([0.01, 0.02, np.nan, 0.04])
    # Cohort with finite MI: {0.01, 0.02, 0.04}, median = 0.02
    cap = derive_conditional_unc_cap(p, unc, p_threshold=0.5, q=0.5)
    assert cap == pytest.approx(0.02, abs=1e-9)


def test_summarize_gate_overlap_consistency():
    p = np.array([0.1, 0.5, 0.9, 0.95])
    unc = np.array([0.05, 0.005, 0.05, 0.005])
    info = summarize_gate_overlap(p, unc, p_threshold=0.5, unc_cap=0.01)
    # score gate passes rows 1,2,3 → 3
    assert info["n_score_pass"] == 3
    # unc gate passes rows 1, 3 → 2
    assert info["n_unc_pass"] == 2
    # joint pass: rows where BOTH p>=0.5 AND unc<=0.01 → rows 1, 3
    assert info["n_joint_pass"] == 2
    assert info["unc_pass_rate_given_score"] == pytest.approx(2 / 3, abs=1e-9)
