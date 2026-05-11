"""Tests for purged-split + label-uniqueness sampling utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analytics.sampling import (
    compute_uniqueness_weights,
    concurrency_counts,
    label_intervals,
    purged_chronological_splits,
    purged_train_indices,
    sample_uniqueness,
)


def test_label_intervals_layout():
    iv = label_intervals(5, M=3, bar_stride=1)
    # Row i looks at [i+1, i+3] inclusive  ==>  [i+1, i+4) exclusive
    assert iv.shape == (5, 2)
    assert iv[0, 0] == 1 and iv[0, 1] == 4
    assert iv[4, 0] == 5 and iv[4, 1] == 8


def test_concurrency_at_1min_cadence_peaks_at_M():
    """At 1-min cadence with M=20, interior bars are covered by exactly M
    samples (the M consecutive rows that include that bar in their horizon).
    """
    n_rows = 1000
    M = 20
    iv = label_intervals(n_rows, M=M, bar_stride=1)
    c = concurrency_counts(iv)
    # Interior bars (far from edges) should all have concurrency == M
    interior = c[M:-M]
    assert (interior == M).all(), (
        f"interior concurrency should be {M}; got unique values "
        f"{np.unique(interior)}"
    )


def test_uniqueness_at_1min_cadence_approaches_1_over_M():
    n_rows = 500
    M = 10
    iv = label_intervals(n_rows, M=M, bar_stride=1)
    c = concurrency_counts(iv)
    u = sample_uniqueness(iv, c)
    # Interior samples should each have uniqueness ≈ 1/M (every future bar
    # is shared with M-1 others)
    interior = u[2 * M : -2 * M]
    assert np.allclose(interior, 1.0 / M, atol=1e-9), (
        f"interior uniqueness should be {1.0/M}; got {np.unique(np.round(interior, 9))}"
    )


def test_uniqueness_at_boundary_cadence_is_1():
    """When samples are non-overlapping (bar_stride==M), every sample has
    its own private future window, so uniqueness must be exactly 1.
    """
    n_rows = 100
    M = 5
    iv = label_intervals(n_rows, M=M, bar_stride=M)
    c = concurrency_counts(iv)
    u = sample_uniqueness(iv, c)
    assert np.allclose(u, 1.0, atol=1e-12)


def test_compute_uniqueness_weights_normalized_mean_1():
    w = compute_uniqueness_weights(500, M=10, bar_stride=1, normalize=True)
    # After normalization, weights should average to 1 — preserves the
    # sum-of-weights effective sample size CatBoost uses.
    assert math.isclose(float(w.mean()), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_purged_train_indices_drops_overlapping_tail():
    """Train tail samples whose label window spills into the test region
    must be purged. With M=10 and bar_stride=1, train rows whose
    intervals overlap the first test row's interval are removed."""
    M = 10
    train = np.arange(0, 100, dtype=np.int64)
    test = np.arange(100, 130, dtype=np.int64)
    kept = purged_train_indices(train, test, M=M, bar_stride=1, embargo_rows=0)
    # Train row i has interval [i+1, i+10]. Test starts at row 100 with
    # interval [101, 110]. The earliest train row whose interval reaches
    # row 101 has i+10 >= 101, i.e. i >= 91. So rows 91..99 must be dropped.
    assert kept.max() < 91
    # And nothing below 91 should be dropped
    assert (kept == np.arange(0, 91)).all()


def test_purged_train_indices_applies_embargo_after_test():
    """If train rows exist after the test region, embargo_rows immediate
    successors must be dropped."""
    M = 5
    test = np.arange(50, 60, dtype=np.int64)
    train_post = np.arange(60, 80, dtype=np.int64)
    kept = purged_train_indices(
        train_post, test, M=M, bar_stride=1, embargo_rows=5
    )
    # Test ends at row 59; embargo drops rows 60..64. Rows 65..79 survive.
    assert kept.min() == 65
    assert (kept == np.arange(65, 80)).all()


def test_purged_chronological_splits_total_count_minus_purged():
    n = 1000
    M = 20
    tr, va, te = purged_chronological_splits(
        n, val_frac=0.2, test_frac=0.2, M=M, bar_stride=1, embargo_rows=M
    )
    # Train and val are reduced by purging+embargo near the val and test
    # boundaries; the train+val+test set must be <= n.
    assert len(tr) + len(va) + len(te) <= n
    # The sets must be disjoint and chronologically ordered.
    assert tr.max() < va.min()
    assert va.max() < te.min()
    # Test fold size is approximately n * test_frac.
    assert math.isclose(len(te), int(round(n * 0.2)))
