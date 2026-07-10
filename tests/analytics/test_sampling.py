"""Tests for purged-split + label-uniqueness sampling utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analytics.sampling import (
    _interval_overlaps_any,
    compute_uniqueness_weights,
    concurrency_counts,
    label_intervals,
    purged_chronological_splits,
    purged_train_indices,
    sample_uniqueness,
)

# Belt-and-braces marker: conftest also stamps this file with
# ``analytics_bootstrap`` by path, but declaring the marker explicitly makes
# the file's selection contract visible at the top.
pytestmark = pytest.mark.analytics_bootstrap


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


def test_purged_train_indices_non_contiguous_test_set_local_purge():
    """Regression: with a non-contiguous test set [50, 200] and M=20, the
    correct purge drops only ~M rows around EACH test row (rows 31..49 around
    row 50, and 181..199 around row 200), NOT every row between them.

    The previous implementation collapsed the "other" intervals to a single
    bounding box and would have purged everything in [31, 199] — 169 rows
    instead of ~38.
    """
    M = 20
    train = np.arange(0, 300, dtype=np.int64)
    test = np.array([50, 200], dtype=np.int64)
    kept = purged_train_indices(train, test, M=M, bar_stride=1, embargo_rows=0)
    # Around test row 50: train row i has interval [i+1, i+20]; it overlaps
    # test row 50's interval [51, 70] iff i+1 < 71 and i+20 > 51, i.e.
    # 32 <= i <= 69 (and i != 50 since 50 is in test, but train passes ALL
    # of 0..299). Bound the dropped slice on each side.
    # Around test row 200: similar logic gives 182 <= i <= 219.
    # In the gap (rows 70..181) nothing should be dropped on overlap grounds.
    assert (kept == 100).sum() == 1, "row 100 (well between test points) must survive"
    assert (kept == 150).sum() == 1, "row 150 (well between test points) must survive"
    # Rough cardinality check: at most ~80 rows dropped (≈M+M each side of two
    # test rows, with overlap), not 169. (Exact count depends on how the
    # overlap math rounds — assert << 169.)
    n_dropped = len(train) - len(kept)
    assert n_dropped < 100, (
        f"expected ~40 dropped (M+M around two non-contiguous test rows), "
        f"got {n_dropped} — the bounding-box bug would drop 169"
    )
    # And it must drop a non-zero number (the overlap must actually fire).
    assert n_dropped >= 30


def test_purged_train_indices_contiguous_test_unchanged_by_new_algorithm():
    """The new per-interval overlap check must produce IDENTICAL results to
    the bounding-box approach on the contiguous test set that
    purged_chronological_splits feeds (it always passes ``np.arange(...)``)."""
    M = 20
    train = np.arange(0, 800, dtype=np.int64)
    test = np.arange(800, 950, dtype=np.int64)
    kept = purged_train_indices(train, test, M=M, bar_stride=1, embargo_rows=0)
    # Train row i has half-open label interval [i+1, i+M+1); first test row
    # 800 has interval [801, 821). Overlap iff (i+1) < 821 AND (i+M+1) > 801,
    # i.e. i+21 > 801 -> i >= 781. So rows 0..780 survive; 781..799 are dropped.
    assert kept.max() == 780
    assert (kept == np.arange(0, 781)).all()


def test_interval_overlaps_any_correctness_under_disjoint_others():
    """Direct unit test of the per-interval overlap function on a synthetic
    disjoint-others case. Train intervals straddle gaps; only those that
    actually touch one of the other intervals are reported."""
    # Queries
    q_starts = np.array([0, 30, 60, 90, 200, 500])
    q_ends = q_starts + 10
    # Disjoint others: [50, 70), [100, 120), [300, 320)
    other_starts = np.array([50, 100, 300])
    other_ends = np.array([70, 120, 320])
    out = _interval_overlaps_any(q_starts, q_ends, other_starts, other_ends)
    # [0,10): no overlap. [30,40): no overlap. [60,70): overlaps [50,70).
    # [90,100): just touches [100,120) — NO overlap (half-open).
    # [200,210): no overlap. [500,510): no overlap.
    expected = np.array([False, False, True, False, False, False])
    np.testing.assert_array_equal(out, expected)


def test_concurrency_counts_rejects_out_of_range_intervals():
    """Negative starts or ends > n_bars must raise rather than silently clip."""
    bad = np.array([[0, 5], [3, 12]])  # second end exceeds n_bars=10
    with pytest.raises(ValueError, match="n_bars"):
        concurrency_counts(bad, n_bars=10)
    bad2 = np.array([[-1, 5], [2, 8]])
    with pytest.raises(ValueError, match="< 0"):
        concurrency_counts(bad2, n_bars=10)
