"""Purged chronological splits + label-uniqueness / concurrency weights.

For overlapping-window barrier labels (1-min cadence with M-bar horizon),
two pieces of standard financial-ML hygiene apply:

1. **Purged splits.** Each sample ``i`` has a label interval
   ``I_i = [i+1, i+M]`` (the future bars its label looks at). Any
   train sample whose label interval overlaps the validation / test
   label intervals leaks future information and must be removed.
   In addition, an *embargo* of ``M`` rows after the end of a test
   fold suppresses immediate serial correlation between the tail of
   the test fold and the head of the next train fold. See López de
   Prado, "Advances in Financial Machine Learning", chapter 7.

2. **Label uniqueness / concurrency.** Adjacent overlapping labels
   describe the same economic event; a 20-bar monotonic move
   contributes 20 highly redundant rows to training. Weight each
   sample by its *uniqueness*

       u_i = (1 / |I_i|) * sum_{l in I_i} 1 / c_l
       c_l = number of i for which l in I_i

   so that an isolated event gets weight 1 and a sample fully
   redundant with its neighbours gets weight close to 1/M. See the
   MlFinLab implementation for the canonical formula.

Both utilities operate on row indices alone — they are cadence-agnostic
and do not need timestamps. They assume rows are sorted chronologically.

Public API
----------

``label_intervals(n_rows, M, bar_stride=1)`` -> array of (start, end_exclusive)
``concurrency_counts(intervals, n_bars)`` -> per-future-bar concurrency vector
``sample_uniqueness(intervals, concurrency)`` -> per-sample uniqueness in (0, 1]
``compute_uniqueness_weights(n_rows, M, bar_stride=1, normalize=True)`` ->
    convenience wrapper returning the per-sample weight array
``purged_train_indices(train_idx, test_idx, M, bar_stride=1, embargo_rows=None)``
    -> ndarray of train rows surviving purging + embargo
``purged_chronological_splits(n_rows, val_frac, test_frac, M, bar_stride=1,
    embargo_rows=None)`` -> (train_idx, val_idx, test_idx) tuples
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def label_intervals(
    n_rows: int, M: int, *, bar_stride: int = 1
) -> np.ndarray:
    """Return the ``(n_rows, 2)`` array of label intervals.

    Row ``i``'s label looks at raw bars ``[i*bar_stride + 1, i*bar_stride + M]``
    (inclusive). The returned array stores ``[start_inclusive, end_exclusive]``
    in raw-bar units so it can be used directly with ``np.intersect1d`` /
    range arithmetic.
    """
    if M <= 0:
        raise ValueError(f"M must be > 0, got {M}")
    if bar_stride <= 0:
        raise ValueError(f"bar_stride must be > 0, got {bar_stride}")
    starts = np.arange(n_rows, dtype=np.int64) * int(bar_stride) + 1
    ends = starts + int(M)  # exclusive
    return np.column_stack([starts, ends])


def concurrency_counts(intervals: np.ndarray, n_bars: Optional[int] = None) -> np.ndarray:
    """Number of overlapping samples per future bar.

    ``c[l] = #{i : l in I_i} = #{i : starts[i] <= l < ends[i]}``.

    Linear-time difference-array implementation.
    """
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals must have shape (n, 2)")
    if n_bars is None:
        n_bars = int(intervals[:, 1].max()) if len(intervals) else 0
    if n_bars <= 0:
        return np.zeros(0, dtype=np.int64)
    # Reject out-of-range inputs explicitly rather than silently clamping —
    # negative start or end > n_bars means the caller passed inconsistent
    # data and a clamped result would corrupt downstream concurrency math.
    if len(intervals):
        if int(intervals[:, 0].min()) < 0:
            raise ValueError(
                f"intervals[:, 0].min()={int(intervals[:, 0].min())} < 0"
            )
        if int(intervals[:, 1].max()) > n_bars:
            raise ValueError(
                f"intervals[:, 1].max()={int(intervals[:, 1].max())} "
                f"> n_bars={n_bars}"
            )
    diff = np.zeros(n_bars + 1, dtype=np.int64)
    starts = intervals[:, 0].astype(np.int64)
    ends = intervals[:, 1].astype(np.int64)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    return np.cumsum(diff)[:n_bars]


def sample_uniqueness(
    intervals: np.ndarray, concurrency: np.ndarray
) -> np.ndarray:
    """Per-sample uniqueness ``u_i = mean(1 / c_l) over l in I_i``.

    ``c_l`` should be the concurrency vector returned by
    ``concurrency_counts(intervals)``. Output is in ``(0, 1]``: an
    isolated sample whose interval is shared with no one gets 1.0; a
    sample whose every future bar is shared with M-1 others approaches
    1/M.
    """
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals must have shape (n, 2)")
    n_bars = len(concurrency)
    # Use a prefix sum of 1/c so we can compute mean-on-interval in O(1)
    # per sample. We treat concurrency==0 as 1 to avoid division blow-ups
    # for empty/invalid bars; those bars contribute 0 to the active rows
    # that include them in practice (active rows always have c >= 1).
    inv = np.where(concurrency > 0, 1.0 / np.maximum(concurrency, 1), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    out = np.zeros(len(intervals), dtype=float)
    starts = np.clip(intervals[:, 0], 0, n_bars).astype(np.int64)
    ends = np.clip(intervals[:, 1], 0, n_bars).astype(np.int64)
    lengths = ends - starts
    valid = lengths > 0
    if valid.any():
        s = starts[valid]
        e = ends[valid]
        out[valid] = (cum[e] - cum[s]) / lengths[valid]
    return out


def compute_uniqueness_weights(
    n_rows: int,
    M: int,
    *,
    bar_stride: int = 1,
    normalize: bool = False,
) -> np.ndarray:
    """Per-row label-uniqueness weight vector ``u_i = mean(1/c_l, l in I_i)``.

    Use the default ``normalize=False`` for any overlapping-label model
    (1-min cadence, M-bar barrier labels, CPCV folds). The raw
    uniqueness for an interior 1-min row is ``~1/M``; multiplying it
    onto a CatBoost training-weight vector shrinks effective N from
    ``n`` to roughly ``n/M`` — which is the *correct* number of
    independent labels and what gradient boosting must "see" to avoid
    overfitting to redundant adjacent rows.

    ``normalize=True`` divides ``u`` by its mean, mapping every interior
    row back to ~1 with only boundary rows carrying ``>1``. That cancels
    the M-fold downweighting and lets the model treat overlapping labels
    as independent — the empirical signature is best-iteration < 50 on
    a high-frequency cache, a tiny model file, and a virtual ensemble
    whose snapshots cluster too tightly to give useful epistemic
    uncertainty. **Do not enable unless you have a specific reason and
    have verified the consequences end-to-end.** It is retained only
    for the unusual case of using uniqueness as the sole weight (with
    no barrier-distance multiplier), where rescaling to mean-1 keeps
    the per-iteration learning-rate effective.

    See ``memory/overlapping_target_refactor.md`` for the diagnosis of
    the original normalize=True regression.
    """
    intervals = label_intervals(n_rows, M, bar_stride=bar_stride)
    n_bars = int(intervals[:, 1].max()) if len(intervals) else 0
    c = concurrency_counts(intervals, n_bars=n_bars)
    u = sample_uniqueness(intervals, c)
    if normalize:
        mean_u = float(u.mean()) if len(u) else 1.0
        if mean_u > 0:
            u = u / mean_u
    return u


# ---------------------------------------------------------------------------
# Purged splits
# ---------------------------------------------------------------------------


def _interval_overlaps_any(starts: np.ndarray, ends: np.ndarray,
                            other_starts: np.ndarray, other_ends: np.ndarray) -> np.ndarray:
    """Vectorized per-test-interval overlap check.

    For each query interval ``[starts[i], ends[i])``, returns True iff it
    overlaps any "other" interval ``[other_starts[j], other_ends[j])``.

    Implementation: sort the "other" intervals by start. For a query
    ``[s, e)``, use ``np.searchsorted`` to find ``k`` = number of "other"
    intervals whose start is ``< e`` (i.e. start strictly to the left of
    the query end — touching does not overlap for half-open intervals).
    Among those first ``k`` sorted by start, overlap exists iff the
    *maximum end* is ``> s``. We precompute the cumulative-max of ends
    sorted by start so each query is answered in O(log N) by a single
    searchsorted + array lookup.

    This is the correct algorithm when "other" intervals can be
    non-contiguous (e.g. a test set that is the union of multiple disjoint
    folds). For the contiguous-fold case from ``purged_chronological_splits``,
    it produces the same result as the simpler bounding-box check.
    """
    if len(other_starts) == 0:
        return np.zeros(len(starts), dtype=bool)
    order = np.argsort(other_starts, kind="mergesort")
    os_sorted = other_starts[order].astype(np.int64)
    oe_sorted = other_ends[order].astype(np.int64)
    cummax_end = np.maximum.accumulate(oe_sorted)
    # side="left" so other_start == ends[i] is NOT counted (half-open).
    k = np.searchsorted(os_sorted, ends, side="left")
    out = np.zeros(len(starts), dtype=bool)
    has_any = k > 0
    idx = np.where(has_any)[0]
    if len(idx) > 0:
        out[idx] = cummax_end[k[idx] - 1] > starts[idx]
    return out


def purged_train_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    M: int,
    *,
    bar_stride: int = 1,
    embargo_rows: Optional[int] = None,
) -> np.ndarray:
    """Drop any train rows whose label interval overlaps a test row's
    label interval. Additionally embargo ``embargo_rows`` rows after the
    end of the test region (default: ``M // bar_stride``).

    ``train_idx``, ``test_idx`` are sorted row-index arrays.

    Practically: at 1-min cadence with M=20, an out-of-sample test fold
    starting at row T has label intervals ``[T+1, T+20], [T+2, T+21], ...``;
    a train row at index ``T-1`` has label interval ``[T, T+19]`` which
    overlaps the first test interval — purge it. The full train tail
    purged this way has size ``M - 1`` immediately before the test fold.
    """
    if embargo_rows is None:
        embargo_rows = int(M) // int(bar_stride)
    embargo_rows = int(embargo_rows)

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if len(test_idx) == 0:
        return train_idx
    # Per-row label interval is [i*bar_stride + 1, i*bar_stride + 1 + M) by
    # the definition in :func:`label_intervals`; compute directly to avoid
    # allocating intervals for every index up to max(idx).
    bs = int(bar_stride)
    train_starts = train_idx * bs + 1
    train_ends = train_starts + int(M)
    test_starts = test_idx * bs + 1
    test_ends = test_starts + int(M)

    overlap = _interval_overlaps_any(train_starts, train_ends, test_starts, test_ends)

    # Embargo: drop train rows in the row-index neighborhood of EVERY test
    # row (within ``embargo_rows`` indices AFTER each test row). The
    # before-side is already handled by the overlap test (train rows whose
    # label window spills into a test region). This generalizes to disjoint
    # multi-fold test sets where each fold has its own embargo region.
    if embargo_rows > 0:
        sorted_test = np.sort(test_idx)
        k = np.searchsorted(sorted_test, train_idx, side="left")
        prev_test = np.where(k > 0, sorted_test[np.maximum(k - 1, 0)], -1 - embargo_rows)
        in_embargo = (
            (prev_test >= 0)
            & (train_idx > prev_test)
            & (train_idx - prev_test <= embargo_rows)
        )
    else:
        in_embargo = np.zeros(len(train_idx), dtype=bool)
    keep = (~overlap) & (~in_embargo)
    return train_idx[keep]


def purged_chronological_splits(
    n_rows: int,
    *,
    val_frac: float,
    test_frac: float,
    M: int,
    bar_stride: int = 1,
    embargo_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train / val / test partition with purging + embargo.

    Rows are assigned to ``train`` first, then ``val``, then ``test`` in
    chronological order. The train tail near the val boundary, and the
    val tail near the test boundary, are purged of any sample whose label
    interval overlaps the next region. An additional embargo of
    ``embargo_rows`` rows is dropped from the train head after each test
    region — set ``embargo_rows=0`` to disable.

    Defaults: ``embargo_rows = M // bar_stride``.

    Returns ``(train_idx, val_idx, test_idx)`` as sorted int64 arrays.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    if embargo_rows is None:
        embargo_rows = int(M) // int(bar_stride)

    n_test = int(round(n_rows * float(test_frac)))
    n_val = int(round(n_rows * float(val_frac)))
    n_train = n_rows - n_val - n_test
    if n_train <= 0:
        raise ValueError("computed n_train <= 0; reduce val_frac/test_frac")

    train_idx = np.arange(n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64)

    train_idx = purged_train_indices(
        train_idx, val_idx, M, bar_stride=bar_stride, embargo_rows=embargo_rows
    )
    train_idx = purged_train_indices(
        train_idx, test_idx, M, bar_stride=bar_stride, embargo_rows=embargo_rows
    )
    val_idx = purged_train_indices(
        val_idx, test_idx, M, bar_stride=bar_stride, embargo_rows=embargo_rows
    )
    return train_idx, val_idx, test_idx
