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
    diff = np.zeros(n_bars + 1, dtype=np.int64)
    starts = np.clip(intervals[:, 0], 0, n_bars).astype(np.int64)
    ends = np.clip(intervals[:, 1], 0, n_bars).astype(np.int64)
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
    normalize: bool = True,
) -> np.ndarray:
    """Convenience: per-row label-uniqueness weight vector.

    With ``normalize=True`` the weights are rescaled so their mean is 1
    — this preserves the effective sample size that gradient boosting
    treats as ``sum(weight)``. Without normalization, an M=20 1-min
    dataset would have a mean weight near 1/M and most rows would carry
    near-zero importance from CatBoost's perspective.
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
    """Vectorized overlap check: returns boolean mask of shape (len(starts),).

    For each ``i``, returns True iff ``[starts[i], ends[i])`` overlaps any
    ``[other_starts[j], other_ends[j])``. Implementation: sort the "other"
    intervals by start and, for each query, binary-search for the first
    other interval whose start is < query.end; check whether that other's
    end is > query.start (sufficient because all intervals are aligned).
    Since the other intervals here are themselves overlapping (consecutive
    1-min samples), we use the simpler bound check that overlap with the
    *union* equals overlap with the union's bounding interval expanded
    by an embargo on either side.
    """
    if len(other_starts) == 0:
        return np.zeros(len(starts), dtype=bool)
    union_lo = int(other_starts.min())
    union_hi = int(other_ends.max())
    return (ends > union_lo) & (starts < union_hi)


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
    train_intervals = label_intervals(int(train_idx.max()) + 1, M, bar_stride=bar_stride)
    test_intervals = label_intervals(int(test_idx.max()) + 1, M, bar_stride=bar_stride)
    train_starts = train_intervals[train_idx, 0]
    train_ends = train_intervals[train_idx, 1]
    test_starts = test_intervals[test_idx, 0]
    test_ends = test_intervals[test_idx, 1]

    overlap = _interval_overlaps_any(train_starts, train_ends, test_starts, test_ends)

    # Embargo: drop train rows immediately AFTER the test region (within
    # ``embargo_rows`` row-indices of the last test row). The before-side
    # is already handled by the overlap test (train rows whose label window
    # spills into the test region).
    if embargo_rows > 0:
        test_max = int(test_idx.max())
        in_embargo = (train_idx > test_max) & (train_idx <= test_max + embargo_rows)
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
