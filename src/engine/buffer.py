"""Preallocated trailing bar buffer with M-grid phase discipline.

The live feature path recomputes the research pipeline on a trailing
window and takes the last row. Two properties of that window are
correctness-critical (see docs/ENGINE.md §4):

1. **Depth** — every rolling window/lag must be fully covered, plus
   burn-in for recursive (EWMA) features.
2. **Phase** — boundary-sparse kernels (the quantile family) emit values
   at ``row_index % M == 0`` *of the frame they are given*. Training
   frames start at the raw data's first bar, so the live window's first
   row must sit on the same absolute M-grid ("grid anchor") or those
   features phase-shift into a distribution the model never saw.

``BarBuffer`` guarantees both: O(1) appends into a preallocated sliding
array, and a read path (:meth:`window_frame`) that trims the head to the
first anchor-aligned row before handing the frame to the pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.engine.domain import RAW_DERIV_COLS, RAW_SPOT_COLS, Bar, DerivSnapshot
from src.engine.errors import GridError, PhaseAlignmentError

_MINUTE = pd.Timedelta(minutes=1)


def minutes_since(anchor_ts: pd.Timestamp, ts: pd.Timestamp) -> int:
    """Whole minutes from ``anchor_ts`` to ``ts`` (negative if earlier)."""
    delta = ts - anchor_ts
    total_s = delta.total_seconds()
    m = int(round(total_s / 60.0))
    if abs(total_s - m * 60.0) > 1e-6:
        raise GridError(
            f"timestamp {ts} is not on the 1-minute grid anchored at {anchor_ts}"
        )
    return m


class BarBuffer:
    """Trailing window of raw 1-minute rows (spot + optional derivatives).

    Parameters
    ----------
    capacity : int
        Maximum trailing rows retained. Must be a positive multiple of M.
    m : int
        Decision multiplier (the M of the label horizon / sparse kernels).
    anchor_ts : pd.Timestamp
        Grid anchor — the first raw bar timestamp of the frame the active
        model was trained on. Phase alignment is defined relative to it.
    with_derivatives : bool
        Whether derivative columns are stored (must match the model's
        feature contract).
    """

    def __init__(
        self,
        capacity: int,
        *,
        m: int,
        anchor_ts: pd.Timestamp,
        with_derivatives: bool = True,
        slack: int = 16_384,
    ) -> None:
        if capacity <= 0 or m <= 0:
            raise ValueError(f"capacity and m must be positive; got {capacity}, {m}")
        if capacity % m != 0:
            raise ValueError(f"capacity ({capacity}) must be a multiple of M ({m})")
        if anchor_ts.tzinfo is None:
            raise ValueError("anchor_ts must be tz-aware (UTC)")
        self._capacity = int(capacity)
        self._m = int(m)
        self._anchor = anchor_ts.tz_convert("UTC")
        self._with_deriv = bool(with_derivatives)
        self._cols: tuple[str, ...] = RAW_SPOT_COLS + (
            RAW_DERIV_COLS if with_derivatives else ()
        )
        self._alloc = self._capacity + int(slack)
        self._data = np.full((self._alloc, len(self._cols)), np.nan, dtype=np.float64)
        self._ts_min = np.zeros(self._alloc, dtype=np.int64)  # minutes since anchor
        self._synthetic = np.zeros(self._alloc, dtype=bool)
        self._start = 0  # first valid row
        self._end = 0    # one past last valid row

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def m(self) -> int:
        return self._m

    @property
    def anchor_ts(self) -> pd.Timestamp:
        return self._anchor

    @property
    def columns(self) -> tuple[str, ...]:
        return self._cols

    def __len__(self) -> int:
        return self._end - self._start

    @property
    def last_ts(self) -> Optional[pd.Timestamp]:
        if self._end == self._start:
            return None
        return self._anchor + int(self._ts_min[self._end - 1]) * _MINUTE

    @property
    def last_close(self) -> Optional[float]:
        if self._end == self._start:
            return None
        return float(self._data[self._end - 1, 3])  # close is RAW_SPOT_COLS[3]

    def aligned_len(self) -> int:
        """Rows in the phase-aligned window (what the pipeline will see)."""
        n = len(self)
        if n == 0:
            return 0
        return n - self._head_pad()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def append(self, bar: Bar, deriv: Optional[DerivSnapshot] = None) -> None:
        """Append one closed bar (O(1) amortized).

        Contiguity (`ts == last_ts + 1min`) is asserted as defense in
        depth — the GridGuard upstream is responsible for repairs.
        """
        g = minutes_since(self._anchor, bar.ts)
        if self._end > self._start:
            g_prev = int(self._ts_min[self._end - 1])
            if g != g_prev + 1:
                raise GridError(
                    f"BarBuffer.append: non-contiguous timestamp {bar.ts} "
                    f"(expected anchor+{g_prev + 1}min, got anchor+{g}min); "
                    "the grid guard must repair gaps before the buffer"
                )
        if self._end == self._alloc:
            self._compact()
        row = self._end
        values = bar.values()
        if self._with_deriv:
            d = deriv if deriv is not None else DerivSnapshot()
            values = values + tuple(
                np.nan if v is None else float(v) for v in d.values()
            )
        self._data[row, :] = values
        self._ts_min[row] = g
        self._synthetic[row] = bar.synthetic
        self._end += 1
        if self._end - self._start > self._capacity:
            self._start = self._end - self._capacity

    def bootstrap(self, frame: pd.DataFrame) -> int:
        """Bulk-load a historical prefix (vectorized append).

        ``frame`` must have a tz-aware DatetimeIndex on the 1-minute grid
        and contain at least the spot columns (missing derivative columns
        load as NaN). Returns the number of rows loaded (tail-trimmed to
        capacity + slack headroom).
        """
        if frame.empty:
            return 0
        if frame.index.tz is None:
            raise ValueError("bootstrap frame must have a tz-aware index")
        if len(self) > 0:
            raise GridError("bootstrap must run on an empty buffer")
        frame = frame.tail(self._capacity)
        idx = frame.index.tz_convert("UTC")
        g0 = minutes_since(self._anchor, idx[0])
        n = len(frame)
        g = np.arange(g0, g0 + n, dtype=np.int64)
        actual = (idx.asi8 - self._anchor.value) // 60_000_000_000
        if not np.array_equal(actual, g):
            raise GridError("bootstrap frame is not a contiguous 1-minute grid")
        for j, col in enumerate(self._cols):
            if col in frame.columns:
                self._data[:n, j] = frame[col].to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                self._data[:n, j] = np.nan
        self._ts_min[:n] = g
        self._synthetic[:n] = False
        self._start, self._end = 0, n
        return n

    def _compact(self) -> None:
        """Slide the live window back to the front of the allocation."""
        n = self._end - self._start
        self._data[:n] = self._data[self._start:self._end]
        self._ts_min[:n] = self._ts_min[self._start:self._end]
        self._synthetic[:n] = self._synthetic[self._start:self._end]
        self._start, self._end = 0, n

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def _head_pad(self) -> int:
        """Rows to skip from the window head so row 0 is anchor-aligned."""
        g_first = int(self._ts_min[self._start])
        return (-g_first) % self._m

    def window_frame(self) -> pd.DataFrame:
        """The phase-aligned trailing window as a pipeline-ready frame.

        Returns a pandas DataFrame (one copy) with a tz-aware UTC
        DatetimeIndex named ``ts`` and columns in pipeline order. The
        first row is guaranteed anchor-aligned (``global_index % M == 0``);
        a violation raises :class:`PhaseAlignmentError` (defense in depth —
        unreachable unless internal state was corrupted).
        """
        start = self._start + self._head_pad()
        if start >= self._end:
            raise PhaseAlignmentError(
                "buffer has no anchor-aligned rows (window shorter than M)"
            )
        g_first = int(self._ts_min[start])
        if g_first % self._m != 0:
            raise PhaseAlignmentError(
                f"window head phase {g_first % self._m} != 0 (M={self._m})"
            )
        data = self._data[start:self._end].copy()
        ts = self._anchor + pd.to_timedelta(self._ts_min[start:self._end], unit="m")
        frame = pd.DataFrame(data, columns=list(self._cols), index=ts)
        frame.index.name = "ts"
        return frame

    def tail_closes(self, n: int) -> np.ndarray:
        """Last ``n`` close prices (newest last) — cheap view for label
        maturation and online stats, no frame construction."""
        lo = max(self._start, self._end - n)
        return self._data[lo:self._end, 3]

    def tail_highs(self, n: int) -> np.ndarray:
        lo = max(self._start, self._end - n)
        return self._data[lo:self._end, 1]

    def synthetic_count(self) -> int:
        return int(self._synthetic[self._start:self._end].sum())
