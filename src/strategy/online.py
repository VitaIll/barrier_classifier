"""Causal online statistics used by the strategy at decision time.

All updaters maintain *only* trailing data — no future leakage. Each
exposes:

- ``update(x)``  — feed one observation
- ``value()`` / ``rank(x)`` / etc. — the quantity the simulator reads back
- ``is_warm()`` — False until enough samples have accrued; gates fall back
  to a defined NaN behavior in that case

Wraps ``river.stats``/``river.drift`` where it pays for itself; uses small
hand-rolled trackers (deque + numpy) where River would be heavier than the
op deserves.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import math
import numpy as np


# ---------------------------------------------------------------------------
# Rolling quantile rank: "what fraction of the last N values is <= x?"
# ---------------------------------------------------------------------------


class RollingQuantileRank:
    """Maintain a trailing window of size ``window`` and rank new values.

    ``rank(x)`` returns the empirical CDF at x using the trailing window:
    ``count(values <= x) / window`` — a value in [0, 1].

    The trailing window is *strictly past*: ``update(x)`` after ``rank(x)``
    means the rank lookup is causal (you ranked x against history that did
    not include x). ``rank_and_update(x)`` is the convenience for the
    common case in the simulator.
    """

    def __init__(self, window: int, *, min_warmup: Optional[int] = None) -> None:
        if window <= 0:
            raise ValueError(f"window must be > 0; got {window}")
        self._window = int(window)
        self._buf: deque[float] = deque(maxlen=self._window)
        self._min_warmup = int(min_warmup) if min_warmup is not None else max(30, self._window // 4)

    @property
    def n(self) -> int:
        return len(self._buf)

    def is_warm(self) -> bool:
        return self.n >= self._min_warmup

    def update(self, x: float) -> None:
        if not (x is None or (isinstance(x, float) and math.isnan(x))):
            self._buf.append(float(x))

    def rank(self, x: float) -> float:
        """Fraction of trailing values ``<= x``. Returns NaN until warm.

        Uses ``side='right'`` so that x's tied with values in the buffer
        push the rank upward (consistent with empirical CDF convention).
        """
        if not self.is_warm() or x is None:
            return float("nan")
        if isinstance(x, float) and math.isnan(x):
            return float("nan")
        arr = np.fromiter(self._buf, dtype=float, count=self.n)
        arr.sort()
        return float(np.searchsorted(arr, float(x), side="right")) / float(self.n)

    def rank_and_update(self, x: float) -> float:
        """Compute the rank against the *current* trailing window, then
        absorb x into it. The order matters — this is the causal pattern."""
        r = self.rank(x)
        self.update(x)
        return r


# ---------------------------------------------------------------------------
# Fast volatility (EWMA on log-returns)
# ---------------------------------------------------------------------------


class FastVolEWMA:
    """EWMA estimate of σ_t from a stream of log-returns.

    Parameterized by ``halflife_bars`` rather than alpha (more interpretable):
    ``alpha = 1 - exp(-ln 2 / halflife)``. Tracks E[r²] (zero-mean assumption,
    fine for short horizons) and exposes ``sqrt(E[r²])`` as σ̂.
    """

    def __init__(
        self, halflife_bars: float, *, min_warmup: Optional[int] = None
    ) -> None:
        if halflife_bars <= 0:
            raise ValueError(f"halflife_bars must be > 0; got {halflife_bars}")
        self._halflife = float(halflife_bars)
        self._alpha = 1.0 - math.exp(-math.log(2.0) / self._halflife)
        self._mean_sq = float("nan")
        self._n = 0
        self._min_warmup = (
            int(min_warmup) if min_warmup is not None else max(5, int(self._halflife))
        )

    @property
    def n(self) -> int:
        return self._n

    def is_warm(self) -> bool:
        return self._n >= self._min_warmup

    def update(self, log_return: float) -> None:
        if log_return is None or (isinstance(log_return, float) and math.isnan(log_return)):
            return
        x2 = float(log_return) ** 2
        if math.isnan(self._mean_sq):
            self._mean_sq = x2
        else:
            self._mean_sq = (1.0 - self._alpha) * self._mean_sq + self._alpha * x2
        self._n += 1

    def value(self) -> float:
        if not self.is_warm() or math.isnan(self._mean_sq):
            return float("nan")
        return float(math.sqrt(max(0.0, self._mean_sq)))


# ---------------------------------------------------------------------------
# Drift detector (thin wrapper around river.drift.ADWIN)
# ---------------------------------------------------------------------------


class DriftADWIN:
    """``river.drift.ADWIN`` with a one-line API.

    Feed it a stream of *residuals* (e.g. ``y_k − p̃_k`` once labels mature).
    ``update(x)`` returns whether drift was detected on this step. Internal
    state is opaque — re-fit (instantiate fresh) when you want to forget
    history after a confirmed regime change.
    """

    def __init__(self, *, delta: float = 0.002, grace_period: int = 30) -> None:
        from river.drift import ADWIN as _ADWIN

        self._adwin = _ADWIN(delta=float(delta), grace_period=int(grace_period))
        self._n = 0
        self._n_detections = 0

    @property
    def n(self) -> int:
        return self._n

    @property
    def n_detections(self) -> int:
        return self._n_detections

    def update(self, x: float) -> bool:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return False
        self._adwin.update(float(x))
        self._n += 1
        fired = bool(self._adwin.drift_detected)
        if fired:
            self._n_detections += 1
        return fired


# ---------------------------------------------------------------------------
# Rolling base-rate map (E[y | streaming quantile bin])
# ---------------------------------------------------------------------------


class RollingRegimeBaseRate:
    """Maintain ``E[y | regime_quantile_bin]`` over a trailing window of (q, y).

    Used by ``score_residualized``: the simulator pulls
    ``base_rate_at(regime_q_t)`` and feeds it into the score function. Bins
    are equal-width over [0, 1]. Empty bins return the global trailing mean
    (a safe fallback that doesn't bias the score).
    """

    def __init__(self, window: int = 1000, *, n_bins: int = 5) -> None:
        if window <= 0:
            raise ValueError(f"window must be > 0; got {window}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2; got {n_bins}")
        self._window = int(window)
        self._n_bins = int(n_bins)
        self._buf_q: deque[float] = deque(maxlen=self._window)
        self._buf_y: deque[float] = deque(maxlen=self._window)

    @property
    def n(self) -> int:
        return len(self._buf_q)

    def update(self, q: float, y: float) -> None:
        if q is None or y is None:
            return
        if isinstance(q, float) and math.isnan(q):
            return
        if isinstance(y, float) and math.isnan(y):
            return
        self._buf_q.append(float(q))
        self._buf_y.append(float(y))

    def base_rate_at(self, q: float) -> float:
        """Empirical P(y=1 | regime_quantile_bin(q)). NaN if no data yet."""
        if self.n == 0:
            return float("nan")
        if not np.isfinite(q):
            return float(np.mean(self._buf_y))
        qa = np.fromiter(self._buf_q, dtype=float, count=self.n)
        ya = np.fromiter(self._buf_y, dtype=float, count=self.n)
        bin_idx = min(int(np.clip(q * self._n_bins, 0, self._n_bins - 1)), self._n_bins - 1)
        target_bin = (qa * self._n_bins).clip(0, self._n_bins - 1).astype(int)
        mask = target_bin == bin_idx
        if mask.sum() == 0:
            return float(ya.mean())
        return float(ya[mask].mean())
