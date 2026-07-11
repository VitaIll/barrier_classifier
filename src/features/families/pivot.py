"""Confirmed swing pivots (new family — Round 1c).

A swing low at bar ``i`` is the local minimum of ``log(low)`` over the
symmetric window ``[i - Q, i + Q]``. It is **confirmed** only once bar
``i + Q`` has been observed, since the right-hand half of its window
requires future bars. Symmetric definition for swing highs on
``log(high)``.

Causality contract: at row ``n``, the most recent confirmable swing is
at index ``i*`` with ``i* + Q <= n`` AND ``i* >= n - W + 1`` (lookback
capped at W). Both the pivot *detection* and the pivot *carry-forward*
are shifted by ``Q`` rows so the indicator is only "available" at row
``i + Q``, never at row ``i``.

Emitted columns (per ``W`` in ``WINDOWS_PIVOT`` × ``Q`` in
``PIVOT_Q_VALUES``):

  - pivot__last_low_dist_z__f__w{W}__q{Q}    vol-normalized distance from
                                              current close to most recent
                                              confirmed swing low.
  - pivot__last_low_age__f__w{W}__q{Q}       bars since that swing low (capped
                                              at W). Sentinel value ``W`` means
                                              "no eligible pivot in lookback".
  - pivot__last_high_dist_z__f__w{W}__q{Q}   symmetric — to most recent
                                              confirmed swing high. Sign of the
                                              distance differs from the low
                                              version: positive means current
                                              price is BELOW the high.
  - pivot__last_high_age__f__w{W}__q{Q}      bars since that swing high.

Distance features use ``low`` / ``high`` (not ``close``) so the geometry
aligns with the high-source label. Volatility normalizer is
``vol__rs__f__w{W}``.

Warmup: ``W + Q - 1`` rows (need ``W`` bars of lookback PLUS ``Q`` bars
to confirm the most recent candidate). Age column emits the sentinel
``W`` instead of null when no confirmed pivot exists in the window, so
the model can distinguish "fresh pivot" from "stale / absent pivot"
without the ``undef`` flag firing on every row of a no-pivot regime.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import polars as pl

from src.features.base import Feature
from src.features.config import EPS




def _detect_pivots_np(
    series: np.ndarray, q: int, *, mode: str
) -> np.ndarray:
    """Return an array of length ``n`` carrying, at each index ``n``, the
    INDEX of the most recently confirmed swing pivot at or before ``n``,
    or ``-1`` if none exists yet.

    A pivot at bar ``i`` is confirmed at bar ``i + q``. So at bar ``n``,
    the candidate pivots are exactly the indices ``i <= n - q`` for which:

      - mode = "low":  series[i] = min(series[i - q : i + q + 1])
      - mode = "high": series[i] = max(series[i - q : i + q + 1])

    Implementation: a rolling extremum identifies candidate pivots; the
    indicator is shifted by ``q`` to land on the confirmation row, then
    forward-filled in-place to give the "most recent confirmed pivot
    index at row n" stream.

    The leading ``q`` rows of ``series`` have no left-hand window so any
    candidate pivot they harbor is unconfirmable AND would compare against
    fewer than 2q+1 neighbors — we exclude them explicitly.

    ``mode="low"`` ties: a stretch of identical lows in the symmetric
    window will not be flagged as a pivot because we use strict equality
    against the rolling min only at the centre. If multiple bars share
    the minimum, the leftmost is picked deterministically by
    ``np.argmin`` order.
    """
    n = len(series)
    out = np.full(n, -1, dtype=np.int64)
    if n < 2 * q + 1:
        return out

    # Build a (n - 2q, 2q+1) matrix of symmetric windows.
    # Row index i corresponds to centre bar i + q in the original series.
    centre_idx = np.arange(q, n - q, dtype=np.int64)
    offsets = np.arange(-q, q + 1, dtype=np.int64)
    rows = centre_idx[:, None] + offsets[None, :]
    windows = series[rows]

    centre_vals = series[centre_idx]
    if mode == "low":
        extreme_vals = np.nanmin(windows, axis=1)
        is_pivot = (centre_vals == extreme_vals)
    elif mode == "high":
        extreme_vals = np.nanmax(windows, axis=1)
        is_pivot = (centre_vals == extreme_vals)
    else:
        raise ValueError(f"mode must be 'low' or 'high', got {mode!r}")

    # NaN poisoning: a window containing any NaN poisons its min/max with
    # NaN; the equality check returns False. That correctly excludes
    # those rows from pivot detection.
    has_nan = np.isnan(windows).any(axis=1)
    is_pivot &= ~has_nan

    # Confirmation row = centre + q. Build a "pivot index at confirmation
    # row" stream and forward-fill the latest seen pivot.
    confirm_idx = centre_idx + q
    pivot_indices_at_confirm = np.where(is_pivot, centre_idx, -1)

    # Walk forward through confirm_idx, carrying the most recent positive
    # pivot index. Rows between confirm_idx[i] and confirm_idx[i+1] carry
    # whatever the running pivot was before them.
    running = -1
    cursor = 0  # next index in confirm_idx to consider
    for n_idx in range(n):
        while cursor < len(confirm_idx) and confirm_idx[cursor] <= n_idx:
            if pivot_indices_at_confirm[cursor] >= 0:
                running = int(pivot_indices_at_confirm[cursor])
            cursor += 1
        out[n_idx] = running

    return out


def _pivot_dist_and_age_np(
    series: np.ndarray,
    p_close: np.ndarray,
    q: int,
    w: int,
    *,
    mode: str,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint distance + age streams given confirmed-pivot indices.

    ``series`` is log-low (for swing lows) or log-high (for swing highs).
    ``p_close`` is log-close (used for the distance numerator). Distance
    sign convention:

      - mode="low",  sign=+1.0: dist = p_close[n] - series[i*]
        (positive = price above the trailing low).
      - mode="high", sign=-1.0: dist = p_close[n] - series[i*]
        (negative when below a recent high; we ALSO emit it as
        ``+(high - close)`` by inverting the sign here, so the column
        is always non-negative when price is "interior" relative to the
        extremum).

    Age is bars since the confirmed pivot, capped at W. If no eligible
    pivot exists within the lookback W, age = W (sentinel) and distance
    = NaN (so the engine's imputation pass marks it via ``undef__``).

    Returns (dist_unnormalized, age). The caller normalizes distance by
    rolling vol.
    """
    pivot_idx = _detect_pivots_np(series, q, mode=mode)
    n = len(series)
    dist = np.full(n, np.nan, dtype=float)
    age = np.full(n, float(w), dtype=float)

    for n_idx in range(n):
        i_star = int(pivot_idx[n_idx])
        if i_star < 0:
            continue
        a = n_idx - i_star
        if a > w:
            # Pivot exists but is outside our lookback W -> treat as no
            # eligible pivot (age sentinel; dist remains NaN).
            continue
        age[n_idx] = float(a)
        dist[n_idx] = sign * (p_close[n_idx] - series[i_star])

    return dist, age


class _PivotFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "pivot"
    # Tier 2: distance features divide by vol__rs__f__w{W}.
    tier: ClassVar[int | str] = 2
    windows: ClassVar[tuple[int, ...]] = ()

    # Subclasses iterate (W, Q) pairs to populate ``expanded()`` —
    # resolved from the injected config so window/quantile grids are
    # per-instance, not frozen at import.
    @property
    def _wq_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (w, q)
            for w in self.cfg.windows_pivot
            for q in self.cfg.pivot_q_values
        )

    def expanded(self):
        for w, q in self._wq_pairs:
            yield (w, q), self.column_name((w, q))

    def warmup_for(self, w):
        # ``w`` here is a (W, Q) tuple. Warmup is W + Q - 1 rows.
        if isinstance(w, tuple):
            ww, qq = w
            return int(ww) + int(qq) - 1
        return 0

    def depends_on(self, w=None) -> tuple[str, ...]:
        if isinstance(w, tuple):
            ww, _qq = w
            return self.inputs + (f"vol__rs__f__w{int(ww)}",)
        return self.inputs


class PivotLastLowDistZ(_PivotFeature):
    """Volatility-normalized distance from current close to most recent
    confirmed swing low. Positive when close is above the swing low.

    Null when no eligible pivot exists in the trailing W bars. Pair with
    ``pivot__last_low_age`` (which uses a numeric sentinel) so the model
    can distinguish "no pivot" from "at pivot".
    """

    inputs = ("p", "low")

    def column_name(self, w):
        if isinstance(w, tuple):
            ww, qq = w
            return f"pivot__last_low_dist_z__f__w{ww}__q{qq}"
        return "pivot__last_low_dist_z__f__w0__q0"

    def compute(self, w):
        ww, qq = w
        vol_col = pl.col(f"vol__rs__f__w{ww}")

        def _kernel(s_struct: pl.Series, qq=qq, ww=ww) -> pl.Series:
            # s_struct is a polars Struct Series with fields ``log_low`` and
            # ``p``. ``.struct.field(name)`` returns each as its own Series.
            log_low = s_struct.struct.field("log_low").to_numpy()
            p_close = s_struct.struct.field("p").to_numpy()
            dist, _age = _pivot_dist_and_age_np(
                log_low, p_close, qq, ww, mode="low", sign=+1.0
            )
            return pl.Series(dist)

        struct = pl.struct(
            pl.col("low").log().alias("log_low"),
            pl.col("p").alias("p"),
        )
        raw = struct.map_batches(_kernel, return_dtype=pl.Float64)
        return raw / (vol_col * math.sqrt(int(self.cfg.m)) + EPS)


def _pivot_age_np(series: np.ndarray, q: int, w: int, *, mode: str) -> np.ndarray:
    """Age (in bars) since the most recent confirmed pivot at or before each
    row, capped at ``w``. Sentinel ``w`` when none exists in the lookback.
    """
    pivot_idx = _detect_pivots_np(series, q, mode=mode)
    n = len(series)
    age = np.full(n, float(w), dtype=float)
    for n_idx in range(n):
        i_star = int(pivot_idx[n_idx])
        if i_star < 0:
            continue
        a = n_idx - i_star
        if a > w:
            continue
        age[n_idx] = float(a)
    return age


class PivotLastLowAge(_PivotFeature):
    """Bars since most recent confirmed swing low, capped at W. Emits the
    sentinel ``W`` when no eligible pivot exists in the trailing W bars
    (the dist_z companion is null in that case)."""

    inputs = ("low",)

    def column_name(self, w):
        if isinstance(w, tuple):
            ww, qq = w
            return f"pivot__last_low_age__f__w{ww}__q{qq}"
        return "pivot__last_low_age__f__w0__q0"

    def compute(self, w):
        ww, qq = w

        def _kernel(s: pl.Series, qq=qq, ww=ww) -> pl.Series:
            log_low = s.log().to_numpy()
            return pl.Series(_pivot_age_np(log_low, qq, ww, mode="low"))

        return pl.col("low").map_batches(_kernel, return_dtype=pl.Float64)


class PivotLastHighDistZ(_PivotFeature):
    """Volatility-normalized distance from most recent confirmed swing high
    DOWN to current close. Positive when close is below the swing high.

    Pair with ``pivot__last_high_age`` for the missing-pivot distinction.
    """

    inputs = ("p", "high")

    def column_name(self, w):
        if isinstance(w, tuple):
            ww, qq = w
            return f"pivot__last_high_dist_z__f__w{ww}__q{qq}"
        return "pivot__last_high_dist_z__f__w0__q0"

    def compute(self, w):
        ww, qq = w
        vol_col = pl.col(f"vol__rs__f__w{ww}")

        def _kernel(s_struct: pl.Series, qq=qq, ww=ww) -> pl.Series:
            log_high = s_struct.struct.field("log_high").to_numpy()
            p_close = s_struct.struct.field("p").to_numpy()
            dist, _age = _pivot_dist_and_age_np(
                log_high, p_close, qq, ww, mode="high", sign=-1.0
            )
            return pl.Series(dist)

        struct = pl.struct(
            pl.col("high").log().alias("log_high"),
            pl.col("p").alias("p"),
        )
        raw = struct.map_batches(_kernel, return_dtype=pl.Float64)
        return raw / (vol_col * math.sqrt(int(self.cfg.m)) + EPS)


class PivotLastHighAge(_PivotFeature):
    """Bars since most recent confirmed swing high, capped at W. Sentinel
    ``W`` when no eligible pivot exists."""

    inputs = ("high",)

    def column_name(self, w):
        if isinstance(w, tuple):
            ww, qq = w
            return f"pivot__last_high_age__f__w{ww}__q{qq}"
        return "pivot__last_high_age__f__w0__q0"

    def compute(self, w):
        ww, qq = w

        def _kernel(s: pl.Series, qq=qq, ww=ww) -> pl.Series:
            log_high = s.log().to_numpy()
            return pl.Series(_pivot_age_np(log_high, qq, ww, mode="high"))

        return pl.col("high").map_batches(_kernel, return_dtype=pl.Float64)
