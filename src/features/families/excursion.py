"""Excursion / burstiness features (spec Group O / Section 7.11) — Tier-2.

Mirrors ``utils.compute_excursion_features`` (utils.py:2047-2095).

Output columns (boundary-sparse — only every M-th row is populated, rest null):
  - excursion__max_drawup__f__w{W}    for W in WINDOWS_EXCURSION
  - excursion__max_drawdown__f__w{W}  for W in WINDOWS_EXCURSION

  These are kept for legacy parity with ``utils.compute_excursion_features``
  at boundary cadence. At 1-min cadence they would produce phase artifacts
  (NaN at non-modulo-M rows), so ``run_pipeline`` excludes them from the
  1-min feature list and substitutes the every-row-rolling variants below.

Output columns (every-row trailing — usable at any cadence):
  - excursion__roll_max_drawup__f__w{W}    for W in WINDOWS_EXCURSION
  - excursion__roll_max_drawdown__f__w{W}  for W in WINDOWS_EXCURSION

Output columns (every-row rolling on returns):
  - ret__max1m__f__w{W}  for W in WINDOWS_MAXRET
  - ret__max2m__f__w{W}  (max of 2-bar return = p - p.shift(2))
  - ret__min1m__f__w{W}
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import polars as pl

from src.features.base import Feature
from src.features.config import M, WINDOWS_EXCURSION, WINDOWS_MAXRET
from src.features.primitives import rolling_max, rolling_min


def _drawup_drawdown_boundary_sparse_np(
    p: np.ndarray, w: int, m_stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute max-drawup and max-drawdown on each ``w``-bar window ending
    at indices that are multiples of ``m_stride``. Other rows return NaN.

    Mirrors utils.py:2056-2086.
    """
    n = len(p)
    out_up = np.full(n, np.nan, dtype=float)
    out_dn = np.full(n, np.nan, dtype=float)
    bidx = np.arange(0, n, m_stride, dtype=np.int64)
    eligible = bidx[bidx >= (w - 1)]
    if len(eligible) == 0:
        return out_up, out_dn

    offsets = np.arange(w - 1, -1, -1, dtype=np.int64)
    max_elements = 5_000_000
    chunk_size = int(min(20_000, max(1, max_elements // int(w))))

    for start in range(0, len(eligible), chunk_size):
        idx = eligible[start : start + chunk_size]
        rows = idx[:, None] - offsets[None, :]
        window_vals = p[rows]
        invalid = np.isnan(window_vals).any(axis=1)

        running_min = np.minimum.accumulate(window_vals, axis=1)
        drawups = window_vals - running_min
        max_drawup = np.max(drawups, axis=1)

        running_max = np.maximum.accumulate(window_vals, axis=1)
        drawdowns = running_max - window_vals
        max_drawdown = np.max(drawdowns, axis=1)

        max_drawup[invalid] = np.nan
        max_drawdown[invalid] = np.nan
        out_up[idx] = max_drawup
        out_dn[idx] = max_drawdown

    return out_up, out_dn


def _drawup_drawdown_rolling_np(
    p: np.ndarray, w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every-row trailing max-drawup and max-drawdown over the window
    ``[n - w + 1, n]``. At row ``n`` (for ``n >= w - 1``):

        drawup[n] = max over b in [n - w + 1, n] of
                        (p[b] - min over a in [n - w + 1, b] of p[a])
        drawdown[n] = max over b in [n - w + 1, n] of
                        (max over a in [n - w + 1, b] of p[a] - p[b])

    Vectorized in chunks: each chunk builds a (chunk_size, w) matrix of
    rolling windows and applies cumulative min/max along axis=1. Memory
    is capped at ~5M elements per chunk (same as the boundary-sparse
    variant). At boundary cadence the boundary-sparse and trailing
    values coincide on rows that are multiples of M (the trailing window
    ending at row k*M is identical in both cases), so this function is a
    strict superset of the sparse one and is safe to substitute when the
    every-row coverage is required (1-min cadence).
    """
    n = len(p)
    out_up = np.full(n, np.nan, dtype=float)
    out_dn = np.full(n, np.nan, dtype=float)
    if n < int(w):
        return out_up, out_dn

    eligible = np.arange(int(w) - 1, n, dtype=np.int64)
    offsets = np.arange(int(w) - 1, -1, -1, dtype=np.int64)
    max_elements = 5_000_000
    chunk_size = int(min(20_000, max(1, max_elements // int(w))))

    for start in range(0, len(eligible), chunk_size):
        idx = eligible[start : start + chunk_size]
        rows = idx[:, None] - offsets[None, :]
        window_vals = p[rows]
        invalid = np.isnan(window_vals).any(axis=1)

        running_min = np.minimum.accumulate(window_vals, axis=1)
        drawups = window_vals - running_min
        max_drawup = np.max(drawups, axis=1)

        running_max = np.maximum.accumulate(window_vals, axis=1)
        drawdowns = running_max - window_vals
        max_drawdown = np.max(drawdowns, axis=1)

        max_drawup[invalid] = np.nan
        max_drawdown[invalid] = np.nan
        out_up[idx] = max_drawup
        out_dn[idx] = max_drawdown

    return out_up, out_dn


class _ExcursionFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "excursion"
    tier: ClassVar[int | str] = 2
    inputs = ("p",)
    windows = tuple(WINDOWS_EXCURSION)


class ExcursionMaxDrawup(_ExcursionFeature):
    def column_name(self, w: int | None = None) -> str:
        return f"excursion__max_drawup__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("p").map_batches(
            lambda s: pl.Series(
                _drawup_drawdown_boundary_sparse_np(s.to_numpy(), w, M)[0]
            ),
            return_dtype=pl.Float64,
        )


class ExcursionMaxDrawdown(_ExcursionFeature):
    def column_name(self, w: int | None = None) -> str:
        return f"excursion__max_drawdown__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("p").map_batches(
            lambda s: pl.Series(
                _drawup_drawdown_boundary_sparse_np(s.to_numpy(), w, M)[1]
            ),
            return_dtype=pl.Float64,
        )


class ExcursionRollMaxDrawup(_ExcursionFeature):
    """Every-row trailing max-drawup over the last ``w`` log-prices."""

    def column_name(self, w: int | None = None) -> str:
        return f"excursion__roll_max_drawup__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("p").map_batches(
            lambda s: pl.Series(_drawup_drawdown_rolling_np(s.to_numpy(), w)[0]),
            return_dtype=pl.Float64,
        )


class ExcursionRollMaxDrawdown(_ExcursionFeature):
    """Every-row trailing max-drawdown over the last ``w`` log-prices."""

    def column_name(self, w: int | None = None) -> str:
        return f"excursion__roll_max_drawdown__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("p").map_batches(
            lambda s: pl.Series(_drawup_drawdown_rolling_np(s.to_numpy(), w)[1]),
            return_dtype=pl.Float64,
        )


# -------- Rolling extremes of returns (every-row, not boundary-sparse) -----


class _MaxretFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "excursion"
    tier: ClassVar[int | str] = 2
    windows = tuple(WINDOWS_MAXRET)


class RetMax1m(_MaxretFeature):
    inputs = ("r",)

    def column_name(self, w: int | None = None) -> str:
        return f"ret__max1m__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_max(pl.col("r"), w)


class RetMax2m(_MaxretFeature):
    """Rolling max of 2-bar log return ``p - p.shift(2)``."""

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"ret__max2m__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        r2 = p - p.shift(2)
        return rolling_max(r2, w)


class RetMin1m(_MaxretFeature):
    inputs = ("r",)

    def column_name(self, w: int | None = None) -> str:
        return f"ret__min1m__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_min(pl.col("r"), w)
