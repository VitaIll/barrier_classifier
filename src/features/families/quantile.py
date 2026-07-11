"""Quantile + MAD features (spec Group B+ / Section 7.6).

Mirrors ``utils.compute_quantile_features`` (utils.py:1639-1696).
Boundary-sparse: only computes at indices that are multiples of M;
other rows are null (matches legacy NaN pattern).

Output columns:
  - ret__q10__f__w{W}, ret__q50__f__w{W}, ret__q90__f__w{W}, ret__mad__f__w{W}
  for W in WINDOWS_BPLUS, populated only at every M-th row.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import polars as pl

from src.features.base import Feature


def _quantile_at_boundaries_np(
    r: np.ndarray, w: int, m_stride: int, q: float
) -> np.ndarray:
    """Compute rolling quantile of ``r`` only at boundary indices (every
    ``m_stride``-th row); other rows return NaN. Matches utils.py:1648-1687."""
    n = len(r)
    out = np.full(n, np.nan, dtype=float)
    bidx = np.arange(0, n, m_stride, dtype=np.int64)
    eligible = bidx[bidx >= (w - 1)]
    if len(eligible) == 0:
        return out
    offsets = np.arange(w - 1, -1, -1, dtype=np.int64)
    chunk_size = 2000 if w >= 720 else 5000
    for start in range(0, len(eligible), chunk_size):
        idx = eligible[start : start + chunk_size]
        rows = idx[:, None] - offsets[None, :]
        window_vals = r[rows]
        invalid = np.isnan(window_vals).any(axis=1)
        q_chunk = np.quantile(window_vals, q, axis=1, method="linear")
        q_chunk[invalid] = np.nan
        out[idx] = q_chunk
    return out


def _mad_at_boundaries_np(
    r: np.ndarray, w: int, m_stride: int
) -> np.ndarray:
    """MAD at boundary indices: median(|x - median(x)|)."""
    n = len(r)
    out = np.full(n, np.nan, dtype=float)
    bidx = np.arange(0, n, m_stride, dtype=np.int64)
    eligible = bidx[bidx >= (w - 1)]
    if len(eligible) == 0:
        return out
    offsets = np.arange(w - 1, -1, -1, dtype=np.int64)
    chunk_size = 2000 if w >= 720 else 5000
    for start in range(0, len(eligible), chunk_size):
        idx = eligible[start : start + chunk_size]
        rows = idx[:, None] - offsets[None, :]
        window_vals = r[rows]
        invalid = np.isnan(window_vals).any(axis=1)
        med = np.quantile(window_vals, 0.5, axis=1, method="linear")
        abs_dev = np.abs(window_vals - med[:, None])
        mad_chunk = np.quantile(abs_dev, 0.5, axis=1, method="linear")
        mad_chunk[invalid] = np.nan
        out[idx] = mad_chunk
    return out


class _QuantileFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "quantile"
    tier: ClassVar[int | str] = 1
    inputs = ("r",)
    windows_field: ClassVar[str] = "windows_bplus"

    output_name: ClassVar[str] = ""
    quantile_value: ClassVar[float] = 0.5

    def column_name(self, w: int | None = None) -> str:
        return f"ret__{self.output_name}__f__w{w}"


class QuantileQ10(_QuantileFeature):
    output_name = "q10"
    quantile_value = 0.10

    def compute(self, w: int | None = None) -> pl.Expr:
        q = self.quantile_value
        return pl.col("r").map_batches(
            lambda s, m=self.cfg.m: pl.Series(
                _quantile_at_boundaries_np(s.to_numpy(), w, m, q)
            ),
            return_dtype=pl.Float64,
        )


class QuantileQ50(_QuantileFeature):
    output_name = "q50"
    quantile_value = 0.50

    def compute(self, w: int | None = None) -> pl.Expr:
        q = self.quantile_value
        return pl.col("r").map_batches(
            lambda s, m=self.cfg.m: pl.Series(
                _quantile_at_boundaries_np(s.to_numpy(), w, m, q)
            ),
            return_dtype=pl.Float64,
        )


class QuantileQ90(_QuantileFeature):
    output_name = "q90"
    quantile_value = 0.90

    def compute(self, w: int | None = None) -> pl.Expr:
        q = self.quantile_value
        return pl.col("r").map_batches(
            lambda s, m=self.cfg.m: pl.Series(
                _quantile_at_boundaries_np(s.to_numpy(), w, m, q)
            ),
            return_dtype=pl.Float64,
        )


class QuantileMad(_QuantileFeature):
    output_name = "mad"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("r").map_batches(
            lambda s, m=self.cfg.m: pl.Series(
                _mad_at_boundaries_np(s.to_numpy(), w, m)
            ),
            return_dtype=pl.Float64,
        )
