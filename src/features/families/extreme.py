"""Causal local-extreme features (new family — Round 1a).

Three families of features at Tier-2 that summarize current price's position
within its trailing window:

  - extreme__dist_low_z__f__w{W}    log-distance from current close to trailing
                                    min(log(low)), normalized by horizon vol.
  - extreme__dist_high_z__f__w{W}   symmetric — to trailing max(log(high)).
  - extreme__price_rank__f__w{W}    empirical CDF of current log close within
                                    the trailing W closes.

Causal contract: at row n, every feature uses only bars in [n-W+1, n]. No
forward shifts. Normalisation reuses ``vol__rs__f__w{W}`` already produced
by Tier-1 of the ``vol`` family, so the scale matches ``barrier__z_tight``
and the model can compare them directly.

Distance features use ``low`` / ``high`` (not ``close``) so the geometry
aligns with the high-source label that backs the production strategy
(``construct_labels_pl(..., barrier_source='high')``).

Null policy:
  - First ``W-1`` rows of every column are null (warmup).
  - When the rolling volatility is zero (constant-price window), the
    distance features divide by ``EPS`` only — they do not null out — so
    the magnitude is bounded but possibly large. Imputation does not see
    a null in that case.
  - Price rank is always in [0, 1] for valid rows.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import polars as pl

from src.features.base import Feature
from src.features.config import EPS, M, WINDOWS_EXTREME
from src.features.primitives import rolling_max, rolling_min


_SQRT_M = math.sqrt(int(M))


class _ExtremeFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "extreme"
    # Tier 2: depends on ``vol__rs__f__wW`` emitted by Tier 1 of the vol
    # family. Tier ordering in the engine ensures vol__rs is on the frame
    # before this expression evaluates.
    tier: ClassVar[int | str] = 2
    windows: ClassVar[tuple[int, ...]] = tuple(WINDOWS_EXTREME)


class ExtremeDistLowZ(_ExtremeFeature):
    """Volatility-normalized log-distance from current close to trailing low.

        D^low_{W,n} = (p_n - min_{i in [n-W+1, n]} log(L_i)) / (sigma_W,n * sqrt(M) + EPS)

    Uses the same ``vol__rs__f__wW`` denominator as ``barrier__z_tight`` so
    the model sees both on the same scale. Positive — near zero means
    price is close to its trailing low.
    """

    inputs = ("p", "low")

    def column_name(self, w: int | None = None) -> str:
        return f"extreme__dist_low_z__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        log_low = pl.col("low").log()
        trailing_min = rolling_min(log_low, w)
        vol_col = pl.col(f"vol__rs__f__w{w}")
        dist = pl.col("p") - trailing_min
        return dist / (vol_col * _SQRT_M + EPS)


class ExtremeDistHighZ(_ExtremeFeature):
    """Volatility-normalized log-distance from trailing high to current close.

        D^high_{W,n} = (max_{i in [n-W+1, n]} log(H_i) - p_n) / (sigma_W,n * sqrt(M) + EPS)

    Positive — near zero means price is close to its trailing high.
    """

    inputs = ("p", "high")

    def column_name(self, w: int | None = None) -> str:
        return f"extreme__dist_high_z__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        log_high = pl.col("high").log()
        trailing_max = rolling_max(log_high, w)
        vol_col = pl.col(f"vol__rs__f__w{w}")
        dist = trailing_max - pl.col("p")
        return dist / (vol_col * _SQRT_M + EPS)


def _rolling_rank_of_current_np(p: np.ndarray, w: int) -> np.ndarray:
    """Empirical CDF of ``p[n]`` inside the window ``[n-w+1, n]``.

    At row n (for n >= w-1):
        out[n] = (1/w) * count(i in [n-w+1, n]: p[i] <= p[n])

    Vectorized in chunks: each chunk builds a (chunk_size, w) matrix of
    rolling windows and compares each row's last column to the rest of
    its window. Memory capped at ~5M elements per chunk (matches the
    excursion family's chunking strategy).
    """
    n = len(p)
    out = np.full(n, np.nan, dtype=float)
    if n < int(w):
        return out

    eligible = np.arange(int(w) - 1, n, dtype=np.int64)
    offsets = np.arange(int(w) - 1, -1, -1, dtype=np.int64)
    max_elements = 5_000_000
    chunk_size = int(min(20_000, max(1, max_elements // int(w))))

    for start in range(0, len(eligible), chunk_size):
        idx = eligible[start : start + chunk_size]
        rows = idx[:, None] - offsets[None, :]
        window_vals = p[rows]
        invalid = np.isnan(window_vals).any(axis=1)
        last = window_vals[:, -1:]
        counts = (window_vals <= last).sum(axis=1)
        ranks = counts.astype(float) / float(w)
        ranks[invalid] = np.nan
        out[idx] = ranks

    return out


class ExtremePriceRank(_ExtremeFeature):
    """Empirical CDF of current log close inside the trailing window.

        R_{W,n} = (1/W) * sum_{i in [n-W+1, n]} 1{p_i <= p_n}

    Bounded in (0, 1]. Robust to volatility level; complementary to the
    two distance features. Uses <= so ties go up, which means the
    minimum a valid rank can take is 1/W (the row itself counts).

    Tier-2 in name only — this feature does not actually depend on any
    Tier-1 emitted column. Kept here for family-namespace consistency.
    """

    inputs = ("p",)

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("p").map_batches(
            lambda s: pl.Series(_rolling_rank_of_current_np(s.to_numpy(), w)),
            return_dtype=pl.Float64,
        )

    def column_name(self, w: int | None = None) -> str:
        return f"extreme__price_rank__f__w{w}"
