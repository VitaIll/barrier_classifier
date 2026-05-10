"""Rolling stats features (spec Group B / Section 7.5).

Mirrors ``utils.compute_rolling_stats`` (utils.py:1603-1632). Nine Feature
classes, one per output column, sharing ``windows = WINDOWS_F``.

Output columns:
  - ret__mean__f__w{W}     = r.rolling(W).mean()
  - ret__std__f__w{W}      = r.rolling(W).std(ddof=0)
  - ret__rms__f__w{W}      = sqrt((r**2).rolling(W).mean())
  - absret__mean__f__w{W}  = |r|.rolling(W).mean()
  - ret__posfrac__f__w{W}  = ((r > 0).astype(float)).rolling(W).mean()
  - range__mean__f__w{W}   = rho.rolling(W).mean()
  - logvol__mean__f__w{W}  = logvol.rolling(W).mean()
  - logvol__std__f__w{W}   = logvol.rolling(W).std(ddof=0)
  - ofi__std__f__w{W}      = ofi.rolling(W).std(ddof=0)
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from src.features.base import Feature
from src.features.config import WINDOWS_F
from src.features.primitives import clip_pos, rolling_mean, rolling_std_pop


class _RollingFeature(Feature):
    """Abstract base: emit one column per window with custom ``output_prefix``.

    Column naming follows the legacy spec convention
    ``{prefix}__f__w{w}`` rather than the canonical default.
    """

    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "rolling"
    tier: ClassVar[int | str] = 1
    windows: ClassVar[tuple[int, ...]] = tuple(WINDOWS_F)

    output_prefix: ClassVar[str] = ""  # e.g. "ret__mean"

    def column_name(self, w: int | None = None) -> str:
        return f"{self.output_prefix}__f__w{w}"


class RollingRetMean(_RollingFeature):
    inputs = ("r",)
    output_prefix = "ret__mean"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_mean(pl.col("r"), w)


class RollingRetStd(_RollingFeature):
    inputs = ("r",)
    output_prefix = "ret__std"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_std_pop(pl.col("r"), w)


class RollingRetRms(_RollingFeature):
    """Root-mean-square of returns: ``sqrt(rolling_mean(r², w))``.

    Polars-specific guard: `clip_pos` before `sqrt`. Polars' rolling
    online algorithm can produce a tiny-negative result on all-zero
    windows (constant-price bars) due to float cancellation; sqrt of a
    tiny negative is NaN. Numpy/pandas don't hit this because their
    rolling mean uses a different summation. The clamp matches what
    every other variance-shaped feature does (VolGk / VolRs /
    VolSemivar*) and is mathematically a no-op for r² ≥ 0.
    """

    inputs = ("r",)
    output_prefix = "ret__rms"

    def compute(self, w: int | None = None) -> pl.Expr:
        return clip_pos(rolling_mean(pl.col("r") ** 2, w)).sqrt()


class RollingAbsretMean(_RollingFeature):
    inputs = ("r",)
    output_prefix = "absret__mean"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_mean(pl.col("r").abs(), w)


class RollingRetPosfrac(_RollingFeature):
    """Fraction of positive returns over the window.

    Parity note: legacy ``(r > 0).astype(float)`` treats NaN as False (→ 0.0)
    because pandas/numpy ``NaN > 0`` evaluates to False. After the engine
    pre-converts NaN to null, ``null > 0`` evaluates to null in polars and
    would propagate to a null indicator — divergent. ``fill_null(0.0)``
    before the comparison reproduces the legacy ``NaN → 0.0`` behavior.
    """

    inputs = ("r",)
    output_prefix = "ret__posfrac"

    def compute(self, w: int | None = None) -> pl.Expr:
        pos = (pl.col("r").fill_null(0.0) > 0).cast(pl.Float64)
        return rolling_mean(pos, w)


class RollingRangeMean(_RollingFeature):
    inputs = ("rho",)
    output_prefix = "range__mean"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_mean(pl.col("rho"), w)


class RollingLogvolMean(_RollingFeature):
    inputs = ("logvol",)
    output_prefix = "logvol__mean"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_mean(pl.col("logvol"), w)


class RollingLogvolStd(_RollingFeature):
    inputs = ("logvol",)
    output_prefix = "logvol__std"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_std_pop(pl.col("logvol"), w)


class RollingOfiStd(_RollingFeature):
    inputs = ("ofi",)
    output_prefix = "ofi__std"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_std_pop(pl.col("ofi"), w)
