"""Activity / flow / liquidity features (spec Group F / Section 7.13).

Mirrors ``utils.compute_activity_flow`` (utils.py:1884-1920).

Output columns:
  - tb_ratio__inst__f__w0, ofi__inst__f__w0,
    qpertrade__inst__f__w0, vwapdev__inst__f__w0  (instantaneous passthrough)
  - logvol__z__f__w{W}                            for W in (60, 120, 240)
  - liq__quote_per_absret__f__w{W}                for W in (60, 120, 240)
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from src.features.base import FRACTION, Domain, Feature
from src.features.config import EPS
from src.features.primitives import rolling_sum, z_score_rolling


# Hardcoded — the notebook calls `compute_activity_flow(df, [60, 120, 240])`
# directly; not a module-level constant in utils.py.
_WINDOWS_ACTIVITY: tuple[int, ...] = (60, 120, 240)


# -------- Instantaneous (passthrough) --------------------------------------


class _ActivityInstFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "activity"
    tier: ClassVar[int | str] = 1
    windows: ClassVar[tuple[int, ...]] = ()

    output_prefix: ClassVar[str] = ""

    def column_name(self, w: int | None = None) -> str:
        return f"{self.output_prefix}__inst__f__w0"


class ActivityTbRatioInst(_ActivityInstFeature):
    domain: ClassVar[Domain] = FRACTION  # taker-buy share
    inputs = ("b",)
    output_prefix = "tb_ratio"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("b")


class ActivityOfiInst(_ActivityInstFeature):
    inputs = ("ofi",)
    output_prefix = "ofi"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("ofi")


class ActivityQpertradeInst(_ActivityInstFeature):
    inputs = ("qpertrade",)
    output_prefix = "qpertrade"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("qpertrade")


class ActivityVwapdevInst(_ActivityInstFeature):
    inputs = ("vwapdev",)
    output_prefix = "vwapdev"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("vwapdev")


# -------- Rolling -----------------------------------------------------------


class ActivityLogvolZ(Feature):
    """Rolling z-score of logvol (sigma==0 → null)."""

    family: ClassVar[str] = "activity"
    tier: ClassVar[int | str] = 1
    inputs = ("logvol",)
    windows: ClassVar[tuple[int, ...]] = _WINDOWS_ACTIVITY

    def column_name(self, w: int | None = None) -> str:
        return f"logvol__z__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return z_score_rolling(pl.col("logvol"), w)


class ActivityLiqQuotePerAbsret(Feature):
    """Liquidity proxy: rolling sum of quote volume / rolling sum of |r|."""

    family: ClassVar[str] = "activity"
    tier: ClassVar[int | str] = 1
    inputs = ("quote_volume", "r")
    windows: ClassVar[tuple[int, ...]] = _WINDOWS_ACTIVITY

    def column_name(self, w: int | None = None) -> str:
        return f"liq__quote_per_absret__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        num = rolling_sum(pl.col("quote_volume"), w)
        den = rolling_sum(pl.col("r").abs(), w)
        return num / (den + EPS)
