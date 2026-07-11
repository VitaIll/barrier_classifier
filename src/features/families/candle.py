"""Candle geometry + breakout features (spec Group D / Section 7.10).

Mirrors ``utils.compute_candle_geometry`` (utils.py:1782-1820).

Output columns:
  - clv__inst__f__w0, bodyfrac__inst__f__w0, wickup__inst__f__w0,
    wickdn__inst__f__w0, gap__inst__f__w0   (instantaneous passthrough)
  - clv__mean__f__w{W}    for W in WINDOWS_CANDLE_ROLL
  - logp__pos__f__w{W}, logp__dd__f__w{W}, logp__du__f__w{W}
    for W in WINDOWS_BREAKOUT
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from src.features.base import FRACTION, Domain, Feature
from src.features.primitives import rolling_max, rolling_mean, rolling_min


# -------- Instantaneous (single-column passthrough) ------------------------


class _CandleInstFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "candle"
    tier: ClassVar[int | str] = 1
    windows: ClassVar[tuple[int, ...]] = ()

    output_prefix: ClassVar[str] = ""

    def column_name(self, w: int | None = None) -> str:
        return f"{self.output_prefix}__inst__f__w0"


class CandleClvInst(_CandleInstFeature):
    inputs = ("clv",)
    output_prefix = "clv"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("clv")


class CandleBodyfracInst(_CandleInstFeature):
    inputs = ("bodyfrac",)
    output_prefix = "bodyfrac"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("bodyfrac")


class CandleWickupInst(_CandleInstFeature):
    inputs = ("wickup",)
    output_prefix = "wickup"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("wickup")


class CandleWickdnInst(_CandleInstFeature):
    inputs = ("wickdn",)
    output_prefix = "wickdn"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("wickdn")


class CandleGapInst(_CandleInstFeature):
    inputs = ("g",)
    output_prefix = "gap"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("g")


# -------- Rolling mean of CLV ----------------------------------------------


class CandleClvMean(Feature):
    family: ClassVar[str] = "candle"
    tier: ClassVar[int | str] = 1
    inputs = ("clv",)
    windows_field: ClassVar[str] = "windows_candle_roll"

    def column_name(self, w: int | None = None) -> str:
        return f"clv__mean__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_mean(pl.col("clv"), w)


# -------- Breakout features (rolling min/max of log price) -----------------


class _CandleBreakoutFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "candle"
    tier: ClassVar[int | str] = 1
    inputs = ("p",)
    windows_field: ClassVar[str] = "windows_breakout"


class CandleLogpPos(_CandleBreakoutFeature):
    domain: ClassVar[Domain] = FRACTION  # range position; 0.5 = mid
    """Position of current log price within the rolling [min, max] band.

    Null where the band collapses (max == min).
    """

    def column_name(self, w: int | None = None) -> str:
        return f"logp__pos__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        p_max = rolling_max(p, w)
        p_min = rolling_min(p, w)
        denom = p_max - p_min
        return pl.when(denom != 0.0).then((p - p_min) / denom).otherwise(None)


class CandleLogpDd(_CandleBreakoutFeature):
    """Drawdown from rolling max log price."""

    def column_name(self, w: int | None = None) -> str:
        return f"logp__dd__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_max(pl.col("p"), w) - pl.col("p")


class CandleLogpDu(_CandleBreakoutFeature):
    """Drawup from rolling min log price."""

    def column_name(self, w: int | None = None) -> str:
        return f"logp__du__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("p") - rolling_min(pl.col("p"), w)
