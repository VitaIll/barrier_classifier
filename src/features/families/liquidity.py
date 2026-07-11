"""Enhanced liquidity features (spec Group P / Section 7.14) — Tier-2.

Mirrors ``utils.compute_enhanced_liquidity`` (utils.py:2098-2128).

Output columns:
  - liq__amihud__f__w{W}             for W in WINDOWS_LIQ_AMIHUD
  - liq__range_per_vol__f__w{W}      for W in WINDOWS_LIQ_RPV
  - ofi__delta__f__w{W}              for W in WINDOWS_OFI_IMPULSE
  - ofi__max__f__w{W}, ofi__min__f__w{W}, ofi__ret_interaction__f__w{W}
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from src.features.base import Feature
from src.features.config import EPS
from src.features.primitives import rolling_max, rolling_min, rolling_sum


# -------- Amihud illiquidity ------------------------------------------------


class LiqAmihud(Feature):
    family: ClassVar[str] = "liquidity"
    # Tier 1: ``r`` is a base-series column on the bars frame at engine
    # entry, not an engine-emitted column. No tier-2 dependency exists.
    tier: ClassVar[int | str] = 1
    inputs = ("r", "volume")
    windows_field: ClassVar[str] = "windows_liq_amihud"

    def column_name(self, w: int | None = None) -> str:
        return f"liq__amihud__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        num = rolling_sum(pl.col("r").abs(), w)
        den = rolling_sum(pl.col("volume"), w)
        return num / (den + EPS)


# -------- Range-per-volume --------------------------------------------------


class LiqRangePerVol(Feature):
    family: ClassVar[str] = "liquidity"
    # Tier 1: reads raw OHLCV only, no engine-emitted dependency.
    tier: ClassVar[int | str] = 1
    inputs = ("high", "low", "volume")
    windows_field: ClassVar[str] = "windows_liq_rpv"

    def column_name(self, w: int | None = None) -> str:
        return f"liq__range_per_vol__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        num = rolling_sum(pl.col("high") - pl.col("low"), w)
        den = rolling_sum(pl.col("volume"), w)
        return num / (den + EPS)


# -------- OFI impulse ------------------------------------------------------


class _OfiImpulseFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "liquidity"
    # Tier 1: ``ofi`` is a base-series column on the bars frame, not an
    # engine-emitted column.
    tier: ClassVar[int | str] = 1
    inputs = ("ofi",)
    windows_field: ClassVar[str] = "windows_ofi_impulse"


class OfiDelta(_OfiImpulseFeature):
    def column_name(self, w: int | None = None) -> str:
        return f"ofi__delta__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        ofi = pl.col("ofi")
        return ofi - ofi.shift(w)


class OfiMax(_OfiImpulseFeature):
    def column_name(self, w: int | None = None) -> str:
        return f"ofi__max__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_max(pl.col("ofi"), w)


class OfiMin(_OfiImpulseFeature):
    def column_name(self, w: int | None = None) -> str:
        return f"ofi__min__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_min(pl.col("ofi"), w)


class OfiRetInteraction(_OfiImpulseFeature):
    inputs = ("ofi", "r")

    def column_name(self, w: int | None = None) -> str:
        return f"ofi__ret_interaction__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_sum(pl.col("ofi") * pl.col("r").abs(), w)
