"""Serial-dependence and correlation features (spec Group G / Section 7.15).

Mirrors ``utils.compute_correlations`` (utils.py:1943-1958). Four features
× len(WINDOWS_CORR) = 48 columns. All use ``population_corr`` (ddof=0) per
the legacy ``_rolling_corr_population`` definition.

Output columns:
  - ret__acf1__f__w{W}            = pop_corr(r, r.shift(1), W)
  - ret__corr_logvol__f__w{W}     = pop_corr(r, logvol, W)
  - absret__corr_logvol__f__w{W}  = pop_corr(|r|, logvol, W)
  - ret__corr_ofi__f__w{W}        = pop_corr(r, ofi, W)
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from src.features.base import Feature
from src.features.config import WINDOWS_CORR
from src.features.primitives import population_corr


class _CorrFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "correlation"
    tier: ClassVar[int | str] = 1
    windows = tuple(WINDOWS_CORR)

    output_name: ClassVar[str] = ""

    def column_name(self, w: int | None = None) -> str:
        return f"{self.output_name}__f__w{w}"


class CorrAcf1(_CorrFeature):
    inputs = ("r",)
    output_name = "ret__acf1"

    def compute(self, w: int | None = None) -> pl.Expr:
        r = pl.col("r")
        return population_corr(r, r.shift(1), w)


class CorrRetLogvol(_CorrFeature):
    inputs = ("r", "logvol")
    output_name = "ret__corr_logvol"

    def compute(self, w: int | None = None) -> pl.Expr:
        return population_corr(pl.col("r"), pl.col("logvol"), w)


class CorrAbsretLogvol(_CorrFeature):
    inputs = ("r", "logvol")
    output_name = "absret__corr_logvol"

    def compute(self, w: int | None = None) -> pl.Expr:
        return population_corr(pl.col("r").abs(), pl.col("logvol"), w)


class CorrRetOfi(_CorrFeature):
    inputs = ("r", "ofi")
    output_name = "ret__corr_ofi"

    def compute(self, w: int | None = None) -> pl.Expr:
        return population_corr(pl.col("r"), pl.col("ofi"), w)
