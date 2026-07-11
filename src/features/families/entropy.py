"""Permutation entropy features (spec Group H / Section 7.16).

Mirrors ``utils.compute_permutation_entropy`` (utils.py:2245-2302) with
``m=3, tau=1``. One feature × len(WINDOWS_PENTROPY) = 4 columns.

Output columns:
  - pentropy_norm__inst__f__w{W}__m3__tau1
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from src.features.base import FRACTION, Domain, Feature
from src.features.primitives import perm_entropy_m3


class EntropyPermNorm(Feature):
    domain: ClassVar[Domain] = FRACTION  # normalized entropy
    family: ClassVar[str] = "entropy"
    tier: ClassVar[int | str] = 1
    inputs = ("r",)
    windows_field: ClassVar[str] = "windows_pentropy"

    def column_name(self, w: int | None = None) -> str:
        return f"pentropy_norm__inst__f__w{w}__m3__tau1"

    def compute(self, w: int | None = None) -> pl.Expr:
        return perm_entropy_m3(pl.col("r"), w)
