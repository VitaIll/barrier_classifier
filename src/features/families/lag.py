"""Lag features (spec Group A / Section 7.4).

Mirrors ``utils.compute_lag_features`` (utils.py:1576-1600). One Feature class
per (input column, output prefix) pair; ``windows`` holds the lag values.

Output columns:
  - ret__lag{L}__f__w0       =  r.shift(L)
  - absret__lag{L}__f__w0    =  r.shift(L).abs()
  - range__lag{L}__f__w0     =  rho.shift(L)
  - clv__lag{L}__f__w0       =  clv.shift(L)
  - logvol__lag{L}__f__w0    =  logvol.shift(L)
  - logtrades__lag{L}__f__w0 =  logtrades.shift(L)
  - ofi__lag{L}__f__w0       =  ofi.shift(L)
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from src.features.base import Feature


class _LagFeature(Feature):
    """Abstract base: shift one column by ``L`` rows.

    ``windows`` is reused to mean lag values — the framework's window
    expansion semantics are identical (one column per element). The custom
    ``column_name`` follows the legacy spec convention rather than the
    canonical ``{family}__{name}__w{w}`` because ``compute_lag_features``
    emits ``{prefix}__lag{L}__f__w0``.
    """

    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "lag"
    tier: ClassVar[int | str] = 1
    windows_field: ClassVar[str] = "lags_f"

    output_prefix: ClassVar[str] = ""  # set by subclass

    def column_name(self, w: int | None = None) -> str:
        return f"{self.output_prefix}__lag{w}__f__w0"

    def warmup_for(self, w: int | None) -> int:
        # shift(L) nulls the first L rows (not L-1).
        return w if w else 0


class LagRet(_LagFeature):
    inputs = ("r",)
    output_prefix = "ret"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("r").shift(w)


class LagAbsret(_LagFeature):
    inputs = ("r",)
    output_prefix = "absret"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("r").shift(w).abs()


class LagRange(_LagFeature):
    inputs = ("rho",)
    output_prefix = "range"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("rho").shift(w)


class LagClv(_LagFeature):
    inputs = ("clv",)
    output_prefix = "clv"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("clv").shift(w)


class LagLogvol(_LagFeature):
    inputs = ("logvol",)
    output_prefix = "logvol"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("logvol").shift(w)


class LagLogtrades(_LagFeature):
    inputs = ("logtrades",)
    output_prefix = "logtrades"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("logtrades").shift(w)


class LagOfi(_LagFeature):
    inputs = ("ofi",)
    output_prefix = "ofi"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("ofi").shift(w)
