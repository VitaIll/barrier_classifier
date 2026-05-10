"""Seasonality / time-context features (spec Group Q / Section 7.17).

Mirrors ``utils.compute_seasonality`` (utils.py:2029-2044). Sin/cos
encodings of minute-of-day and day-of-week from a timestamp column.

The polars frame must have a ``ts`` column of dtype Datetime (the
parity tests use ``df.reset_index(names="ts")`` to convert from the
pandas DatetimeIndex used in the legacy).

Day-of-week parity note: pandas ``DatetimeIndex.dayofweek`` is 0-indexed
(Monday=0, Sunday=6). polars ``Expr.dt.weekday()`` returns ISO 1-indexed
(Monday=1, Sunday=7). The features subtract 1 from polars to match.

Output columns:
  - time__sin_minute__f__w0, time__cos_minute__f__w0
  - time__sin_dow__f__w0, time__cos_dow__f__w0
"""

from __future__ import annotations

import math
from typing import ClassVar

import polars as pl

from src.features.base import Feature


class _SeasonalityFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "seasonality"
    tier: ClassVar[int | str] = 1
    windows: ClassVar[tuple[int, ...]] = ()
    inputs = ("ts",)

    output_prefix: ClassVar[str] = ""

    def column_name(self, w: int | None = None) -> str:
        return f"time__{self.output_prefix}__f__w0"


def _minute_of_day(ts: pl.Expr) -> pl.Expr:
    """Minute of day, 0..1439.

    Cast to Int32 first — polars ``dt.hour()`` returns Int8, and
    ``23 * 60 = 1380`` overflows that range silently. The cast must
    happen on each component before the arithmetic.
    """
    return ts.dt.hour().cast(pl.Int32) * 60 + ts.dt.minute().cast(pl.Int32)


def _dow_pandas(ts: pl.Expr) -> pl.Expr:
    """Day of week in pandas convention: Monday=0, Sunday=6."""
    return ts.dt.weekday().cast(pl.Int32) - 1


class SeasonalitySinMinute(_SeasonalityFeature):
    output_prefix = "sin_minute"

    def compute(self, w: int | None = None) -> pl.Expr:
        m = _minute_of_day(pl.col("ts"))
        return (2.0 * math.pi * m / 1440.0).sin()


class SeasonalityCosMinute(_SeasonalityFeature):
    output_prefix = "cos_minute"

    def compute(self, w: int | None = None) -> pl.Expr:
        m = _minute_of_day(pl.col("ts"))
        return (2.0 * math.pi * m / 1440.0).cos()


class SeasonalitySinDow(_SeasonalityFeature):
    output_prefix = "sin_dow"

    def compute(self, w: int | None = None) -> pl.Expr:
        d = _dow_pandas(pl.col("ts"))
        return (2.0 * math.pi * d / 7.0).sin()


class SeasonalityCosDow(_SeasonalityFeature):
    output_prefix = "cos_dow"

    def compute(self, w: int | None = None) -> pl.Expr:
        d = _dow_pandas(pl.col("ts"))
        return (2.0 * math.pi * d / 7.0).cos()
