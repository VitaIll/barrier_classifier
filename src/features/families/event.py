"""Event / streak features (spec Group J / Section 7.19).

Mirrors ``utils.compute_event_features`` (utils.py:1962-2000). Three
columns from one signed-streak state machine over ``r``.

Output columns:
  - event__run_dir__f__w0     (Int8;  -1 / 0 / +1)
  - event__run_len__f__w0     (Int32; bars in current run)
  - event__run_cumret__f__w0  (Float64; cumulative return within run)
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from src.features.base import Feature
from src.features.primitives import (
    signed_run_cumret,
    signed_run_dir,
    signed_run_length,
)


class _EventFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "event"
    tier: ClassVar[int | str] = 1
    windows: ClassVar[tuple[int, ...]] = ()
    inputs = ("r",)

    output_prefix: ClassVar[str] = ""

    def column_name(self, w: int | None = None) -> str:
        return f"event__{self.output_prefix}__f__w0"


class EventRunDir(_EventFeature):
    output_prefix = "run_dir"

    def compute(self, w: int | None = None) -> pl.Expr:
        return signed_run_dir(pl.col("r"))


class EventRunLen(_EventFeature):
    output_prefix = "run_len"

    def compute(self, w: int | None = None) -> pl.Expr:
        return signed_run_length(pl.col("r"))


class EventRunCumret(_EventFeature):
    output_prefix = "run_cumret"

    def compute(self, w: int | None = None) -> pl.Expr:
        return signed_run_cumret(pl.col("r"))
