"""Trend / momentum features (spec Group E / Section 7.12).

Mirrors ``utils.compute_trend_momentum`` (utils.py:1847-1881).

Output columns:
  - logp__z__f__w{W}                     for W in WINDOWS_LOGP_Z
  - logp__ema_spread__f__w0__fast{X}__slow{Y}   for (10,60), (20,120), (60,240)
  - ret__rsi__f__w{W}                    for W in WINDOWS_RSI

The RSI feature uses the documented ``shift(-1) / shift(1)`` sandwich
around ``wilder_smooth`` to reproduce the legacy ``_wilder_rsi`` seed
offset (seed at index W from ``mean(gain[1..W])``, skipping ``r[0]``
which is null from log_return.diff).
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from src.features.base import Feature
from src.features.config import EPS, WINDOWS_LOGP_Z, WINDOWS_RSI
from src.features.primitives import ewm_mean, wilder_smooth, z_score_rolling


# -------- Log-price z-score -------------------------------------------------


class TrendLogpZ(Feature):
    family: ClassVar[str] = "trend"
    tier: ClassVar[int | str] = 1
    inputs = ("p",)
    windows = tuple(WINDOWS_LOGP_Z)

    def column_name(self, w: int | None = None) -> str:
        return f"logp__z__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return z_score_rolling(pl.col("p"), w)


# -------- EMA-spread features (3 fixed pairs) ------------------------------


class _TrendEmaSpread(Feature):
    """One column per fixed (fast, slow) EMA pair on log price."""

    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "trend"
    tier: ClassVar[int | str] = 1
    windows: ClassVar[tuple[int, ...]] = ()
    inputs = ("p",)

    fast: ClassVar[int] = 0
    slow: ClassVar[int] = 0

    def column_name(self, w: int | None = None) -> str:
        return f"logp__ema_spread__f__w0__fast{self.fast}__slow{self.slow}"

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        return ewm_mean(p, span=self.fast) - ewm_mean(p, span=self.slow)


class TrendEmaSpread10_60(_TrendEmaSpread):
    fast = 10
    slow = 60


class TrendEmaSpread20_120(_TrendEmaSpread):
    fast = 20
    slow = 120


class TrendEmaSpread60_240(_TrendEmaSpread):
    fast = 60
    slow = 240


# -------- Wilder RSI --------------------------------------------------------


class TrendRsi(Feature):
    """Wilder's RSI on log returns.

    Compute path:
      1. r_filled = pl.col("r").fill_null(0.0)        (null → 0 to match legacy NaN→0)
      2. gain = where(r_filled > 0, r_filled, 0.0)
         loss = where(r_filled < 0, -r_filled, 0.0)
      3. avg_gain = wilder_smooth(gain.shift(-1), W).shift(1)
         avg_loss = wilder_smooth(loss.shift(-1), W).shift(1)
         (shift trick reproduces legacy seed at index W using mean(gain[1..W]))
      4. rs = avg_gain / (avg_loss + EPS)
         rsi = 100 - 100 / (1 + rs)

    Warmup is W (rows 0..W-1 are missing; first valid output at row W),
    matching legacy ``_wilder_rsi`` (utils.py:1843).
    """

    family: ClassVar[str] = "trend"
    tier: ClassVar[int | str] = 1
    inputs = ("r",)
    windows = tuple(WINDOWS_RSI)

    def column_name(self, w: int | None = None) -> str:
        return f"ret__rsi__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        return w if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        r_filled = pl.col("r").fill_null(0.0)
        gain = pl.when(r_filled > 0).then(r_filled).otherwise(0.0)
        loss = pl.when(r_filled < 0).then(-r_filled).otherwise(0.0)
        avg_gain = wilder_smooth(gain.shift(-1), w).shift(1)
        avg_loss = wilder_smooth(loss.shift(-1), w).shift(1)
        rs = avg_gain / (avg_loss + EPS)
        return 100.0 - 100.0 / (1.0 + rs)
