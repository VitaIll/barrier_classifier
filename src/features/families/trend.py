"""Trend / momentum features (spec Group E / Section 7.12).

Mirrors ``utils.compute_trend_momentum`` (utils.py:1847-1881).

Output columns:
  - logp__z__f__w{W}                     for W in WINDOWS_LOGP_Z
  - logp__ema_spread__f__w0__fast{X}__slow{Y}   for (10,60), (20,120), (60,240)
  - ret__rsi__f__w{W}                    for W in WINDOWS_RSI

Round 1b additions (Tier-2, depend on ``vol__rs__f__w{W}``):
  - trend__quad_slope_z__f__w{W}         volatility-normalized linear coefficient
                                          of a least-squares quadratic fit to log
                                          price over the trailing W bars.
  - trend__quad_curv_z__f__w{W}          volatility-normalized quadratic coefficient
                                          of the same fit. Captures inflection.

The RSI feature uses the documented ``shift(-1) / shift(1)`` sandwich
around ``wilder_smooth`` to reproduce the legacy ``_wilder_rsi`` seed
offset (seed at index W from ``mean(gain[1..W])``, skipping ``r[0]``
which is null from log_return.diff).
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import polars as pl

from src.features.base import OSCILLATOR_0_100, Domain, Feature
from src.features.config import EPS
from src.features.primitives import ewm_mean, wilder_smooth, z_score_rolling


# -------- Log-price z-score -------------------------------------------------


class TrendLogpZ(Feature):
    family: ClassVar[str] = "trend"
    tier: ClassVar[int | str] = 1
    inputs = ("p",)
    windows_field: ClassVar[str] = "windows_logp_z"

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
    domain: ClassVar[Domain] = OSCILLATOR_0_100  # RSI
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
    windows_field: ClassVar[str] = "windows_rsi"

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


# -------- Quadratic trend (slope + curvature) ------------------------------


def _quad_trend_np(p: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form least-squares quadratic fit on each trailing w-bar window.

    The design uses centred ``u_j = j - (w - 1) / 2`` so ``sum(u) = 0`` and
    ``sum(u^3) = 0``. That makes the linear coefficient decouple from the
    intercept and the quadratic term:

        beta1 = sum(u * p) / sum(u^2)

    and the quadratic + intercept satisfy the 2x2 normal equations:

        [[w,        sum(u^2)],
         [sum(u^2), sum(u^4)]] * [alpha, beta2]^T = [sum(p), sum(u^2 * p)]^T

    So:

        det   = w * sum(u^4) - sum(u^2)^2
        beta2 = (w * sum(u^2 * p) - sum(u^2) * sum(p)) / det

    Returns (slope, curvature) — both raw coefficients, NOT yet normalized
    by volatility. Caller applies the vol normalization in polars.
    """
    n = len(p)
    slope = np.full(n, np.nan, dtype=float)
    curv = np.full(n, np.nan, dtype=float)
    w_int = int(w)
    if n < w_int:
        return slope, curv

    # Fixed design moments (depend only on w)
    u = np.arange(w_int, dtype=float) - (w_int - 1) / 2.0
    u2 = u ** 2
    sum_u2 = float(u2.sum())
    sum_u4 = float((u2 ** 2).sum())
    det = float(w_int) * sum_u4 - sum_u2 ** 2

    eligible = np.arange(w_int - 1, n, dtype=np.int64)
    offsets = np.arange(w_int - 1, -1, -1, dtype=np.int64)
    max_elements = 5_000_000
    chunk_size = int(min(20_000, max(1, max_elements // w_int)))

    for start in range(0, len(eligible), chunk_size):
        idx = eligible[start : start + chunk_size]
        rows = idx[:, None] - offsets[None, :]
        # ``offsets`` reverses so column 0 is the oldest bar and column w-1
        # is the newest. ``u`` was built for j=0..w-1 in the same order
        # (oldest -> newest), so no flip needed.
        window_vals = p[rows]
        invalid = np.isnan(window_vals).any(axis=1)

        S1 = window_vals.sum(axis=1)
        Suy = (window_vals * u).sum(axis=1)
        Su2y = (window_vals * u2).sum(axis=1)

        beta1 = Suy / sum_u2
        beta2 = (float(w_int) * Su2y - sum_u2 * S1) / det

        beta1[invalid] = np.nan
        beta2[invalid] = np.nan
        slope[idx] = beta1
        curv[idx] = beta2

    return slope, curv


class _TrendQuad(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "trend"
    # Tier 2: depends on ``vol__rs__f__w{W}`` for normalization.
    tier: ClassVar[int | str] = 2
    inputs = ("p",)
    windows_field: ClassVar[str] = "windows_quad_trend"

    # Sub-class picks 0 (slope) or 1 (curvature) — same kernel, two outputs.
    _select: ClassVar[int] = 0


class TrendQuadSlopeZ(_TrendQuad):
    """Volatility-normalized linear coefficient of the quadratic fit.

        slope = beta1 * W / (sigma_W * sqrt(W) + EPS) = beta1 * sqrt(W) / sigma_W
    """

    _select = 0

    def column_name(self, w: int | None = None) -> str:
        return f"trend__quad_slope_z__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        raw = pl.col("p").map_batches(
            lambda s, ww=w: pl.Series(_quad_trend_np(s.to_numpy(), ww)[0]),
            return_dtype=pl.Float64,
        )
        vol_col = pl.col(f"vol__rs__f__w{w}")
        return raw * float(w) / (vol_col * math.sqrt(int(w)) + EPS)


class TrendQuadCurvZ(_TrendQuad):
    """Volatility-normalized quadratic coefficient of the quadratic fit.

        curv = beta2 * W^2 / (sigma_W * sqrt(W) + EPS)
    """

    _select = 1

    def column_name(self, w: int | None = None) -> str:
        return f"trend__quad_curv_z__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        raw = pl.col("p").map_batches(
            lambda s, ww=w: pl.Series(_quad_trend_np(s.to_numpy(), ww)[1]),
            return_dtype=pl.Float64,
        )
        vol_col = pl.col(f"vol__rs__f__w{w}")
        return raw * (float(w) ** 2) / (vol_col * math.sqrt(int(w)) + EPS)
