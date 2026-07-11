"""Volatility families (spec Group C / Section 7.7-7.8).

Tier-1: OHLC variance estimators (Parkinson, Garman-Klass, Rogers-Satchell)
        — utils.compute_volatility_ohlc + compute_volatility_rs_only.

Tier-2: variance decomposition (BPV ratio, semivariance up/down/ratio,
        VoV) — utils.compute_volatility_decomposition. Depends on
        ret__rms__f__w20 from the rolling family (Tier-1 output).

Output columns (Tier-1):
  - vol__parkinson__f__w{W}  = sqrt(rolling_mean(log_hl² / (4·ln2)))
  - vol__gk__f__w{W}         = sqrt(max(0, rolling_mean(0.5·log_hl² - (2·ln2-1)·log_co²)))
  - vol__rs__f__w{W}         = sqrt(max(0, rolling_mean(rs_variance(o,h,l,c))))

Output columns (Tier-2):
  - vol__bpv_ratio__f__w{W}     = rv / (bpv + EPS)
  - vol__semivar_down__f__w{W}  = sqrt(max(0, rolling_mean(min(r,0)²)))
  - vol__semivar_up__f__w{W}    = sqrt(max(0, rolling_mean(max(r,0)²)))
  - vol__semivar_ratio__f__w{W} = semidown / (semiup + EPS)
  - vol__vov__f__w{W}           = rolling_std_pop(ret__rms__f__w20, W)
"""

from __future__ import annotations

import math
from typing import ClassVar

import polars as pl

from src.features.base import RATIO, Domain, Feature
from src.features.config import EPS
from src.features.primitives import (
    clip_pos,
    rolling_mean,
    rolling_std_pop,
    rolling_sum,
    rs_variance,
    safe_log_ratio,
)


# =============================================================================
# Tier-1: OHLC volatility estimators
# =============================================================================


class _VolOhlcFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "vol"
    tier: ClassVar[int | str] = 1
    windows_field: ClassVar[str] = "windows_vol_ohlc"

    output_name: ClassVar[str] = ""

    def column_name(self, w: int | None = None) -> str:
        return f"vol__{self.output_name}__f__w{w}"


class VolParkinson(_VolOhlcFeature):
    """Parkinson volatility: sqrt(mean(log²(h/l) / (4·ln2))).

    log_hl is null when ``high <= low`` (degenerate candle); legacy guard
    ``np.where(high > low, log(h/l), NaN)`` at utils.py:1702.

    ``clip_pos`` before sqrt for the same reason as ``RollingRetRms`` /
    ``VolGk`` / ``VolRs`` / ``VolSemivar*``: polars' rolling-mean online
    algorithm can produce tiny-negative cancellation residuals on
    all-zero windows (e.g. a flat-bar stretch where every ``log_hl`` is
    null or zero), and sqrt of a tiny negative is NaN. The clamp is a
    mathematical no-op for the non-degenerate path.
    """

    inputs = ("high", "low")
    output_name = "parkinson"

    def compute(self, w: int | None = None) -> pl.Expr:
        log_hl = safe_log_ratio(
            pl.col("high"),
            pl.col("low"),
            when=pl.col("high") > pl.col("low"),
        )
        var_p = (log_hl ** 2) / (4.0 * math.log(2.0))
        return clip_pos(rolling_mean(var_p, w)).sqrt()


class VolGk(_VolOhlcFeature):
    """Garman-Klass volatility.

    var_gk = 0.5·log²(h/l) - (2·ln2 - 1)·log²(c/o); can go negative, so
    ``clip_pos`` before sqrt. legacy guards: ``high > low`` for log_hl,
    ``open > 0`` for log_co (utils.py:1702-1703).
    """

    inputs = ("open", "high", "low", "close")
    output_name = "gk"

    def compute(self, w: int | None = None) -> pl.Expr:
        log_hl = safe_log_ratio(
            pl.col("high"),
            pl.col("low"),
            when=pl.col("high") > pl.col("low"),
        )
        log_co = safe_log_ratio(
            pl.col("close"),
            pl.col("open"),
            when=pl.col("open") > 0,
        )
        var_gk = 0.5 * (log_hl ** 2) - (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)
        return clip_pos(rolling_mean(var_gk, w)).sqrt()


class VolRs(_VolOhlcFeature):
    """Rogers-Satchell volatility.

    Uses unguarded :func:`rs_variance` — matches utils.py:1707-1710 which
    is also unguarded. RS variance can go negative on degenerate ticks;
    ``clip_pos`` before sqrt.
    """

    inputs = ("open", "high", "low", "close")
    output_name = "rs"

    def compute(self, w: int | None = None) -> pl.Expr:
        var_rs = rs_variance(
            pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")
        )
        return clip_pos(rolling_mean(var_rs, w)).sqrt()


# =============================================================================
# Tier-2: volatility decomposition
# =============================================================================


class _VolDecompFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "vol"
    tier: ClassVar[int | str] = 2
    windows_field: ClassVar[str] = "windows_vol_decomp"

    output_name: ClassVar[str] = ""

    def column_name(self, w: int | None = None) -> str:
        return f"vol__{self.output_name}__f__w{w}"


class VolBpvRatio(_VolDecompFeature):
    domain: ClassVar[Domain] = RATIO  # bipower/realized; 1 = no jumps
    """Bipower-variation ratio: realized variance over scaled bipower variation.

    rv = rolling_mean(r², w)
    bpv = (π/2) · rolling_mean(|r|·|r.shift(1)|, w-1)
    ratio = rv / (bpv + EPS)

    Legacy uses a window of ``W-1`` for bpv (utils.py:1761). Effective
    warmup is ``w`` because:
      * r[0] is null (from log_return.diff)
      * ``|r|·|r.shift(1)|`` is null at rows 0 and 1
      * rv = rolling_mean(r², w) first valid at row w (one null in window)
      * bpv = rolling_mean(prod, w-1) first valid at row w
    """

    inputs = ("r",)
    output_name = "bpv_ratio"

    def compute(self, w: int | None = None) -> pl.Expr:
        r = pl.col("r")
        rv = rolling_mean(r ** 2, w)
        prod = r.abs() * r.abs().shift(1)
        bpv = (math.pi / 2.0) * rolling_mean(prod, w - 1)
        return rv / (bpv + EPS)

    def warmup_for(self, w: int | None) -> int:
        return w if w else 0


class VolSemivarDown(_VolDecompFeature):
    """Downside semivariance: sqrt(mean(min(r, 0)²))."""

    inputs = ("r",)
    output_name = "semivar_down"

    def compute(self, w: int | None = None) -> pl.Expr:
        down = pl.min_horizontal(pl.col("r"), pl.lit(0.0)) ** 2
        return clip_pos(rolling_mean(down, w)).sqrt()


class VolSemivarUp(_VolDecompFeature):
    """Upside semivariance: sqrt(mean(max(r, 0)²))."""

    inputs = ("r",)
    output_name = "semivar_up"

    def compute(self, w: int | None = None) -> pl.Expr:
        up = pl.max_horizontal(pl.col("r"), pl.lit(0.0)) ** 2
        return clip_pos(rolling_mean(up, w)).sqrt()


class VolSemivarRatio(_VolDecompFeature):
    domain: ClassVar[Domain] = RATIO  # down/up semivariance parity
    """Down/up semivariance ratio: semidown / (semiup + EPS).

    Computed inline rather than referencing the up/down feature columns
    so this Feature is self-contained — both expressions evaluate
    against the same input column ``r`` in the tier-2 with_columns pass.
    """

    inputs = ("r",)
    output_name = "semivar_ratio"

    def compute(self, w: int | None = None) -> pl.Expr:
        down = pl.min_horizontal(pl.col("r"), pl.lit(0.0)) ** 2
        up = pl.max_horizontal(pl.col("r"), pl.lit(0.0)) ** 2
        semidown = clip_pos(rolling_mean(down, w)).sqrt()
        semiup = clip_pos(rolling_mean(up, w)).sqrt()
        return semidown / (semiup + EPS)


class VolVov(_VolDecompFeature):
    """Volatility-of-volatility: rolling pop-std of ret__rms__f__w20.

    Depends on a Tier-1 output (rolling family). The engine runs Tier-1
    first, so by the time this expression evaluates, ``ret__rms__f__w20``
    is a column on the frame.
    """

    inputs = ("ret__rms__f__w20",)
    output_name = "vov"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_std_pop(pl.col("ret__rms__f__w20"), w)


# =============================================================================
# Round 1d additions
# =============================================================================


class VolSemivarSigned(Feature):
    """Signed semivariance ratio in [-1, 1]:

        SVS_{W,n} = (RV+ - RV-) / (RV+ + RV- + EPS)

    where ``RV+`` and ``RV-`` are the sums of squared positive / negative
    returns over the trailing W bars. Positive when upside variance
    dominates; negative when downside dominates.

    Sister to ``vol__semivar_ratio`` (the ratio of magnitudes); this
    variant is bounded and zero-centred, which trees split on more
    cleanly.
    """

    family: ClassVar[str] = "vol"
    tier: ClassVar[int | str] = 2
    inputs = ("r",)
    windows_field: ClassVar[str] = "windows_vol_signed"

    def column_name(self, w: int | None = None) -> str:
        return f"vol__semivar_signed__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        r = pl.col("r")
        pos_sq = pl.when(r > 0).then(r ** 2).otherwise(0.0)
        neg_sq = pl.when(r < 0).then(r ** 2).otherwise(0.0)
        rv_pos = rolling_sum(pos_sq, w)
        rv_neg = rolling_sum(neg_sq, w)
        return (rv_pos - rv_neg) / (rv_pos + rv_neg + EPS)


class VolJumpRatio(Feature):
    """Bipower-jump share clipped to [0, 1]:

        RV   = sum(r^2) over [n-W+1, n]
        BV   = (pi / 2) * sum(|r_i| * |r_{i-1}|) over [n-W+1, n] (W-1 terms)
        J    = clip((RV - BV) / (RV + EPS), 0, 1)

    Captures the share of trailing variance attributable to jumps under
    the Barndorff-Nielsen-Shephard decomposition. Differs from
    ``vol__bpv_ratio`` in that this one is clamped to [0, 1] and is a
    fraction rather than a ratio.

    Warmup is ``W + 1`` rows: ``r[0]`` is null from ``log_return``, and
    the bipower term needs ``r_{i-1}`` so the bipower sum's first valid
    row is one later than the realized-variance sum.
    """

    family: ClassVar[str] = "vol"
    tier: ClassVar[int | str] = 2
    inputs = ("r",)
    windows_field: ClassVar[str] = "windows_vol_jump"

    def column_name(self, w: int | None = None) -> str:
        return f"vol__jump_ratio__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        # r[0] is null (log_return). r² rolling_sum(w) first valid at row n=w.
        # prod = |r| * |r.shift(1)| is null at rows 0 and 1; rolling_sum(w-1)
        # first valid at row n where window [n-w+2..n] starts at index ≥ 2,
        # i.e., n ≥ w. So jump_ratio first valid at row n=w (rows 0..w-1 null).
        return w if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        r = pl.col("r")
        rv = rolling_sum(r ** 2, w)
        prod = r.abs() * r.abs().shift(1)
        bv = (math.pi / 2.0) * rolling_sum(prod, w - 1)
        raw = (rv - bv) / (rv + EPS)
        return raw.clip(lower_bound=0.0, upper_bound=1.0)
