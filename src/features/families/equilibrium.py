"""Equilibrium-residual feature family (Round 2).

Builds several local fair-value proxies from past-only windows (the strict
``H_W(n) = {n-W, ..., n-1}`` set) and compares the current log close against
them on different scales:

  - mean / VWAP / range-midpoint residuals normalized by horizon volatility
  - median residual normalized by MAD
  - linear-trend residual normalized by the fit's own residual std
  - pre-update EWMA innovation normalized by EWMA-of-r² scale
  - barrier / pullback / overextension interactions with the project ``PHI``

Tier-1 emits the raw proxy values and the scale columns (the ``mu`` and
``sigma`` columns). Tier-2 reads those by name and emits the residual /
interaction features. All columns end in ``__f__…`` so the existing causal
audit treats them as frozen-up-to-now.

Past-only window convention: every rolling primitive output gets
``.shift(1)``. At row ``n`` the value is what ``rolling_X(x, w)`` produced
at row ``n-1`` — i.e. computed over rows ``[n-w, n-1]``. The first valid
output is therefore at row ``w`` (not ``w-1``); ``warmup_for(w)`` returns
``w`` for every window-indexed feature in this family.
"""

from __future__ import annotations

import math
from typing import ClassVar

import polars as pl

from src.features.base import Feature
from src.features.config import (
    EPS,
    HALFLIVES_EQ,
    M,
    PHI,
    WINDOWS_EQ,
    WINDOWS_EQ_PAIRS,
)
from src.features.primitives import (
    ewm_mean_halflife,
    past_only_linear_trend_mu,
    past_only_linear_trend_sresid,
    rolling_max,
    rolling_mad,
    rolling_mean,
    rolling_min,
    rolling_quantile,
    rolling_std_pop,
    rolling_sum,
)


_SQRT_M = math.sqrt(float(M))


# =============================================================================
# Tier-1: equilibrium proxies and scales
#
# Every feature here applies ``.shift(1)`` to its rolling primitive so the
# window at row ``n`` covers rows ``[n-w, n-1]`` strictly — the current bar
# is excluded from the equilibrium estimate. ``warmup_for(w) = w`` accounts
# for the shift.
# =============================================================================


class _EqTier1Window(Feature):
    """Window-indexed tier-1 base. All eq tier-1 features iterate ``WINDOWS_EQ``."""

    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "eq"
    tier: ClassVar[int | str] = 1
    windows: ClassVar[tuple[int, ...]] = tuple(WINDOWS_EQ)

    def warmup_for(self, w: int | None) -> int:
        return w if w else 0


class _EqTier1HalfLife(Feature):
    """Half-life-indexed tier-1 base for the EWMA proxies. Iterates
    ``HALFLIVES_EQ``; the variable is a half-life ``H`` rather than a window
    width, surfaced as ``__h{H}`` in the column suffix."""

    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "eq"
    tier: ClassVar[int | str] = 1
    windows: ClassVar[tuple[int, ...]] = tuple(HALFLIVES_EQ)

    def warmup_for(self, w: int | None) -> int:
        # EWMA with adjust=False seeds at row 0 (y_0 = x_0) and never
        # produces null for valid inputs. The single warmup row comes
        # from the .shift(1) we apply in compute() to get the pre-update
        # state. Declaring more than 1 would over-trim valid rows when
        # the engine is run with only halflife features and no large-W
        # window features driving the trim.
        return 1


# -------- 1. Mean proxy -----------------------------------------------------


class EqMuMean(_EqTier1Window):
    """Past-only rolling arithmetic mean of log close."""

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__mu_mean__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_mean(pl.col("p"), w).shift(1)


# -------- 2. Median proxy ---------------------------------------------------


class EqMuMedian(_EqTier1Window):
    """Past-only rolling median of log close.

    Robust to one-bar spikes / wicks — the typical pull a flash crash has on
    the simple mean is absorbed here, giving a cleaner ``where is the central
    traded region?`` signal.
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__mu_median__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_quantile(pl.col("p"), w, 0.5).shift(1)


# -------- 3. VWAP proxy -----------------------------------------------------


class EqMuVwap(_EqTier1Window):
    """Past-only VWAP equilibrium: ``log(sum(close·volume) / sum(volume))``.

    Bars with non-positive close, negative volume, or non-finite values
    contribute ``null`` and propagate through ``rolling_sum`` so any
    contaminated window is null. Windows where the trailing volume sum is
    exactly zero (flat-volume stretch) also return null — VWAP undefined.

    Volume contributes its raw scale (not log1p). The ratio cancels the
    multiplicative scaling so the VWAP price is denominator-free of units.
    """

    inputs = ("close", "volume")

    def column_name(self, w: int | None = None) -> str:
        return f"eq__mu_vwap__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        close = pl.col("close")
        volume = pl.col("volume")
        safe_close = (
            pl.when(close.is_finite() & (close > 0)).then(close).otherwise(None)
        )
        safe_volume = (
            pl.when(volume.is_finite() & (volume >= 0))
            .then(volume)
            .otherwise(None)
        )
        num = rolling_sum(safe_close * safe_volume, w).shift(1)
        denom = rolling_sum(safe_volume, w).shift(1)
        # When denom > 0 the window contains at least one positive-volume
        # bar (with a valid close), so num > 0 and the log is defined.
        return (
            pl.when(denom > 0)
            .then((num / denom).log())
            .otherwise(None)
        )


# -------- 4. Range midpoint -------------------------------------------------


class EqMuRange(_EqTier1Window):
    """Midpoint of the trailing past-only [min, max] band of log close."""

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__mu_range__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        return ((rolling_max(p, w) + rolling_min(p, w)) / 2.0).shift(1)


# -------- 5. Linear-trend proxy + residual scale ---------------------------


class EqMuTrend(_EqTier1Window):
    """Linear-OLS fair value extrapolated to the current bar.

    Fits ``p_i = a + b·x_i`` over the past-only window with
    ``x_i = i - n ∈ [-W, -1]`` and reports the intercept at ``x = 0`` —
    i.e. the trend line's value at the current-bar position.

    See :func:`src.features.primitives._past_only_linear_trend_np` for the
    closed-form OLS kernel. ``.fill_nan(None)`` normalizes the kernel's
    warmup NaNs to polars null so downstream tier-2 features propagate
    null (not NaN) and the engine's per-feature null-count contract is
    honoured. This matches the project's NaN-vs-null discipline: numpy
    kernels naturally emit NaN, but every column the engine emits should
    use null for "missing" so the impute layer can flag it correctly.
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__mu_trend__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return past_only_linear_trend_mu(pl.col("p"), w).fill_nan(None)


class EqTrendSresid(_EqTier1Window):
    """Residual std (population) from the same past-only linear OLS fit.

    Used as the denominator of ``eq__trend_resid_z`` — the in-window
    natural scale of departures from the local trend line. Zero or
    tiny-negative residual variance (perfect-line windows / float
    cancellation) is clamped at zero by the kernel before sqrt.
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__trend_sresid__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return past_only_linear_trend_sresid(pl.col("p"), w).fill_nan(None)


# -------- 6. Return volatility scale ---------------------------------------


class EqSigmaR(_EqTier1Window):
    """Past-only population std of 1-bar log returns.

    Multiplied by ``sqrt(M)`` at the use site to get the M-bar horizon
    volatility (the natural denominator for residual-to-barrier-scale
    normalization).

    Warmup is ``w + 1``: ``r[0]`` is null from ``log_return.diff``, so
    ``rolling_std_pop(r, w)``'s first valid row is ``w`` (one row later
    than the pure-``p`` rolling features); applying ``.shift(1)`` for
    past-only semantics adds another row. First non-null output at row
    ``w + 1``.
    """

    inputs = ("r",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__sigma_r__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        return (w + 1) if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_std_pop(pl.col("r"), w).shift(1)


# -------- 7. MAD scale on log price ----------------------------------------


class EqMadP(_EqTier1Window):
    """Past-only MAD of log close, scaled by 1.4826 to match Gaussian std.

    Robust denominator paired with the median proxy. Coefficient 1.4826 is
    the canonical MAD-to-std conversion for Gaussian-distributed inputs
    (``1 / Φ⁻¹(0.75)`` rounded).
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__mad_p__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        # rolling_mad uses a numpy kernel that emits NaN during warmup;
        # the post-shift result is a mix of NaN (kernel warmup) and null
        # (shift-induced at row 0). Coerce to a uniform null pattern so
        # downstream tier-2 propagates cleanly and null_count is correct.
        return (1.4826 * rolling_mad(pl.col("p"), w)).shift(1).fill_nan(None)


# -------- 8-9. EWMA proxy and EWMA-of-r² scale ----------------------------


class EqMuEwm(_EqTier1HalfLife):
    """Pre-update EWMA of log close at half-life H.

    ``ewm_mean(p, half_life=H, adjust=False).shift(1)``: the shift makes the
    state at row ``n`` reflect the EWMA from row ``n-1``, before the current
    bar's contribution. The recursion is
    ``mu_n = α·p_n + (1-α)·mu_{n-1}`` with ``α = 1 - 2^(-1/H)``.
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__mu_ewm__f__h{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return ewm_mean_halflife(pl.col("p"), half_life=w).shift(1)


class EqSigmaREwm(_EqTier1HalfLife):
    """Pre-update EWMA-of-r² scale at half-life H, square-rooted.

    ``r`` is null at row 0 (log_return.diff). We fill that single null with
    0 so the recursion seeds cleanly at zero variance, then let the EWMA
    catch up — mirrors the way ``RollingRetPosfrac`` handles the same
    edge case for legacy parity (``r.fill_null(0.0)``).
    """

    inputs = ("r",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__sigma_r_ewm__f__h{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        r_filled = pl.col("r").fill_null(0.0)
        ewm_var = ewm_mean_halflife(r_filled ** 2, half_life=w)
        # clip at zero before sqrt (defense in depth against float cancellation)
        return ewm_var.clip(lower_bound=0.0).sqrt().shift(1)


# =============================================================================
# Tier-2: residuals and interactions
#
# Each reads tier-1 proxy / scale columns by name. The engine runs tier-1
# before tier-2, so by the time these expressions evaluate the proxy
# columns already exist on the frame.
# =============================================================================


class _EqTier2Window(Feature):
    """Window-indexed tier-2 base."""

    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "eq"
    tier: ClassVar[int | str] = 2
    windows: ClassVar[tuple[int, ...]] = tuple(WINDOWS_EQ)

    def warmup_for(self, w: int | None) -> int:
        return w if w else 0


class _EqTier2HalfLife(Feature):
    """Half-life-indexed tier-2 base."""

    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "eq"
    tier: ClassVar[int | str] = 2
    windows: ClassVar[tuple[int, ...]] = tuple(HALFLIVES_EQ)

    def warmup_for(self, w: int | None) -> int:
        # Same as the tier-1 EWMA base: only the .shift(1) contributes a
        # warmup row. The tier-2 residual is null exactly when its mu/sigma
        # inputs are null, i.e. at row 0.
        return 1


# -------- Feature 1: mean residual / horizon vol ---------------------------


class EqMeanResidHz(_EqTier2Window):
    """``(p - eq__mu_mean) / (eq__sigma_r · √M + EPS)``.

    Tells the classifier the size, in M-bar volatility units, of price's
    deviation from the recent average clearing level. Negative means
    current price is below the local average; positive means above.
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__mean_resid_hz__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        # sigma_r warmup is w+1 (r[0] null + shift); binding constraint here.
        return (w + 1) if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        mu = pl.col(f"eq__mu_mean__f__w{w}")
        sigma = pl.col(f"eq__sigma_r__f__w{w}")
        return (p - mu) / (sigma * _SQRT_M + EPS)


# -------- Feature 2: median residual / MAD ---------------------------------


class EqMedianResidMadz(_EqTier2Window):
    """``(p - eq__mu_median) / (eq__mad_p + EPS)``.

    Robust z-score against a local median fair value. Outlier-resistant
    counterpart to ``eq__mean_resid_hz``.
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__median_resid_madz__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        mu = pl.col(f"eq__mu_median__f__w{w}")
        scale = pl.col(f"eq__mad_p__f__w{w}")
        return (p - mu) / (scale + EPS)


# -------- Feature 3: VWAP residual / horizon vol ---------------------------


class EqVwapResidHz(_EqTier2Window):
    """``(p - eq__mu_vwap) / (eq__sigma_r · √M + EPS)``.

    Where price sits relative to the recent volume-weighted clearing
    level. Below VWAP → potential reversion room; far above → either
    breakout demand or overextension (the model decides via interactions).
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__vwap_resid_hz__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        return (w + 1) if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        mu = pl.col(f"eq__mu_vwap__f__w{w}")
        sigma = pl.col(f"eq__sigma_r__f__w{w}")
        return (p - mu) / (sigma * _SQRT_M + EPS)


# -------- Feature 4: range-midpoint residual / horizon vol -----------------


class EqRangeMidResidHz(_EqTier2Window):
    """``(p - eq__mu_range) / (eq__sigma_r · √M + EPS)``.

    Microstructure-style equilibrium — midpoint of the recent [min, max]
    band. Normalized so low-vol and high-vol regimes are comparable.
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__range_mid_resid_hz__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        return (w + 1) if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        mu = pl.col(f"eq__mu_range__f__w{w}")
        sigma = pl.col(f"eq__sigma_r__f__w{w}")
        return (p - mu) / (sigma * _SQRT_M + EPS)


# -------- Feature 5: linear-trend residual / fit residual scale ------------


class EqTrendResidZ(_EqTier2Window):
    """``(p - eq__mu_trend) / (eq__trend_sresid + EPS)``.

    The crucial trend-aware residual: distinguishes ``below flat equilibrium``
    (small magnitude) from ``below rising equilibrium`` (negative) from
    ``below falling equilibrium`` (large negative — the trend itself is
    moving away). For a long upper-barrier classifier the third case is a
    much worse setup than the second, and this feature is the cleanest
    way to tell them apart.
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__trend_resid_z__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        mu = pl.col(f"eq__mu_trend__f__w{w}")
        scale = pl.col(f"eq__trend_sresid__f__w{w}")
        return (p - mu) / (scale + EPS)


# -------- Feature 6: EWMA innovation / EWMA-vol scale ---------------------


class EqEwmInnovHz(_EqTier2HalfLife):
    """``(p - eq__mu_ewm) / (eq__sigma_r_ewm · √M + EPS)``.

    Innovation against the pre-update EWMA equilibrium, normalized by the
    pre-update EWMA-of-r² scale. The pre-update construction is causal —
    the current bar cannot pull the fair-value estimate toward itself
    before the residual is measured.
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__ewm_innov_hz__f__h{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        mu = pl.col(f"eq__mu_ewm__f__h{w}")
        sigma = pl.col(f"eq__sigma_r_ewm__f__h{w}")
        return (p - mu) / (sigma * _SQRT_M + EPS)


# -------- Feature 7: upside-to-equilibrium / phi ----------------------------


class _EqUpsideToEqOverPhi(_EqTier2Window):
    """``(mu_proxy - p) / (PHI + EPS)``.

    Positive when the equilibrium proxy is above current price.
    ``UER = 1`` means returning to equilibrium would equal exactly the
    profit barrier — strong support for a long classifier's positive call.
    Negative values mean the equilibrium is below current price, so
    "reverting" wouldn't help.

    Emitted in two flavors — one via the trend proxy and one via the VWAP
    proxy. Pick the one that aligns best with the rest of the
    feature set during model audit.
    """

    __abstract__: ClassVar[bool] = True
    inputs = ("p",)
    proxy: ClassVar[str] = ""  # subclass picks: "trend" or "vwap"

    def column_name(self, w: int | None = None) -> str:
        return f"eq__upside_to_eq_over_phi__via_{self.proxy}__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        mu = pl.col(f"eq__mu_{self.proxy}__f__w{w}")
        return (mu - p) / (PHI + EPS)


class EqUpsideToEqOverPhiViaTrend(_EqUpsideToEqOverPhi):
    proxy = "trend"


class EqUpsideToEqOverPhiViaVwap(_EqUpsideToEqOverPhi):
    proxy = "vwap"


# -------- Feature 8: barrier-vs-equilibrium scaled by horizon vol ----------


class _EqBarrierVsEqHz(_EqTier2Window):
    """``(mu_proxy - (p + PHI)) / (eq__sigma_r · √M + EPS)``.

    Positive: the upper barrier sits *below* the estimated fair value, so
    barrier-hit is structurally supported by reversion alone. Negative:
    the barrier requires price to move *beyond* equilibrium, which is
    still possible in momentum regimes but less naturally supported.

    Emitted via the trend and VWAP proxies for the same reason as
    feature 7.
    """

    __abstract__: ClassVar[bool] = True
    inputs = ("p",)
    proxy: ClassVar[str] = ""

    def column_name(self, w: int | None = None) -> str:
        return f"eq__barrier_vs_eq_hz__via_{self.proxy}__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        return (w + 1) if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        mu = pl.col(f"eq__mu_{self.proxy}__f__w{w}")
        sigma = pl.col(f"eq__sigma_r__f__w{w}")
        return (mu - (p + PHI)) / (sigma * _SQRT_M + EPS)


class EqBarrierVsEqHzViaTrend(_EqBarrierVsEqHz):
    proxy = "trend"


class EqBarrierVsEqHzViaVwap(_EqBarrierVsEqHz):
    proxy = "vwap"


# -------- Feature 9: cross-proxy dispersion / horizon vol ------------------


class EqProxyDispersionHz(_EqTier2Window):
    """Population std across the five proxies, normalized by horizon vol.

    Zero / small: all proxies agree, so ``distance from equilibrium`` is a
    well-defined concept and the residual features are trustworthy.
    Large: proxies disagree (e.g. mean and VWAP say one thing, range-mid
    and trend say another), meaning fair value is locally unstable.

    Computed with ``pl.concat_list(...).list.std(ddof=0)``. Drops nulls
    before aggregation, then forces null when fewer than two valid
    proxies survive — a 1-element population-std is 0, but the spec
    treats one proxy as insufficient.
    """

    inputs = ("p",)

    def column_name(self, w: int | None = None) -> str:
        return f"eq__proxy_dispersion_hz__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        return (w + 1) if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        proxy_cols = [
            pl.col(f"eq__mu_mean__f__w{w}"),
            pl.col(f"eq__mu_median__f__w{w}"),
            pl.col(f"eq__mu_vwap__f__w{w}"),
            pl.col(f"eq__mu_trend__f__w{w}"),
            pl.col(f"eq__mu_range__f__w{w}"),
        ]
        sigma = pl.col(f"eq__sigma_r__f__w{w}")
        proxies = pl.concat_list(proxy_cols).list.drop_nulls()
        n_valid = proxies.list.len()
        disp_std = proxies.list.std(ddof=0)
        guarded = pl.when(n_valid >= 2).then(disp_std).otherwise(None)
        return guarded / (sigma * _SQRT_M + EPS)


# -------- Feature 10: rising-equilibrium pullback --------------------------


class _EqPullbackRisingEq(Feature):
    """``max(0, (mu_S - p)/(sigma_L·√M+EPS)) · max(0, (mu_S - mu_L)/(sigma_L·√M+EPS))``.

    Positive only when (a) current price is below the *short-horizon* mean
    equilibrium AND (b) short-horizon equilibrium is itself above the
    long-horizon equilibrium (local fair value is rising). Aims at the
    ``pullback inside an uptrend`` setup that flat residuals can't pick out.

    Pair-windowed: one concrete class per ``(S, L) ∈ WINDOWS_EQ_PAIRS``.
    """

    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "eq"
    tier: ClassVar[int | str] = 2
    windows: ClassVar[tuple[int, ...]] = ()
    inputs = ("p",)

    S: ClassVar[int] = 0
    L: ClassVar[int] = 0

    def column_name(self, w: int | None = None) -> str:
        return f"eq__pullback_rising_eq__f__w{self.S}__l{self.L}"

    def warmup_for(self, w: int | None) -> int:
        # The long-window proxy decides warmup; the short window warms up
        # earlier so it's not the binding constraint. ``sigma_r_L`` has
        # warmup ``L + 1`` (r[0] null + past-only shift), so that is the
        # binding row count here.
        return int(self.L) + 1

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        mu_s = pl.col(f"eq__mu_mean__f__w{self.S}")
        mu_l = pl.col(f"eq__mu_mean__f__w{self.L}")
        sigma_l = pl.col(f"eq__sigma_r__f__w{self.L}")
        denom = sigma_l * _SQRT_M + EPS
        # ``clip(lower_bound=0)`` preserves null on null inputs — necessary
        # for the warmup contract. ``pl.max_horizontal(lit(0), null) = 0``
        # would silently emit 0 during warmup and mask the missingness.
        below_short = ((mu_s - p) / denom).clip(lower_bound=0.0)
        short_rising = ((mu_s - mu_l) / denom).clip(lower_bound=0.0)
        return below_short * short_rising


# -------- Feature 11: falling-equilibrium overextension --------------------


class _EqAboveFallingEq(Feature):
    """``max(0, (p - mu_S)/(sigma_L·√M+EPS)) · max(0, (mu_L - mu_S)/(sigma_L·√M+EPS))``.

    Mirror of feature 10. Positive only when current price is *above* the
    short equilibrium AND the short equilibrium has fallen *below* the
    long one (local fair value drifting down). The natural false-positive
    region for a long classifier.
    """

    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "eq"
    tier: ClassVar[int | str] = 2
    windows: ClassVar[tuple[int, ...]] = ()
    inputs = ("p",)

    S: ClassVar[int] = 0
    L: ClassVar[int] = 0

    def column_name(self, w: int | None = None) -> str:
        return f"eq__above_falling_eq__f__w{self.S}__l{self.L}"

    def warmup_for(self, w: int | None) -> int:
        # Same as the pullback feature: sigma_r_L drives the warmup at L+1.
        return int(self.L) + 1

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("p")
        mu_s = pl.col(f"eq__mu_mean__f__w{self.S}")
        mu_l = pl.col(f"eq__mu_mean__f__w{self.L}")
        sigma_l = pl.col(f"eq__sigma_r__f__w{self.L}")
        denom = sigma_l * _SQRT_M + EPS
        above_short = ((p - mu_s) / denom).clip(lower_bound=0.0)
        short_falling = ((mu_l - mu_s) / denom).clip(lower_bound=0.0)
        return above_short * short_falling


# -------- Concrete pair subclasses for features 10 & 11 --------------------
#
# Generated programmatically from ``WINDOWS_EQ_PAIRS`` so adding a pair to
# the config registers two new feature columns automatically. The ``type()``
# call triggers ``Feature.__init_subclass__`` which auto-registers each
# concrete class.

for _s, _l in WINDOWS_EQ_PAIRS:
    _pullback_cls = type(
        f"EqPullbackRisingEq_{_s}_{_l}",
        (_EqPullbackRisingEq,),
        {"S": _s, "L": _l},
    )
    _above_cls = type(
        f"EqAboveFallingEq_{_s}_{_l}",
        (_EqAboveFallingEq,),
        {"S": _s, "L": _l},
    )
    globals()[_pullback_cls.__name__] = _pullback_cls
    globals()[_above_cls.__name__] = _above_cls

del _s, _l, _pullback_cls, _above_cls
