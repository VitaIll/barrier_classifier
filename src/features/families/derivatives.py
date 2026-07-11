"""Derivatives feature families (spec Appendix E).

Six sub-families ported from utils.compute_basis_features /
compute_flow_features / compute_oi_features / compute_funding_features /
compute_options_features / compute_vol_index_features.

All are Tier-1; derivatives base series (basis_abs, basis_pct,
tb_ratio_fut, net_vol_fut, pcr_oi, pcr_vol) is computed by the legacy
``utils.compute_derivatives_base_series`` and assumed already on the
bars frame at engine entry. We do not re-implement that base step in
the engine; it is base-series machinery (like compute_base_series).

Output columns and parity hazards documented at each Feature class.
"""

from __future__ import annotations

import math
from typing import ClassVar

import polars as pl

from src.features.base import Feature
from src.features.config import EPS
from src.features.primitives import (
    ewm_mean,
    rolling_mean,
    rolling_std_pop,
    rolling_sum,
)


# =============================================================================
# basis family (perpetual futures vs spot)
# =============================================================================


class BasisAbs(Feature):
    family: ClassVar[str] = "deriv_basis"
    tier: ClassVar[int | str] = 1
    inputs = ("basis_abs",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "basis__abs__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("basis_abs")


class BasisPct(Feature):
    family: ClassVar[str] = "deriv_basis"
    tier: ClassVar[int | str] = 1
    inputs = ("basis_pct",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "basis__pct__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("basis_pct")


class BasisAnnYield(Feature):
    """Annualised yield from spot↔fut basis with funding-aware time scaling.

    tau (minutes-to-next-funding) clipped to [60, 480]; legacy uses funding
    boundaries at 8h, 16h, 24h UTC (utils.py:2572-2581). Cast hour/minute
    to Int32 to avoid Int8 overflow on hour*60.
    """

    family: ClassVar[str] = "deriv_basis"
    tier: ClassVar[int | str] = 1
    inputs = ("ts", "close", "close_fut")
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "basis__ann_yield__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        ts = pl.col("ts")
        minute = ts.dt.hour().cast(pl.Int32) * 60 + ts.dt.minute().cast(pl.Int32)
        next_funding = (
            pl.when(minute < 8 * 60).then(8 * 60)
            .when(minute < 16 * 60).then(16 * 60)
            .otherwise(24 * 60)
        )
        tau = (next_funding - minute).cast(pl.Float64).clip(lower_bound=60.0, upper_bound=480.0)
        basis_ratio = (pl.col("close_fut") - pl.col("close")) / (pl.col("close") + EPS)
        return basis_ratio * (365.0 * 24.0 * 60.0 / tau) * 100.0


class BasisChg(Feature):
    family: ClassVar[str] = "deriv_basis"
    tier: ClassVar[int | str] = 1
    inputs = ("basis_pct",)
    windows: ClassVar[tuple[int, ...]] = (5,)

    def column_name(self, w: int | None = None) -> str:
        return f"basis__chg__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        b = pl.col("basis_pct")
        return b - b.shift(w)


class BasisMean(Feature):
    family: ClassVar[str] = "deriv_basis"
    tier: ClassVar[int | str] = 1
    inputs = ("basis_pct",)
    windows: ClassVar[tuple[int, ...]] = (5, 60)

    def column_name(self, w: int | None = None) -> str:
        return f"basis__mean__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_mean(pl.col("basis_pct"), w)


class BasisStd(Feature):
    family: ClassVar[str] = "deriv_basis"
    tier: ClassVar[int | str] = 1
    inputs = ("basis_pct",)
    windows: ClassVar[tuple[int, ...]] = (5, 60)

    def column_name(self, w: int | None = None) -> str:
        return f"basis__std__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_std_pop(pl.col("basis_pct"), w)


# =============================================================================
# flow family
# =============================================================================


class FlowTakerBuyRatio(Feature):
    family: ClassVar[str] = "deriv_flow"
    tier: ClassVar[int | str] = 1
    inputs = ("tb_ratio_fut",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "flow__taker_buy_ratio__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("tb_ratio_fut")


class FlowNetVolBtcs(Feature):
    family: ClassVar[str] = "deriv_flow"
    tier: ClassVar[int | str] = 1
    inputs = ("net_vol_fut",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "flow__net_vol_btcs__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("net_vol_fut")


class FlowNetVolCsum(Feature):
    family: ClassVar[str] = "deriv_flow"
    tier: ClassVar[int | str] = 1
    inputs = ("net_vol_fut",)
    windows: ClassVar[tuple[int, ...]] = (5, 10, 20)

    def column_name(self, w: int | None = None) -> str:
        return f"flow__net_vol_csum__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return rolling_sum(pl.col("net_vol_fut"), w)


class FlowFutVsSpotVol(Feature):
    """fut quote-vol over spot quote-vol (null when spot ≤ 0)."""

    family: ClassVar[str] = "deriv_flow"
    tier: ClassVar[int | str] = 1
    inputs = ("quote_volume", "quote_volume_fut")
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "liq__fut_vs_spot_vol__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        qv = pl.col("quote_volume")
        qvf = pl.col("quote_volume_fut")
        return pl.when(qv > 0).then(qvf / (qv + EPS)).otherwise(None)


class FlowAvgTradeSizeInst(Feature):
    """Per-bar avg trade size (volume / num_trades), null when num_trades ≤ 0."""

    family: ClassVar[str] = "deriv_flow"
    tier: ClassVar[int | str] = 1
    inputs = ("volume_fut", "num_trades_fut")
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "liq__avg_trade_size__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        nt = pl.col("num_trades_fut")
        return pl.when(nt > 0).then(pl.col("volume_fut") / (nt + EPS)).otherwise(None)


class FlowAvgTradeSize15(Feature):
    """15-bar windowed avg trade size."""

    family: ClassVar[str] = "deriv_flow"
    tier: ClassVar[int | str] = 1
    inputs = ("volume_fut", "num_trades_fut")
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "liq__avg_trade_size__f__w15"

    def compute(self, w: int | None = None) -> pl.Expr:
        vol_15 = rolling_sum(pl.col("volume_fut"), 15)
        trades_15 = rolling_sum(pl.col("num_trades_fut"), 15)
        return pl.when(trades_15 > 0).then(vol_15 / (trades_15 + EPS)).otherwise(None)


class FlowTradesZscore30(Feature):
    """30-bar z-score of futures trades count.

    Legacy guards with strict ``std_30 > 0`` and uses ``+ EPS`` in the
    denominator (utils.py:2651-2655). Reproduce both.
    """

    family: ClassVar[str] = "deriv_flow"
    tier: ClassVar[int | str] = 1
    inputs = ("num_trades_fut",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "activity__trades_zscore__f__w30"

    def compute(self, w: int | None = None) -> pl.Expr:
        n = pl.col("num_trades_fut")
        mu = rolling_mean(n, 30)
        sigma = rolling_std_pop(n, 30)
        return pl.when(sigma > 0).then((n - mu) / (sigma + EPS)).otherwise(None)


# =============================================================================
# oi family
# =============================================================================


class OiTotalUsd(Feature):
    family: ClassVar[str] = "deriv_oi"
    tier: ClassVar[int | str] = 1
    inputs = ("oi_usd",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "oi__total_usd__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("oi_usd")


class OiChg(Feature):
    family: ClassVar[str] = "deriv_oi"
    tier: ClassVar[int | str] = 1
    inputs = ("oi_usd",)
    windows: ClassVar[tuple[int, ...]] = (60,)

    def column_name(self, w: int | None = None) -> str:
        return f"oi__chg__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        oi = pl.col("oi_usd")
        return oi - oi.shift(w)


class OiChgPct(Feature):
    family: ClassVar[str] = "deriv_oi"
    tier: ClassVar[int | str] = 1
    inputs = ("oi_usd",)
    windows: ClassVar[tuple[int, ...]] = (60,)

    def column_name(self, w: int | None = None) -> str:
        return f"oi__chg_pct__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        oi = pl.col("oi_usd")
        prev = oi.shift(w)
        return (oi - prev) / (prev + EPS) * 100.0


class OiVolRatio(Feature):
    family: ClassVar[str] = "deriv_oi"
    tier: ClassVar[int | str] = 1
    inputs = ("oi_usd", "quote_volume_fut")
    windows: ClassVar[tuple[int, ...]] = (60,)

    def column_name(self, w: int | None = None) -> str:
        return f"oi__vol_ratio__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        oi = pl.col("oi_usd")
        qv = rolling_sum(pl.col("quote_volume_fut"), w)
        return pl.when(qv > 0).then(oi / (qv + EPS)).otherwise(None)


class OiPriceCorr(Feature):
    """Sample (ddof=1) correlation between diff(oi) and r over 120 bars.

    Parity note: legacy uses pandas ``.corr()`` which is **sample**
    correlation (ddof=1) — diverges from the population convention used
    in compute_correlations. Reproduce sample correlation here via
    polars' native ``rolling_corr`` (which is also sample, ddof=1).
    """

    family: ClassVar[str] = "deriv_oi"
    tier: ClassVar[int | str] = 1
    inputs = ("oi_usd", "r")
    windows: ClassVar[tuple[int, ...]] = (120,)

    def column_name(self, w: int | None = None) -> str:
        return f"oi__price_corr__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        oi_diff = pl.col("oi_usd").diff()
        return pl.rolling_corr(oi_diff, pl.col("r"), window_size=w, min_samples=w, ddof=1)


# -------- OI regime decomposition (Round 4a) ------------------------------
#
# Four non-overlapping quadrants of the (price-change, OI-change) plane,
# each non-negative and bounded by ``tanh`` smoothing:
#
#     dp_z = tanh((p_n - p_{n-W}) / (sigma_W,n * sqrt(W) + EPS))
#     doi_z = tanh((OI_n - OI_{n-W}) / (sigma_OI_W,n + EPS))
#
#     long_build  = max(0,  dp_z) * max(0,  doi_z)   price up + OI up    → new longs
#     short_build = max(0, -dp_z) * max(0,  doi_z)   price down + OI up  → new shorts
#     short_cover = max(0,  dp_z) * max(0, -doi_z)   price up + OI down  → covering shorts
#     long_liq    = max(0, -dp_z) * max(0, -doi_z)   price down + OI down → liquidations
#
# The four columns are 1 of N: only ONE quadrant is non-zero on any
# given row. The model can split on any of them; useful for learning
# regime-conditional effects (e.g. a high score might be more reliable
# under ``short_cover`` than under ``long_build``).
#
# Normalizer for the OI side uses ``rolling_std_pop(oi_usd.diff(), W)``
# — population std of 1-bar OI changes over the trailing W bars. This is
# faster than MAD and adequate given the ``tanh`` saturation.


class _OiRegimeFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "deriv_oi"
    # Tier 2: depends on ``ret__rms__f__w{W}`` for price normalization.
    tier: ClassVar[int | str] = 2
    inputs = ("p", "oi_usd")
    windows_field: ClassVar[str] = "windows_oi_regime"


def _oi_regime_components(w: int) -> tuple[pl.Expr, pl.Expr]:
    """Shared building blocks: signed normalized price change and OI change."""
    p = pl.col("p")
    dp = p - p.shift(w)
    # Use ret__rms as the per-bar return scale (consistent with flow features).
    sigma_r = pl.col(f"ret__rms__f__w{w}")
    sqrt_w = float(w) ** 0.5
    dp_z = (dp / (sigma_r * sqrt_w + EPS)).tanh()

    oi = pl.col("oi_usd")
    doi = oi - oi.shift(w)
    sigma_oi = rolling_std_pop(oi.diff(), w)
    # 1-bar OI delta std rescaled to W-bar by sqrt(W), matching the
    # Brownian-style normalization used on the price side.
    doi_z = (doi / (sigma_oi * sqrt_w + EPS)).tanh()
    return dp_z, doi_z


class OiLongBuild(_OiRegimeFeature):
    """Price up × OI up — new longs adding leverage."""

    def column_name(self, w: int | None = None) -> str:
        return f"oi__long_build__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        # ``dp = p - p.shift(w)`` is null until row w; ``ret__rms`` and the
        # OI std are likewise null until row w. Binding warmup is w.
        return w if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        dp_z, doi_z = _oi_regime_components(int(w))
        # ``clip`` (not ``max_horizontal``) so null inputs propagate to
        # null output. See ``FlowSellAbsorption`` for the rationale.
        return dp_z.clip(lower_bound=0.0) * doi_z.clip(lower_bound=0.0)


class OiShortBuild(_OiRegimeFeature):
    """Price down × OI up — new shorts adding leverage."""

    def column_name(self, w: int | None = None) -> str:
        return f"oi__short_build__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        return w if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        dp_z, doi_z = _oi_regime_components(int(w))
        return (-dp_z).clip(lower_bound=0.0) * doi_z.clip(lower_bound=0.0)


class OiShortCover(_OiRegimeFeature):
    """Price up × OI down — shorts covering, deleveraging into rally."""

    def column_name(self, w: int | None = None) -> str:
        return f"oi__short_cover__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        return w if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        dp_z, doi_z = _oi_regime_components(int(w))
        return dp_z.clip(lower_bound=0.0) * (-doi_z).clip(lower_bound=0.0)


class OiLongLiq(_OiRegimeFeature):
    """Price down × OI down — longs liquidating, deleveraging into drop."""

    def column_name(self, w: int | None = None) -> str:
        return f"oi__long_liq__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        return w if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        dp_z, doi_z = _oi_regime_components(int(w))
        return (-dp_z).clip(lower_bound=0.0) * (-doi_z).clip(lower_bound=0.0)


# =============================================================================
# funding family
# =============================================================================


class FundingRate(Feature):
    family: ClassVar[str] = "deriv_funding"
    tier: ClassVar[int | str] = 1
    inputs = ("funding_rate",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "funding__rate__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("funding_rate") * 100.0


class FundingEwma(Feature):
    family: ClassVar[str] = "deriv_funding"
    tier: ClassVar[int | str] = 1
    inputs = ("funding_rate",)
    windows: ClassVar[tuple[int, ...]] = (1440,)

    def column_name(self, w: int | None = None) -> str:
        return f"funding__ewma__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        return ewm_mean(pl.col("funding_rate") * 100.0, span=w, adjust=False)


class _FundingPhaseBase(Feature):
    """Common machinery for funding-cycle phase features.

    Phase = (minutes to next scheduled funding) / 480
          ∈ (0, 1], with values near 1 just AFTER a settlement and values
          near 0 just BEFORE the next settlement. Calendar-causal — uses
          ``ts`` only, no realized funding rate.
    """

    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "deriv_funding"
    tier: ClassVar[int | str] = 1
    inputs = ("ts",)
    windows: ClassVar[tuple[int, ...]] = ()

    @staticmethod
    def _phase_expr() -> pl.Expr:
        ts = pl.col("ts")
        minute = ts.dt.hour().cast(pl.Int32) * 60 + ts.dt.minute().cast(pl.Int32)
        next_funding = (
            pl.when(minute < 8 * 60).then(8 * 60)
            .when(minute < 16 * 60).then(16 * 60)
            .otherwise(24 * 60)
        )
        # tau is in (0, 480] minutes. Divide by 480 to get the unit phase.
        tau = (next_funding - minute).cast(pl.Float64)
        return tau / 480.0


class FundingPhase(_FundingPhaseBase):
    """Unit phase to next funding settlement, in (0, 1]."""

    def column_name(self, w: int | None = None) -> str:
        return "funding__phase__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return self._phase_expr()


class FundingPhaseSin(_FundingPhaseBase):
    """Sine of 2*pi*phase — bounded continuous representation of the
    cyclic phase. Combined with the cosine partner, gives the model
    distance-on-the-cycle without a discontinuity at phase = 0/1."""

    def column_name(self, w: int | None = None) -> str:
        return "funding__phase_sin__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return (2.0 * math.pi * self._phase_expr()).sin()


class FundingPhaseCos(_FundingPhaseBase):
    """Cosine of 2*pi*phase. See ``FundingPhaseSin`` for rationale."""

    def column_name(self, w: int | None = None) -> str:
        return "funding__phase_cos__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return (2.0 * math.pi * self._phase_expr()).cos()


class FundingTrend(Feature):
    family: ClassVar[str] = "deriv_funding"
    tier: ClassVar[int | str] = 1
    inputs = ("funding_rate",)
    windows: ClassVar[tuple[int, ...]] = (4320,)

    def column_name(self, w: int | None = None) -> str:
        return f"funding__trend__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        rate = pl.col("funding_rate") * 100.0
        return rate - rolling_mean(rate, w)


# =============================================================================
# options family
# =============================================================================


class OptPcrOi(Feature):
    family: ClassVar[str] = "deriv_options"
    tier: ClassVar[int | str] = 1
    inputs = ("pcr_oi",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "opt_pcr__oi__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("pcr_oi")


class OptPcrVol(Feature):
    family: ClassVar[str] = "deriv_options"
    tier: ClassVar[int | str] = 1
    inputs = ("pcr_vol",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "opt_pcr__vol__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("pcr_vol")


class OptPcrOiChg(Feature):
    family: ClassVar[str] = "deriv_options"
    tier: ClassVar[int | str] = 1
    inputs = ("pcr_oi",)
    windows: ClassVar[tuple[int, ...]] = (1440,)

    def column_name(self, w: int | None = None) -> str:
        return f"opt_pcr__oi_chg__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        p = pl.col("pcr_oi")
        return p - p.shift(w)


class OptOiTotalUsd(Feature):
    family: ClassVar[str] = "deriv_options"
    tier: ClassVar[int | str] = 1
    inputs = ("opt_oi",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "opt_oi__total_usd__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("opt_oi")


class OptVol24hUsd(Feature):
    family: ClassVar[str] = "deriv_options"
    tier: ClassVar[int | str] = 1
    inputs = ("opt_volume",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "opt_vol__24h_usd__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("opt_volume")


# =============================================================================
# vol_idx family
# =============================================================================


class VolIdxBvol30d(Feature):
    family: ClassVar[str] = "deriv_volidx"
    tier: ClassVar[int | str] = 1
    inputs = ("bvol",)
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "vol_idx__bvol30d__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("bvol")


class VolIdxBvolChg(Feature):
    family: ClassVar[str] = "deriv_volidx"
    tier: ClassVar[int | str] = 1
    inputs = ("bvol",)
    windows: ClassVar[tuple[int, ...]] = (1440,)

    def column_name(self, w: int | None = None) -> str:
        return f"vol_idx__bvol_chg__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        b = pl.col("bvol")
        return b - b.shift(w)


class VolRealized30d(Feature):
    """Annualised realised vol from r over 30 days = 43200 minutes.

    Declares ``windows=(43200,)`` so the engine's warmup tracker includes
    the 30-day window — and so dependent features
    (``VolRiskPremiumDiff``/``VolRiskPremiumRatio``) can read the
    precomputed column instead of re-evaluating a 43200-row rolling_std.
    """

    family: ClassVar[str] = "deriv_volidx"
    tier: ClassVar[int | str] = 1
    inputs = ("r",)
    windows: ClassVar[tuple[int, ...]] = (43200,)

    def column_name(self, w: int | None = None) -> str:
        return "vol_realized__30d__f__w43200"

    def compute(self, w: int | None = None) -> pl.Expr:
        # 525600 minutes per year; std × sqrt(525600) × 100.
        return rolling_std_pop(pl.col("r"), 43200) * math.sqrt(525600.0) * 100.0


class VolRiskPremiumDiff(Feature):
    """Reads the precomputed ``vol_realized__30d__f__w43200`` (tier-1) so the
    43200-row rolling_std evaluates exactly once across the three vol-idx
    features that depend on it."""

    family: ClassVar[str] = "deriv_volidx"
    tier: ClassVar[int | str] = 2
    inputs = ("bvol", "vol_realized__30d__f__w43200")
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "vol_risk_premium__diff__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        return pl.col("bvol") - pl.col("vol_realized__30d__f__w43200")


class VolRiskPremiumRatio(Feature):
    """See ``VolRiskPremiumDiff`` for the shared-rv rationale."""

    family: ClassVar[str] = "deriv_volidx"
    tier: ClassVar[int | str] = 2
    inputs = ("bvol", "vol_realized__30d__f__w43200")
    windows: ClassVar[tuple[int, ...]] = ()

    def column_name(self, w: int | None = None) -> str:
        return "vol_risk_premium__ratio__f__w0"

    def compute(self, w: int | None = None) -> pl.Expr:
        rv = pl.col("vol_realized__30d__f__w43200")
        return pl.col("bvol") / (rv + EPS)
