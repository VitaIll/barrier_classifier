"""Rolling signed-flow features (new family — Round 1e).

Three Tier-2 features that summarize taker-side flow over a trailing
window:

  - flow__pressure__f__w{W}          volume-weighted signed-pressure index in [-1, 1].
                                      Positive = net taker-buy; negative = net
                                      taker-sell. Equivalent to
                                      ``2 * sum(taker_buy_base) / sum(volume) - 1``.
  - flow__sell_absorption__f__w{W}   in [0, 1]. Positive when net sell pressure
                                      coincides with non-negative trailing price
                                      response (sellers fail to push price down).
  - flow__buy_exhaustion__f__w{W}    in [0, 1]. Positive when net buy pressure
                                      coincides with non-positive trailing price
                                      response (buyers fail to push price up).

All three are causal: at row n they use only bars ``[n-W+1, n]``.
``taker_buy_base`` and ``volume`` are base columns on the bars frame;
``p`` (log close) is also a base column.

These differ from the existing ``ofi``-based instantaneous features:
  - ``ofi__inst__h__w0`` is per-bar signed-flow rate, no rolling window.
  - ``flow__pressure`` is the volume-WEIGHTED rolling pressure over W bars,
    which is more stable than averaging ``ofi`` (which is itself a ratio).

For low-volume windows ``sum(volume)`` can approach zero and the
denominator is bounded by EPS — the magnitude is still finite, but the
ratio loses meaning. The ``flow__pressure__cum__f__w{W}`` companion
emits the raw signed volume (numerator only) so the model can compare.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl

from src.features.base import Feature
from src.features.config import EPS
from src.features.primitives import rolling_sum, rolling_std_pop


class _FlowFeature(Feature):
    __abstract__: ClassVar[bool] = True
    family: ClassVar[str] = "flow"
    # Tier 2: ``sell_absorption`` and ``buy_exhaustion`` divide trailing
    # price change by trailing volatility — they need a tier-1 vol column
    # (``vol__rs``). ``pressure`` could be tier-1 but stays here for
    # family-namespace consistency.
    tier: ClassVar[int | str] = 2
    windows_field: ClassVar[str] = "windows_flow_pressure"


class FlowPressure(_FlowFeature):
    """Volume-weighted signed pressure in [-1, 1].

        FP_{W,n} = 2 * sum_{i in [n-W+1, n]} taker_buy_base_i
                   / (sum_{i in [n-W+1, n]} volume_i + EPS) - 1

    Algebraically equivalent to ``sum(s_i * V_i) / sum(V_i)`` with
    ``s_i = 2 * taker_buy_base_i / V_i - 1`` (the per-bar signed pressure).
    Empty windows (``sum(volume) == 0``) yield ``-1`` due to the
    additive EPS guard — that's the same as "all sells" by convention;
    extremely low-volume windows behave similarly. Pair with the
    ``cum`` companion for the absolute scale.
    """

    inputs = ("taker_buy_base", "volume")

    def column_name(self, w: int | None = None) -> str:
        return f"flow__pressure__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        sum_tbb = rolling_sum(pl.col("taker_buy_base"), w)
        sum_vol = rolling_sum(pl.col("volume"), w)
        return 2.0 * sum_tbb / (sum_vol + EPS) - 1.0


class FlowPressureCum(_FlowFeature):
    """Raw signed volume (numerator of ``flow__pressure``).

        FPC_{W,n} = 2 * sum(taker_buy_base) - sum(volume)
                  = sum(s_i * V_i)   over [n-W+1, n]

    Useful when the per-bar pressure is well-defined but the
    normalised ``flow__pressure`` saturates (e.g. very-low-volume
    windows). Tree models can compare the two via interaction splits.
    """

    inputs = ("taker_buy_base", "volume")

    def column_name(self, w: int | None = None) -> str:
        return f"flow__pressure_cum__f__w{w}"

    def compute(self, w: int | None = None) -> pl.Expr:
        sum_tbb = rolling_sum(pl.col("taker_buy_base"), w)
        sum_vol = rolling_sum(pl.col("volume"), w)
        return 2.0 * sum_tbb - sum_vol


class FlowSellAbsorption(_FlowFeature):
    def depends_on(self, w: int | None = None) -> tuple[str, ...]:
        return self.inputs + (f"ret__rms__f__w{w}",)
    """Sell-absorption signal in [0, 1].

        d_{W,n} = tanh((p_n - p_{n-W}) / (sigma_W,n * sqrt(W) + EPS))
        SA_{W,n} = max(0, -FP_{W,n}) * max(0, d_{W,n})

    Positive when net sell pressure (FP < 0) coincides with non-negative
    trailing price response. Both factors are bounded in [0, 1], so the
    product is bounded in [0, 1]. Sellers are aggressing but price is
    not falling -> passive demand absorbing supply.

    Uses ``ret__rms__f__w{W}`` as the volatility normalizer for the
    price-response term. ``ret__rms`` (rolling RMS of returns) is a
    Tier-1 column emitted by the rolling family; the engine's tier
    ordering ensures it is present by the time this Tier-2 expression
    evaluates.
    """

    inputs = ("p", "taker_buy_base", "volume")

    def column_name(self, w: int | None = None) -> str:
        return f"flow__sell_absorption__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        # ``dp = p - p.shift(w)`` is null until row n=w; ``ret__rms`` is null
        # until row n=w. The rolling-pressure sum is null until row n=w-1.
        # The binding warmup is therefore w (rows 0..w-1 emit null).
        return w if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        sum_tbb = rolling_sum(pl.col("taker_buy_base"), w)
        sum_vol = rolling_sum(pl.col("volume"), w)
        fp = 2.0 * sum_tbb / (sum_vol + EPS) - 1.0
        # trailing log-price change over W bars
        p = pl.col("p")
        dp = p - p.shift(w)
        # ret__rms is sqrt(rolling_mean(r^2, W)); rolling_std_pop on p would
        # double-count the level. Instead use ret__rms__f__w{W} so the
        # normalizer is on the same scale as ``barrier__z_tight``.
        sigma = pl.col(f"ret__rms__f__w{w}")
        # sqrt(W) because we are normalising a cumulative W-bar price change
        # by per-bar return RMS.
        d = (dp / (sigma * (float(w) ** 0.5) + EPS)).tanh()
        # Use ``clip(lower_bound=0.0)`` rather than ``pl.max_horizontal``:
        # ``max_horizontal(lit(0.0), null)`` silently returns 0.0, hiding
        # the warmup region from the engine's null-pattern trim AND the
        # imputation step's ``undef__`` flag. ``clip`` propagates null,
        # which is what we want.
        return (-fp).clip(lower_bound=0.0) * d.clip(lower_bound=0.0)


class FlowBuyExhaustion(_FlowFeature):
    def depends_on(self, w: int | None = None) -> tuple[str, ...]:
        return self.inputs + (f"ret__rms__f__w{w}",)
    """Buy-exhaustion signal in [0, 1] — opposite of sell absorption.

        BE_{W,n} = max(0, FP_{W,n}) * max(0, -d_{W,n})

    Positive when net buy pressure coincides with non-positive trailing
    price response. Buyers aggressing but price not rising -> supply
    absorbing demand.
    """

    inputs = ("p", "taker_buy_base", "volume")

    def column_name(self, w: int | None = None) -> str:
        return f"flow__buy_exhaustion__f__w{w}"

    def warmup_for(self, w: int | None) -> int:
        return w if w else 0

    def compute(self, w: int | None = None) -> pl.Expr:
        sum_tbb = rolling_sum(pl.col("taker_buy_base"), w)
        sum_vol = rolling_sum(pl.col("volume"), w)
        fp = 2.0 * sum_tbb / (sum_vol + EPS) - 1.0
        p = pl.col("p")
        dp = p - p.shift(w)
        sigma = pl.col(f"ret__rms__f__w{w}")
        d = (dp / (sigma * (float(w) ** 0.5) + EPS)).tanh()
        # See FlowSellAbsorption — clip preserves nulls, max_horizontal does not.
        return fp.clip(lower_bound=0.0) * (-d).clip(lower_bound=0.0)
