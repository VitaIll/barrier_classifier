"""Live strategy core — a streaming driver over the shared BoundaryStep.

``LiveTrader.on_boundary`` and the offline ``simulate()`` now execute the
SAME per-boundary implementation — :class:`src.strategy.step.BoundaryStep`
— so live≡offline ledger parity holds by construction (and remains pinned
by ``tests/engine/test_parity_simulator.py`` as regression insurance).
This module owns only what is live-specific: the single-bar intra span,
the probability feed binding, halt/degraded entry suppression, and
label feedback at MATURITY.

That last one is the one deliberate causality difference: the offline
simulator feeds boundary ``k``'s *label* into its monitoring stats at row
``k`` (the label is only knowable at ``k+M`` in real time). The live
trader feeds labels at maturity instead. Specs that consume label-fed
stats (``score_residualized`` base rates, drift) therefore see slightly
*older* information live; the production spec consumes neither, so its
decisions are identical.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.engine.domain import Bar
from src.strategy.inventory import ClosedPosition, Portfolio, Position
from src.strategy.online import RollingRegimeBaseRate
from src.strategy.policy import (
    IntraBar,
    RiskConfig,
    StrategySpec,
    gate_score_above,
    make_exit_let_winners_run,
    make_exit_let_winners_run_monotonic,
    score_raw_p,
    size_clip,
    size_constant,
)
from src.strategy.simulator import SimConfig
from src.strategy.step import BoundaryInputs, BoundaryStep, TradingState


class LiveProbFeed:
    """``{ts -> p}`` mapping fed by the engine each bar.

    Drop-in replacement for the precomputed prediction Series the offline
    exit closures were built around (``p_map[intra_bar.ts]``). Missing
    timestamps return NaN via KeyError — the closures' documented safe
    fallback (treat as "no conviction signal").
    """

    def __init__(self, retention: int = 4_096) -> None:
        self._map: dict[int, float] = {}
        self._order: deque[int] = deque()
        self._retention = int(retention)

    def update(self, ts: pd.Timestamp, p: float) -> None:
        key = int(pd.Timestamp(ts).value)
        if key not in self._map:
            self._order.append(key)
            while len(self._order) > self._retention:
                self._map.pop(self._order.popleft(), None)
        self._map[key] = float(p)

    def __getitem__(self, ts: pd.Timestamp) -> float:
        return self._map[int(pd.Timestamp(ts).value)]


def make_live_production_spec(
    prob_feed: LiveProbFeed,
    *,
    p_threshold: float,
    lot_size: float = 0.02,
    max_concurrent: int = 50,
    cost_per_trade: float = 0.0005,
    exit_variant: str = "threshold",
    sl_log_return: Optional[float] = None,
    name: str = "production_P1_P3_live",
) -> StrategySpec:
    """The researched winning spec (P1+P3), bound to a live probability feed.

    Mirrors ``scripts/run_winning_strategy_charts.py`` / notebook 05
    construction exactly: top-q selective entry (``p >= p_threshold``),
    let-winners-run exit holding while conviction stays above the same
    threshold, constant lots, no stop-loss, no time expiry.

    ``exit_variant="monotonic"`` swaps in the stricter cousin that holds
    only while conviction makes new highs.
    """
    if exit_variant == "threshold":
        exit_policy = make_exit_let_winners_run(
            prob_feed, hold_threshold=p_threshold, sl_log_return=sl_log_return
        )
    elif exit_variant == "monotonic":
        exit_policy = make_exit_let_winners_run_monotonic(prob_feed)
    else:
        raise ValueError(
            f"exit_variant must be 'threshold' or 'monotonic', got {exit_variant!r}"
        )
    return StrategySpec(
        name=name,
        requires=(),
        score_fn=score_raw_p,
        entry_gates=(lambda s, t=float(p_threshold): gate_score_above(s, t),),
        sizer=lambda s, sz=float(lot_size): size_clip(
            size_constant(s, default=sz), max_size=1.0
        ),
        exit_policy=exit_policy,
        bulk_close=lambda s: None,
        risk=RiskConfig(
            cost_per_trade=float(cost_per_trade),
            max_open_positions=int(max_concurrent),
            max_gross_size=int(max_concurrent) * float(lot_size) + 1e-6,
            max_horizon_boundaries=1_000_000,
            position_mtm_floor_log_return=None,
        ),
        description=(
            "P1: top-q selective entry; P3: let-winners-run while conviction "
            "holds; no SL; no time expiry (live binding)"
        ),
    )


@dataclass(frozen=True)
class EquitySnapshot:
    ts: pd.Timestamp
    k: int
    realized_cum: float
    unrealized: float
    n_open: int
    gross_size: float

    @property
    def equity(self) -> float:
        return self.realized_cum + self.unrealized


@dataclass(frozen=True)
class BoundaryResult:
    """Everything one boundary step produced (for persistence/events)."""

    ts: pd.Timestamp
    k: int
    closed: tuple[ClosedPosition, ...]
    opened: Optional[Position]
    bulk_reason: Optional[str]
    score: float
    equity: EquitySnapshot
    entered: bool


@dataclass
class _PendingLabelCtx:
    """Decision-time context retained until the label matures."""

    ts: pd.Timestamp
    p: float
    regime_q: float


class LiveTrader:
    """Streaming driver over the shared :class:`BoundaryStep`."""

    def __init__(
        self,
        spec: StrategySpec,
        *,
        sim_config: Optional[SimConfig] = None,
    ) -> None:
        self.spec = spec
        self.cfg = sim_config or SimConfig()
        self._state = TradingState.fresh(self.cfg)
        self.cost_per_trade = (
            self.cfg.cost_per_trade_override
            if self.cfg.cost_per_trade_override is not None
            else spec.risk.cost_per_trade
        )
        self._step = BoundaryStep(spec, cost_per_trade=self.cost_per_trade)
        self.base_rate = RollingRegimeBaseRate(
            window=self.cfg.base_rate_window, n_bins=self.cfg.base_rate_n_bins
        )
        # Decision contexts pending label maturity (monitoring feed).
        self._pending_ctx: deque[_PendingLabelCtx] = deque(maxlen=4_096)

    # -- state the engine reads ---------------------------------------- #

    @property
    def portfolio(self) -> Portfolio:
        return self._state.portfolio

    @property
    def realized_cum(self) -> float:
        return self._state.realized_cum

    def restore_realized(self, realized_cum: float) -> None:
        """Named mutation point for crash-safe resume.

        The engine restores the persisted running P&L here after reopening
        the position snapshot; everything else about ``TradingState`` is
        rebuilt by replaying boundaries. Only the resume path may call this.
        """
        if not math.isfinite(realized_cum):
            raise ValueError(
                f"restore_realized requires a finite value; got {realized_cum!r}"
            )
        self._state.realized_cum = float(realized_cum)

    # ------------------------------------------------------------------ #

    def on_boundary(
        self,
        *,
        k: int,
        ts: pd.Timestamp,
        p: float,
        bar: Bar,
        regime_value: float,
        phi: float,
        mean_p_ve: float = float("nan"),
        knowledge_unc: float = float("nan"),
        allow_entry: bool = True,
    ) -> BoundaryResult:
        """One boundary step: exits → bulk → expiry → decide → maybe enter.

        ``allow_entry=False`` runs the full exit path but suppresses new
        entries (used for degraded bars — e.g. feature failure with p=NaN —
        and for halt states). Exit policies see NaN conviction on such bars
        and fail safe (close at market on TP touch).
        """
        st = self._state
        bar_close = float(bar.close)

        # Live path span = the single boundary bar, present only once a
        # previous boundary exists and inventory is open — the exact
        # condition the batch driver applies to its multi-bar span.
        if st.prev_ts is not None and st.portfolio.n_open() > 0:
            intra_bars = [
                IntraBar(
                    n=-1,
                    ts=ts,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=bar_close,
                )
            ]
        else:
            intra_bars = []

        outcome = self._step.run(
            st,
            BoundaryInputs(
                k=k,
                ts=ts,
                p=p,
                regime_value=regime_value,
                phi=phi,
                bar_close=bar_close,
                bar_high=float(bar.high),
                bar_low=float(bar.low),
                intra_bars=intra_bars,
                mean_p_ve=mean_p_ve,
                knowledge_unc=knowledge_unc,
                allow_entry=allow_entry,
            ),
        )

        # Retain decision context for label-maturity feedback.
        self._pending_ctx.append(
            _PendingLabelCtx(ts=ts, p=p, regime_q=outcome.regime_q)
        )

        equity = EquitySnapshot(
            ts=ts,
            k=k,
            realized_cum=st.realized_cum,
            unrealized=(
                st.portfolio.mtm_log_return(bar_close) if bar_close > 0 else 0.0
            ),
            n_open=st.portfolio.n_open(),
            gross_size=st.portfolio.gross_size(),
        )
        return BoundaryResult(
            ts=ts,
            k=k,
            closed=outcome.closed,
            opened=outcome.opened,
            bulk_reason=outcome.bulk_reason,
            score=outcome.score,
            equity=equity,
            entered=outcome.opened is not None,
        )

    # ------------------------------------------------------------------ #

    def feed_matured_label(self, entry_ts: pd.Timestamp, y: int) -> None:
        """Fold a matured label into monitoring stats (base rate).

        Live counterpart of the batch driver's label feedback — fed at
        maturity (``t+M``) with the decision-time regime quantile, which is
        the strictly-causal version of the same update.
        """
        key = pd.Timestamp(entry_ts)
        while self._pending_ctx and pd.Timestamp(self._pending_ctx[0].ts) < key:
            self._pending_ctx.popleft()
        if not self._pending_ctx or pd.Timestamp(self._pending_ctx[0].ts) != key:
            return
        ctx = self._pending_ctx.popleft()
        if not math.isnan(ctx.regime_q):
            self.base_rate.update(ctx.regime_q, float(y))
