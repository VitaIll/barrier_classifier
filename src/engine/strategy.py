"""Live strategy core — the simulator's boundary step, one bar at a time.

``LiveTrader.on_boundary`` re-expresses ``src.strategy.simulator.simulate``'s
per-boundary steps 1–10 incrementally, reusing the *same* primitives
(``Portfolio``, ``State``, ``StrategySpec`` gates/sizers/exits, the public
``resolve_intra_path_exits``/``resolve_expiries`` helpers, and the online
stats from ``src.strategy.online``). Given the same stream of
``(ts, p, bar)`` it produces the same ledger as the offline backtest —
enforced by ``tests/engine/test_parity_simulator.py``.

One deliberate causality difference: the offline simulator feeds boundary
``k``'s *label* into its monitoring stats at row ``k`` (the label is only
knowable at ``k+M`` in real time). The live trader feeds labels at
maturity instead. Specs that consume label-fed stats (``score_residualized``
base rates, drift) therefore see slightly *older* information live; the
production spec consumes neither, so its decisions are identical.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.engine.domain import Bar
from src.strategy.inventory import ClosedPosition, Portfolio, Position
from src.strategy.online import FastVolEWMA, RollingQuantileRank, RollingRegimeBaseRate
from src.strategy.policy import (
    IntraBar,
    RiskConfig,
    State,
    StrategySpec,
    gate_score_above,
    make_exit_let_winners_run,
    make_exit_let_winners_run_monotonic,
    score_raw_p,
    size_clip,
    size_constant,
)
from src.strategy.simulator import (
    SimConfig,
    resolve_expiries,
    resolve_intra_path_exits,
)


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
    """Incremental boundary stepper over an unmodified :class:`StrategySpec`."""

    def __init__(
        self,
        spec: StrategySpec,
        *,
        sim_config: Optional[SimConfig] = None,
    ) -> None:
        self.spec = spec
        self.cfg = sim_config or SimConfig()
        self.portfolio = Portfolio()
        self.regime_rank = RollingQuantileRank(
            window=self.cfg.quantile_window, min_warmup=self.cfg.quantile_min_warmup
        )
        self.score_rank = RollingQuantileRank(
            window=self.cfg.quantile_window, min_warmup=self.cfg.quantile_min_warmup
        )
        self.unc_rank = RollingQuantileRank(
            window=self.cfg.quantile_window, min_warmup=self.cfg.quantile_min_warmup
        )
        self.fast_vol = FastVolEWMA(
            halflife_bars=self.cfg.fast_vol_halflife,
            min_warmup=self.cfg.fast_vol_min_warmup,
        )
        self.base_rate = RollingRegimeBaseRate(
            window=self.cfg.base_rate_window, n_bins=self.cfg.base_rate_n_bins
        )
        self.realized_cum = 0.0
        self.cost_per_trade = (
            self.cfg.cost_per_trade_override
            if self.cfg.cost_per_trade_override is not None
            else spec.risk.cost_per_trade
        )
        self._prev_ts: Optional[pd.Timestamp] = None
        self._prev_close: Optional[float] = None
        # Cluster bookkeeping (mirrors simulate()).
        self._cluster_id = 0
        self._cluster_open_id: Optional[int] = None
        self._cluster_streak = 0
        self._cluster_pnl = 0.0
        self._cluster_entry_count = 0
        # Decision contexts pending label maturity (monitoring feed).
        self._pending_ctx: deque[_PendingLabelCtx] = deque(maxlen=4_096)

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
        bar_close = float(bar.close)
        bar_high = float(bar.high)
        bar_low = float(bar.low)

        # ---- 1. Online stats (rank-then-update = causal) -----------------
        if self._prev_close is not None and self._prev_close > 0 and bar_close > 0:
            self.fast_vol.update(math.log(bar_close / self._prev_close))
        regime_q = self.regime_rank.rank_and_update(regime_value)
        unc_q = (
            self.unc_rank.rank_and_update(knowledge_unc)
            if not math.isnan(knowledge_unc) else float("nan")
        )

        # ---- 2. Resolve elapsed TP/SL exits BEFORE composing State -------
        closed_this_step: list[ClosedPosition] = []
        bulk_reason: Optional[str] = None
        if self._prev_ts is not None and self.portfolio.n_open() > 0:
            intra = [IntraBar(
                n=-1, ts=ts, open=float(bar.open), high=bar_high,
                low=bar_low, close=bar_close,
            )]
            sketch = State(
                k=k, ts=ts, p=p, p_calibrated=p,
                bar_close=bar_close, bar_high=bar_high, bar_low=bar_low,
                regime_value=regime_value, regime_quantile=regime_q,
                fast_sigma=self.fast_vol.value(),
                n_open_positions=self.portfolio.n_open(),
                cluster_pnl=0.0,
                cluster_streak=self._cluster_streak,
                mean_p_ve=mean_p_ve,
                knowledge_unc=knowledge_unc,
                knowledge_unc_quantile=unc_q,
            )
            closed_this_step.extend(resolve_intra_path_exits(
                self.portfolio, self.spec, intra_bars=intra, k_now=k,
                state_for_records=sketch,
            ))

        # ---- 3-4. Bulk-close on post-exit inventory ----------------------
        cluster_pnl_pre_bulk = (
            sum(pos.size * pos.mtm_log_return(bar_close)
                for pos in self.portfolio.open_positions)
            if self.portfolio.n_open() > 0 else 0.0
        )
        state_pre_bulk = State(
            k=k, ts=ts, p=p, p_calibrated=p,
            bar_close=bar_close, bar_high=bar_high, bar_low=bar_low,
            regime_value=regime_value, regime_quantile=regime_q,
            fast_sigma=self.fast_vol.value(),
            n_open_positions=self.portfolio.n_open(),
            cluster_pnl=cluster_pnl_pre_bulk,
            cluster_streak=self._cluster_streak,
            inventory_gross_size=self.portfolio.gross_size(),
            mean_p_ve=mean_p_ve,
            knowledge_unc=knowledge_unc,
            knowledge_unc_quantile=unc_q,
        )
        if self.portfolio.n_open() > 0:
            bulk_reason = self.spec.bulk_close(state_pre_bulk)
        if bulk_reason is not None:
            closed_this_step.extend(self.portfolio.close_all(
                k_exit=k, ts_exit=ts, exit_price=bar_close, exit_reason=bulk_reason
            ))

        # ---- 5. Expiry-by-time -------------------------------------------
        if self.portfolio.n_open() > 0:
            closed_this_step.extend(resolve_expiries(
                self.portfolio, self.spec, boundary_close_price=bar_close,
                k_now=k, ts_now=ts, state_for_records=state_pre_bulk,
            ))

        for c in closed_this_step:
            self.realized_cum += c.weighted_net_log_return(self.cost_per_trade)

        # ---- 6. Cluster bookkeeping --------------------------------------
        if self.portfolio.n_open() == 0:
            if self._cluster_open_id is not None:
                self._cluster_open_id = None
                self._cluster_pnl = 0.0
                self._cluster_entry_count = 0
                self._cluster_streak = 0
        else:
            self._cluster_streak += 1

        # ---- 7. Compose the entry-decision State -------------------------
        cluster_pnl_at_decision = (
            sum(pos.size * pos.mtm_log_return(bar_close)
                for pos in self.portfolio.open_positions)
            if self.portfolio.n_open() > 0 else 0.0
        )
        state_no_score = State(
            k=k, ts=ts, p=p, p_calibrated=p,
            bar_close=bar_close, bar_high=bar_high, bar_low=bar_low,
            regime_value=regime_value, regime_quantile=regime_q,
            fast_sigma=self.fast_vol.value(),
            n_open_positions=self.portfolio.n_open(),
            cluster_pnl=cluster_pnl_at_decision,
            cluster_streak=self._cluster_streak,
            inventory_gross_size=self.portfolio.gross_size(),
            mean_p_ve=mean_p_ve,
            knowledge_unc=knowledge_unc,
            knowledge_unc_quantile=unc_q,
        )
        s_score = float(self.spec.score_fn(state_no_score))
        state = State(**{**vars(state_no_score), "score": s_score})
        self.score_rank.update(s_score)

        # ---- 8. Entry decision -------------------------------------------
        opened: Optional[Position] = None
        if (
            allow_entry
            and self.spec.evaluate_entry(state)
            and self.portfolio.n_open() < self.spec.risk.max_open_positions
            and self.portfolio.gross_size() < self.spec.risk.max_gross_size
            and bar_close > 0
        ):
            size = float(self.spec.sizer(state))
            size = min(size, self.spec.risk.max_size_per_lot)
            size = min(size, self.spec.risk.max_gross_size - self.portfolio.gross_size())
            if size > 0:
                # Mirror of the simulator's guard (parity contract): a
                # non-finite phi must never become a NaN take-profit.
                if not (math.isfinite(phi) and phi > 0):
                    raise ValueError(
                        f"entry at k={k} (ts={ts}) with non-finite or "
                        f"non-positive phi={phi!r} — refusing to open a "
                        "position with an undefined take-profit"
                    )
                tp_price = bar_close * math.exp(phi)
                if self.spec.risk.position_mtm_floor_log_return is not None:
                    sl_price = bar_close * math.exp(
                        float(self.spec.risk.position_mtm_floor_log_return)
                    )
                else:
                    sl_price = None
                opened = Position(
                    k_entry=k, ts_entry=ts, side=1, size=size,
                    entry_price=bar_close, tp_price=tp_price, sl_price=sl_price,
                    expiry_k=k + int(self.spec.risk.max_horizon_boundaries),
                )
                self.portfolio.open_one(opened)
                if self._cluster_open_id is None:
                    self._cluster_open_id = self._cluster_id
                    self._cluster_id += 1
                    self._cluster_streak = 1
                self._cluster_entry_count += 1

        if self._cluster_open_id is not None and closed_this_step:
            for c in closed_this_step:
                self._cluster_pnl += c.weighted_net_log_return(self.cost_per_trade)

        # ---- 9. Retain decision context for label-maturity feedback ------
        self._pending_ctx.append(_PendingLabelCtx(ts=ts, p=p, regime_q=regime_q))

        self._prev_ts = ts
        self._prev_close = bar_close

        equity = EquitySnapshot(
            ts=ts, k=k, realized_cum=self.realized_cum,
            unrealized=(
                self.portfolio.mtm_log_return(bar_close) if bar_close > 0 else 0.0
            ),
            n_open=self.portfolio.n_open(),
            gross_size=self.portfolio.gross_size(),
        )
        return BoundaryResult(
            ts=ts, k=k, closed=tuple(closed_this_step), opened=opened,
            bulk_reason=bulk_reason, score=s_score, equity=equity,
            entered=opened is not None,
        )

    # ------------------------------------------------------------------ #

    def feed_matured_label(self, entry_ts: pd.Timestamp, y: int) -> None:
        """Fold a matured label into monitoring stats (base rate).

        Live counterpart of the simulator's step 9 — fed at maturity
        (``t+M``) with the decision-time regime quantile, which is the
        strictly-causal version of the same update.
        """
        key = pd.Timestamp(entry_ts)
        while self._pending_ctx and pd.Timestamp(self._pending_ctx[0].ts) < key:
            self._pending_ctx.popleft()
        if not self._pending_ctx or pd.Timestamp(self._pending_ctx[0].ts) != key:
            return
        ctx = self._pending_ctx.popleft()
        if not math.isnan(ctx.regime_q):
            self.base_rate.update(ctx.regime_q, float(y))
