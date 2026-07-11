"""The boundary step — ONE implementation of the per-boundary trading loop.

Before this module existed, the per-boundary sequence (resolve path exits →
bulk-close → expiries → compose state → gate/size → open → cluster/ledger
bookkeeping) lived twice: once in ``simulate()`` and once, hand-mirrored, in
the live trader — with parity maintained by test rather than by
construction. :class:`BoundaryStep` is that sequence, extracted verbatim;
``simulate()`` (batch) and ``LiveTrader`` (streaming) are now thin drivers
over it. The recorded golden ledgers (``tests/strategy/golden/``) and the
live≡offline parity suite pin the extraction.

Mutation contract (TARGET_ARCHITECTURE.md LAW 5): :class:`TradingState` is
the run's accumulator — created by exactly one driver, mutated only inside
:meth:`BoundaryStep.run`. Everything else here is immutable input/output.

Causality contract (pinned by tests/strategy/test_simulator.py): exits over
the elapsed path resolve BEFORE the decision state is composed, bulk-close
sees post-exit inventory, expiries run after bulk, and the entry decision
sees the final post-resolution world. Label feedback is deliberately NOT
part of the step — the batch driver feeds labels at the decision row, the
live driver at maturity (documented divergence; see engine.strategy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.strategy.inventory import ClosedPosition, Portfolio, Position
from src.strategy.online import FastVolEWMA, RollingQuantileRank
from src.strategy.policy import IntraBar, State, StrategySpec


# ---------------------------------------------------------------------------
# Exit resolution over a path (moved from simulator.py; simulator re-exports)
# ---------------------------------------------------------------------------


def resolve_intra_path_exits(
    portfolio: Portfolio,
    spec: StrategySpec,
    *,
    intra_bars: Sequence[IntraBar],
    k_now: int,
    state_for_records: State,
) -> list[ClosedPosition]:
    """Phase 1 of exit resolution: walk path bars, close on TP/SL only.

    Calls ``spec.exit_policy(pos, bar, k_now)`` for each intra-bar in
    order; the first firing closes that position at the appropriate
    barrier price. Positions that survive the path-walk stay open and
    are dealt with by :func:`resolve_expiries` after bulk-close evaluation.
    """
    closed: list[ClosedPosition] = []
    for pos in list(portfolio.open_positions):
        for bar in intra_bars:
            r = spec.exit_policy(pos, bar, k_now)
            if r is None:
                continue
            if r == "tp":
                exit_price = pos.tp_price
            elif r == "sl":
                exit_price = pos.sl_price if pos.sl_price is not None else bar.close
            else:
                # Market-fill reasons ("tp_market", "tp_or_sl", ...) use the
                # bar close. This branch is load-bearing, not defensive.
                exit_price = bar.close
            c = portfolio.close_one(
                pos,
                k_exit=k_now,
                ts_exit=bar.ts,
                exit_price=exit_price,
                exit_reason=r,
                p_at_entry=state_for_records.p,
                knowledge_unc_at_entry=state_for_records.knowledge_unc,
                regime_quantile_at_entry=state_for_records.regime_quantile,
            )
            closed.append(c)
            break
    return closed


def resolve_expiries(
    portfolio: Portfolio,
    spec: StrategySpec,
    *,
    boundary_close_price: float,
    k_now: int,
    ts_now: pd.Timestamp,
    state_for_records: State,
) -> list[ClosedPosition]:
    """Phase 2 of exit resolution: end-of-path expiry check.

    Calls ``spec.exit_policy(pos, intra_bar=None, k_now)`` once per
    open position. Closes any that return a non-None reason (typically
    ``"expiry"``) at ``boundary_close_price``.
    """
    closed: list[ClosedPosition] = []
    for pos in list(portfolio.open_positions):
        r = spec.exit_policy(pos, None, k_now)
        if r is None:
            continue
        c = portfolio.close_one(
            pos,
            k_exit=k_now,
            ts_exit=ts_now,
            exit_price=boundary_close_price,
            exit_reason=r,
            p_at_entry=state_for_records.p,
            knowledge_unc_at_entry=state_for_records.knowledge_unc,
            regime_quantile_at_entry=state_for_records.regime_quantile,
        )
        closed.append(c)
    return closed


# ---------------------------------------------------------------------------
# Cluster bookkeeping — one object instead of loose locals in two files
# ---------------------------------------------------------------------------


@dataclass
class ClusterTracker:
    """Overlapping-exposure ("cluster") bookkeeping across boundaries.

    A cluster spans from the first entry after a flat state until the
    portfolio is flat again. Owned by :class:`TradingState`; mutated only
    from :meth:`BoundaryStep.run`. ``rows`` accumulates one record per
    completed cluster (the batch driver exports them as the cluster log;
    the live driver may persist or ignore them — a record is a small dict,
    one per cluster, so growth is negligible even over long sessions).
    """

    next_id: int = 0
    open_id: Optional[int] = None
    streak: int = 0
    pnl: float = 0.0
    entry_count: int = 0
    start_ts: Optional[pd.Timestamp] = None
    rows: list[dict] = field(default_factory=list)

    def on_flat_boundary(
        self,
        ts: pd.Timestamp,
        bulk_reason: Optional[str],
        closed_this_step: Sequence[ClosedPosition],
    ) -> None:
        """Portfolio is flat after this boundary's exits: record + reset."""
        if self.open_id is None:
            return
        self.rows.append(
            {
                "cluster_id": self.open_id,
                "ts_start": self.start_ts,
                "ts_end": ts,
                "n_entries": self.entry_count,
                "duration_boundaries": self.streak,
                "end_reason": (
                    bulk_reason
                    or (
                        closed_this_step[-1].exit_reason
                        if closed_this_step
                        else "expiry_flat"
                    )
                ),
                "cluster_pnl": self.pnl,
            }
        )
        self.open_id = None
        self.pnl = 0.0
        self.entry_count = 0
        self.streak = 0
        self.start_ts = None

    def on_open_boundary(self) -> None:
        """At least one position survived this boundary: streak continues."""
        self.streak += 1

    def on_entry(self, ts: pd.Timestamp) -> None:
        """An entry fired; open a fresh cluster if none is running."""
        if self.open_id is None:
            self.open_id = self.next_id
            self.next_id += 1
            self.start_ts = ts
            self.streak = 1
        self.entry_count += 1

    def accrue_closed(
        self,
        closed_this_step: Sequence[ClosedPosition],
        cost_per_trade: float,
    ) -> None:
        """Fold this boundary's realized closes into the open cluster's P&L.

        Faithful to the historical ordering quirk: this runs AFTER the entry
        step, so when one boundary both flattens a cluster and opens a new
        one, the old cluster's closes accrue to the NEW cluster's P&L. The
        golden corpus pins this behavior; change it only deliberately.
        """
        if self.open_id is None or not closed_this_step:
            return
        for c in closed_this_step:
            self.pnl += c.weighted_net_log_return(cost_per_trade)

    def flush_end_of_data(self, ts_end: Optional[pd.Timestamp]) -> None:
        """Record a still-open cluster at the end of a batch run."""
        if self.open_id is None:
            return
        self.rows.append(
            {
                "cluster_id": self.open_id,
                "ts_start": self.start_ts,
                "ts_end": ts_end,
                "n_entries": self.entry_count,
                "duration_boundaries": self.streak,
                "end_reason": "end_of_data",
                "cluster_pnl": self.pnl,
            }
        )


# ---------------------------------------------------------------------------
# O(raw) intra-path lookup
# ---------------------------------------------------------------------------


class PathIndex:
    """Pre-indexed 1-min path bars with O(log n) span extraction.

    Replaces the per-boundary full-frame boolean mask (O(boundaries × raw))
    with one upfront normalization plus two ``searchsorted`` calls per
    span. Semantics identical to the historical ``get_intra_bars``:
    bars with ``ts_after < ts <= ts_through``, in order; tz-aware inputs
    compare in UTC.
    """

    def __init__(self, raw_bars: pd.DataFrame) -> None:
        idx = raw_bars.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        self._ts_ns = idx.asi8
        if len(self._ts_ns) > 1 and bool((np.diff(self._ts_ns) < 0).any()):
            raise ValueError("PathIndex requires a monotonically sorted ts index")
        self._open = raw_bars["open"].to_numpy(dtype=float)
        self._high = raw_bars["high"].to_numpy(dtype=float)
        self._low = raw_bars["low"].to_numpy(dtype=float)
        self._close = raw_bars["close"].to_numpy(dtype=float)

    @staticmethod
    def _as_ns(ts: pd.Timestamp) -> int:
        # Timestamp.value is UTC ns for aware stamps and wall-clock ns for
        # naive ones — matching the tz-stripped index on both sides.
        return int(pd.Timestamp(ts).value)

    def span(self, ts_after: pd.Timestamp, ts_through: pd.Timestamp) -> list[IntraBar]:
        lo = int(np.searchsorted(self._ts_ns, self._as_ns(ts_after), side="right"))
        hi = int(np.searchsorted(self._ts_ns, self._as_ns(ts_through), side="right"))
        return [
            IntraBar(
                n=-1,
                ts=pd.Timestamp(self._ts_ns[i]),
                open=float(self._open[i]),
                high=float(self._high[i]),
                low=float(self._low[i]),
                close=float(self._close[i]),
            )
            for i in range(lo, hi)
        ]


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------


@dataclass
class TradingState:
    """Per-run accumulator: portfolio, cluster, online stats, running P&L.

    Created by exactly one driver, mutated only by :meth:`BoundaryStep.run`
    (LAW 5). Both drivers construct it via :meth:`fresh` from the same
    ``SimConfig`` so their streaming estimators are identically configured.
    """

    portfolio: Portfolio
    cluster: ClusterTracker
    regime_rank: RollingQuantileRank
    score_rank: RollingQuantileRank
    unc_rank: RollingQuantileRank
    fast_vol: FastVolEWMA
    realized_cum: float = 0.0
    prev_ts: Optional[pd.Timestamp] = None
    prev_close: Optional[float] = None

    @staticmethod
    def fresh(cfg) -> "TradingState":
        """Build a zeroed state from a ``SimConfig``-shaped object."""
        return TradingState(
            portfolio=Portfolio(),
            cluster=ClusterTracker(),
            regime_rank=RollingQuantileRank(
                window=cfg.quantile_window, min_warmup=cfg.quantile_min_warmup
            ),
            score_rank=RollingQuantileRank(
                window=cfg.quantile_window, min_warmup=cfg.quantile_min_warmup
            ),
            unc_rank=RollingQuantileRank(
                window=cfg.quantile_window, min_warmup=cfg.quantile_min_warmup
            ),
            fast_vol=FastVolEWMA(
                halflife_bars=cfg.fast_vol_halflife,
                min_warmup=cfg.fast_vol_min_warmup,
            ),
        )


_EMPTY_VE: np.ndarray = np.empty(0)


def _empty_ve() -> np.ndarray:
    return _EMPTY_VE


@dataclass(frozen=True)
class BoundaryInputs:
    """Everything one boundary decision may observe. Immutable."""

    k: int
    ts: pd.Timestamp
    p: float
    regime_value: float
    phi: float
    bar_close: float
    bar_high: float
    bar_low: float
    intra_bars: Sequence[IntraBar] = ()
    mean_p_ve: float = float("nan")
    knowledge_unc: float = float("nan")
    p_ve_samples: np.ndarray = field(default_factory=_empty_ve)
    allow_entry: bool = True


@dataclass(frozen=True)
class StepOutcome:
    """What one boundary produced. Immutable; drivers render it their way."""

    closed: tuple[ClosedPosition, ...]
    opened: Optional[Position]
    bulk_reason: Optional[str]
    score: float
    regime_q: float
    unc_q: float
    fast_sigma: float
    state: State  # final decision-time state (post-resolution, scored)


class BoundaryStep:
    """The per-boundary sequence, verbatim from the historical loops."""

    def __init__(self, spec: StrategySpec, *, cost_per_trade: float) -> None:
        self.spec = spec
        self.cost_per_trade = float(cost_per_trade)
        # Optional per-position exit-state cleanup, attached by exit-policy
        # factories that keep closure state (e.g. the monotonic
        # let-winners-run tracker). Called for EVERY close, including bulk
        # and SL paths that never route through the exit policy itself —
        # this is what prevents the state from leaking (review N10).
        self._on_position_closed = getattr(
            spec.exit_policy, "on_position_closed", None
        )

    def run(self, st: TradingState, inp: BoundaryInputs) -> StepOutcome:
        spec = self.spec
        k, ts = inp.k, inp.ts
        p = inp.p
        bar_close, bar_high, bar_low = inp.bar_close, inp.bar_high, inp.bar_low

        # ---- 1. Online stats (rank-then-update = causal) -------------------
        if (
            st.prev_close is not None
            and st.prev_close > 0
            and bar_close > 0
        ):
            st.fast_vol.update(math.log(bar_close / st.prev_close))
        regime_q = st.regime_rank.rank_and_update(inp.regime_value)
        unc_q = (
            st.unc_rank.rank_and_update(inp.knowledge_unc)
            if not math.isnan(inp.knowledge_unc)
            else float("nan")
        )

        # ---- 2. Resolve elapsed TP/SL exits BEFORE composing State ---------
        closed_this_step: list[ClosedPosition] = []
        if inp.intra_bars and st.portfolio.n_open() > 0:
            sketch = State(
                k=k, ts=ts, p=p, p_calibrated=p,
                bar_close=bar_close, bar_high=bar_high, bar_low=bar_low,
                regime_value=inp.regime_value, regime_quantile=regime_q,
                fast_sigma=st.fast_vol.value(),
                n_open_positions=st.portfolio.n_open(),
                cluster_pnl=0.0,
                cluster_streak=st.cluster.streak,
                mean_p_ve=inp.mean_p_ve,
                knowledge_unc=inp.knowledge_unc,
                knowledge_unc_quantile=unc_q,
            )
            closed_this_step.extend(
                resolve_intra_path_exits(
                    st.portfolio,
                    spec,
                    intra_bars=inp.intra_bars,
                    k_now=k,
                    state_for_records=sketch,
                )
            )

        # ---- 3. Pre-bulk State on post-exit inventory -----------------------
        cluster_pnl_pre_bulk = (
            sum(
                pos.size * pos.mtm_log_return(bar_close)
                for pos in st.portfolio.open_positions
            )
            if st.portfolio.n_open() > 0
            else 0.0
        )
        state_pre_bulk = State(
            k=k, ts=ts, p=p, p_calibrated=p,
            bar_close=bar_close, bar_high=bar_high, bar_low=bar_low,
            regime_value=inp.regime_value, regime_quantile=regime_q,
            fast_sigma=st.fast_vol.value(),
            n_open_positions=st.portfolio.n_open(),
            cluster_pnl=cluster_pnl_pre_bulk,
            cluster_streak=st.cluster.streak,
            inventory_gross_size=st.portfolio.gross_size(),
            mean_p_ve=inp.mean_p_ve,
            knowledge_unc=inp.knowledge_unc,
            knowledge_unc_quantile=unc_q,
            p_ve_samples=inp.p_ve_samples,
        )

        # ---- 4. Bulk-close trigger (pre-expiry; affects survivors) ---------
        bulk_reason = (
            spec.bulk_close(state_pre_bulk) if st.portfolio.n_open() > 0 else None
        )
        if bulk_reason is not None:
            closed_this_step.extend(
                st.portfolio.close_all(
                    k_exit=k, ts_exit=ts, exit_price=bar_close,
                    exit_reason=bulk_reason,
                )
            )

        # ---- 5. Expiry-by-time for any positions still open ----------------
        if st.portfolio.n_open() > 0:
            closed_this_step.extend(
                resolve_expiries(
                    st.portfolio,
                    spec,
                    boundary_close_price=bar_close,
                    k_now=k,
                    ts_now=ts,
                    state_for_records=state_pre_bulk,
                )
            )

        # ---- realized P&L accrual (order matters for float parity) ---------
        for c in closed_this_step:
            st.realized_cum += c.weighted_net_log_return(self.cost_per_trade)

        # ---- 6. Cluster bookkeeping -----------------------------------------
        if st.portfolio.n_open() == 0:
            st.cluster.on_flat_boundary(ts, bulk_reason, closed_this_step)
        else:
            st.cluster.on_open_boundary()

        # ---- 7. Compose the entry-decision State (post-resolution) ----------
        cluster_pnl_at_decision = (
            sum(
                pos.size * pos.mtm_log_return(bar_close)
                for pos in st.portfolio.open_positions
            )
            if st.portfolio.n_open() > 0
            else 0.0
        )
        state_no_score = State(
            k=k, ts=ts, p=p, p_calibrated=p,
            bar_close=bar_close, bar_high=bar_high, bar_low=bar_low,
            regime_value=inp.regime_value, regime_quantile=regime_q,
            fast_sigma=st.fast_vol.value(),
            n_open_positions=st.portfolio.n_open(),
            cluster_pnl=cluster_pnl_at_decision,
            cluster_streak=st.cluster.streak,
            inventory_gross_size=st.portfolio.gross_size(),
            mean_p_ve=inp.mean_p_ve,
            knowledge_unc=inp.knowledge_unc,
            knowledge_unc_quantile=unc_q,
            p_ve_samples=inp.p_ve_samples,
        )
        s_score = float(spec.score_fn(state_no_score))
        state = State(**{**vars(state_no_score), "score": s_score})
        st.score_rank.update(s_score)

        # ---- 8. Entry decision ----------------------------------------------
        opened: Optional[Position] = None
        if (
            inp.allow_entry
            and spec.evaluate_entry(state)
            and st.portfolio.n_open() < spec.risk.max_open_positions
            and st.portfolio.gross_size() < spec.risk.max_gross_size
            and bar_close > 0
        ):
            size = float(spec.sizer(state))
            size = min(size, spec.risk.max_size_per_lot)
            size = min(size, spec.risk.max_gross_size - st.portfolio.gross_size())
            if size > 0:
                # A non-finite phi would produce a NaN tp_price whose TP
                # condition (high >= NaN) never fires — a silently corrupt
                # position. Refuse loudly; the input row is broken.
                if not (math.isfinite(inp.phi) and inp.phi > 0):
                    raise ValueError(
                        f"entry at k={k} (ts={ts}) with non-finite or "
                        f"non-positive phi={inp.phi!r} — refusing to open a "
                        "position with an undefined take-profit"
                    )
                tp_price = bar_close * math.exp(inp.phi)  # long TP at +phi
                if spec.risk.position_mtm_floor_log_return is not None:
                    sl_price = bar_close * math.exp(
                        float(spec.risk.position_mtm_floor_log_return)
                    )
                else:
                    sl_price = None
                opened = Position(
                    k_entry=k,
                    ts_entry=ts,
                    side=1,
                    size=size,
                    entry_price=bar_close,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    expiry_k=k + int(spec.risk.max_horizon_boundaries),
                )
                st.portfolio.open_one(opened)
                st.cluster.on_entry(ts)

        # ---- cluster P&L accrual (after entry — see ClusterTracker note) ---
        st.cluster.accrue_closed(closed_this_step, self.cost_per_trade)

        # ---- exit-state cleanup for stateful exit policies (N10) -----------
        if self._on_position_closed is not None and closed_this_step:
            for c in closed_this_step:
                self._on_position_closed(c)

        st.prev_ts = ts
        st.prev_close = bar_close

        return StepOutcome(
            closed=tuple(closed_this_step),
            opened=opened,
            bulk_reason=bulk_reason,
            score=s_score,
            regime_q=regime_q,
            unc_q=unc_q,
            fast_sigma=st.fast_vol.value(),
            state=state,
        )
