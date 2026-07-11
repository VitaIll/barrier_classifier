"""Bar-by-bar honest backtest driver.

Walks boundaries chronologically. At each boundary k:

1. Walk the M intra-horizon 1-min bars in ``(n_{k-1}, n_k]`` and resolve any
   open position whose horizon ended in that window (TP / SL).
2. Mark remaining inventory at ``close[n_k]`` and update online stats
   (fast vol, regime quantile, MI quantile).
3. Compose the decision-time ``State`` AFTER elapsed exits have resolved
   — so the spec's gates and sizer see the post-resolution
   ``n_open_positions`` and ``cluster_pnl`` rather than a stale snapshot.
4. Evaluate the spec's ``bulk_close``; if it fires, flatten what survived
   step 1 at ``close[n_k]``.
5. Resolve expiry-by-time for any positions still open.
6. Update cluster bookkeeping.
7. Evaluate the spec's entry gates; if they pass, size the new lot, cap to
   risk limits, and open a position at ``close[n_k]``.

Causality: every decision uses only data observable at or before the
boundary's close. No primitive reads ``y_k`` or any future price during
entry decision; ``y_k`` only affects the simulator after the position has
closed (it's recorded into the closed-trade ledger for analytics).

The simulator is a pure function: ``simulate(cache, raw_bars, spec) -> SimResult``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.inventory import ClosedPosition, Portfolio, Position
from src.strategy.online import (
    DriftADWIN,
    FastVolEWMA,
    RollingQuantileRank,
    RollingRegimeBaseRate,
)
from src.strategy.policy import IntraBar, RiskConfig, State, StrategySpec


# ---------------------------------------------------------------------------
# Output bundle
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    """Output of one ``simulate`` call.

    Attributes
    ----------
    spec_name : str
    closed : pd.DataFrame
        One row per closed trade (output of ``Portfolio.closed_to_frame``).
    equity : pd.DataFrame
        Per-boundary realized + unrealized equity curve. Columns:
        ``ts, k, realized_log_ret, unrealized_log_ret, equity, n_open,
        gross_size, n_trades_closed, regime_quantile, p, mean_p_ve,
        knowledge_unc, knowledge_unc_quantile``.
    cluster_log : pd.DataFrame
        One row per cluster (sequence of boundary entries + their exits),
        with cumulative P&L, n_entries, duration_boundaries, end_reason.
    diagnostics_used : dict[str, bool]
        Snapshot of which diagnostic flags were declared by the spec.
    config : dict
        Echo of run-time config — sigma_target, reference_window, etc.
    """

    spec_name: str
    closed: pd.DataFrame
    equity: pd.DataFrame
    cluster_log: pd.DataFrame
    diagnostics_used: dict
    config: dict


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    """Run-time knobs that aren't part of the StrategySpec.

    Kept here (not in StrategySpec) because they're per-RUN, not per-strategy:
    a sweep over ``cost_per_trade`` or ``sigma_target`` re-uses the same spec.
    """

    M: int = 20
    quantile_window: int = 1000          # rolling window for streaming quantile rank
    quantile_min_warmup: int = 100
    fast_vol_halflife: float = 30.0      # bars; ~ 10 hours at M=20-min boundaries
    fast_vol_min_warmup: int = 30
    base_rate_window: int = 1000
    base_rate_n_bins: int = 5
    sigma_target: float = 0.001          # used by size_voltarget_overlay if spec opts in
    cost_per_trade_override: Optional[float] = None  # overrides spec.risk.cost_per_trade
    # Per-row cadence in minutes. 20.0 = legacy 20-min-boundary mode; 1.0 =
    # canonical 1-min spec from the overlapping-target refactor. Used by
    # reporting.headline_row for Sharpe annualization. Echoed onto
    # SimResult.config so reporting can pick it up without an extra arg.
    cadence_minutes: float = 20.0


def _bar_index_at_boundary(boundary_index_k: int, M: int, k_offset: int = 0) -> int:
    """Compute the 1-min bar index ``n_k = (k - k_offset) * M``.

    ``k_offset`` accommodates a cache that starts at ``K_WARMUP`` while
    raw_bars start at index 0; in practice we resolve via ts join and
    don't use this helper for path lookup.
    """
    return (int(boundary_index_k) - int(k_offset)) * int(M)


# ---------------------------------------------------------------------------
# Path-bar lookup
# ---------------------------------------------------------------------------


def get_intra_bars(
    raw_bars: pd.DataFrame,
    *,
    ts_after: pd.Timestamp,
    ts_through: pd.Timestamp,
) -> list[IntraBar]:
    """Return the list of 1-min IntraBars with ``ts > ts_after`` and ``ts <= ts_through``,
    in chronological order. ``raw_bars`` must be ts-indexed (UTC or naive)."""
    if raw_bars.index.tz is not None:
        # Strip tz for comparison consistency with cache (which has tz-naive ts)
        idx = raw_bars.index.tz_localize(None)
    else:
        idx = raw_bars.index
    after = pd.Timestamp(ts_after)
    if after.tzinfo is not None:
        after = after.tz_convert(None) if after.tz is not None else after.tz_localize(None)
    thru = pd.Timestamp(ts_through)
    if thru.tzinfo is not None:
        thru = thru.tz_convert(None) if thru.tz is not None else thru.tz_localize(None)
    mask = (idx > after) & (idx <= thru)
    sub = raw_bars.loc[mask]
    out: list[IntraBar] = []
    for ts, row in sub.iterrows():
        out.append(
            IntraBar(
                n=-1,  # bar index not needed at this layer
                ts=pd.Timestamp(ts),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Position resolution over a path
# ---------------------------------------------------------------------------


def resolve_intra_path_exits(
    portfolio: Portfolio,
    spec: StrategySpec,
    *,
    intra_bars: list[IntraBar],
    k_now: int,
    state_for_records: State,
) -> list[ClosedPosition]:
    """Phase 1 of exit resolution: walk path bars, close on TP/SL only.

    Calls ``spec.exit_policy(pos, bar, k_now)`` for each intra-bar in
    order; the first firing closes that position at the appropriate
    barrier price. Positions that survive the path-walk stay open and
    are dealt with by ``_resolve_expiries`` after bulk-close evaluation.
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
                # Defensive: any other reason during path-walk uses bar close.
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


# Backwards-compatible aliases (pre-engine private names).
_get_intra_bars = get_intra_bars
_resolve_intra_path_exits = resolve_intra_path_exits
_resolve_expiries = resolve_expiries


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def simulate(
    cache: pd.DataFrame,
    raw_bars: pd.DataFrame,
    spec: StrategySpec,
    *,
    config: Optional[SimConfig] = None,
    p_ve_samples: Optional[np.ndarray] = None,
) -> SimResult:
    """Run the spec over the cache + raw_bars in chronological order.

    ``cache`` must carry at least: ``ts, k, p, regime, phi`` (and ``y, m_k``
    for analytics). Optional: ``mean_p_ve, knowledge_unc``. ``p_ve_samples``,
    if supplied, must be aligned row-wise to ``cache`` (shape ``(N, K_ve)``).

    The ``cache`` is sorted by ``ts`` internally; ``raw_bars`` must be a
    ts-indexed OHLCV DataFrame whose index covers the cache's time range.
    """
    cfg = config or SimConfig()

    if cache.empty:
        return SimResult(
            spec_name=spec.name,
            closed=pd.DataFrame(),
            equity=pd.DataFrame(),
            cluster_log=pd.DataFrame(),
            diagnostics_used={r: True for r in spec.requires},
            config=vars(cfg).copy(),
        )

    cache = cache.sort_values("ts").reset_index(drop=True)
    has_ve = "mean_p_ve" in cache.columns and "knowledge_unc" in cache.columns

    # --- streaming online stats ------------------------------------------------
    regime_rank = RollingQuantileRank(
        window=cfg.quantile_window, min_warmup=cfg.quantile_min_warmup
    )
    score_rank = RollingQuantileRank(
        window=cfg.quantile_window, min_warmup=cfg.quantile_min_warmup
    )
    unc_rank = RollingQuantileRank(
        window=cfg.quantile_window, min_warmup=cfg.quantile_min_warmup
    )
    fast_vol = FastVolEWMA(
        halflife_bars=cfg.fast_vol_halflife, min_warmup=cfg.fast_vol_min_warmup
    )
    base_rate = RollingRegimeBaseRate(window=cfg.base_rate_window, n_bins=cfg.base_rate_n_bins)
    drift = DriftADWIN()  # exposed via state via diagnostics; not consumed by v1 specs

    portfolio = Portfolio()
    cost_per_trade = (
        cfg.cost_per_trade_override
        if cfg.cost_per_trade_override is not None
        else spec.risk.cost_per_trade
    )

    # Cluster bookkeeping
    cluster_id = 0
    cluster_streak = 0
    cluster_pnl = 0.0
    cluster_open_id: Optional[int] = None
    cluster_log_rows: list[dict] = []
    cluster_entry_count = 0
    cluster_start_ts: Optional[pd.Timestamp] = None
    cluster_end_reason = ""

    equity_rows: list[dict] = []
    realized_cumulative = 0.0

    prev_ts: Optional[pd.Timestamp] = None
    prev_close: Optional[float] = None

    n = len(cache)
    for i in range(n):
        row = cache.iloc[i]
        k = int(row["k"])
        ts = pd.Timestamp(row["ts"])
        p = float(row["p"])
        regime_value = float(row["regime"])
        phi = float(row["phi"])
        bar_close = float(row["close"]) if "close" in cache.columns else float("nan")
        bar_high = float(row["high"]) if "high" in cache.columns else float("nan")
        bar_low = float(row["low"]) if "low" in cache.columns else float("nan")
        if math.isnan(bar_close):
            # Caller must augment the cache with boundary OHLC upfront via
            # ``augment_cache_with_boundary_ohlc`` — silent fallback to a
            # raw_bars lookup hid a bug once and we don't want to repeat it.
            raise ValueError(
                f"boundary row at ts={ts} (k={k}) is missing 'close' — augment "
                f"the cache with ``augment_cache_with_boundary_ohlc`` first"
            )

        mean_p_ve = float(row["mean_p_ve"]) if has_ve else float("nan")
        knowledge_unc = float(row["knowledge_unc"]) if has_ve else float("nan")
        y = float(row["y"]) if "y" in cache.columns and not math.isnan(row.get("y", float("nan"))) else float("nan")

        # ---- 1. Update online stats with current observations ---------------
        # Causality: we update stats with values observable at close[n_k]
        # BEFORE building the State that downstream gates read.
        if prev_close is not None and prev_close > 0 and bar_close > 0:
            log_ret = math.log(bar_close / prev_close)
            fast_vol.update(log_ret)
        regime_q = regime_rank.rank_and_update(regime_value)
        unc_q = unc_rank.rank_and_update(knowledge_unc) if has_ve else float("nan")

        # ---- 2. Resolve elapsed TP/SL exits BEFORE composing State ---------
        # The docstring contract says exits over the (prev_ts, ts] path are
        # resolved first, so that the State the strategy sees reflects
        # post-resolution inventory and cluster PnL — not stale pre-exit
        # values. This is still strictly causal: every exit uses only the
        # intra-bar OHLC that finished before ts, and the State sees a
        # snapshot of the world at ts that includes those resolutions.
        if prev_ts is not None and portfolio.n_open() > 0:
            intra_bars = get_intra_bars(raw_bars, ts_after=prev_ts, ts_through=ts)
        else:
            intra_bars = []
        closed_this_step: list[ClosedPosition] = []
        if intra_bars and portfolio.n_open() > 0:
            # State-for-records uses a minimal sketch — we only need the
            # diagnostic fields recorded on the closed-trade ledger.
            sketch = State(
                k=k, ts=ts, p=p, p_calibrated=p,
                bar_close=bar_close, bar_high=bar_high, bar_low=bar_low,
                regime_value=regime_value, regime_quantile=regime_q,
                fast_sigma=fast_vol.value(),
                n_open_positions=portfolio.n_open(),
                cluster_pnl=0.0,
                cluster_streak=cluster_streak,
                mean_p_ve=mean_p_ve,
                knowledge_unc=knowledge_unc,
                knowledge_unc_quantile=unc_q,
            )
            closed_this_step.extend(
                resolve_intra_path_exits(
                    portfolio,
                    spec,
                    intra_bars=intra_bars,
                    k_now=k,
                    state_for_records=sketch,
                )
            )

        # ---- 3. Pre-bulk State for bulk_close trigger ----------------------
        # Bulk-close inspects post-elapsed-exit inventory but does NOT need to
        # see the score (it's keyed on regime / unc / cluster P&L). Compose a
        # minimal pre-bulk state for the bulk_close primitive only.
        cluster_pnl_pre_bulk = (
            sum(
                pos.size * pos.mtm_log_return(bar_close)
                for pos in portfolio.open_positions
            )
            if portfolio.n_open() > 0
            else 0.0
        )
        state_pre_bulk = State(
            k=k,
            ts=ts,
            p=p,
            p_calibrated=p,
            bar_close=bar_close,
            bar_high=bar_high,
            bar_low=bar_low,
            regime_value=regime_value,
            regime_quantile=regime_q,
            fast_sigma=fast_vol.value(),
            n_open_positions=portfolio.n_open(),
            cluster_pnl=cluster_pnl_pre_bulk,
            cluster_streak=cluster_streak,
            inventory_gross_size=portfolio.gross_size(),
            mean_p_ve=mean_p_ve,
            knowledge_unc=knowledge_unc,
            knowledge_unc_quantile=unc_q,
            p_ve_samples=(
                np.asarray(p_ve_samples[i], dtype=float)
                if p_ve_samples is not None
                else np.empty(0)
            ),
        )

        # ---- 4. Bulk-close trigger (pre-expiry; affects survivors) ---------
        bulk_reason = spec.bulk_close(state_pre_bulk) if portfolio.n_open() > 0 else None
        if bulk_reason is not None:
            bulk_closed = portfolio.close_all(
                k_exit=k, ts_exit=ts, exit_price=bar_close, exit_reason=bulk_reason
            )
            closed_this_step.extend(bulk_closed)

        # ---- 5. Expiry-by-time for any positions still open ----------------
        if portfolio.n_open() > 0:
            closed_this_step.extend(
                resolve_expiries(
                    portfolio,
                    spec,
                    boundary_close_price=bar_close,
                    k_now=k,
                    ts_now=ts,
                    state_for_records=state_pre_bulk,
                )
            )

        # Accumulate realized P&L from any closes this step
        if closed_this_step:
            for c in closed_this_step:
                realized_cumulative += c.weighted_net_log_return(cost_per_trade)

        # ---- 6. Cluster bookkeeping -----------------------------------------
        if portfolio.n_open() == 0:
            # Cluster ended this step (or we're flat)
            if cluster_open_id is not None:
                cluster_log_rows.append(
                    {
                        "cluster_id": cluster_open_id,
                        "ts_start": cluster_start_ts,
                        "ts_end": ts,
                        "n_entries": cluster_entry_count,
                        "duration_boundaries": cluster_streak,
                        "end_reason": (
                            bulk_reason or
                            (closed_this_step[-1].exit_reason if closed_this_step else "expiry_flat")
                        ),
                        "cluster_pnl": cluster_pnl,
                    }
                )
                cluster_open_id = None
                cluster_pnl = 0.0
                cluster_entry_count = 0
                cluster_streak = 0
                cluster_start_ts = None
        else:
            # Streak continues while at least one position remains open
            # after bulk_close + expiry at the current boundary. Reset
            # to 1 when a fresh cluster opens (handled in the entry
            # branch below — see ``cluster_streak = 1`` there).
            cluster_streak += 1

        # ---- 7. Compose entry-decision State (post-bulk, post-expiry) -------
        # The gate/sizer must see the FINAL post-resolution inventory: after
        # elapsed-path exits, bulk_close, and expiry have all fired. Rebuild
        # State here rather than reuse state_pre_bulk so n_open_positions,
        # cluster_pnl, inventory_gross_size, and cluster_streak all reflect
        # the world entering the decision call.
        cluster_pnl_at_decision = (
            sum(
                pos.size * pos.mtm_log_return(bar_close)
                for pos in portfolio.open_positions
            )
            if portfolio.n_open() > 0
            else 0.0
        )
        state_no_score = State(
            k=k,
            ts=ts,
            p=p,
            p_calibrated=p,
            bar_close=bar_close,
            bar_high=bar_high,
            bar_low=bar_low,
            regime_value=regime_value,
            regime_quantile=regime_q,
            fast_sigma=fast_vol.value(),
            n_open_positions=portfolio.n_open(),
            cluster_pnl=cluster_pnl_at_decision,
            cluster_streak=cluster_streak,
            inventory_gross_size=portfolio.gross_size(),
            mean_p_ve=mean_p_ve,
            knowledge_unc=knowledge_unc,
            knowledge_unc_quantile=unc_q,
            p_ve_samples=(
                np.asarray(p_ve_samples[i], dtype=float)
                if p_ve_samples is not None
                else np.empty(0)
            ),
        )
        s_score = float(spec.score_fn(state_no_score))
        state = State(**{**vars(state_no_score), "score": s_score})
        # Update score quantile (post-rank, in case future specs key off it)
        score_rank.update(s_score)

        # ---- 8. Entry decision ----------------------------------------------
        opened = False
        if (
            spec.evaluate_entry(state)
            and portfolio.n_open() < spec.risk.max_open_positions
            and portfolio.gross_size() < spec.risk.max_gross_size
            and bar_close > 0
        ):
            size = float(spec.sizer(state))
            size = min(size, spec.risk.max_size_per_lot)
            size = min(size, spec.risk.max_gross_size - portfolio.gross_size())
            if size > 0:
                # A non-finite phi would produce a NaN tp_price whose TP
                # condition (high >= NaN) never fires — a silently corrupt
                # position. Refuse loudly; the cache row is broken.
                if not (math.isfinite(phi) and phi > 0):
                    raise ValueError(
                        f"entry at k={k} (ts={ts}) with non-finite or "
                        f"non-positive phi={phi!r} — cache 'phi' column is "
                        "corrupt; refusing to open a position with an "
                        "undefined take-profit"
                    )
                tp_price = bar_close * math.exp(phi)  # long TP at +phi
                # Per-position MTM floor → SL price (if configured in RiskConfig)
                if spec.risk.position_mtm_floor_log_return is not None:
                    sl_price = bar_close * math.exp(
                        float(spec.risk.position_mtm_floor_log_return)
                    )
                else:
                    sl_price = None
                position = Position(
                    k_entry=k,
                    ts_entry=ts,
                    side=1,
                    size=size,
                    entry_price=bar_close,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    # Horizon comes from RiskConfig; default 1 = label-aligned;
                    # higher = patient wait-for-TP. Bulk-close triggers still
                    # apply across the longer horizon.
                    expiry_k=k + int(spec.risk.max_horizon_boundaries),
                )
                portfolio.open_one(position)
                opened = True
                if cluster_open_id is None:
                    cluster_open_id = cluster_id
                    cluster_id += 1
                    cluster_start_ts = ts
                    cluster_streak = 1
                cluster_entry_count += 1

        # Update cluster_pnl tally (counts realized closes during the cluster)
        if cluster_open_id is not None and closed_this_step:
            for c in closed_this_step:
                cluster_pnl += c.weighted_net_log_return(cost_per_trade)

        # ---- 9. Online label feedback (post-decision; drift + base-rate) ----
        # ``y`` is the LABEL of boundary k, which the spec did NOT see at
        # decision time. We feed it AFTER the entry decision so that for
        # ``score_residualized`` specs, the regime-conditional base rate
        # used by the score at boundary k does NOT depend on y[k]. The
        # label is only folded into the rolling accumulators consumed at
        # boundary k+1 onwards — consistent with the "label available
        # retrospectively, not at decision time" causality contract.
        if not math.isnan(y):
            if not math.isnan(p):
                drift.update(y - p)
            if not math.isnan(regime_q):
                base_rate.update(regime_q, y)

        # ---- 10. Equity row -------------------------------------------------
        equity_rows.append(
            {
                "ts": ts,
                "k": k,
                "realized_cum": realized_cumulative,
                "unrealized": (
                    portfolio.mtm_log_return(bar_close) if bar_close > 0 else 0.0
                ),
                "equity": realized_cumulative + (
                    portfolio.mtm_log_return(bar_close) if bar_close > 0 else 0.0
                ),
                "n_open": portfolio.n_open(),
                "gross_size": portfolio.gross_size(),
                "n_trades_closed_step": len(closed_this_step),
                "n_trades_closed_cum": len(portfolio.closed_positions),
                "regime_quantile": regime_q,
                "p": p,
                "mean_p_ve": mean_p_ve,
                "knowledge_unc": knowledge_unc,
                "knowledge_unc_quantile": unc_q,
                "fast_sigma": fast_vol.value(),
                "score": s_score,
                "opened_this_step": opened,
                "bulk_close_reason": bulk_reason,
            }
        )

        prev_ts = ts
        prev_close = bar_close

    # --- finalize: flush any open cluster --------------------------------------
    if cluster_open_id is not None:
        cluster_log_rows.append(
            {
                "cluster_id": cluster_open_id,
                "ts_start": cluster_start_ts,
                "ts_end": prev_ts,
                "n_entries": cluster_entry_count,
                "duration_boundaries": cluster_streak,
                "end_reason": "end_of_data",
                "cluster_pnl": cluster_pnl,
            }
        )

    closed_df = portfolio.closed_to_frame()
    equity_df = pd.DataFrame(equity_rows)
    cluster_df = pd.DataFrame(cluster_log_rows)
    return SimResult(
        spec_name=spec.name,
        closed=closed_df,
        equity=equity_df,
        cluster_log=cluster_df,
        diagnostics_used={r: True for r in spec.requires},
        config=vars(cfg).copy(),
    )


# ---------------------------------------------------------------------------
# Spec filter — used by the calibration notebook to drop specs whose
# diagnostic prerequisites failed.
# ---------------------------------------------------------------------------


def filter_specs_by_diagnostics(
    specs: list[StrategySpec], diagnostics_passed: dict[str, bool]
) -> list[StrategySpec]:
    """Return only specs whose every ``requires`` entry is True in the dict.

    Missing keys are treated as False — a strict requires-check.
    """
    out = []
    for spec in specs:
        if all(diagnostics_passed.get(r, False) for r in spec.requires):
            out.append(spec)
    return out
