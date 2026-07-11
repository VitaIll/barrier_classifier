"""Bar-by-bar honest backtest driver.

``simulate()`` is now a thin BATCH DRIVER over
:class:`src.strategy.step.BoundaryStep` — the single implementation of the
per-boundary sequence shared with the live trader (see step.py for the
sequence and its causality contract). This module owns only what is
batch-specific:

- column extraction from the decision cache (numpy, once — not per-row),
- intra-path span lookup via :class:`~src.strategy.step.PathIndex`
  (O(log n) per boundary instead of a full-frame mask),
- the offline label-feedback convention: boundary ``k``'s label feeds the
  monitoring stats (drift, regime base-rate) at row ``k`` — the live
  driver feeds at label maturity instead (documented divergence; the
  production spec consumes neither),
- equity/cluster frame assembly into :class:`SimResult`.

Causality: every decision uses only data observable at or before the
boundary's close. No primitive reads ``y_k`` or any future price during
entry decision; ``y_k`` only affects the simulator after the position has
closed (it's recorded into the closed-trade ledger for analytics).

The simulator is a pure function: ``simulate(cache, raw_bars, spec) -> SimResult``.
Behavior is pinned by tests/strategy/test_simulator.py (ordering, no-peek,
hand-computed P&L) and tests/strategy/test_golden_ledgers.py (recorded
ledger corpus).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.online import DriftADWIN, RollingRegimeBaseRate
from src.strategy.policy import IntraBar, State, StrategySpec  # noqa: F401  (State re-exported)
from src.strategy.step import (
    BoundaryInputs,
    BoundaryStep,
    PathIndex,
    TradingState,
    resolve_expiries,
    resolve_intra_path_exits,
)

__all__ = [
    "SimConfig",
    "SimResult",
    "simulate",
    "get_intra_bars",
    "resolve_intra_path_exits",
    "resolve_expiries",
    "filter_specs_by_diagnostics",
]


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
        ``ts, k, realized_cum, unrealized, equity, n_open, gross_size,
        n_trades_closed_step, n_trades_closed_cum, regime_quantile, p,
        mean_p_ve, knowledge_unc, knowledge_unc_quantile, fast_sigma,
        score, opened_this_step, bulk_close_reason``.
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
# Path-bar lookup (legacy per-call API; the hot loop uses PathIndex)
# ---------------------------------------------------------------------------


def get_intra_bars(
    raw_bars: pd.DataFrame,
    *,
    ts_after: pd.Timestamp,
    ts_through: pd.Timestamp,
) -> list[IntraBar]:
    """Return the list of 1-min IntraBars with ``ts > ts_after`` and ``ts <= ts_through``,
    in chronological order. ``raw_bars`` must be ts-indexed (UTC or naive).

    One-shot convenience API. ``simulate()`` builds a
    :class:`~src.strategy.step.PathIndex` once and queries spans instead —
    same semantics, O(log n) per boundary.
    """
    return PathIndex(raw_bars).span(ts_after, ts_through)


# Backwards-compatible aliases (pre-engine private names).
_get_intra_bars = get_intra_bars
_resolve_intra_path_exits = resolve_intra_path_exits
_resolve_expiries = resolve_expiries


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _column(
    cache: pd.DataFrame, name: str, n: int, *, default: float = float("nan")
) -> np.ndarray:
    if name in cache.columns:
        return cache[name].to_numpy(dtype=float)
    return np.full(n, default)


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
    n = len(cache)

    # --- columnar extraction (once, not per-row) ---------------------------
    ts_index = pd.DatetimeIndex(cache["ts"])
    k_arr = cache["k"].to_numpy()
    p_arr = _column(cache, "p", n)
    regime_arr = _column(cache, "regime", n)
    phi_arr = _column(cache, "phi", n)
    close_arr = _column(cache, "close", n)
    high_arr = _column(cache, "high", n)
    low_arr = _column(cache, "low", n)
    y_arr = _column(cache, "y", n)
    mean_p_ve_arr = _column(cache, "mean_p_ve", n) if has_ve else np.full(n, np.nan)
    unc_arr = _column(cache, "knowledge_unc", n) if has_ve else np.full(n, np.nan)

    # --- state + step + monitoring -----------------------------------------
    st = TradingState.fresh(cfg)
    base_rate = RollingRegimeBaseRate(
        window=cfg.base_rate_window, n_bins=cfg.base_rate_n_bins
    )
    drift = DriftADWIN()  # exposed via state via diagnostics; not consumed by v1 specs
    cost_per_trade = (
        cfg.cost_per_trade_override
        if cfg.cost_per_trade_override is not None
        else spec.risk.cost_per_trade
    )
    step = BoundaryStep(spec, cost_per_trade=cost_per_trade)
    path = PathIndex(raw_bars)

    equity_rows: list[dict] = []
    empty_ve = np.empty(0)

    for i in range(n):
        k = int(k_arr[i])
        ts = ts_index[i]
        bar_close = float(close_arr[i])
        if math.isnan(bar_close):
            # Caller must augment the cache with boundary OHLC upfront via
            # ``augment_cache_with_boundary_ohlc`` — silent fallback to a
            # raw_bars lookup hid a bug once and we don't want to repeat it.
            raise ValueError(
                f"boundary row at ts={ts} (k={k}) is missing 'close' — augment "
                f"the cache with ``augment_cache_with_boundary_ohlc`` first"
            )

        intra_bars = (
            path.span(st.prev_ts, ts)
            if st.prev_ts is not None and st.portfolio.n_open() > 0
            else []
        )
        outcome = step.run(
            st,
            BoundaryInputs(
                k=k,
                ts=ts,
                p=float(p_arr[i]),
                regime_value=float(regime_arr[i]),
                phi=float(phi_arr[i]),
                bar_close=bar_close,
                bar_high=float(high_arr[i]),
                bar_low=float(low_arr[i]),
                intra_bars=intra_bars,
                mean_p_ve=float(mean_p_ve_arr[i]),
                knowledge_unc=float(unc_arr[i]),
                p_ve_samples=(
                    np.asarray(p_ve_samples[i], dtype=float)
                    if p_ve_samples is not None
                    else empty_ve
                ),
            ),
        )

        # ---- Offline label feedback (post-decision; drift + base-rate) ----
        # ``y`` is the LABEL of boundary k, which the spec did NOT see at
        # decision time. Fed AFTER the entry decision so residualized-score
        # specs never see y[k] in the base rate they consume at boundary k.
        y = float(y_arr[i])
        if not math.isnan(y):
            if not math.isnan(outcome.state.p):
                drift.update(y - outcome.state.p)
            if not math.isnan(outcome.regime_q):
                base_rate.update(outcome.regime_q, y)

        # ---- Equity row -----------------------------------------------------
        unrealized = (
            st.portfolio.mtm_log_return(bar_close) if bar_close > 0 else 0.0
        )
        equity_rows.append(
            {
                "ts": ts,
                "k": k,
                "realized_cum": st.realized_cum,
                "unrealized": unrealized,
                "equity": st.realized_cum + unrealized,
                "n_open": st.portfolio.n_open(),
                "gross_size": st.portfolio.gross_size(),
                "n_trades_closed_step": len(outcome.closed),
                "n_trades_closed_cum": len(st.portfolio.closed_positions),
                "regime_quantile": outcome.regime_q,
                "p": float(p_arr[i]),
                "mean_p_ve": float(mean_p_ve_arr[i]),
                "knowledge_unc": float(unc_arr[i]),
                "knowledge_unc_quantile": outcome.unc_q,
                "fast_sigma": outcome.fast_sigma,
                "score": outcome.score,
                "opened_this_step": outcome.opened is not None,
                "bulk_close_reason": outcome.bulk_reason,
            }
        )

    # --- finalize: flush any open cluster --------------------------------------
    st.cluster.flush_end_of_data(st.prev_ts)

    return SimResult(
        spec_name=spec.name,
        closed=st.portfolio.closed_to_frame(),
        equity=pd.DataFrame(equity_rows),
        cluster_log=pd.DataFrame(st.cluster.rows),
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
