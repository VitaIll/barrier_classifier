"""Golden-ledger scenario definitions (shared by generator + test).

Each scenario builds a fresh ``(cache, raw_bars, spec, SimConfig)`` tuple —
fresh, because exit policies carry closure state that must never leak
between runs. The matrix is chosen to exercise every exit path the
simulator supports:

- ``threshold_expiry``       — plain TP-or-time-expiry, several lots
- ``tp_sl_floor``            — per-position stop-loss via the MTM floor
- ``monotonic_bulk``         — let-winners-run (monotonic) + bulk-close on
                               cluster loss: closure state, ``tp_market``,
                               bulk paths
- ``production_1min``        — the P1+P3 shape at 1-min cadence: top-q
                               entry, let-winners-run threshold exit,
                               50-lot cluster stacking

The goldens recorded from these scenarios pin the simulator's ledger,
equity curve, and cluster log across the BoundaryStep re-architecture:
any behavioral drift fails the comparison test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.strategy.policy import (
    RiskConfig,
    StrategySpec,
    bulk_on_cluster_loss,
    exit_tp_or_expiry,
    exit_tp_sl_or_expiry,
    gate_score_above,
    make_exit_let_winners_run,
    make_exit_let_winners_run_monotonic,
    score_raw_p,
    size_clip,
    size_constant,
)
from src.strategy.simulator import SimConfig

PHI = 0.0025


@dataclass(frozen=True)
class Scenario:
    name: str
    build: Callable[[], tuple[pd.DataFrame, pd.DataFrame, StrategySpec, SimConfig]]


def _market(
    n_boundaries: int,
    m: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic 1-min bars + boundary cache with autocorrelated p."""
    rng = np.random.default_rng(seed)
    n_raw = n_boundaries * m
    r = rng.normal(0.0, 0.0006, n_raw)
    close = 30_000.0 * np.exp(np.cumsum(r))
    high = close * np.exp(np.abs(rng.normal(0.0, 0.0004, n_raw)))
    low = close * np.exp(-np.abs(rng.normal(0.0, 0.0004, n_raw)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    raw_index = pd.date_range("2025-03-01", periods=n_raw, freq="1min")
    raw_bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=raw_index,
    )

    # Autocorrelated conviction in (0, 1): AR(1) latent + logistic squash.
    z = np.empty(n_boundaries)
    z[0] = 0.0
    eps = rng.normal(0.0, 0.35, n_boundaries)
    for i in range(1, n_boundaries):
        z[i] = 0.92 * z[i - 1] + eps[i]
    p = 1.0 / (1.0 + np.exp(-(z - 1.0)))

    b_idx = np.arange(n_boundaries) * m
    ts = raw_index[b_idx]
    ret = pd.Series(close).pct_change().to_numpy()
    regime = (
        pd.Series(np.abs(ret)).rolling(60, min_periods=1).std().to_numpy()[b_idx]
    )
    cache = pd.DataFrame(
        {
            "ts": ts,
            "k": np.arange(n_boundaries),
            "p": p,
            "regime": np.nan_to_num(regime, nan=0.0),
            "phi": PHI,
            "close": close[b_idx],
            "high": high[b_idx],
            "low": low[b_idx],
            "y": (rng.uniform(0, 1, n_boundaries) < 0.08).astype(float),
            "m_k": np.zeros(n_boundaries),
        }
    )
    return cache, raw_bars


def _p_map(cache: pd.DataFrame) -> pd.Series:
    return pd.Series(cache["p"].to_numpy(), index=pd.DatetimeIndex(cache["ts"]))


def _threshold_expiry() -> tuple[pd.DataFrame, pd.DataFrame, StrategySpec, SimConfig]:
    cache, raw = _market(800, 20, seed=101)
    spec = StrategySpec(
        name="golden_threshold_expiry",
        entry_gates=(lambda s: gate_score_above(s, 0.55),),
        score_fn=score_raw_p,
        sizer=lambda s: size_clip(size_constant(s, default=0.1), max_size=1.0),
        exit_policy=exit_tp_or_expiry,
        risk=RiskConfig(
            max_open_positions=5,
            max_gross_size=3.0,
            max_horizon_boundaries=3,
            cost_per_trade=0.0005,
        ),
    )
    return cache, raw, spec, SimConfig(M=20, cadence_minutes=20.0)


def _tp_sl_floor() -> tuple[pd.DataFrame, pd.DataFrame, StrategySpec, SimConfig]:
    cache, raw = _market(800, 20, seed=202)
    spec = StrategySpec(
        name="golden_tp_sl_floor",
        entry_gates=(lambda s: gate_score_above(s, 0.5),),
        score_fn=score_raw_p,
        sizer=lambda s: size_clip(size_constant(s, default=0.15), max_size=1.0),
        exit_policy=exit_tp_sl_or_expiry,
        risk=RiskConfig(
            max_open_positions=4,
            max_gross_size=2.0,
            max_horizon_boundaries=5,
            cost_per_trade=0.0005,
            position_mtm_floor_log_return=-0.004,
        ),
    )
    return cache, raw, spec, SimConfig(M=20, cadence_minutes=20.0)


def _monotonic_bulk() -> tuple[pd.DataFrame, pd.DataFrame, StrategySpec, SimConfig]:
    cache, raw = _market(800, 20, seed=303)
    spec = StrategySpec(
        name="golden_monotonic_bulk",
        entry_gates=(lambda s: gate_score_above(s, 0.55),),
        score_fn=score_raw_p,
        sizer=lambda s: size_clip(size_constant(s, default=0.1), max_size=1.0),
        exit_policy=make_exit_let_winners_run_monotonic(_p_map(cache)),
        bulk_close=lambda s: bulk_on_cluster_loss(s, cap_log_return=-0.006),
        risk=RiskConfig(
            max_open_positions=8,
            max_gross_size=2.0,
            max_horizon_boundaries=1_000_000,
            cost_per_trade=0.0005,
        ),
    )
    return cache, raw, spec, SimConfig(M=20, cadence_minutes=20.0)


def _production_1min() -> tuple[pd.DataFrame, pd.DataFrame, StrategySpec, SimConfig]:
    cache, raw = _market(6_000, 1, seed=404)
    p_map = _p_map(cache)
    p_threshold = float(np.quantile(cache["p"].to_numpy(), 0.99))
    lot = 0.02
    spec = StrategySpec(
        name="golden_production_1min",
        entry_gates=(lambda s, t=p_threshold: gate_score_above(s, t),),
        score_fn=score_raw_p,
        sizer=lambda s, sz=lot: size_clip(size_constant(s, default=sz), max_size=1.0),
        exit_policy=make_exit_let_winners_run(
            p_map, hold_threshold=p_threshold, sl_log_return=None
        ),
        risk=RiskConfig(
            max_open_positions=50,
            max_gross_size=50 * lot + 1e-6,
            max_horizon_boundaries=1_000_000,
            cost_per_trade=0.0005,
        ),
    )
    return cache, raw, spec, SimConfig(M=1, cadence_minutes=1.0)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("threshold_expiry", _threshold_expiry),
    Scenario("tp_sl_floor", _tp_sl_floor),
    Scenario("monotonic_bulk", _monotonic_bulk),
    Scenario("production_1min", _production_1min),
)
