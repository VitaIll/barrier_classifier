"""Ledger parity: LiveTrader (incremental) ≡ simulate() (batch).

The live engine's whole claim to correctness is that trading the stream
bar-by-bar produces the same trades as the researched backtest. These
tests drive both implementations with identical synthetic inputs and
require the closed-trade ledgers and equity curves to match exactly.

Covered specs:
- the production P1+P3 spec (threshold let-winners-run, no SL/expiry),
- its strict monotonic variant,
- the cluster-aware spec (bulk-close + cluster-marginal sizing) — a much
  stricter exercise of State composition (cluster_pnl, gross size,
  drawdown gate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.domain import Bar
from src.engine.strategy import LiveProbFeed, LiveTrader, make_live_production_spec
from src.strategy.policy import make_1min_cluster_aware_spec
from src.strategy.simulator import SimConfig, simulate

pytestmark = pytest.mark.engine

START = pd.Timestamp("2025-05-01 00:01:00")  # tz-naive: simulator convention
N = 900
M = 20
PHI = 0.0025


def synthetic_market(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(cache, raw_bars) shaped exactly like the research inputs."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=N, freq="1min")
    # Volatile random walk with drift bursts so TP zones actually get hit.
    ret = rng.normal(0, 8e-4, N) + np.where(rng.random(N) < 0.03, 25e-4, 0.0)
    close = 100.0 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 6e-4, N)) + 3e-4
    high = close * np.exp(spread)
    low = close * np.exp(-spread)
    open_ = np.concatenate([[close[0]], close[:-1]])
    raw = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )
    # Model probabilities: autocorrelated with occasional conviction spikes.
    base = 0.30 + 0.25 * np.abs(np.sin(np.arange(N) / 37.0))
    noise = rng.normal(0, 0.05, N)
    p = np.clip(base + noise + np.where(rng.random(N) < 0.05, 0.35, 0.0), 0.01, 0.99)
    regime = pd.Series(np.abs(ret)).rolling(30, min_periods=1).mean().to_numpy()
    cache = pd.DataFrame({
        "ts": idx,
        "k": np.arange(N, dtype=np.int64),
        "p": p,
        "regime": regime,
        "phi": PHI,
        "y": rng.integers(0, 2, N).astype(float),
        "open": open_, "high": high, "low": low, "close": close,
    })
    return cache, raw


def bars_from_raw(raw: pd.DataFrame) -> list[Bar]:
    out = []
    for ts, row in raw.iterrows():
        out.append(Bar(
            ts=ts, open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]), volume=1.0,
            quote_volume=1.0, num_trades=1.0, taker_buy_base=0.5,
            taker_buy_quote=0.5,
        ))
    return out


def run_live(cache: pd.DataFrame, raw: pd.DataFrame, make_spec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drive LiveTrader bar-by-bar with the same inputs simulate() gets."""
    feed = LiveProbFeed()
    spec = make_spec(feed)
    trader = LiveTrader(spec, sim_config=SimConfig(M=M, cadence_minutes=1.0))
    equity_rows = []
    bars = bars_from_raw(raw)
    for i, bar in enumerate(bars):
        p = float(cache["p"].iloc[i])
        feed.update(bar.ts, p)
        result = trader.on_boundary(
            k=int(cache["k"].iloc[i]), ts=bar.ts, p=p, bar=bar,
            regime_value=float(cache["regime"].iloc[i]), phi=PHI,
        )
        equity_rows.append({
            "ts": result.ts, "realized_cum": result.equity.realized_cum,
            "unrealized": result.equity.unrealized, "n_open": result.equity.n_open,
            "gross_size": result.equity.gross_size,
        })
    closed = trader.portfolio.closed_to_frame()
    return closed, pd.DataFrame(equity_rows)


LEDGER_COLS = [
    "k_entry", "ts_entry", "size", "entry_price", "tp_price",
    "k_exit", "ts_exit", "exit_price", "exit_reason", "gross_log_return",
]


def assert_ledgers_equal(offline: pd.DataFrame, live: pd.DataFrame) -> None:
    assert len(offline) == len(live), (
        f"trade count differs: offline={len(offline)}, live={len(live)}"
    )
    if offline.empty:
        return
    o = offline[LEDGER_COLS].reset_index(drop=True)
    l = live[LEDGER_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(o, l, check_exact=True)


@pytest.mark.parametrize("exit_variant", ["threshold", "monotonic"])
def test_production_spec_parity(exit_variant):
    cache, raw = synthetic_market()
    p_map = pd.Series(cache["p"].to_numpy(), index=pd.DatetimeIndex(cache["ts"]))
    p_th = float(np.quantile(cache["p"], 0.90))

    def make_offline_spec():
        feed_like = p_map
        return make_live_production_spec(
            feed_like, p_threshold=p_th, lot_size=0.02, max_concurrent=50,
            cost_per_trade=0.0005, exit_variant=exit_variant,
        )

    offline = simulate(cache, raw, make_offline_spec(),
                       config=SimConfig(M=M, cadence_minutes=1.0))

    live_closed, live_equity = run_live(
        cache, raw,
        lambda feed: make_live_production_spec(
            feed, p_threshold=p_th, lot_size=0.02, max_concurrent=50,
            cost_per_trade=0.0005, exit_variant=exit_variant,
        ),
    )
    assert len(offline.closed) > 3, "degenerate scenario: almost no trades"
    assert_ledgers_equal(offline.closed, live_closed)

    eq_off = offline.equity[["realized_cum", "unrealized", "n_open", "gross_size"]]
    eq_live = live_equity[["realized_cum", "unrealized", "n_open", "gross_size"]]
    pd.testing.assert_frame_equal(
        eq_off.reset_index(drop=True), eq_live.reset_index(drop=True),
        check_exact=False, rtol=0, atol=1e-12,
    )


def test_cluster_aware_spec_parity():
    cache, raw = synthetic_market(seed=11)
    spec = make_1min_cluster_aware_spec(
        M=M, threshold=float(np.quantile(cache["p"], 0.85)),
        cluster_target_size=0.5, first_lot_size=0.1, cost_per_trade=0.0005,
        cluster_loss_cap=0.02,
    )
    offline = simulate(cache, raw, spec, config=SimConfig(M=M, cadence_minutes=1.0))

    # The cluster spec has no p_map dependency — same spec object drives both
    # (it is stateless: plain functions over State).
    feed = LiveProbFeed()
    trader = LiveTrader(spec, sim_config=SimConfig(M=M, cadence_minutes=1.0))
    for i, bar in enumerate(bars_from_raw(raw)):
        p = float(cache["p"].iloc[i])
        feed.update(bar.ts, p)
        trader.on_boundary(
            k=int(cache["k"].iloc[i]), ts=bar.ts, p=p, bar=bar,
            regime_value=float(cache["regime"].iloc[i]), phi=PHI,
        )
    live_closed = trader.portfolio.closed_to_frame()
    assert len(offline.closed) > 3, "degenerate scenario: almost no trades"
    assert_ledgers_equal(offline.closed, live_closed)
    assert trader.realized_cum == pytest.approx(
        float(offline.equity["realized_cum"].iloc[-1]), abs=1e-12
    )


def test_degraded_bar_fails_safe():
    """NaN conviction on a bar (feature failure) must not freeze exits:
    a TP-zone touch closes at market (the documented safe fallback)."""
    cache, raw = synthetic_market(seed=3)
    feed = LiveProbFeed()
    p_th = 0.5
    spec = make_live_production_spec(feed, p_threshold=p_th, lot_size=0.02)
    trader = LiveTrader(spec, sim_config=SimConfig(M=M, cadence_minutes=1.0))
    bars = bars_from_raw(raw)

    # Bar 0: force an entry.
    feed.update(bars[0].ts, 0.9)
    r0 = trader.on_boundary(k=0, ts=bars[0].ts, p=0.9, bar=bars[0],
                            regime_value=0.001, phi=PHI)
    assert r0.entered

    # Find/construct a TP-touching bar with NO probability in the feed.
    pos = trader.portfolio.open_positions[0]
    tp = pos.tp_price
    touch = Bar(
        ts=bars[1].ts, open=tp * 0.999, high=tp * 1.001, low=tp * 0.998,
        close=tp * 1.0005, volume=1.0, quote_volume=1.0, num_trades=1.0,
        taker_buy_base=0.5, taker_buy_quote=0.5,
    )
    r1 = trader.on_boundary(k=1, ts=touch.ts, p=float("nan"), bar=touch,
                            regime_value=0.001, phi=PHI, allow_entry=False)
    assert len(r1.closed) == 1
    assert r1.closed[0].exit_reason == "tp_market"
    assert not r1.entered
