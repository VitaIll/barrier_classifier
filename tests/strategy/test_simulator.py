"""Simulator integration tests on hand-crafted micro-fixtures.

The fixtures are designed so that the expected closed-trade ledger is
fully predictable from the fixture itself — no model, no random walk."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.strategy.policy import (
    RiskConfig,
    State,
    StrategySpec,
    bulk_on_regime_drop,
    exit_tp_or_expiry,
    gate_regime_high,
    gate_score_above,
    make_1min_cluster_aware_spec,
    score_raw_p,
    size_clip,
    size_constant,
)
from src.strategy.simulator import (
    SimConfig,
    SimResult,
    filter_specs_by_diagnostics,
    simulate,
)

pytestmark = pytest.mark.strategy_v1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_bars(
    *,
    n_bars: int = 100,
    start_ts: str = "2025-06-01 00:00:00",
    seed: int = 0,
    sigma: float = 0.0005,
    drift: float = 0.0,
) -> pd.DataFrame:
    """Generate 1-min bars with intra-bar OHLC from a per-bar log-return."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, sigma, size=n_bars)
    log_p = np.cumsum(log_returns)
    closes = 100.0 * np.exp(log_p)
    opens = np.r_[100.0, closes[:-1]]
    highs = np.maximum(opens, closes) * np.exp(np.abs(rng.normal(0, sigma / 2, size=n_bars)))
    lows = np.minimum(opens, closes) * np.exp(-np.abs(rng.normal(0, sigma / 2, size=n_bars)))
    ts = pd.date_range(start=start_ts, periods=n_bars, freq="1min")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes}, index=ts
    )


def _make_cache_from_bars(
    raw_bars: pd.DataFrame,
    *,
    M: int = 20,
    p_func=None,
    regime_func=None,
    phi: float = 0.0025,
) -> pd.DataFrame:
    """Boundary cache: every M-th bar carries (k, ts, p, regime, phi, OHLC)."""
    boundary = raw_bars.iloc[::M].copy().reset_index(names="ts")
    n = len(boundary)
    boundary["k"] = np.arange(n)
    if p_func is None:
        boundary["p"] = 0.4
    else:
        boundary["p"] = [p_func(i) for i in range(n)]
    if regime_func is None:
        boundary["regime"] = 0.001
    else:
        boundary["regime"] = [regime_func(i) for i in range(n)]
    boundary["phi"] = phi
    # Synthetic y is not used by the simulator's entry decision but the
    # simulator does feed it forward to the drift detector and base-rate map.
    boundary["y"] = 0.0
    return boundary[
        ["k", "ts", "y", "p", "regime", "phi", "open", "high", "low", "close"]
    ]


# ---------------------------------------------------------------------------
# Smoke + structural assertions
# ---------------------------------------------------------------------------


def test_simulate_empty_cache_returns_empty_result():
    raw = _make_raw_bars(n_bars=10)
    spec = StrategySpec(name="empty", entry_gates=())
    res = simulate(pd.DataFrame(columns=["k", "ts", "p", "regime", "phi"]), raw, spec)
    assert isinstance(res, SimResult)
    assert res.equity.empty
    assert res.closed.empty
    assert res.cluster_log.empty


def test_simulate_no_entry_gate_yields_zero_trades():
    raw = _make_raw_bars(n_bars=200, sigma=0.0005)
    cache = _make_cache_from_bars(raw, M=20)
    spec = StrategySpec(name="noop", entry_gates=())
    res = simulate(cache, raw, spec)
    assert len(res.closed) == 0
    assert res.equity["equity"].iloc[-1] == pytest.approx(0.0)
    assert (res.equity["n_open"] == 0).all()


# ---------------------------------------------------------------------------
# Causal correctness: the simulator must use only past data
# ---------------------------------------------------------------------------


def test_entry_decision_does_not_peek_at_label():
    """If the entry gate decides to enter when y=1 was set in the cache,
    we should still observe entries — proving the gate didn't gate ON y.
    Conversely, if we set y=0 everywhere but the gate fires anyway, the
    simulator must allow it. Causality: y is not in State."""
    raw = _make_raw_bars(n_bars=200, sigma=0.0005)
    cache = _make_cache_from_bars(raw, M=20, p_func=lambda i: 0.5)
    cache["y"] = 0.0  # all losers
    spec = StrategySpec(
        name="always_enter_above_0_3",
        entry_gates=(lambda s: gate_score_above(s, threshold=0.3),),
        score_fn=score_raw_p,
        sizer=lambda s: 0.5,
        exit_policy=exit_tp_or_expiry,
    )
    res = simulate(cache, raw, spec)
    assert len(res.closed) >= 5  # entries fired despite y=0 throughout


# ---------------------------------------------------------------------------
# Exit semantics
# ---------------------------------------------------------------------------


def test_position_expires_after_one_boundary_when_no_tp():
    """Tight σ + zero drift => no TP hit; every position expires at the next boundary."""
    raw = _make_raw_bars(n_bars=200, sigma=0.00001, drift=0.0, seed=1)
    cache = _make_cache_from_bars(raw, M=20, p_func=lambda i: 0.5)
    spec = StrategySpec(
        name="enter_always",
        entry_gates=(lambda s: gate_score_above(s, threshold=0.3),),
        score_fn=score_raw_p,
        sizer=lambda s: 0.5,
        exit_policy=exit_tp_or_expiry,
    )
    res = simulate(cache, raw, spec)
    # Expect ~ N-1 closed expiries (last one stays open at end of cache)
    assert len(res.closed) >= 5
    # All should have exit_reason='expiry' (no TP under tiny sigma)
    assert (res.closed["exit_reason"] == "expiry").mean() > 0.95


def test_tp_fires_on_steady_drift_up():
    """Strong upward drift => TP at every step."""
    raw = _make_raw_bars(n_bars=400, sigma=0.0001, drift=0.0005, seed=2)
    cache = _make_cache_from_bars(raw, M=20, p_func=lambda i: 0.5, phi=0.0025)
    spec = StrategySpec(
        name="enter_always",
        entry_gates=(lambda s: gate_score_above(s, threshold=0.3),),
        score_fn=score_raw_p,
        sizer=lambda s: 1.0,
        exit_policy=exit_tp_or_expiry,
    )
    res = simulate(cache, raw, spec, config=SimConfig(M=20))
    assert len(res.closed) >= 5
    # Most should hit TP
    tp_rate = (res.closed["exit_reason"] == "tp").mean()
    assert tp_rate > 0.5


def test_realized_pnl_matches_handcomputed_for_single_trade():
    """Single-entry deterministic test."""
    n_bars = 60
    raw = _make_raw_bars(n_bars=n_bars, sigma=1e-8, drift=0.0, seed=3)
    # Pin the path: bar at t=10 close ↑ to a +1% jump, then flat
    raw.loc[raw.index[10:], "close"] = raw.iloc[9]["close"] * math.exp(0.01)
    raw.loc[raw.index[10:], "open"] = raw.iloc[9]["close"]
    raw.loc[raw.index[10:], "high"] = raw.iloc[9]["close"] * math.exp(0.01)
    raw.loc[raw.index[10:], "low"] = raw.iloc[9]["close"]
    cache = _make_cache_from_bars(raw, M=20, p_func=lambda i: 0.5 if i == 0 else 0.0)
    # Force entry only at boundary 0 (p>=0.3 only there)
    cache.loc[0, "p"] = 0.5
    cache.loc[1:, "p"] = 0.0

    spec = StrategySpec(
        name="single_entry",
        entry_gates=(lambda s: gate_score_above(s, threshold=0.3),),
        score_fn=score_raw_p,
        sizer=lambda s: 1.0,
        exit_policy=exit_tp_or_expiry,
        risk=RiskConfig(cost_per_trade=0.0),
    )
    res = simulate(cache, raw, spec, config=SimConfig(M=20))
    # We expect exactly one closed trade
    assert len(res.closed) == 1
    c = res.closed.iloc[0]
    # The 1% jump exceeds phi=0.0025, so exit_reason should be 'tp'
    assert c["exit_reason"] == "tp"
    # gross log return should be ~ phi
    assert c["gross_log_return"] == pytest.approx(0.0025, rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# Bulk-close
# ---------------------------------------------------------------------------


def test_bulk_close_on_regime_drop_unwinds_open_positions():
    """Open many positions during high regime, then drop the regime mid-stream
    and confirm we unwind in a single boundary."""
    n_bars = 200
    raw = _make_raw_bars(n_bars=n_bars, sigma=1e-6, drift=0.0, seed=4)
    cache = _make_cache_from_bars(raw, M=20)
    # First half: high regime (causes regime_quantile -> high). Second half: low.
    cache["regime"] = [0.001] * 5 + [0.0001] * (len(cache) - 5)

    def my_bulk(state):
        # Mimic threshold-based regime drop with a low quantile cutoff
        q = state.regime_quantile
        if not np.isfinite(q):
            return None
        return "bulk_regime" if q < 0.4 else None

    spec = StrategySpec(
        name="vol_gate_with_bulk",
        entry_gates=(
            lambda s: gate_score_above(s, threshold=0.3),
            lambda s: gate_regime_high(s, q_min=0.5),
        ),
        score_fn=score_raw_p,
        sizer=lambda s: 0.5,
        exit_policy=exit_tp_or_expiry,
        bulk_close=my_bulk,
    )
    res = simulate(
        cache,
        raw,
        spec,
        config=SimConfig(M=20, quantile_window=20, quantile_min_warmup=3),
    )
    # We expect at least one bulk_regime exit reason in the ledger
    assert "bulk_regime" in res.closed["exit_reason"].values


# ---------------------------------------------------------------------------
# Risk caps
# ---------------------------------------------------------------------------


def test_max_open_positions_caps_concurrency():
    """At any boundary, n_open should never exceed max_open_positions."""
    n_bars = 200
    raw = _make_raw_bars(n_bars=n_bars, sigma=1e-7, drift=0.0)  # no TP
    cache = _make_cache_from_bars(raw, M=20, p_func=lambda i: 0.5)
    # Allow 2 open at most; horizon is 1 boundary, so at most 1 stays open
    spec = StrategySpec(
        name="capped",
        entry_gates=(lambda s: gate_score_above(s, threshold=0.3),),
        score_fn=score_raw_p,
        sizer=lambda s: 0.5,
        exit_policy=exit_tp_or_expiry,
        risk=RiskConfig(max_open_positions=2),
    )
    res = simulate(cache, raw, spec)
    assert (res.equity["n_open"] <= 2).all()


# ---------------------------------------------------------------------------
# Spec filtering
# ---------------------------------------------------------------------------


def test_filter_specs_by_diagnostics():
    a = StrategySpec(name="a", requires=())
    b = StrategySpec(name="b", requires=("ve_diag",))
    c = StrategySpec(name="c", requires=("ve_diag", "vol_gate"))
    out = filter_specs_by_diagnostics(
        [a, b, c], diagnostics_passed={"ve_diag": True, "vol_gate": False}
    )
    names = [s.name for s in out]
    assert "a" in names
    assert "b" in names
    assert "c" not in names


def test_state_sees_post_elapsed_exit_inventory():
    """The State the spec evaluates at boundary k must reflect inventory
    AFTER elapsed-path TP / SL exits have resolved, not before. We force
    a TP-hitting upward move on the path between two boundaries: the
    position opened at boundary k-1 should be closed (TP) before
    boundary k's gates fire, so ``state.n_open_positions`` at k is 0."""
    n_bars = 60
    raw = _make_raw_bars(n_bars=n_bars, sigma=1e-8, drift=0.0, seed=42)
    # Pin a +1% pop on bar 10 (within the path from boundary 0 to boundary 1)
    pop = raw.iloc[9]["close"] * math.exp(0.01)
    raw.loc[raw.index[10], "high"] = pop
    raw.loc[raw.index[10], "close"] = pop
    raw.loc[raw.index[10], "open"] = raw.iloc[9]["close"]
    raw.loc[raw.index[10], "low"] = raw.iloc[9]["close"]
    raw.loc[raw.index[11:], "close"] = pop
    raw.loc[raw.index[11:], "open"] = pop
    raw.loc[raw.index[11:], "high"] = pop
    raw.loc[raw.index[11:], "low"] = pop
    cache = _make_cache_from_bars(raw, M=20, p_func=lambda i: 0.5, phi=0.0025)
    n_open_seen: list[int] = []

    def recording_gate(state):
        n_open_seen.append(int(state.n_open_positions))
        return gate_score_above(state, threshold=0.3)

    spec = StrategySpec(
        name="record_n_open",
        entry_gates=(recording_gate,),
        score_fn=score_raw_p,
        sizer=lambda s: 1.0,
        exit_policy=exit_tp_or_expiry,
        risk=RiskConfig(cost_per_trade=0.0),
    )
    res = simulate(cache, raw, spec, config=SimConfig(M=20))
    # Boundary 0 opens a position. Bar 10's high crosses TP, so the
    # position closes before boundary 1's State is built. Therefore the
    # gate at boundary 1 sees n_open_positions == 0.
    assert len(n_open_seen) >= 2
    assert n_open_seen[0] == 0  # nothing open at boundary 0 yet
    assert n_open_seen[1] == 0, (
        "boundary 1 saw n_open=%d; the path TP should have closed the lot "
        "before State was built" % n_open_seen[1]
    )
    # Sanity: at least one closed TP trade
    assert (res.closed["exit_reason"] == "tp").any()


def test_size_cluster_marginal_clips_to_remaining_headroom():
    """A cluster-aware sizer caps total cluster exposure at the target,
    so subsequent entries within the same cluster add only the marginal
    headroom."""
    from src.strategy.policy import size_cluster_marginal

    # First entry, flat state: returns base_size (or target if base is None)
    s_flat = State(
        k=0, ts=pd.Timestamp("2025-01-01"), p=0.5, p_calibrated=0.5,
        bar_close=100.0, bar_high=100.0, bar_low=100.0,
        regime_value=0.0, regime_quantile=0.5, fast_sigma=0.001,
        n_open_positions=0, cluster_pnl=0.0, cluster_streak=0,
        inventory_gross_size=0.0,
    )
    assert size_cluster_marginal(s_flat, target_cluster_size=2.0, base_size=0.5) == 0.5
    # No base_size override -> first lot is the full cluster cap
    assert size_cluster_marginal(s_flat, target_cluster_size=2.0) == 2.0

    # Mid-cluster: inventory below target -> marginal headroom
    s_mid = State(
        k=1, ts=pd.Timestamp("2025-01-01"), p=0.5, p_calibrated=0.5,
        bar_close=100.0, bar_high=100.0, bar_low=100.0,
        regime_value=0.0, regime_quantile=0.5, fast_sigma=0.001,
        n_open_positions=2, cluster_pnl=0.0, cluster_streak=2,
        inventory_gross_size=1.5,
    )
    assert size_cluster_marginal(s_mid, target_cluster_size=2.0) == pytest.approx(0.5)

    # Inventory at or above target -> zero
    s_full = State(
        k=2, ts=pd.Timestamp("2025-01-01"), p=0.5, p_calibrated=0.5,
        bar_close=100.0, bar_high=100.0, bar_low=100.0,
        regime_value=0.0, regime_quantile=0.5, fast_sigma=0.001,
        n_open_positions=4, cluster_pnl=0.0, cluster_streak=4,
        inventory_gross_size=2.5,
    )
    assert size_cluster_marginal(s_full, target_cluster_size=2.0) == 0.0


def test_1min_cluster_aware_spec_caps_total_exposure_at_target():
    """The 1-min cluster-aware spec must cap total open exposure at
    cluster_target_size. With cluster_target_size=1.0 and first_lot=0.4,
    subsequent overlapping high-score rows add at most 0.6 of marginal
    size before further entries are zero-sized."""
    n_bars = 200
    raw = _make_raw_bars(n_bars=n_bars, sigma=1e-8, drift=0.0, seed=5)
    cache = _make_cache_from_bars(raw, M=1, p_func=lambda i: 0.5, phi=0.0025)
    spec = make_1min_cluster_aware_spec(
        M=1,
        threshold=0.3,
        cluster_target_size=1.0,
        first_lot_size=0.4,
        cost_per_trade=0.0,
        max_horizon_boundaries=20,
    )
    res = simulate(cache, raw, spec, config=SimConfig(M=1))
    assert res.equity["gross_size"].max() <= 1.0 + 1e-12, (
        f"gross size cap violated: max={res.equity['gross_size'].max()}"
    )
    nonzero = res.equity[res.equity["gross_size"] > 0]
    if not nonzero.empty:
        first_gross = float(nonzero["gross_size"].iloc[0])
        assert first_gross == pytest.approx(0.4, abs=1e-12), (
            f"first cluster entry should be first_lot_size=0.4, got {first_gross}"
        )


def test_1min_cluster_aware_spec_accumulates_marginal_lots_correctly():
    """Strong test: the lot-size SEQUENCE must be (first_lot, marginal,
    0, 0, ...) — proving the marginal sizer actually adds incremental
    headroom, not just clipping a constant via the simulator's gross
    cap.

    With cluster_target_size=1.0 and first_lot_size=0.4: opening lots
    should be exactly [0.4, 0.6, 0, 0, ...] within a single cluster.
    A buggy sizer that returned `first_lot_size` every time and relied
    on the simulator's `max_gross_size` to clip would produce
    [0.4, 0.4, 0.2, 0, ...] — and this test catches it.
    """
    n_bars = 200
    raw = _make_raw_bars(n_bars=n_bars, sigma=1e-8, drift=0.0, seed=6)
    cache = _make_cache_from_bars(raw, M=1, p_func=lambda i: 0.5, phi=0.0025)
    spec = make_1min_cluster_aware_spec(
        M=1,
        threshold=0.3,
        cluster_target_size=1.0,
        first_lot_size=0.4,
        cost_per_trade=0.0,
        max_horizon_boundaries=200,  # never expire — keep one cluster the whole run
    )
    res = simulate(cache, raw, spec, config=SimConfig(M=1))
    # Reconstruct the sequence of OPENING lot sizes from the equity log:
    # each row where opened_this_step is True corresponds to a new lot,
    # whose size = gross_size[k] - gross_size[k-1] (one cluster -> no
    # closures in between).
    eq = res.equity.copy()
    gross = eq["gross_size"].to_numpy()
    opened = eq["opened_this_step"].to_numpy()
    lots: list[float] = []
    prev = 0.0
    for i in range(len(eq)):
        if opened[i]:
            lot = float(gross[i] - prev)
            lots.append(lot)
        prev = float(gross[i])
    # The marginal sizer's design is: first lot = 0.4, second lot = 0.6
    # (the cluster cap is reached), subsequent entries return size=0 and
    # are skipped by the simulator's `if size > 0` guard. So we expect
    # exactly two opening events with sizes (0.4, 0.6).
    assert len(lots) == 2, (
        f"expected exactly 2 opening events for cluster_target=1.0 + first_lot=0.4; "
        f"got {len(lots)} lots: {lots}"
    )
    assert lots[0] == pytest.approx(0.4, abs=1e-12), (
        f"lot[0] = first_lot_size = 0.4; got {lots[0]}"
    )
    assert lots[1] == pytest.approx(0.6, abs=1e-12), (
        f"lot[1] = cluster_target_size - first_lot_size = 0.6; got {lots[1]}. "
        "If this is 0.4, the sizer is ignoring inventory_gross_size and "
        "returning first_lot_size every time."
    )
    # After the cap is reached, gross must remain at 1.0 (no decay, no
    # extra entries) for the rest of the run — proving the sizer
    # consistently returns 0 marginal for every subsequent high-score row.
    second_idx = next(i for i in range(len(eq)) if eq["gross_size"].iloc[i] >= 1.0 - 1e-12)
    post = gross[second_idx:]
    assert (post >= 1.0 - 1e-12).all() and (post <= 1.0 + 1e-12).all(), (
        "after the marginal lot fills the cap, gross must stay at the cap "
        "for the entire remaining run"
    )


def test_filter_specs_missing_diagnostic_treated_as_failed():
    a = StrategySpec(name="a", requires=("missing_key",))
    out = filter_specs_by_diagnostics([a], diagnostics_passed={})
    assert out == []
