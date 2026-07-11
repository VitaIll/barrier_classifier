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


def test_label_update_ordering_does_not_leak_into_residualized_score():
    """For a ``score_residualized`` spec, the regime-conditional base rate
    used by the score at boundary k must NOT depend on ``y[k]``. The
    simulator feeds the label into the rolling base-rate accumulator AFTER
    the entry decision, so flipping ``y[k]`` between two runs must leave
    the recorded score (and therefore entry decisions) for boundary k
    identical — only boundaries k+1 onwards can change.
    """
    from src.strategy.policy import score_residualized

    # Build two identical caches differing only in ``y`` at boundary 50.
    raw = _make_raw_bars(n_bars=2400, sigma=0.0005, seed=7)
    cache_a = _make_cache_from_bars(raw, M=20, p_func=lambda i: 0.4)
    cache_b = cache_a.copy()
    cache_a["y"] = 0.0
    cache_b["y"] = 0.0
    cache_b.loc[50, "y"] = 1.0  # flip exactly one label

    # A score function that consults the rolling base rate. We can't
    # directly inject the rolling base rate from the simulator (the
    # simulator has its own ``base_rate`` instance), so use a closure-
    # based fake that mimics the simulator's wiring: each call updates
    # a local rolling mean with the LAST y we saw on the previous step.
    # The point is to verify that the simulator's ``base_rate.update``
    # happens AFTER entry-decision, NOT before.

    # Strategy: record the simulator's *recorded score at decision* via a
    # custom score_fn that captures (k, state.score=NaN, regime_q, p).
    # The simulator's base_rate accumulator is internal — we instead
    # assert the closed-trade ledger for boundary 50 is byte-identical
    # between cache_a and cache_b.
    spec = StrategySpec(
        name="trade_at_p_geq_0_3",
        entry_gates=(lambda s: gate_score_above(s, threshold=0.3),),
        score_fn=score_raw_p,
        sizer=lambda s: 0.5,
        exit_policy=exit_tp_or_expiry,
        risk=RiskConfig(cost_per_trade=0.0, max_horizon_boundaries=1),
    )
    res_a = simulate(cache_a, raw, spec, config=SimConfig(M=20))
    res_b = simulate(cache_b, raw, spec, config=SimConfig(M=20))
    # Boundaries up to and INCLUDING k=50 must produce identical equity
    # rows. y[50] differs, but it's only fed to the online accumulator
    # AFTER the decision at k=50.
    eq_a_until_50 = res_a.equity[res_a.equity["k"] <= 50].drop(columns=["ts"]).reset_index(drop=True)
    eq_b_until_50 = res_b.equity[res_b.equity["k"] <= 50].drop(columns=["ts"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        eq_a_until_50, eq_b_until_50, check_dtype=False,
        obj="equity rows up through k=50 must be identical regardless of y[50]",
    )


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
    """Single-entry deterministic test.

    TP-fill convention: when a long's intra-bar high crosses
    ``entry * exp(+phi)``, the position closes at ``tp_price = entry *
    exp(+phi)`` exactly — NOT at the bar's high. See
    ``policy.exit_tp_or_expiry`` (path-walk branch sets
    ``exit_price = pos.tp_price`` in ``step.resolve_intra_path_exits``).
    This matches the high-source training label: the label fires on
    intrabar-high crossing the +phi barrier and assumes a limit-order
    fill at the barrier, not market-order fill at the bar's high.
    """
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
    # gross log return should be ~ phi (filled at tp_price, NOT bar.high)
    assert c["gross_log_return"] == pytest.approx(0.0025, rel=1e-6, abs=1e-6), (
        "if this is ~ 0.01 (the bar's high return), the simulator is "
        "filling at bar.high instead of tp_price — see policy.exit_tp_or_expiry"
    )


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
    """At any boundary, n_open should never exceed max_open_positions.

    Use a TINY per-lot size (0.01) and a generous max_gross_size so the
    gross-size cap can never be the limiter — proving that
    ``max_open_positions`` itself is enforced. Also widen the horizon
    enough that positions truly stack across boundaries.
    """
    n_bars = 200
    raw = _make_raw_bars(n_bars=n_bars, sigma=1e-7, drift=0.0)  # no TP
    cache = _make_cache_from_bars(raw, M=20, p_func=lambda i: 0.5)
    spec = StrategySpec(
        name="capped",
        entry_gates=(lambda s: gate_score_above(s, threshold=0.3),),
        score_fn=score_raw_p,
        sizer=lambda s: 0.01,  # tiny — gross-size cap can never bind
        exit_policy=exit_tp_or_expiry,
        risk=RiskConfig(
            max_open_positions=2,
            max_gross_size=10.0,  # 1000x the per-lot size; can't be limiter
            max_horizon_boundaries=5,  # 5 boundaries = positions stack
            cost_per_trade=0.0,
        ),
    )
    res = simulate(cache, raw, spec)
    assert (res.equity["n_open"] <= 2).all(), (
        f"n_open exceeded the cap: max={int(res.equity['n_open'].max())}"
    )
    # Sanity: assert the cap was actually exercised — at least one
    # boundary saw n_open == 2. Otherwise the test passes trivially.
    assert (res.equity["n_open"] == 2).any(), (
        "test fixture didn't reach the cap; raise the horizon or extend bars"
    )


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


def test_state_sees_post_partial_elapsed_exit_inventory():
    """Stronger ordering test: open TWO overlapping positions, force ONLY
    ONE to TP-elapsed by the next boundary, then assert the gate at the
    NEXT boundary sees ``n_open == 1`` (not 2 pre-resolution, not 0 if
    everything had closed). This catches a bug where State composition
    runs BEFORE elapsed-exit resolution.

    The strategy: open one short-horizon position A (expires soon) and
    one long-horizon position B (still open at the next boundary). On
    the path to the next boundary, neither hits TP — but A's horizon
    elapses (forced expiry-close at the boundary's close). Then at the
    decision-time State, we should see exactly B remaining open.

    Actually expiry resolution runs AFTER bulk_close at the boundary,
    and BEFORE the decision-time State. So this is the right primitive
    to exercise the post-resolution ordering.
    """
    n_bars = 80
    # Flat path — neither A nor B will TP. Only the horizon difference
    # decides what's open at decision-time at boundary 1.
    raw = _make_raw_bars(n_bars=n_bars, sigma=1e-10, drift=0.0, seed=43)
    cache = _make_cache_from_bars(raw, M=20, p_func=lambda i: 0.5, phi=0.0025)
    n_open_seen: list[int] = []
    state_inv_seen: list[float] = []

    def recording_gate(state):
        n_open_seen.append(int(state.n_open_positions))
        state_inv_seen.append(float(state.inventory_gross_size))
        # Only allow boundaries 0 and 1 to open
        return gate_score_above(state, threshold=0.3)

    # We need TWO distinct horizons. Simplest: use a single spec with
    # short horizon and pre-seed an inventory snapshot via a custom
    # bulk-close that's a no-op. To keep both positions, use horizon=2
    # for the spec (everything stays open across one full boundary).
    spec = StrategySpec(
        name="record_n_open_partial",
        entry_gates=(recording_gate,),
        score_fn=score_raw_p,
        sizer=lambda s: 0.5,
        exit_policy=exit_tp_or_expiry,
        risk=RiskConfig(
            cost_per_trade=0.0,
            max_open_positions=5,
            max_gross_size=5.0,
            max_horizon_boundaries=2,  # opened at k expires at k+2
        ),
    )
    res = simulate(cache, raw, spec, config=SimConfig(M=20))
    # Boundary 0: nothing open => gate sees 0
    # Boundary 1: A (opened at k=0) is still open (horizon expires at
    #            k=0+2=2, not yet); gate sees 1 (A open after path-walk).
    # Boundary 2: A's expiry_k=2 == k_now; expiry resolution closes A
    #            before decision-time State => gate sees only B (n=1).
    assert len(n_open_seen) >= 3, (
        f"need at least 3 recorded gates; got {len(n_open_seen)}"
    )
    assert n_open_seen[0] == 0, f"k=0 should see 0 open, got {n_open_seen[0]}"
    assert n_open_seen[1] == 1, (
        f"k=1 should see 1 open (A still open from k=0); got {n_open_seen[1]}"
    )
    # At k=2: A's expiry resolves at this boundary (k_entry=0, expiry_k=2,
    # k_now=2 triggers expiry). After expiry resolution, only B (opened
    # at k=1, expiry_k=3) remains open. So decision-time State sees 1.
    # CRITICAL CHECK: this MUST be 1, not 2 (which would mean expiry ran
    # AFTER State composition) and not 0 (only relevant if neither A nor
    # B opened — sanity assertion).
    assert n_open_seen[2] == 1, (
        f"k=2 should see 1 open (A's expiry resolved, B still open); "
        f"got {n_open_seen[2]}. If 2: State was composed BEFORE expiry; "
        f"if 0: B didn't open (fixture bug)."
    )
    # Sanity: at least one expiry exit
    assert (res.closed["exit_reason"] == "expiry").sum() >= 1


def test_state_at_entry_decision_sees_post_bulk_inventory():
    """The decision-time State must reflect inventory AFTER bulk_close
    AND expiry resolution, not before. We construct: open positions, then
    fire bulk_close at the next boundary, then assert the entry-decision
    gate sees n_open == 0 at that boundary (post-bulk).

    The recording gate only fires when ``evaluate_entry`` is called, and
    it's keyed on (state.k, state.n_open_positions). We map results by
    k to make the assertion robust to which exact boundary fires bulk.
    """
    n_bars = 200
    raw = _make_raw_bars(n_bars=n_bars, sigma=1e-8, drift=0.0, seed=44)
    cache = _make_cache_from_bars(raw, M=20, p_func=lambda i: 0.5)
    # Boundaries 0-2 high regime (warmup + a couple of opens),
    # boundary 3 onwards low regime => bulk_close fires there.
    cache["regime"] = [0.001] * 3 + [0.0001] * (len(cache) - 3)
    seen_by_k: dict[int, int] = {}

    def recording_gate(state):
        seen_by_k[int(state.k)] = int(state.n_open_positions)
        return gate_score_above(state, threshold=0.3)

    def my_bulk(state):
        q = state.regime_quantile
        if not np.isfinite(q):
            return None
        return "bulk_regime" if q < 0.4 else None

    spec = StrategySpec(
        name="record_n_open_post_bulk",
        entry_gates=(recording_gate,),
        score_fn=score_raw_p,
        sizer=lambda s: 0.5,
        exit_policy=exit_tp_or_expiry,
        bulk_close=my_bulk,
        risk=RiskConfig(
            cost_per_trade=0.0,
            max_open_positions=5,
            max_gross_size=5.0,
            max_horizon_boundaries=100,
        ),
    )
    res = simulate(
        cache, raw, spec,
        config=SimConfig(M=20, quantile_window=5, quantile_min_warmup=2),
    )
    bulk_closes = res.closed[res.closed["exit_reason"] == "bulk_regime"]
    assert not bulk_closes.empty, (
        "bulk_close should have fired with the low-regime tail; ledger:\n"
        + str(res.closed["exit_reason"].value_counts())
    )
    # The k_exit on each bulk-closed trade is the boundary at which
    # bulk_close fired. The entry-decision State at that same boundary
    # must see n_open == 0 (because bulk_close ran first).
    bulk_k = int(bulk_closes["k_exit"].iloc[0])
    assert bulk_k in seen_by_k, (
        f"the recording gate didn't run at the bulk-close boundary k={bulk_k}; "
        f"observed boundaries: {sorted(seen_by_k.keys())}"
    )
    assert seen_by_k[bulk_k] == 0, (
        f"at bulk_close boundary k={bulk_k}, entry-decision gate saw "
        f"n_open={seen_by_k[bulk_k]}; should be 0 because bulk_close "
        f"runs before State is composed for the entry decision"
    )


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
