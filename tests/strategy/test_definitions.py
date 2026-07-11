"""Declarative strategy definitions + Phase-2 decision-domain guards.

Covers:
- StrategyDefinition: registry validation, JSON round-trip, sweep identity
- build() ≡ hand-wired spec: identical ledgers on the production scenario
- ExitReason enum: typo rejection at the close_position choke point
- N10: monotonic exit-state released on ANY close path (the closure leak)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.errors import ConfigError
from src.strategy.definitions import (
    ComponentRef,
    StrategyDefinition,
    bulk,
    exit_rule,
    gate,
    production_definition,
    sizer,
)
from src.strategy.inventory import ExitReason, Portfolio, Position, close_position
from src.strategy.policy import (
    IntraBar,
    RiskConfig,
    StrategySpec,
    make_exit_let_winners_run_monotonic,
    score_raw_p,
)
from src.strategy.simulator import SimConfig, simulate
from src.strategy.step import BoundaryInputs, BoundaryStep, TradingState
from tests.strategy.golden_scenarios import _market, _p_map

pytestmark = pytest.mark.strategy


# ---------------------------------------------------------------------------
# StrategyDefinition
# ---------------------------------------------------------------------------


def _definition(**over) -> StrategyDefinition:
    base = dict(
        name="unit",
        gates=(gate("score_above", threshold=0.6),),
        sizer=sizer("constant_clipped", size=0.1, max_size=1.0),
        exit=exit_rule("tp_or_expiry"),
        bulk=bulk("on_cluster_loss", cap_log_return=-0.01),
        risk=RiskConfig(max_open_positions=3),
    )
    base.update(over)
    return StrategyDefinition(**base)


class TestStrategyDefinition:
    def test_unknown_kinds_fail_eagerly_with_known_list(self):
        with pytest.raises(ConfigError, match="known:"):
            gate("score_over", threshold=0.5)
        with pytest.raises(ConfigError, match="sizer"):
            _definition(sizer=ComponentRef("martingale", {}))
        with pytest.raises(ConfigError, match="score"):
            _definition(score="magic")

    def test_json_round_trip_is_identity(self):
        d = _definition()
        d2 = StrategyDefinition.from_json(d.to_json())
        assert d2 == d
        assert d2.key() == d.key()

    def test_key_distinguishes_parameter_changes(self):
        a = _definition()
        b = _definition(gates=(gate("score_above", threshold=0.7),))
        assert a.key() != b.key()

    def test_build_produces_working_spec(self):
        spec = _definition().build()
        assert isinstance(spec, StrategySpec)
        assert spec.name == "unit"
        assert len(spec.entry_gates) == 1

    def test_feed_requiring_exit_without_feed_is_config_error(self):
        d = _definition(exit=exit_rule("let_winners_run", hold_threshold=0.6))
        with pytest.raises(ConfigError, match="probability feed"):
            d.build()

    def test_production_definition_matches_hand_wired_spec_ledger(self):
        # The declarative P1+P3 must trade identically to the closure-wired
        # construction used by the notebooks/scripts/live engine.
        cache, raw = _market(4_000, 1, seed=777)
        p_map = _p_map(cache)
        p_th = float(np.quantile(cache["p"].to_numpy(), 0.99))

        d = production_definition(p_threshold=p_th, lot_size=0.02,
                                  max_concurrent=50, cost_per_trade=0.0005)
        spec_decl = d.build(prob_feed=p_map)

        from src.strategy.policy import (
            gate_score_above,
            make_exit_let_winners_run,
            size_clip,
            size_constant,
        )
        spec_hand = StrategySpec(
            name="hand",
            entry_gates=(lambda s, t=p_th: gate_score_above(s, t),),
            score_fn=score_raw_p,
            sizer=lambda s: size_clip(size_constant(s, default=0.02), max_size=1.0),
            exit_policy=make_exit_let_winners_run(
                p_map, hold_threshold=p_th, sl_log_return=None
            ),
            risk=d.risk,
        )
        cfg = SimConfig(M=1, cadence_minutes=1.0)
        r_decl = simulate(cache, raw, spec_decl, config=cfg)
        r_hand = simulate(cache, raw, spec_hand, config=cfg)
        assert len(r_decl.closed) == len(r_hand.closed)
        if len(r_decl.closed):
            pd.testing.assert_frame_equal(
                r_decl.closed.reset_index(drop=True),
                r_hand.closed.reset_index(drop=True),
                check_exact=True,
            )


# ---------------------------------------------------------------------------
# ExitReason enum
# ---------------------------------------------------------------------------


class TestExitReason:
    def _pos(self) -> Position:
        return Position(
            k_entry=1, ts_entry=pd.Timestamp("2025-01-01"), side=1, size=0.1,
            entry_price=100.0, tp_price=100.3, sl_price=None, expiry_k=5,
        )

    def test_members_compare_equal_to_plain_strings(self):
        assert ExitReason.TP == "tp"
        assert str(ExitReason.TP_MARKET) == "tp_market"

    def test_typo_rejected_at_close(self):
        with pytest.raises(ValueError, match="pt"):
            close_position(
                self._pos(), k_exit=2, ts_exit=pd.Timestamp("2025-01-02"),
                exit_price=100.3, exit_reason="pt",  # typo of "tp"
            )

    def test_known_reasons_normalize(self):
        c = close_position(
            self._pos(), k_exit=2, ts_exit=pd.Timestamp("2025-01-02"),
            exit_price=100.3, exit_reason="bulk_cluster_loss",
        )
        assert c.exit_reason is ExitReason.BULK_CLUSTER_LOSS
        assert c.exit_reason == "bulk_cluster_loss"


# ---------------------------------------------------------------------------
# N10 — monotonic exit state must release on every close path
# ---------------------------------------------------------------------------


def _tp_zone_bar(ts: pd.Timestamp, pos: Position) -> IntraBar:
    return IntraBar(
        n=-1, ts=ts, open=pos.entry_price,
        high=pos.tp_price * 1.001, low=pos.entry_price * 0.999,
        close=pos.entry_price * 1.002,  # profit side of entry
    )


class TestMonotonicExitStateRelease:
    def _setup(self):
        ts0 = pd.Timestamp("2025-01-01 00:00")
        ts1 = pd.Timestamp("2025-01-01 00:01")
        p_map = pd.Series([0.9, 0.85], index=pd.DatetimeIndex([ts0, ts1]))
        exit_fn = make_exit_let_winners_run_monotonic(p_map)
        pos = Position(
            k_entry=7, ts_entry=ts0, side=1, size=0.1,
            entry_price=100.0, tp_price=100.25, sl_price=None,
            expiry_k=10_000,
        )
        return exit_fn, pos, ts0, ts1

    def test_leak_scenario_without_cleanup_takes_market_exit(self):
        exit_fn, pos, ts0, ts1 = self._setup()
        assert exit_fn(pos, _tp_zone_bar(ts0, pos), 1) is None  # p_max := 0.9
        # Conviction dropped from its peak -> closes at market.
        assert exit_fn(pos, _tp_zone_bar(ts1, pos), 2) == "tp_market"

    def test_on_position_closed_releases_state(self):
        exit_fn, pos, ts0, ts1 = self._setup()
        assert exit_fn(pos, _tp_zone_bar(ts0, pos), 1) is None  # p_max := 0.9
        closed = close_position(
            pos, k_exit=8, ts_exit=ts0, exit_price=100.0,
            exit_reason="bulk_regime",  # a path that bypasses the exit policy
        )
        exit_fn.on_position_closed(closed)
        # State released: 0.85 > fresh -inf -> "still growing" -> hold.
        assert exit_fn(pos, _tp_zone_bar(ts1, pos), 2) is None

    def test_boundary_step_invokes_hook_on_bulk_close(self):
        calls: list[int] = []

        def exit_never(pos, bar, k):
            return None

        exit_never.on_position_closed = lambda c: calls.append(c.k_entry)

        spec = StrategySpec(
            name="bulk_hook",
            entry_gates=(),
            score_fn=score_raw_p,
            exit_policy=exit_never,
            bulk_close=lambda s: "bulk_regime",
            risk=RiskConfig(),
        )
        st = TradingState.fresh(SimConfig())
        pos = Position(
            k_entry=3, ts_entry=pd.Timestamp("2025-01-01"), side=1, size=0.1,
            entry_price=100.0, tp_price=100.3, sl_price=None, expiry_k=99,
        )
        st.portfolio.open_one(pos)
        step = BoundaryStep(spec, cost_per_trade=0.0)
        outcome = step.run(
            st,
            BoundaryInputs(
                k=4, ts=pd.Timestamp("2025-01-01 00:20"), p=0.5,
                regime_value=1.0, phi=0.0025,
                bar_close=100.0, bar_high=100.1, bar_low=99.9,
            ),
        )
        assert [c.exit_reason for c in outcome.closed] == ["bulk_regime"]
        assert calls == [3]
