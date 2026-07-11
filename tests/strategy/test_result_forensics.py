"""SimResult owns its trade forensics (composition, entry quality, sampling).

Behaviors ported from notebook 05's inline helpers to their natural owner;
exercised on a real golden-scenario run so shapes and invariants hold on
production-shaped ledgers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.simulator import simulate
from tests.strategy.golden_scenarios import SCENARIOS

pytestmark = pytest.mark.strategy


@pytest.fixture(scope="module")
def run():
    scenario = next(s for s in SCENARIOS if s.name == "threshold_expiry")
    cache, raw_bars, spec, cfg = scenario.build()
    return simulate(cache, raw_bars, spec, config=cfg), raw_bars


BIN_EDGES = np.array([-np.inf, -0.005, -0.0025, 0.0, 0.0025, 0.005, np.inf])


class TestComposition:
    def test_shapes_and_bucket_consistency(self, run):
        result, raw_bars = run
        comp = result.composition(raw_bars, bin_edges=BIN_EDGES)
        n_steps = len(result.equity)
        assert comp["counts"].shape == (n_steps, len(BIN_EDGES) - 1)
        # Bucket counts must sum to the open-lot count at every step.
        np.testing.assert_array_equal(comp["counts"].sum(axis=1), comp["n_open"])
        # worst_mtm defined exactly where lots are open.
        open_mask = comp["n_open"] > 0
        assert np.isfinite(comp["worst_mtm"][open_mask]).all()
        assert np.isnan(comp["worst_mtm"][~open_mask]).all()
        # Composition reflects real trading in this scenario.
        assert comp["n_open"].max() > 0

    def test_empty_ledger_is_all_zero(self, run):
        result, raw_bars = run
        empty = result.closed.iloc[0:0]
        from src.strategy.reporting import trade_composition

        comp = trade_composition(
            result.equity, empty, raw_bars, bin_edges=BIN_EDGES
        )
        assert comp["n_open"].sum() == 0
        assert np.isnan(comp["worst_mtm"]).all()


class TestEntryLocalMinRank:
    def test_ranks_bounded_and_populated(self, run):
        result, raw_bars = run
        ranks = result.entry_local_min_rank(raw_bars, window_minutes=60)
        assert len(ranks) == len(result.closed)
        finite = ranks[np.isfinite(ranks)]
        assert len(finite) > 0
        assert ((finite >= 0.0) & (finite <= 1.0)).all()

    def test_entry_at_window_minimum_ranks_zero(self):
        from src.strategy.reporting import entry_local_min_rank

        idx = pd.date_range("2025-01-01", periods=121, freq="1min")
        close = np.full(121, 101.0)
        close[60] = 100.0  # the entry bar IS the local minimum
        raw = pd.DataFrame({"close": close, "low": close}, index=idx)
        closed = pd.DataFrame(
            {"ts_entry": [idx[60]], "entry_price": [100.0]}
        )
        ranks = entry_local_min_rank(closed, raw, window_minutes=60)
        assert ranks[0] == 0.0


class TestSampleTrades:
    def test_groups_and_no_mutation(self, run):
        result, raw_bars = run
        before = result.closed.copy()
        sample = result.sample_trades(raw_bars, n_per_group=4, seed=0)
        pd.testing.assert_frame_equal(result.closed, before)  # no mutation
        assert {"hold_min", "worst_mtm", "group"} <= set(sample.columns)
        assert set(sample["group"]) <= {"fastest", "slowest", "worst-MTM", "random"}
        assert 1 <= len(sample) <= 16
        # Deduplicated by entry timestamp.
        assert sample["ts_entry"].is_unique

    def test_seeded_determinism(self, run):
        result, raw_bars = run
        a = result.sample_trades(raw_bars, seed=7)
        b = result.sample_trades(raw_bars, seed=7)
        pd.testing.assert_frame_equal(a, b)
