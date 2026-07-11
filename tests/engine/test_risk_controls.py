"""Pre-trade risk controls: the institutional entry gate.

Unit-level on the governor (event-time budget/collar/loss/notional +
UTC-day reset) and integration on the engine (controls veto entries;
disabled() preserves research parity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.errors import ConfigError
from src.engine.risk import EntryControls, EntryGovernor

pytestmark = pytest.mark.engine

TS = pd.Timestamp("2025-06-01 00:00:00", tz="UTC")


class TestEntryControlsValidation:
    def test_bad_values_rejected(self):
        with pytest.raises(ConfigError):
            EntryControls(max_daily_entries=0)
        with pytest.raises(ConfigError):
            EntryControls(max_bar_move=-0.01)
        with pytest.raises(ConfigError):
            EntryControls(max_daily_loss=float("nan"))

    def test_disabled_is_all_none(self):
        d = EntryControls.disabled()
        assert (d.max_daily_entries, d.max_entry_capital_frac,
                d.max_bar_move, d.max_daily_loss) == (None, None, None, None)


class TestEntryGovernor:
    def _gov(self, **kw) -> EntryGovernor:
        return EntryGovernor(EntryControls(**kw))

    def _ok(self, gov, **over):
        base = dict(ts=TS, prev_close=100.0, bar_close=100.0,
                    total_equity=0.0, entry_size=0.02)
        base.update(over)
        return gov.check_entry(**base)

    def test_permits_normal_entry(self):
        assert self._ok(self._gov()) is None

    def test_daily_entry_budget(self):
        gov = self._gov(max_daily_entries=3, max_bar_move=None,
                        max_daily_loss=None, max_entry_capital_frac=None)
        for _ in range(3):
            assert self._ok(gov) is None
            gov.record_entry(TS)
        assert "budget exhausted" in self._ok(gov)
        assert gov.vetoes["daily_entries"] == 1

    def test_budget_resets_next_utc_day(self):
        gov = self._gov(max_daily_entries=1, max_bar_move=None,
                        max_daily_loss=None, max_entry_capital_frac=None)
        gov.record_entry(TS)
        assert self._ok(gov) is not None  # exhausted today
        tomorrow = TS + pd.Timedelta(days=1)
        assert self._ok(gov, ts=tomorrow) is None  # fresh budget

    def test_dislocation_collar(self):
        gov = self._gov(max_bar_move=0.02, max_daily_entries=None,
                        max_daily_loss=None, max_entry_capital_frac=None)
        # 3% move > 2% collar.
        veto = self._ok(gov, prev_close=100.0, bar_close=103.0)
        assert "dislocation" in veto
        assert self._ok(gov, prev_close=100.0, bar_close=101.0) is None

    def test_daily_loss_stop(self):
        gov = self._gov(max_daily_loss=0.05, max_daily_entries=None,
                        max_bar_move=None, max_entry_capital_frac=None)
        # Day opens at equity 0 (first check sets the open).
        assert self._ok(gov, total_equity=0.0) is None
        # Down 0.06 from open -> stop.
        veto = self._ok(gov, total_equity=-0.06)
        assert "daily loss stop" in veto

    def test_notional_cap(self):
        gov = self._gov(max_entry_capital_frac=0.05, max_daily_entries=None,
                        max_bar_move=None, max_daily_loss=None)
        assert self._ok(gov, entry_size=0.05) is None
        assert "per-order cap" in self._ok(gov, entry_size=0.10)

    def test_disabled_never_vetoes(self):
        gov = EntryGovernor(EntryControls.disabled())
        assert self._ok(gov, bar_close=1_000_000.0, entry_size=99.0,
                        total_equity=-100.0) is None


class TestEngineWiring:
    def test_defaults_on(self):
        # Production posture: controls and periodic reconcile are ON by
        # default; opting out is the deliberate act (EntryControls.disabled()
        # / --no-risk-controls). The e2e suite runs disabled() and proves
        # that path reproduces the offline simulator exactly.
        from src.engine.engine import EngineConfig

        cfg = EngineConfig()
        assert cfg.entry_controls.max_daily_entries is not None
        assert cfg.entry_controls.max_bar_move is not None
        assert cfg.entry_controls.max_daily_loss is not None
        assert cfg.reconcile_every_bars is not None

    def test_governor_summary_shape(self):
        gov = EntryGovernor(EntryControls())
        gov.record_entry(TS)
        s = gov.summary()
        assert s["entries_today"] == 1
        assert set(s["vetoes"]) == {"daily_entries", "bar_move", "daily_loss"}
