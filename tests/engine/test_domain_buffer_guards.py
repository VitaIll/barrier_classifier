"""Domain model, bar buffer, and boundary guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.buffer import BarBuffer, minutes_since
from src.engine.domain import Bar, DerivSnapshot, MarketUpdate
from src.engine.errors import BarSchemaError, GridError, PhaseAlignmentError
from src.engine.guards import BarSchemaGuard, GridGuard, WarmupGuard

pytestmark = pytest.mark.engine

ANCHOR = pd.Timestamp("2025-01-01 00:01:00", tz="UTC")


def mk_bar(ts: pd.Timestamp, close: float = 100.0, **kw) -> Bar:
    defaults = dict(
        open=close, high=close, low=close, close=close, volume=1.0,
        quote_volume=close, num_trades=5.0, taker_buy_base=0.5,
        taker_buy_quote=50.0,
    )
    defaults.update(kw)
    return Bar(ts=ts, **defaults)


def ts_at(i: int) -> pd.Timestamp:
    return ANCHOR + pd.Timedelta(minutes=i)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


def test_flat_synthetic_bar_follows_spec_gap_rule():
    b = Bar.flat_synthetic(ts_at(5), prev_close=123.0)
    assert b.open == b.high == b.low == b.close == 123.0
    assert b.volume == 0.0 and b.num_trades == 0.0
    assert b.synthetic


def test_deriv_snapshot_forward_fill_semantics():
    prev = DerivSnapshot(funding_rate=1e-4, oi_usd=5e9)
    cur = DerivSnapshot(oi_usd=6e9)
    merged = cur.merged_over(prev)
    assert merged.funding_rate == 1e-4      # carried
    assert merged.oi_usd == 6e9             # fresher wins
    assert merged.bvol is None              # both dark


def test_minutes_since_rejects_off_grid():
    with pytest.raises(GridError):
        minutes_since(ANCHOR, ANCHOR + pd.Timedelta(seconds=30))


# ---------------------------------------------------------------------------
# BarBuffer
# ---------------------------------------------------------------------------


def test_buffer_append_window_and_phase_alignment():
    buf = BarBuffer(40, m=4, anchor_ts=ANCHOR, with_derivatives=False, slack=8)
    for i in range(10):
        buf.append(mk_bar(ts_at(i), close=100 + i))
    frame = buf.window_frame()
    # First row phase-aligned: global index of row 0 must be ≡ 0 (mod 4).
    assert minutes_since(ANCHOR, frame.index[0]) % 4 == 0
    assert frame.index[-1] == ts_at(9)
    assert list(frame.columns)[:4] == ["open", "high", "low", "close"]
    assert frame["close"].iloc[-1] == 109


def test_buffer_trims_to_capacity_and_stays_aligned():
    buf = BarBuffer(8, m=4, anchor_ts=ANCHOR, with_derivatives=False, slack=4)
    for i in range(23):
        buf.append(mk_bar(ts_at(i)))
    assert len(buf) == 8
    frame = buf.window_frame()
    # After arbitrary trimming/compaction the aligned window head is still
    # on the M-grid, and covers at least capacity - M + 1 rows.
    assert minutes_since(ANCHOR, frame.index[0]) % 4 == 0
    assert len(frame) >= 8 - 4 + 1
    assert buf.last_ts == ts_at(22)


def test_buffer_rejects_non_contiguous_appends():
    buf = BarBuffer(8, m=4, anchor_ts=ANCHOR, with_derivatives=False)
    buf.append(mk_bar(ts_at(0)))
    with pytest.raises(GridError):
        buf.append(mk_bar(ts_at(2)))  # gap must be repaired upstream
    with pytest.raises(GridError):
        buf.append(mk_bar(ts_at(0)))  # duplicate


def test_buffer_bootstrap_and_tail_views():
    buf = BarBuffer(16, m=4, anchor_ts=ANCHOR, with_derivatives=False)
    idx = pd.date_range(ts_at(0), periods=12, freq="1min", tz="UTC")
    frame = pd.DataFrame({
        "open": 1.0, "high": 2.0, "low": 0.5, "close": np.arange(12, dtype=float) + 1,
        "volume": 1.0, "quote_volume": 1.0, "num_trades": 1.0,
        "taker_buy_base": 0.5, "taker_buy_quote": 0.5,
    }, index=idx)
    n = buf.bootstrap(frame)
    assert n == 12
    assert buf.last_close == 12.0
    np.testing.assert_allclose(buf.tail_closes(3), [10.0, 11.0, 12.0])
    buf.append(mk_bar(ts_at(12), close=13.0))
    assert buf.last_close == 13.0


def test_buffer_deriv_columns_roundtrip():
    buf = BarBuffer(8, m=4, anchor_ts=ANCHOR, with_derivatives=True)
    buf.append(mk_bar(ts_at(0)), DerivSnapshot(funding_rate=2e-4))
    frame = buf.window_frame()
    assert frame["funding_rate"].iloc[0] == pytest.approx(2e-4)
    assert np.isnan(frame["bvol"].iloc[0])  # dark source → NaN → undef path


# ---------------------------------------------------------------------------
# GridGuard
# ---------------------------------------------------------------------------


def _upd(i: int, close: float = 100.0) -> MarketUpdate:
    return MarketUpdate(bar=mk_bar(ts_at(i), close=close))


def test_grid_guard_passes_contiguous_and_repairs_gaps():
    events = []
    g = GridGuard(max_repair_gap=5, sink=events.append)
    assert len(g.admit(_upd(0))) == 1
    assert len(g.admit(_upd(1))) == 1
    out = g.admit(_upd(4, close=110.0))  # minutes 2,3 missing
    assert [minutes_since(ANCHOR, u.ts) for u in out] == [2, 3, 4]
    assert out[0].bar.synthetic and out[1].bar.synthetic and not out[2].bar.synthetic
    assert out[0].bar.close == 100.0  # flat at previous close
    assert g.n_repaired == 2
    assert events and events[0].guard == "grid"


def test_grid_guard_drops_duplicates_and_rejects_out_of_order():
    g = GridGuard()
    g.admit(_upd(0))
    g.admit(_upd(1))
    assert g.admit(_upd(1)) == []          # duplicate dropped
    assert g.n_duplicates == 1
    with pytest.raises(GridError):
        g.admit(_upd(0))                    # rewind refused


def test_grid_guard_gap_kill_switch():
    g = GridGuard(max_repair_gap=3)
    g.admit(_upd(0))
    with pytest.raises(GridError):
        g.admit(_upd(5))  # 4 missing > 3


def test_grid_guard_forward_fills_deriv_state():
    g = GridGuard()
    g.admit(MarketUpdate(bar=mk_bar(ts_at(0)), deriv=DerivSnapshot(funding_rate=1e-4)))
    out = g.admit(MarketUpdate(bar=mk_bar(ts_at(1))))  # deriv dark this minute
    assert out[0].deriv.funding_rate == 1e-4


# ---------------------------------------------------------------------------
# BarSchemaGuard
# ---------------------------------------------------------------------------


def test_schema_guard_clamps_repairable_violations():
    events = []
    guard = BarSchemaGuard(sink=events.append)
    bad = mk_bar(ts_at(0), open=100.0, close=102.0, high=101.0, low=99.0,
                 volume=1.0, taker_buy_base=5.0)
    fixed = guard.admit(bad)
    assert fixed.high == 102.0             # high >= max(o, c)
    assert fixed.taker_buy_base == fixed.volume  # taker <= volume
    assert guard.n_repaired == 1 and events


def test_schema_guard_rejects_structural_violations():
    guard = BarSchemaGuard()
    with pytest.raises(BarSchemaError):
        guard.admit(mk_bar(ts_at(0), close=-5.0))
    with pytest.raises(BarSchemaError):
        guard.admit(mk_bar(ts_at(0), open=float("nan")))


def test_schema_guard_passthrough_is_identity():
    guard = BarSchemaGuard()
    good = mk_bar(ts_at(0), open=100.0, close=101.0, high=101.5, low=99.5)
    assert guard.admit(good) is good
    assert guard.n_repaired == 0


# ---------------------------------------------------------------------------
# WarmupGuard
# ---------------------------------------------------------------------------


def test_warmup_guard():
    w = WarmupGuard(10)
    assert not w.ready(9)
    assert w.ready(10)
    assert w.deficit(3) == 7
