"""SQLite store: round-trips, batching, retention, reader access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.domain import (
    Action,
    Bar,
    Decision,
    Fill,
    GuardEvent,
    Order,
    OrderKind,
    Prediction,
    Trade,
)
from src.engine.store import SQLiteStore

pytestmark = pytest.mark.engine

TS = pd.Timestamp("2025-06-01 12:00:00", tz="UTC")


def mk_bar(i: int, close: float = 100.0) -> Bar:
    return Bar(
        ts=TS + pd.Timedelta(minutes=i), open=close, high=close + 1,
        low=close - 1, close=close, volume=2.0, quote_volume=200.0,
        num_trades=5.0, taker_buy_base=1.0, taker_buy_quote=100.0,
    )


@pytest.fixture()
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


def test_roundtrip_all_tables(store):
    store.record_bar(mk_bar(0))
    store.record_prediction(Prediction(ts=TS, p=0.61, model_version="v0001",
                                       feature_ms=12.0, predict_ms=1.0))
    store.record_decision(Decision(ts=TS, action=Action.ENTER, size=0.02, p=0.61,
                                   score=0.61, threshold=0.55, n_open=1,
                                   gross_size=0.02, reason=""))
    store.record_order(Order(order_id=1, ts=TS, side=1, size=0.02,
                             kind=OrderKind.MARKET), status="filled")
    store.record_fill(Fill(order_id=1, ts=TS, price=100.0, size=0.02, side=1))
    store.record_trade(Trade(
        trade_id=1, k_entry=10, ts_entry=TS, entry_price=100.0, size=0.02,
        k_exit=15, ts_exit=TS + pd.Timedelta(minutes=5), exit_price=100.3,
        exit_reason="tp_market", gross_log_return=0.003, net_log_return=0.0025,
        weighted_net_log_return=0.00005, p_at_entry=0.61, model_version="v0001",
    ))
    store.record_equity(TS, realized_cum=0.001, unrealized=0.0005, n_open=1,
                        gross_size=0.02)
    store.record_label(TS, y=1, m_k=0.004)
    store.record_guard_event(GuardEvent(ts=TS, guard="grid", severity="warning",
                                        message="repaired 1"))
    store.flush()

    counts = store.counts()
    assert counts["bars"] == 1
    assert counts["predictions"] == 1
    assert counts["trades"] == 1
    assert counts["guard_events"] == 1

    trades = store.trades_frame()
    assert trades.loc[0, "exit_reason"] == "tp_market"
    assert trades.loc[0, "ts_exit"] == TS + pd.Timedelta(minutes=5)
    eq = store.equity_frame()
    assert eq.loc[0, "equity"] == pytest.approx(0.0015)


def test_bars_frame_bulk_and_windowed(store):
    idx = pd.date_range(TS, periods=10, freq="1min", tz="UTC")
    frame = pd.DataFrame({
        "open": 1.0, "high": 2.0, "low": 0.5, "close": np.arange(10.0) + 1,
        "volume": 1.0, "quote_volume": 1.0, "num_trades": 1.0,
        "taker_buy_base": 0.5, "taker_buy_quote": 0.5,
    }, index=idx)
    n = store.record_bars_frame(frame)
    assert n == 10
    # Idempotent on ts (INSERT OR IGNORE)
    assert store.record_bars_frame(frame) == 10
    assert store.counts()["bars"] == 10

    out = store.bars_frame(start=idx[3], end=idx[7])
    assert len(out) == 4
    assert out.index[0] == idx[3]
    assert list(out.columns)[:4] == ["open", "high", "low", "close"]
    assert out.index.tz is not None


def test_model_and_retrain_bookkeeping(store):
    store.record_model_version("v0001", created_ts=TS, path="models/v0001",
                               metrics={"val_roc_auc": 0.76},
                               thresholds={"p_threshold": 0.55})
    store.mark_model_activated("v0001", TS)
    run_id = store.open_retrain_run(TS)
    store.close_retrain_run(run_id, status="published", n_rows=1000,
                            best_iter=42, gate_passed=True, new_version="v0002")
    counts = store.counts()
    assert counts["model_versions"] == 1
    assert counts["retrain_runs"] == 1


def test_meta_roundtrip(store):
    assert store.get_meta("session") is None
    store.set_meta("session", "abc")
    assert store.get_meta("session") == "abc"


def test_prune_keeps_audit_trail(store):
    for i in range(10):
        store.record_bar(mk_bar(i))
        store.record_equity(TS + pd.Timedelta(minutes=i), 0.0, 0.0, 0, 0.0)
    store.record_trade(Trade(
        trade_id=1, k_entry=0, ts_entry=TS, entry_price=100.0, size=0.02,
        k_exit=1, ts_exit=TS + pd.Timedelta(minutes=1), exit_price=100.1,
        exit_reason="tp", gross_log_return=0.001, net_log_return=0.0005,
        weighted_net_log_return=0.00001, p_at_entry=0.6, model_version="v0001",
    ))
    removed = store.prune_before(TS + pd.Timedelta(minutes=5))
    assert removed["bars"] == 5 and removed["equity"] == 5
    counts = store.counts()
    assert counts["bars"] == 5
    assert counts["trades"] == 1  # audit trail never pruned


def test_file_store_wal_and_read_connection(tmp_path):
    path = tmp_path / "engine.db"
    store = SQLiteStore(path)
    try:
        store.record_bar(mk_bar(0))
        store.flush()
        conn = store.read_connection()
        try:
            n = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
            assert n == 1
        finally:
            conn.close()
    finally:
        store.close()
