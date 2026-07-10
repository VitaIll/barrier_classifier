"""Data sources: replay windowing/bootstrap and the push-queue adapter."""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from src.engine.domain import Bar, MarketUpdate
from src.engine.errors import ConfigError
from src.engine.sources import CallbackSource, DataSource, ReplaySource

pytestmark = pytest.mark.engine

START = pd.Timestamp("2025-03-01 00:01:00", tz="UTC")


@pytest.fixture()
def spot_parquet(tmp_path):
    idx = pd.date_range(START, periods=300, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.1, size=len(idx)))
    frame = pd.DataFrame({
        "open": close, "high": close + 0.05, "low": close - 0.05, "close": close,
        "volume": 1.0, "quote_volume": close, "num_trades": 10.0,
        "taker_buy_base": 0.5, "taker_buy_quote": 50.0,
    }, index=idx)
    frame.index.name = "ts"
    path = tmp_path / "klines_1m.parquet"
    frame.to_parquet(path)
    return path, frame


def test_replay_source_streams_window_and_bootstraps(spot_parquet):
    path, frame = spot_parquet
    start = frame.index[100]
    end = frame.index[200]
    src = ReplaySource(path, start=start, end=end)
    assert isinstance(src, DataSource)  # runtime_checkable protocol

    boot = src.bootstrap(50)
    assert len(boot) == 50
    assert boot.index[-1] == frame.index[99]  # strictly before the stream

    updates = list(src.stream())
    assert len(updates) == 100
    assert updates[0].ts == start
    assert updates[-1].ts == frame.index[199]  # end exclusive
    assert updates[0].bar.close == pytest.approx(frame["close"].iloc[100])


def test_replay_source_full_frame_covers_history_plus_live(spot_parquet):
    path, frame = spot_parquet
    src = ReplaySource(path, start=frame.index[100])
    full = src.full_frame()
    assert len(full) == len(frame)
    pd.testing.assert_index_equal(full.index, frame.index)


def test_replay_source_rejects_empty_window(spot_parquet):
    path, frame = spot_parquet
    with pytest.raises(ConfigError):
        ReplaySource(path, start=frame.index[-1] + pd.Timedelta(minutes=5))


def test_replay_source_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        ReplaySource(tmp_path / "nope.parquet")


def test_callback_source_push_stream_close():
    src = CallbackSource()
    bar = Bar(ts=START, open=1, high=1, low=1, close=1, volume=0,
              quote_volume=0, num_trades=0, taker_buy_base=0, taker_buy_quote=0)
    received = []

    def consume():
        received.extend(u.ts for u in src.stream())

    t = threading.Thread(target=consume)
    t.start()
    src.push(MarketUpdate(bar=bar))
    src.push(MarketUpdate(bar=Bar(**{**vars_bar(bar), "ts": START + pd.Timedelta(minutes=1)})))
    src.close()
    t.join(timeout=5)
    assert not t.is_alive()
    assert received == [START, START + pd.Timedelta(minutes=1)]
    with pytest.raises(RuntimeError):
        src.push(MarketUpdate(bar=bar))


def vars_bar(bar: Bar) -> dict:
    return {
        "ts": bar.ts, "open": bar.open, "high": bar.high, "low": bar.low,
        "close": bar.close, "volume": bar.volume, "quote_volume": bar.quote_volume,
        "num_trades": bar.num_trades, "taker_buy_base": bar.taker_buy_base,
        "taker_buy_quote": bar.taker_buy_quote,
    }
