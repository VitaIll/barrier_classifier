"""Feed service: durable store, writer backfill/append, and process isolation.

Verifies the external-service contract end to end on a REALISTIC bar
stream: a Binance kline poller (fake transport, real payload shapes)
drives the writer into a store; a separate FeedSource reader — the engine
side — tails it and reconstructs identical bars. Writer and reader share
only the SQLite file, never in-process state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.data.binance as bz
from src.core.errors import ConfigError
from src.data.binance import BinanceClient, BinanceKlineSource
from src.data.feed import FeedStore, FeedWriter
from src.engine.sources import FeedSource

pytestmark = pytest.mark.data

MIN_MS = 60_000
BASE_MS = (1_700_000_000_000 // MIN_MS) * MIN_MS


def _realistic_klines(start_ms: int, n: int, seed: int = 0) -> list:
    """Binance kline arrays with the exact 12-field shape + string types."""
    rng = np.random.default_rng(seed)
    close = 50_000 + np.cumsum(rng.normal(0, 15, n))
    rows = []
    for i in range(n):
        o = float(close[i - 1]) if i else float(close[0])
        c = float(close[i])
        hi = max(o, c) + abs(rng.normal(0, 5))
        lo = min(o, c) - abs(rng.normal(0, 5))
        vol = abs(rng.normal(8, 2))
        open_ms = start_ms + i * MIN_MS
        rows.append([
            open_ms, f"{o:.2f}", f"{hi:.2f}", f"{lo:.2f}", f"{c:.2f}",
            f"{vol:.6f}", open_ms + MIN_MS - 1, f"{vol * c:.6f}",
            int(rng.integers(50, 5000)), f"{vol * 0.5:.6f}",
            f"{vol * c * 0.5:.6f}", "0",
        ])
    return rows


class FakeTransport:
    def __init__(self, rows: list) -> None:
        self.rows = rows

    def __call__(self, method, url, params, headers, timeout):
        start = params.get("startTime")
        end = params.get("endTime")
        out = [
            r for r in self.rows
            if (start is None or r[0] >= start) and (end is None or r[0] <= end)
        ]
        return 200, out[: params.get("limit", 1000)]


class TestFeedStore:
    def test_append_is_idempotent(self, tmp_path):
        store = FeedStore(tmp_path / "feed.db")
        frame = _frame(BASE_MS, 10)
        assert store.append_frame(frame) == 10
        assert store.append_frame(frame) == 0  # re-send = no-op (INSERT OR IGNORE)
        assert store.count() == 10
        store.close()

    def test_read_tail_and_since(self, tmp_path):
        store = FeedStore(tmp_path / "feed.db")
        store.append_frame(_frame(BASE_MS, 20))
        tail = store.read_tail(5)
        assert len(tail) == 5
        cutoff = pd.Timestamp(BASE_MS + 15 * MIN_MS, unit="ms", tz="UTC")
        since = store.read_since(cutoff)
        assert (since.index > cutoff).all() and len(since) == 4
        store.close()

    def test_read_only_requires_existing(self, tmp_path):
        with pytest.raises(ConfigError, match="does not exist"):
            FeedStore(tmp_path / "nope.db", read_only=True)


class TestWriterReaderChain:
    def test_binance_poller_through_store_to_feedsource(self, tmp_path, monkeypatch):
        # Freeze the clock so exactly `n_closed` bars are "closed".
        n_hist, n_stream = 30, 5
        total = n_hist + n_stream
        rows = _realistic_klines(BASE_MS, total + 1)  # +1 = still-open candle
        now_ms = BASE_MS + total * MIN_MS + 30_000  # mid of the open candle
        monkeypatch.setattr(bz.time, "time", lambda: now_ms / 1000.0)

        client = BinanceClient(testnet=True, transport=FakeTransport(rows),
                               api_key="k", api_secret="s")
        source = BinanceKlineSource(client, poll_seconds=0.001)

        # --- producer process side: writer drains the poller into the store
        store = FeedStore(tmp_path / "feed.db")
        writer = FeedWriter(source, store, backfill_rows=n_hist)
        source.close()  # stream() will exit immediately; backfill still runs
        writer.run(max_bars=0)
        assert store.count() >= n_hist
        store.close()

        # --- consumer process side: a FRESH read-only store, no shared state
        reader_store = FeedStore(tmp_path / "feed.db", read_only=True)
        feed = FeedSource(reader_store, poll_seconds=0.001, idle_timeout_seconds=0.01)
        boot = feed.bootstrap(10)
        assert len(boot) == 10
        assert str(boot.index.tz) == "UTC"
        # Round-trip: what the consumer reads equals what the writer stored,
        # and every stored bar traces back to the exchange payload exactly.
        stored_tail = FeedStore(tmp_path / "feed.db", read_only=True).read_tail(10)
        np.testing.assert_array_equal(
            boot["close"].to_numpy(), stored_tail["close"].to_numpy()
        )
        by_ts = {r[0] + MIN_MS: float(r[4]) for r in rows}  # bar-complete ms -> close
        for ts, close in zip(boot.index, boot["close"]):
            assert float(close) == by_ts[int(ts.value // 1_000_000)]
        reader_store.close()

    def test_writer_backfill_resumes_without_gap(self, tmp_path, monkeypatch):
        rows = _realistic_klines(BASE_MS, 40)
        now_ms = BASE_MS + 40 * MIN_MS + 30_000
        monkeypatch.setattr(bz.time, "time", lambda: now_ms / 1000.0)
        path = tmp_path / "feed.db"

        # First writer run persists 20 bars, then "crashes".
        store = FeedStore(path)
        store.append_frame(_frame_from_klines(rows[:20]))
        first_last = store.last_ts()
        store.close()

        # Second writer run: backfill only fetches bars AFTER the store's tail.
        client = BinanceClient(testnet=True, transport=FakeTransport(rows),
                               api_key="k", api_secret="s")
        source = BinanceKlineSource(client, poll_seconds=0.001)
        store2 = FeedStore(path)
        writer = FeedWriter(source, store2, backfill_rows=50)
        source.close()
        writer.run(max_bars=0)
        # No gap, no duplication: contiguous 1-min grid across the restart.
        frame = store2.read_tail(1000)
        diffs = frame.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(minutes=1)).all()
        assert frame.index[0] <= first_last
        store2.close()


class TestFeedSourceConsumer:
    def test_stream_tails_appended_bars(self, tmp_path):
        store = FeedStore(tmp_path / "feed.db")
        store.append_frame(_frame(BASE_MS, 3))
        reader = FeedStore(tmp_path / "feed.db", read_only=True)
        feed = FeedSource(reader, poll_seconds=0.001, idle_timeout_seconds=0.02)
        feed.bootstrap(3)
        # Producer appends two more bars after the consumer bootstrapped.
        store.append_frame(_frame(BASE_MS + 3 * MIN_MS, 2))
        got = list(feed.stream())
        assert [u.bar.ts for u in got] == [
            pd.Timestamp(BASE_MS + 3 * MIN_MS, unit="ms", tz="UTC"),
            pd.Timestamp(BASE_MS + 4 * MIN_MS, unit="ms", tz="UTC"),
        ]
        store.close()
        reader.close()

    def test_idle_timeout_ends_session(self, tmp_path):
        store = FeedStore(tmp_path / "feed.db")
        store.append_frame(_frame(BASE_MS, 2))
        store.close()
        reader = FeedStore(tmp_path / "feed.db", read_only=True)
        feed = FeedSource(reader, poll_seconds=0.001, idle_timeout_seconds=0.01)
        feed.bootstrap(2)
        assert list(feed.stream()) == []  # silent feed -> clean end
        reader.close()


# -- helpers ---------------------------------------------------------------


def _frame(start_ms: int, n: int) -> pd.DataFrame:
    idx = pd.date_range(
        pd.Timestamp(start_ms, unit="ms", tz="UTC"), periods=n, freq="1min"
    )
    close = 50_000.0 + np.arange(n)
    return pd.DataFrame(
        {
            "open": close, "high": close + 1, "low": close - 1, "close": close,
            "volume": np.full(n, 5.0), "quote_volume": close * 5,
            "num_trades": np.full(n, 10.0), "taker_buy_base": np.full(n, 2.5),
            "taker_buy_quote": close * 2.5,
        },
        index=idx,
    )


def _frame_from_klines(rows: list) -> pd.DataFrame:
    from src.data.binance import klines_to_frame

    return klines_to_frame(rows)
