"""The bar feed: market-data acquisition as an EXTERNAL service.

Topology (docs/PRODUCTION.md): a :class:`FeedWriter` process owns the
exchange connection and appends every closed bar to a durable
:class:`FeedStore` (its own SQLite file, WAL). The trading engine runs as
a separate process and consumes the store through
``src.engine.sources.FeedSource`` — it holds no exchange-data code, no
credentials for data, and no shared in-process state with acquisition.

What the isolation buys operationally:

- the engine can restart/upgrade while bars keep landing (no data loss,
  no re-backfill — resume tails the feed from where it stopped);
- a data-side failure cannot corrupt engine state and vice versa — the
  only contract between them is rows in a WAL store;
- any producer works: the Binance poller, a replay of recorded history,
  or another venue — the engine cannot tell the difference.

The store is append-only from one writer; readers open their own
connections (WAL allows concurrent read/write across processes).
"""

from __future__ import annotations

import logging
import signal
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.errors import ConfigError

logger = logging.getLogger("src.data.feed")

SPOT_COLS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ts_ms INTEGER PRIMARY KEY,
    open REAL NOT NULL, high REAL NOT NULL,
    low REAL NOT NULL, close REAL NOT NULL,
    volume REAL NOT NULL, quote_volume REAL NOT NULL,
    num_trades REAL NOT NULL, taker_buy_base REAL NOT NULL,
    taker_buy_quote REAL NOT NULL,
    synthetic INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS feed_meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
"""


def _ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def _frame_from_rows(rows: list, columns: tuple[str, ...] = SPOT_COLS) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["ts_ms", *columns, "synthetic"])
    frame.index = pd.to_datetime(frame.pop("ts_ms"), unit="ms", utc=True)
    frame.index.name = "ts"
    return frame.drop(columns=["synthetic"])


class FeedStore:
    """Durable bar store shared BETWEEN processes (writer appends, engine
    tails). One writer, any number of readers; SQLite WAL is the
    cross-process contract."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = bool(read_only)
        if read_only:
            if not self.path.exists():
                raise ConfigError(f"feed store does not exist: {self.path}")
            self._conn = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.commit()

    # -- writer side ---------------------------------------------------------

    def append_frame(self, frame: pd.DataFrame) -> int:
        """Idempotent append (``INSERT OR IGNORE`` on the ts key): the
        writer may safely re-send overlap after its own restart."""
        if self.read_only:
            raise ConfigError("append on a read-only FeedStore")
        if frame.empty:
            return 0
        rows = [
            (
                _ms(ts),
                *(float(frame.iloc[i][c]) for c in SPOT_COLS),
                int(frame.iloc[i].get("synthetic", 0)),
            )
            for i, ts in enumerate(frame.index)
        ]
        cur = self._conn.executemany(
            f"INSERT OR IGNORE INTO bars (ts_ms, {', '.join(SPOT_COLS)}, synthetic) "
            f"VALUES ({', '.join('?' * (len(SPOT_COLS) + 2))})",
            rows,
        )
        self._conn.commit()
        return int(cur.rowcount if cur.rowcount is not None else 0)

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO feed_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    # -- reader side ---------------------------------------------------------

    def last_ts(self) -> Optional[pd.Timestamp]:
        row = self._conn.execute("SELECT MAX(ts_ms) FROM bars").fetchone()
        if row is None or row[0] is None:
            return None
        return pd.Timestamp(int(row[0]), unit="ms", tz="UTC")

    def read_tail(self, n_rows: int) -> pd.DataFrame:
        rows = self._conn.execute(
            f"SELECT ts_ms, {', '.join(SPOT_COLS)}, synthetic FROM bars "
            "ORDER BY ts_ms DESC LIMIT ?",
            (int(n_rows),),
        ).fetchall()
        return _frame_from_rows(rows[::-1])

    def read_since(self, ts: Optional[pd.Timestamp], *, limit: int = 10_000) -> pd.DataFrame:
        if ts is None:
            return self.read_tail(limit)
        rows = self._conn.execute(
            f"SELECT ts_ms, {', '.join(SPOT_COLS)}, synthetic FROM bars "
            "WHERE ts_ms > ? ORDER BY ts_ms ASC LIMIT ?",
            (_ms(ts), int(limit)),
        ).fetchall()
        return _frame_from_rows(rows)

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FeedStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class FeedWriter:
    """Drives any bar producer into the store — the service's main loop.

    The producer is anything with the engine's ``DataSource`` shape
    (``bootstrap(n)`` + ``stream()``): the Binance poller in production,
    a replay of recorded history for rehearsals. On startup the writer
    backfills the gap between the store's last row and now, so a writer
    restart loses nothing.
    """

    def __init__(self, source, store: FeedStore, *, backfill_rows: int = 50_000) -> None:
        self.source = source
        self.store = store
        self.backfill_rows = int(backfill_rows)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        close = getattr(self.source, "close", None)
        if callable(close):
            close()

    def run(self, *, max_bars: Optional[int] = None) -> int:
        """Backfill, then append every streamed bar until stopped.

        Returns the number of bars appended from the stream (backfill not
        counted). ``max_bars`` bounds the stream for rehearsals/tests.
        """
        last = self.store.last_ts()
        boot = self.source.bootstrap(self.backfill_rows)
        if not boot.empty:
            if last is not None:
                boot = boot[boot.index > last]
            appended = self.store.append_frame(boot)
            logger.info(
                "feed backfill: %d new bar(s) (store now ends %s)",
                appended, self.store.last_ts(),
            )
        n = 0
        for update in self.source.stream():
            if self._stop.is_set():
                break
            bar = update.bar
            frame = pd.DataFrame(
                {c: [getattr(bar, c)] for c in SPOT_COLS},
                index=pd.DatetimeIndex([bar.ts], name="ts"),
            )
            frame["synthetic"] = int(getattr(bar, "synthetic", False))
            self.store.append_frame(frame)
            n += 1
            if n % 60 == 0:
                logger.info("feed: %d bars appended (last %s)", n, bar.ts)
            if max_bars is not None and n >= max_bars:
                break
        logger.info("feed writer stopped after %d streamed bar(s)", n)
        return n


def install_graceful_stop(writer: FeedWriter) -> None:
    def _handler(signum, frame):  # noqa: ANN001
        logger.warning("signal %s — stopping feed writer", signum)
        writer.stop()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
