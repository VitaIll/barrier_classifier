"""Fast local persistence — SQLite in WAL mode.

The store is the engine's *working set*, not its archive: everything the
session produces (bars, predictions, decisions, fills, trades, equity,
matured labels, guard events, model/retrain bookkeeping) lands here with
batched single-writer inserts, and event-time retention pruning keeps it
bounded. Long-term truth stays in parquet and the model registry.

Concurrency contract: the engine thread owns the writer connection; any
other thread (retrainer, dashboards) uses :meth:`read_connection` /
:meth:`bars_frame`, which open their own read-only handles — WAL makes
concurrent readers safe.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.engine.domain import (
    Bar,
    Decision,
    Fill,
    GuardEvent,
    Order,
    Prediction,
    Trade,
)
from src.engine.errors import StoreError

logger = logging.getLogger("src.engine.store")

# How long SQLite waits on a locked DB before raising (ms). Covers the brief
# windows where a reader (retrain worker / dashboard) holds the file.
_BUSY_TIMEOUT_MS = 5_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ts_ms INTEGER PRIMARY KEY,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume REAL NOT NULL, quote_volume REAL NOT NULL, num_trades REAL NOT NULL,
    taker_buy_base REAL NOT NULL, taker_buy_quote REAL NOT NULL,
    synthetic INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS predictions (
    ts_ms INTEGER PRIMARY KEY,
    p REAL NOT NULL,
    model_version TEXT NOT NULL,
    feature_ms REAL,
    predict_ms REAL
);
CREATE TABLE IF NOT EXISTS decisions (
    ts_ms INTEGER PRIMARY KEY,
    action TEXT NOT NULL,
    size REAL NOT NULL,
    p REAL, score REAL, threshold REAL,
    n_open INTEGER NOT NULL,
    gross_size REAL NOT NULL,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    ts_ms INTEGER NOT NULL,
    side INTEGER NOT NULL,
    size REAL NOT NULL,
    kind TEXT NOT NULL,
    limit_price REAL,
    status TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    order_id INTEGER NOT NULL,
    ts_ms INTEGER NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    side INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    trade_id INTEGER PRIMARY KEY,
    k_entry INTEGER NOT NULL,
    ts_entry_ms INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    size REAL NOT NULL,
    k_exit INTEGER NOT NULL,
    ts_exit_ms INTEGER NOT NULL,
    exit_price REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    gross_log_return REAL NOT NULL,
    net_log_return REAL NOT NULL,
    weighted_net_log_return REAL NOT NULL,
    p_at_entry REAL,
    model_version TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    ts_ms INTEGER PRIMARY KEY,
    realized_cum REAL NOT NULL,
    unrealized REAL NOT NULL,
    equity REAL NOT NULL,
    n_open INTEGER NOT NULL,
    gross_size REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
    ts_ms INTEGER PRIMARY KEY,
    y INTEGER NOT NULL,
    m_k REAL
);
CREATE TABLE IF NOT EXISTS guard_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    guard TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    created_ts_ms INTEGER NOT NULL,
    path TEXT NOT NULL,
    activated_ts_ms INTEGER,
    metrics_json TEXT,
    thresholds_json TEXT
);
CREATE TABLE IF NOT EXISTS retrain_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_ts_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_ms INTEGER,
    finished_ms INTEGER,
    n_rows INTEGER,
    best_iter INTEGER,
    gate_passed INTEGER,
    new_version TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS open_positions (
    k_entry INTEGER PRIMARY KEY,
    ts_entry_ms INTEGER NOT NULL,
    side INTEGER NOT NULL,
    size REAL NOT NULL,
    entry_price REAL NOT NULL,
    tp_price REAL NOT NULL,
    sl_price REAL,
    expiry_k INTEGER NOT NULL,
    p_at_entry REAL,
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ts_exit ON trades(ts_exit_ms);
CREATE INDEX IF NOT EXISTS idx_guard_events_ts ON guard_events(ts_ms);
"""

_BAR_COLS = (
    "ts_ms", "open", "high", "low", "close", "volume", "quote_volume",
    "num_trades", "taker_buy_base", "taker_buy_quote", "synthetic",
)


def _ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def _from_ms(ms: "pd.Series | np.ndarray") -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(ms, dtype=np.int64), unit="ms", utc=True))


class SQLiteStore:
    """Single-writer batched SQLite store (WAL). ``path=':memory:'`` for tests."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._memory = self._path == ":memory:"
        if not self._memory:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._lock = threading.Lock()  # guards the writer connection
        cur = self._conn.cursor()
        if not self._memory:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.executescript(_SCHEMA)
        self._conn.commit()
        # Per-table pending batches (flushed once per bar).
        self._pending: dict[str, list[tuple]] = {
            "bars": [], "predictions": [], "decisions": [], "orders": [],
            "fills": [], "trades": [], "equity": [], "labels": [],
            "guard_events": [],
        }

    # ------------------------------------------------------------------ #
    # Recorders (buffered)
    # ------------------------------------------------------------------ #

    def record_bar(self, bar: Bar) -> None:
        self._pending["bars"].append(
            (_ms(bar.ts),) + bar.values() + (int(bar.synthetic),)
        )

    def record_bars_frame(self, frame: pd.DataFrame) -> int:
        """Bulk-persist a historical prefix (bootstrap). Idempotent on ts."""
        if frame.empty:
            return 0
        ts_ms = (frame.index.tz_convert("UTC").asi8 // 1_000_000).astype(np.int64)
        cols = [frame[c].to_numpy(dtype=np.float64) for c in _BAR_COLS[1:-1]]
        rows = list(zip(ts_ms.tolist(), *[c.tolist() for c in cols], [0] * len(frame)))
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            self._conn.commit()
        return len(rows)

    def record_prediction(self, pred: Prediction) -> None:
        self._pending["predictions"].append(
            (_ms(pred.ts), pred.p, pred.model_version, pred.feature_ms, pred.predict_ms)
        )

    def record_decision(self, d: Decision) -> None:
        self._pending["decisions"].append((
            _ms(d.ts), d.action.value, d.size,
            None if np.isnan(d.p) else d.p,
            None if np.isnan(d.score) else d.score,
            None if np.isnan(d.threshold) else d.threshold,
            d.n_open, d.gross_size, d.reason,
        ))

    def record_order(self, o: Order, status: str) -> None:
        self._pending["orders"].append((
            o.order_id, _ms(o.ts), o.side, o.size, o.kind.value,
            o.limit_price, status, o.note,
        ))

    def record_fill(self, f: Fill) -> None:
        self._pending["fills"].append((f.order_id, _ms(f.ts), f.price, f.size, f.side))

    def record_trade(self, t: Trade) -> None:
        self._pending["trades"].append((
            t.trade_id, t.k_entry, _ms(t.ts_entry), t.entry_price, t.size,
            t.k_exit, _ms(t.ts_exit), t.exit_price, t.exit_reason,
            t.gross_log_return, t.net_log_return, t.weighted_net_log_return,
            None if np.isnan(t.p_at_entry) else t.p_at_entry, t.model_version,
        ))

    def record_equity(
        self, ts: pd.Timestamp, realized_cum: float, unrealized: float,
        n_open: int, gross_size: float,
    ) -> None:
        self._pending["equity"].append((
            _ms(ts), realized_cum, unrealized, realized_cum + unrealized,
            n_open, gross_size,
        ))

    def record_label(self, ts: pd.Timestamp, y: int, m_k: float) -> None:
        self._pending["labels"].append(
            (_ms(ts), int(y), None if np.isnan(m_k) else m_k)
        )

    def record_guard_event(self, e: GuardEvent) -> None:
        self._pending["guard_events"].append(
            (_ms(e.ts), e.guard, e.severity, e.message, e.count)
        )

    _INSERTS = {
        "bars": "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        "predictions": "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?)",
        "decisions": "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
        "orders": "INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?)",
        "fills": "INSERT INTO fills VALUES (?,?,?,?,?)",
        "trades": "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        "equity": "INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?)",
        "labels": "INSERT OR REPLACE INTO labels VALUES (?,?,?)",
        "guard_events": "INSERT INTO guard_events (ts_ms, guard, severity, message, count) VALUES (?,?,?,?,?)",
    }

    def flush(self) -> None:
        """Persist all buffered rows atomically.

        Buffers are cleared only *after* the commit succeeds. On any insert
        or commit failure the transaction is rolled back, the buffered rows
        are retained for a later retry, and a :class:`StoreError` is raised —
        a mid-flush failure never partially commits or silently drops rows.
        """
        with self._lock:
            pending = [(t, rows) for t, rows in self._pending.items() if rows]
            if not pending:
                return
            try:
                for table, rows in pending:
                    self._conn.executemany(self._INSERTS[table], rows)
                self._conn.commit()
            except sqlite3.Error as exc:
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                n = sum(len(rows) for _, rows in pending)
                raise StoreError(
                    f"flush failed ({exc}); {n} buffered row(s) retained for retry"
                ) from exc
            for _table, rows in pending:
                rows.clear()

    # ------------------------------------------------------------------ #
    # Model/retrain bookkeeping (unbuffered — rare events)
    # ------------------------------------------------------------------ #

    def record_model_version(
        self, version: str, *, created_ts: pd.Timestamp, path: str,
        metrics: dict | None = None, thresholds: dict | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?)",
                (version, _ms(created_ts), path, None,
                 json.dumps(metrics or {}), json.dumps(thresholds or {})),
            )
            self._conn.commit()

    def mark_model_activated(self, version: str, ts: pd.Timestamp) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE model_versions SET activated_ts_ms=? WHERE version=?",
                (_ms(ts), version),
            )
            self._conn.commit()

    def open_retrain_run(self, trigger_ts: pd.Timestamp) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO retrain_runs (trigger_ts_ms, status) VALUES (?, 'running')",
                (_ms(trigger_ts),),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def close_retrain_run(
        self, run_id: int, *, status: str, started_ms: Optional[int] = None,
        finished_ms: Optional[int] = None, n_rows: Optional[int] = None,
        best_iter: Optional[int] = None, gate_passed: Optional[bool] = None,
        new_version: Optional[str] = None, notes: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE retrain_runs SET status=?, started_ms=?, finished_ms=?, "
                "n_rows=?, best_iter=?, gate_passed=?, new_version=?, notes=? "
                "WHERE run_id=?",
                (status, started_ms, finished_ms, n_rows, best_iter,
                 None if gate_passed is None else int(gate_passed),
                 new_version, notes, run_id),
            )
            self._conn.commit()

    def snapshot_open_positions(self, rows: list[tuple]) -> None:
        """Atomically replace the open-inventory snapshot (crash/resume state).

        ``rows`` are ``(k_entry, ts_entry_ms, side, size, entry_price,
        tp_price, sl_price, expiry_k, p_at_entry, model_version)`` tuples —
        the engine writes one snapshot per bar *on inventory change* so a
        restart can rebuild the exact live portfolio.
        """
        with self._lock:
            try:
                self._conn.execute("DELETE FROM open_positions")
                if rows:
                    self._conn.executemany(
                        "INSERT INTO open_positions VALUES (?,?,?,?,?,?,?,?,?,?)", rows
                    )
                self._conn.commit()
            except sqlite3.Error as exc:
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                raise StoreError(f"open-position snapshot failed ({exc})") from exc

    def load_open_positions(self) -> list[tuple]:
        """The last open-inventory snapshot (rows as stored, ordered by entry)."""
        cur = self._conn.execute(
            "SELECT k_entry, ts_entry_ms, side, size, entry_price, tp_price, "
            "sl_price, expiry_k, p_at_entry, model_version "
            "FROM open_positions ORDER BY k_entry"
        )
        return list(cur.fetchall())

    def last_realized_cum(self) -> Optional[float]:
        """Realized cumulative log-return at the newest equity row (resume)."""
        row = self._conn.execute(
            "SELECT realized_cum FROM equity ORDER BY ts_ms DESC LIMIT 1"
        ).fetchone()
        return None if row is None else float(row[0])

    def max_total_equity(self) -> Optional[float]:
        """Highest total (realized + unrealized) equity ever recorded —
        seeds the drawdown peak on resume."""
        row = self._conn.execute("SELECT MAX(equity) FROM equity").fetchone()
        return None if row is None or row[0] is None else float(row[0])

    def max_trade_id(self) -> int:
        row = self._conn.execute("SELECT MAX(trade_id) FROM trades").fetchone()
        return 0 if row is None or row[0] is None else int(row[0])

    def max_order_id(self) -> int:
        row = self._conn.execute("SELECT MAX(order_id) FROM orders").fetchone()
        return 0 if row is None or row[0] is None else int(row[0])

    def last_bar_ts(self) -> Optional[pd.Timestamp]:
        row = self._conn.execute("SELECT MAX(ts_ms) FROM bars").fetchone()
        if row is None or row[0] is None:
            return None
        return pd.Timestamp(int(row[0]), unit="ms", tz="UTC")

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value)
            )
            self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def read_connection(self) -> sqlite3.Connection:
        """A fresh connection for reader threads (WAL-safe)."""
        if self._memory:
            return self._conn  # :memory: has a single shared handle
        return sqlite3.connect(f"file:{self._path}?mode=ro", uri=True,
                               check_same_thread=False)

    def bars_frame(
        self,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        *,
        with_synthetic: bool = False,
        conn: Optional[sqlite3.Connection] = None,
    ) -> pd.DataFrame:
        """Spot bars as a tz-aware pipeline-ready frame (retraining input).

        ``with_synthetic=True`` keeps the ``synthetic`` flag column (1 for
        gap-repair bars fabricated by the grid guard). Consumers that train
        on this frame need it — a fabricated flat bar must not become a
        training row indistinguishable from real data.
        """
        q = "SELECT * FROM bars"
        clauses, params = [], []
        if start is not None:
            clauses.append("ts_ms >= ?")
            params.append(_ms(start))
        if end is not None:
            clauses.append("ts_ms < ?")
            params.append(_ms(end))
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY ts_ms"
        frame = pd.read_sql_query(q, conn or self._conn, params=params or None)
        if frame.empty:
            return pd.DataFrame()
        frame.index = _from_ms(frame.pop("ts_ms"))
        frame.index.name = "ts"
        if with_synthetic:
            return frame
        return frame.drop(columns=["synthetic"])

    def trades_frame(self) -> pd.DataFrame:
        self.flush()
        frame = pd.read_sql_query("SELECT * FROM trades ORDER BY ts_exit_ms", self._conn)
        if not frame.empty:
            frame["ts_entry"] = _from_ms(frame.pop("ts_entry_ms"))
            frame["ts_exit"] = _from_ms(frame.pop("ts_exit_ms"))
        return frame

    def equity_frame(self) -> pd.DataFrame:
        self.flush()
        frame = pd.read_sql_query("SELECT * FROM equity ORDER BY ts_ms", self._conn)
        if not frame.empty:
            frame["ts"] = _from_ms(frame.pop("ts_ms"))
        return frame

    def counts(self) -> dict[str, int]:
        self.flush()
        out = {}
        for table in ("bars", "predictions", "decisions", "trades", "equity",
                      "labels", "guard_events", "model_versions", "retrain_runs",
                      "open_positions"):
            out[table] = int(
                self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
        return out

    # ------------------------------------------------------------------ #
    # Retention
    # ------------------------------------------------------------------ #

    _PRUNABLE = ("bars", "predictions", "decisions", "equity", "labels")

    def prune_before(self, cutoff: pd.Timestamp) -> dict[str, int]:
        """Event-time retention: drop high-churn rows older than ``cutoff``.

        Trades, orders, fills, guard events, and bookkeeping are never
        pruned — they are the audit trail.
        """
        self.flush()
        removed: dict[str, int] = {}
        with self._lock:
            for table in self._PRUNABLE:
                cur = self._conn.execute(
                    f"DELETE FROM {table} WHERE ts_ms < ?", (_ms(cutoff),)
                )
                removed[table] = cur.rowcount
            self._conn.commit()
        return removed

    def close(self) -> None:
        """Best-effort final flush, then always release the handle.

        A flush failure during shutdown (disk full, malformed buffered row)
        is logged rather than raised — the connection is closed regardless
        so shutdown always completes.
        """
        try:
            self.flush()
        except StoreError as exc:
            logger.error("store close: final flush failed, buffered rows lost: %s", exc)
        finally:
            with self._lock:
                self._conn.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
