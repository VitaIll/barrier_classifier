"""Validate the external feed topology on REAL data.

Proves the data-source isolation is behavior-preserving: real BTC klines
pushed through the durable FeedStore and consumed by the engine's
FeedSource produce byte-identical predictions to the in-process
ReplaySource on the same window. Writer and reader share only the SQLite
file — the engine holds no market-data connection.

    python scripts/validate_feed_chain.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # Windows consoles default to cp1252; force UTF-8 for the report glyphs.
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.feed import FeedStore  # noqa: E402
from src.engine.engine import Engine, EngineConfig  # noqa: E402
from src.engine.model import ModelRegistry  # noqa: E402
from src.engine.retrain import RetrainPolicy  # noqa: E402
from src.engine.sources import FeedSource, ReplaySource  # noqa: E402

SPOT = ROOT / "data" / "raw_data" / "klines_1m.parquet"
MODELS = ROOT / "models"
N_BARS = 8  # streamed bars; rolling mode ~18s/bar, source-equivalence needs few


def _config(store_path: str, min_ready: int) -> EngineConfig:
    return EngineConfig(
        model_dir=MODELS, feature_mode="rolling",
        store_path=store_path, retention_days=None,
        min_ready_rows=min_ready,
        retrain=RetrainPolicy(enabled=False), retrain_threaded=False,
        log_every_bars=100_000,
    )


def main() -> int:
    if not SPOT.exists():
        print(f"SKIP — real klines not found at {SPOT}")
        return 0
    reg = ModelRegistry(MODELS)
    contract = reg.load(reg.active_version()).contract
    warm = contract.n_warmup + 200

    frame = pd.read_parquet(SPOT)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    frame = frame.sort_index()
    # A window with enough prefix to warm the buffer + N_BARS to stream.
    start_i = contract.n_warmup + 5_000
    window = frame.iloc[start_i - warm : start_i + N_BARS]
    stream_start = window.index[warm]
    print(f"[0] window {window.index[0]} -> {window.index[-1]} "
          f"({len(window):,} bars; stream from {stream_start})")

    import shutil

    td = Path(tempfile.mkdtemp())
    try:
        # --- Reference: in-process ReplaySource ---------------------------
        spot_path = td / "window.parquet"
        window.to_parquet(spot_path)
        ref_source = ReplaySource(spot_path, start=stream_start)
        t0 = time.perf_counter()
        with Engine(_config(str(td / "ref.db"), contract.n_warmup + 1), source=ref_source) as eng:
            eng.run()
        ref_preds = _predictions(td / "ref.db")
        print(f"[1] ReplaySource: {len(ref_preds):,} predictions in "
              f"{time.perf_counter()-t0:.1f}s")

        # --- Under test: FeedStore + FeedSource (external topology) --------
        # Realistic live arrival: the store starts with the warm PREFIX
        # only; a separate producer thread appends the streamed bars one at
        # a time while the engine (a different object, its own read-only
        # connection) consumes them. Writer and reader share only the file.
        import threading

        feed_path = td / "feed.db"
        prefix, streamed = window.iloc[:warm], window.iloc[warm:]
        writer_store = FeedStore(feed_path)
        writer_store.append_frame(prefix)

        def _producer() -> None:
            time.sleep(1.0)  # let the engine bootstrap + start polling
            for i in range(len(streamed)):
                writer_store.append_frame(streamed.iloc[i : i + 1])
                time.sleep(0.2)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        feed_reader = FeedStore(feed_path, read_only=True)
        feed_source = FeedSource(
            feed_reader, poll_seconds=0.1, idle_timeout_seconds=5.0
        )
        t0 = time.perf_counter()
        with Engine(_config(str(td / "feed.db.engine"), contract.n_warmup + 1), source=feed_source) as eng:
            eng.run()
        thread.join(timeout=5.0)
        feed_preds = _predictions(td / "feed.db.engine")
        feed_reader.close()
        writer_store.close()
        print(f"[2] FeedStore->FeedSource: {len(feed_preds):,} predictions in "
              f"{time.perf_counter()-t0:.1f}s")
    finally:
        shutil.rmtree(td, ignore_errors=True)  # WAL files may linger on Windows

    # --- Compare on the shared timestamps -----------------------------------
    merged = ref_preds.merge(feed_preds, on="ts", suffixes=("_ref", "_feed"))
    if merged.empty:
        print("FAIL — no overlapping predictions")
        return 1
    dp = np.abs(merged["p_ref"].to_numpy() - merged["p_feed"].to_numpy())
    exact = int((dp == 0).sum())
    print(f"[3] prediction parity on {len(merged):,} shared bars: "
          f"max|Δp|={dp.max():.3e}, exact={exact:,}/{len(merged):,}")
    ok = dp.max() == 0.0 and len(merged) > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _predictions(db_path: Path) -> pd.DataFrame:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        frame = pd.read_sql_query(
            "SELECT ts_ms, p FROM predictions ORDER BY ts_ms", conn
        )
    finally:
        conn.close()
    frame["ts"] = pd.to_datetime(frame.pop("ts_ms"), unit="ms", utc=True)
    return frame[["ts", "p"]]


if __name__ == "__main__":
    raise SystemExit(main())
