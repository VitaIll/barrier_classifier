"""Data sources — the engine's inbound public interface.

A :class:`DataSource` is anything that can (a) hand the engine a
historical prefix to warm the buffer and (b) stream closed 1-minute bars
in order. Two implementations ship:

- :class:`ReplaySource` — replays validated historical parquet as a
  simulated live stream (the definition-of-done source; also the
  backtest-parity harness).
- :class:`CallbackSource` — a thread-safe push queue. A real exchange
  adapter (e.g. a Binance kline websocket client) only has to translate
  exchange payloads into :class:`MarketUpdate`s and ``push()`` them; the
  engine neither knows nor cares what is on the other side.

Sources emit *closed* bars only. Timestamps are bar-complete UTC
(``open_time + 60s``), matching the research convention everywhere else.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.engine.domain import (
    RAW_DERIV_COLS,
    RAW_SPOT_COLS,
    Bar,
    DerivSnapshot,
    MarketUpdate,
)
from src.engine.errors import ConfigError

# How raw derivative parquets (written by 01_data_download.ipynb) map onto
# the pipeline's expected column names.
_FUTURES_RENAME = {
    "close": "close_fut",
    "volume": "volume_fut",
    "quote_volume": "quote_volume_fut",
    "taker_buy_base": "taker_buy_base_fut",
    "num_trades": "num_trades_fut",
}
_DERIV_FILES = {
    "futures_klines_1m.parquet": _FUTURES_RENAME,
    "funding_rate_1m.parquet": {"funding_rate": "funding_rate"},
    "futures_metrics_1m.parquet": {"oi_usd": "oi_usd"},
    "eoh_summary_1m.parquet": {
        "opt_oi": "opt_oi",
        "put_open_interest": "put_open_interest",
        "call_open_interest": "call_open_interest",
        "opt_volume": "opt_volume",
        "put_volume": "put_volume",
        "call_volume": "call_volume",
    },
    "bvol_index_1m.parquet": {"bvol": "bvol"},
}


def _as_utc(ts: "str | pd.Timestamp | None") -> Optional[pd.Timestamp]:
    """Parse a window bound to tz-aware UTC (naive input = UTC wall time)."""
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


@runtime_checkable
class DataSource(Protocol):
    """Public interface every market-data adapter implements."""

    def bootstrap(self, n_rows: int) -> pd.DataFrame:
        """Historical prefix ending immediately before the first streamed
        bar: a tz-aware 1-minute frame with (at least) the spot columns.
        May return fewer rows than requested (or empty) if history is
        unavailable — the engine then warms up from the stream itself."""
        ...

    def stream(self) -> Iterator[MarketUpdate]:
        """Yield closed bars in chronological order. Returning ends the
        session (replay exhausted / adapter closed)."""
        ...


def _frame_to_updates(frame: pd.DataFrame, with_deriv: bool) -> Iterator[MarketUpdate]:
    """Vectorized frame → MarketUpdate iterator (no per-row pandas access)."""
    idx = frame.index
    spot = {c: frame[c].to_numpy(dtype=np.float64, na_value=np.nan) for c in RAW_SPOT_COLS}
    deriv_arrays = {}
    if with_deriv:
        for c in RAW_DERIV_COLS:
            if c in frame.columns:
                deriv_arrays[c] = frame[c].to_numpy(dtype=np.float64, na_value=np.nan)
    for i in range(len(frame)):
        bar = Bar(
            ts=idx[i],
            open=float(spot["open"][i]), high=float(spot["high"][i]),
            low=float(spot["low"][i]), close=float(spot["close"][i]),
            volume=float(spot["volume"][i]),
            quote_volume=float(spot["quote_volume"][i]),
            num_trades=float(spot["num_trades"][i]),
            taker_buy_base=float(spot["taker_buy_base"][i]),
            taker_buy_quote=float(spot["taker_buy_quote"][i]),
        )
        if deriv_arrays:
            deriv = DerivSnapshot(**{
                c: (None if np.isnan(a[i]) else float(a[i]))
                for c, a in deriv_arrays.items()
            })
        else:
            deriv = DerivSnapshot()
        yield MarketUpdate(bar=bar, deriv=deriv)


class ReplaySource:
    """Replay validated historical parquet as a simulated live stream.

    Parameters
    ----------
    spot_path : Path
        ``klines_1m.parquet`` (tz-aware 1-minute index, spot schema).
    start, end : optional timestamps
        Half-open stream window ``[start, end)`` in bar-complete time.
        Rows before ``start`` are the bootstrap reservoir.
    with_derivatives : bool
        Also load + join the derivative parquets (as-of forward-filled by
        the download pipeline already).
    speed : float | None
        ``None`` (default) replays as fast as the engine consumes;
        a float replays at that multiple of real time (``1.0`` = one bar
        per minute) — useful for demoing the live cadence.
    """

    def __init__(
        self,
        spot_path: str | Path,
        *,
        deriv_dir: str | Path | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        with_derivatives: bool = False,
        speed: Optional[float] = None,
    ) -> None:
        spot_path = Path(spot_path)
        if not spot_path.exists():
            raise ConfigError(f"ReplaySource: spot parquet not found: {spot_path}")
        frame = pd.read_parquet(spot_path)
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        missing = [c for c in RAW_SPOT_COLS if c not in frame.columns]
        if missing:
            raise ConfigError(f"ReplaySource: spot parquet missing columns {missing}")
        if with_derivatives:
            if deriv_dir is None:
                deriv_dir = spot_path.parent / "derivatives"
            frame = self._join_derivatives(frame, Path(deriv_dir))
        frame = frame.sort_index()

        self._start = _as_utc(start)
        self._end = _as_utc(end)

        first_streamed = self._start if self._start is not None else frame.index[0]
        self._history = frame.loc[frame.index < first_streamed]
        live = frame.loc[frame.index >= first_streamed]
        if self._end is not None:
            live = live.loc[live.index < self._end]
        if live.empty:
            raise ConfigError(
                f"ReplaySource: no rows to stream in [{self._start}, {self._end})"
            )
        self._live = live
        self._with_deriv = with_derivatives
        self._speed = speed

    @staticmethod
    def _join_derivatives(frame: pd.DataFrame, deriv_dir: Path) -> pd.DataFrame:
        for fname, rename in _DERIV_FILES.items():
            path = deriv_dir / fname
            if not path.exists():
                continue
            d = pd.read_parquet(path)
            if d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            cols = [c for c in rename if c in d.columns]
            d = d[cols].rename(columns=rename)
            frame = frame.join(d, how="left")
        return frame

    # ------------------------------------------------------------------ #

    @property
    def first_ts(self) -> pd.Timestamp:
        return self._live.index[0]

    @property
    def last_ts(self) -> pd.Timestamp:
        return self._live.index[-1]

    def __len__(self) -> int:
        return len(self._live)

    def bootstrap(self, n_rows: int) -> pd.DataFrame:
        return self._history.tail(n_rows)

    def full_frame(self) -> pd.DataFrame:
        """History + live window in one frame — the batch feature service
        uses this to precompute the replay's features in a single pass."""
        return pd.concat([self._history, self._live])

    def stream(self) -> Iterator[MarketUpdate]:
        if self._speed is None:
            yield from _frame_to_updates(self._live, self._with_deriv)
            return
        interval = 60.0 / float(self._speed)
        for update in _frame_to_updates(self._live, self._with_deriv):
            t0 = time.monotonic()
            yield update
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)


class CallbackSource:
    """Thread-safe push queue — the surface a live exchange adapter targets.

    The producing side (websocket client, REST poller, …) calls
    :meth:`push` with each closed bar and :meth:`close` on shutdown. The
    engine consumes :meth:`stream`. An optional bootstrap frame provides
    the warmup prefix (e.g. fetched once over REST before subscribing).
    """

    _SENTINEL = object()

    def __init__(
        self,
        *,
        bootstrap_frame: pd.DataFrame | None = None,
        maxsize: int = 10_000,
    ) -> None:
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=maxsize)
        self._bootstrap = bootstrap_frame if bootstrap_frame is not None else pd.DataFrame()
        self._closed = threading.Event()

    def push(self, update: MarketUpdate) -> None:
        if self._closed.is_set():
            raise RuntimeError("CallbackSource is closed")
        self._queue.put(update)

    def close(self) -> None:
        if not self._closed.is_set():
            self._closed.set()
            self._queue.put(self._SENTINEL)

    def bootstrap(self, n_rows: int) -> pd.DataFrame:
        if self._bootstrap.empty:
            return self._bootstrap
        return self._bootstrap.tail(n_rows)

    def stream(self) -> Iterator[MarketUpdate]:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                return
            yield item  # type: ignore[misc]
