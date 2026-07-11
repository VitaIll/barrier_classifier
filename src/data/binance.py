"""Binance spot market-data acquisition — the exchange DATA adapter.

Lives in ``src.data`` (not ``src.engine``): this is what the external
feed service uses to pull bars. The trading engine never imports it —
it consumes bars through the feed store. ``BinanceClient`` (the signed
REST primitive) is shared with the engine's execution broker, which is a
separate concern (placing orders, not acquiring data).

Safety, transport injection, and retry policy are documented on the
classes. No code path here touches the network in tests.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
from typing import Callable, Iterator, Optional
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from src.engine.domain import Bar, DerivSnapshot, MarketUpdate
from src.engine.errors import ConfigError, ExchangeError

logger = logging.getLogger("src.data.binance")

LIVE_BASE_URL = "https://api.binance.com"
TESTNET_BASE_URL = "https://testnet.binance.vision"

#: (status_code, parsed_json_or_text) returned by a transport call.
Transport = Callable[[str, str, dict, dict, float], tuple[int, object]]


def _default_transport() -> Transport:
    import requests

    session = requests.Session()

    def transport(
        method: str, url: str, params: dict, headers: dict, timeout: float
    ) -> tuple[int, object]:
        resp = session.request(
            method, url, params=params, headers=headers, timeout=timeout
        )
        try:
            body: object = resp.json()
        except ValueError:
            body = resp.text
        return resp.status_code, body

    return transport


class BinanceClient:
    """Thin signed/unsigned REST client with bounded retries.

    Retries transport failures, HTTP 5xx, and 429/418 (honoring
    ``Retry-After``) with exponential backoff; anything else — including
    4xx order rejections — raises :class:`ExchangeError` immediately with
    Binance's error payload in the message.
    """

    def __init__(
        self,
        *,
        testnet: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 4,
        recv_window_ms: int = 5_000,
        transport: Optional[Transport] = None,
    ) -> None:
        self.base_url = TESTNET_BASE_URL if testnet else LIVE_BASE_URL
        self.testnet = bool(testnet)
        self._api_key = api_key if api_key is not None else os.environ.get("BINANCE_API_KEY", "")
        self._api_secret = (
            api_secret if api_secret is not None else os.environ.get("BINANCE_API_SECRET", "")
        )
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.recv_window_ms = int(recv_window_ms)
        self._transport = transport if transport is not None else _default_transport()

    @property
    def has_credentials(self) -> bool:
        return bool(self._api_key and self._api_secret)

    def _sign(self, params: dict) -> dict:
        if not self.has_credentials:
            raise ConfigError(
                "signed Binance request without credentials — set "
                "BINANCE_API_KEY / BINANCE_API_SECRET (or pass api_key/api_secret)"
            )
        signed = dict(params)
        signed["timestamp"] = int(time.time() * 1000)
        signed["recvWindow"] = self.recv_window_ms
        query = urlencode(signed)
        signed["signature"] = hmac.new(
            self._api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        return signed

    def request(
        self, method: str, path: str, params: Optional[dict] = None, *, signed: bool = False
    ) -> object:
        params = dict(params or {})
        headers = {"X-MBX-APIKEY": self._api_key} if (signed or self._api_key) else {}
        if signed:
            params = self._sign(params)
        url = f"{self.base_url}{path}"
        delay = 0.5
        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                status, body = self._transport(method, url, params, headers, self.timeout)
            except Exception as exc:  # transport-level failure
                last_err = f"transport error: {exc}"
                logger.warning("%s %s attempt %d/%d: %s", method, path, attempt, self.max_retries, last_err)
                time.sleep(delay)
                delay *= 2
                continue
            if status < 300:
                return body
            if status in (429, 418) or status >= 500:
                retry_after = 0.0
                if isinstance(body, dict) and "retryAfter" in body:
                    retry_after = float(body["retryAfter"])
                wait = max(delay, retry_after)
                last_err = f"HTTP {status}: {body}"
                logger.warning(
                    "%s %s attempt %d/%d: %s (waiting %.1fs)",
                    method, path, attempt, self.max_retries, last_err, wait,
                )
                time.sleep(wait)
                delay *= 2
                continue
            raise ExchangeError(f"{method} {path} rejected: HTTP {status}: {body}")
        raise ExchangeError(
            f"{method} {path} failed after {self.max_retries} attempts: {last_err}"
        )


# ---------------------------------------------------------------------------
# Klines → engine bars
# ---------------------------------------------------------------------------

_BAR_MS = 60_000


def klines_to_frame(rows: list) -> pd.DataFrame:
    """Binance kline arrays → the engine's tz-aware spot frame.

    Timestamp convention: bar-complete UTC (``open_time + 60s``) —
    identical to the research pipeline and every other source.
    """
    if not rows:
        return pd.DataFrame()
    arr = np.asarray(rows, dtype=object)
    open_ms = arr[:, 0].astype(np.int64)
    ts = pd.to_datetime(open_ms + _BAR_MS, unit="ms", utc=True)
    frame = pd.DataFrame(
        {
            "open": arr[:, 1].astype(float),
            "high": arr[:, 2].astype(float),
            "low": arr[:, 3].astype(float),
            "close": arr[:, 4].astype(float),
            "volume": arr[:, 5].astype(float),
            "quote_volume": arr[:, 7].astype(float),
            "num_trades": arr[:, 8].astype(float),
            "taker_buy_base": arr[:, 9].astype(float),
            "taker_buy_quote": arr[:, 10].astype(float),
        },
        index=pd.DatetimeIndex(ts, name="ts"),
    )
    return frame


def _frame_to_update(row: pd.Series, ts: pd.Timestamp) -> MarketUpdate:
    bar = Bar(
        ts=ts,
        open=float(row["open"]), high=float(row["high"]),
        low=float(row["low"]), close=float(row["close"]),
        volume=float(row["volume"]), quote_volume=float(row["quote_volume"]),
        num_trades=float(row["num_trades"]),
        taker_buy_base=float(row["taker_buy_base"]),
        taker_buy_quote=float(row["taker_buy_quote"]),
    )
    return MarketUpdate(bar=bar, deriv=DerivSnapshot())


class BinanceKlineSource:
    """Closed 1-minute bars from Binance spot, as a :class:`DataSource`.

    ``bootstrap(n)`` pages ``GET /api/v3/klines`` backwards to warm the
    engine buffer (28 days = ~29 requests at 1000 bars each, trivial
    weight). ``stream()`` polls for newly CLOSED candles every
    ``poll_seconds`` — at 1-minute cadence, polling is the robust choice
    over a websocket (no reconnect state machine; a missed interval is
    caught up by the same backfill path). Bars are always emitted in
    order with no gaps: any jump is filled from REST before newer bars
    are released.
    """

    def __init__(
        self,
        client: BinanceClient,
        *,
        symbol: str = "BTCUSDT",
        poll_seconds: float = 2.0,
        max_backfill_bars: int = 10_000,
    ) -> None:
        self.client = client
        self.symbol = symbol.upper()
        self.poll_seconds = float(poll_seconds)
        self.max_backfill_bars = int(max_backfill_bars)
        self._closed = threading.Event()
        self._last_emitted: Optional[pd.Timestamp] = None

    # -- DataSource ------------------------------------------------------- #

    def bootstrap(self, n_rows: int) -> pd.DataFrame:
        """Trailing ``n_rows`` CLOSED bars ending at the last complete minute."""
        end_open_ms = (int(time.time() * 1000) // _BAR_MS) * _BAR_MS  # current (open) bar
        frames: list[pd.DataFrame] = []
        remaining = int(n_rows)
        cursor_end = end_open_ms  # exclusive: fetch bars with open_time < cursor_end
        while remaining > 0:
            batch = min(1000, remaining)
            start_ms = cursor_end - batch * _BAR_MS
            rows = self.client.request(
                "GET", "/api/v3/klines",
                {
                    "symbol": self.symbol, "interval": "1m",
                    "startTime": start_ms, "endTime": cursor_end - 1,
                    "limit": 1000,
                },
            )
            frame = klines_to_frame(rows if isinstance(rows, list) else [])
            if frame.empty:
                break  # history exhausted (young market/testnet)
            frames.append(frame)
            remaining -= len(frame)
            cursor_end = int(frame.index[0].value // 1_000_000) - _BAR_MS + _BAR_MS
            cursor_end = int((frame.index[0] - pd.Timedelta(minutes=1)).value // 1_000_000)
            if len(frame) < batch:
                break
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames[::-1]).sort_index()
        out = out[~out.index.duplicated(keep="last")].tail(n_rows)
        self._last_emitted = out.index[-1]
        logger.info(
            "BinanceKlineSource.bootstrap: %d bars [%s .. %s]",
            len(out), out.index[0], out.index[-1],
        )
        return out

    def close(self) -> None:
        self._closed.set()

    def stream(self) -> Iterator[MarketUpdate]:
        while not self._closed.is_set():
            frame = self._fetch_since(self._last_emitted)
            for ts, row in frame.iterrows():
                if self._closed.is_set():
                    return
                self._last_emitted = ts
                yield _frame_to_update(row, ts)
            self._closed.wait(self.poll_seconds)

    # -- internals ---------------------------------------------------------- #

    def _fetch_since(self, last_ts: Optional[pd.Timestamp]) -> pd.DataFrame:
        """All CLOSED bars strictly newer than ``last_ts``, in order.

        One request covers both steady-state polling (0-1 new bars) and
        gap catch-up after a stall (up to ``max_backfill_bars``).
        """
        now_ms = int(time.time() * 1000)
        current_open_ms = (now_ms // _BAR_MS) * _BAR_MS
        params: dict = {"symbol": self.symbol, "interval": "1m", "limit": 1000}
        if last_ts is not None:
            start_open_ms = int(last_ts.value // 1_000_000)  # = next bar's open time
            if start_open_ms >= current_open_ms:
                return pd.DataFrame()  # nothing closed since last emission
            n_missing = (current_open_ms - start_open_ms) // _BAR_MS
            if n_missing > self.max_backfill_bars:
                raise ExchangeError(
                    f"stream stalled for {n_missing} bars (> max_backfill_bars="
                    f"{self.max_backfill_bars}) — refusing silent catch-up; "
                    "restart with a fresh bootstrap"
                )
            params["startTime"] = start_open_ms
        rows = self.client.request("GET", "/api/v3/klines", params)
        frame = klines_to_frame(rows if isinstance(rows, list) else [])
        if frame.empty:
            return frame
        # Drop the still-open candle (its close time is in the future) and
        # anything already emitted.
        frame = frame[frame.index <= pd.Timestamp(current_open_ms, unit="ms", tz="UTC")]
        if last_ts is not None:
            frame = frame[frame.index > last_ts]
        return frame

