"""Binance spot adapter: market-data source + order broker.

The exchange-facing implementations of the engine's two ports
(docs/ENGINE.md §2): :class:`BinanceKlineSource` feeds closed 1-minute
bars (REST backfill for the buffer, then closed-candle polling with
automatic gap catch-up), and :class:`BinanceBroker` turns the strategy's
intents into real orders behind the same :class:`~src.engine.execution.Broker`
protocol the paper broker implements.

Safety posture — deliberate, in this order:

1. ``dry_run=True`` is the DEFAULT: orders are built, validated against
   the exchange's real filters, logged — and not sent.
2. ``testnet=True`` targets https://testnet.binance.vision, where the
   full order path runs against play money. Run there first, always.
3. Live trading requires BOTH ``dry_run=False`` and API credentials
   (``BINANCE_API_KEY`` / ``BINANCE_API_SECRET`` env vars by default —
   keys never live in config files).

Execution semantics match the researched strategy exactly: every intent
is a MARKET order decided at bar close (the production P1+P3 spec has no
resting orders). Entries buy by QUOTE amount (``size × trade_capital``);
closes sell the recorded base quantity snapped to the lot filter. A
failed execution after bounded retries raises
:class:`~src.engine.errors.ExecutionError` — the engine halts new entries
and alerts, because a ledger/exchange divergence needs an operator (see
docs/PRODUCTION.md).

Transport is injectable, so every code path here is hermetically
testable; no test talks to the network.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Callable, Iterator, Optional
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from src.engine.domain import Bar, DerivSnapshot, Fill, MarketUpdate, Order, OrderKind
from src.engine.errors import ConfigError, ExchangeError, ExecutionError
from src.strategy.inventory import ClosedPosition, Position

logger = logging.getLogger("src.engine.binance")

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


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolFilters:
    """The exchange trading rules the broker must respect."""

    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal

    def snap_qty(self, qty: Decimal) -> Decimal:
        """Round DOWN to the lot grid (never round an order up)."""
        if self.step_size <= 0:
            return qty
        return (qty / self.step_size).to_integral_value(ROUND_DOWN) * self.step_size

    @staticmethod
    def from_exchange_info(payload: dict, symbol: str) -> "SymbolFilters":
        for entry in payload.get("symbols", []):
            if entry.get("symbol") == symbol:
                step = min_qty = Decimal("0")
                min_notional = Decimal("0")
                for f in entry.get("filters", []):
                    if f.get("filterType") == "LOT_SIZE":
                        step = Decimal(str(f["stepSize"]))
                        min_qty = Decimal(str(f["minQty"]))
                    elif f.get("filterType") in ("NOTIONAL", "MIN_NOTIONAL"):
                        min_notional = Decimal(str(f.get("minNotional", "0")))
                return SymbolFilters(
                    step_size=step, min_qty=min_qty, min_notional=min_notional
                )
        raise ExchangeError(f"symbol {symbol!r} not present in exchangeInfo")


@dataclass(frozen=True)
class ReconcileReport:
    """Exchange-vs-ledger position comparison at startup."""

    expected_base_qty: float
    exchange_base_qty: float
    tolerance: float
    details: str = ""

    @property
    def ok(self) -> bool:
        return abs(self.expected_base_qty - self.exchange_base_qty) <= self.tolerance


@dataclass
class BinanceBroker:
    """Real order execution behind the engine's :class:`Broker` protocol.

    ``trade_capital`` (quote units, e.g. USDT) maps the strategy's
    fractional sizes to money: an entry of ``size=0.02`` buys
    ``0.02 × trade_capital`` worth at market. Closes sell the base
    quantity recorded at entry, snapped down to the lot grid.

    ``dry_run=True`` (default) builds and validates every order against
    the real exchange filters but does not send it — fills are reported
    at the strategy's assumed price, which keeps the ledger identical to
    paper trading while exercising the full order path.
    """

    client: BinanceClient
    symbol: str = "BTCUSDT"
    trade_capital: float = 10_000.0
    dry_run: bool = True
    next_order_id: int = 1
    max_retries: int = 3
    filters: Optional[SymbolFilters] = None
    session_tag: str = ""
    _qty_by_position: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper()
        if self.trade_capital <= 0:
            raise ConfigError(f"trade_capital must be > 0, got {self.trade_capital}")
        if self.next_order_id < 1:
            raise ConfigError(f"next_order_id must be >= 1, got {self.next_order_id}")
        if not self.dry_run and not self.client.has_credentials:
            raise ConfigError(
                "live trading (dry_run=False) requires API credentials — set "
                "BINANCE_API_KEY / BINANCE_API_SECRET"
            )
        if self.filters is None:
            payload = self.client.request(
                "GET", "/api/v3/exchangeInfo", {"symbol": self.symbol}
            )
            self.filters = SymbolFilters.from_exchange_info(payload, self.symbol)
        if not self.session_tag:
            self.session_tag = f"{int(time.time())}"
        logger.info(
            "BinanceBroker ready: %s %s capital=%.2f filters(step=%s minQty=%s "
            "minNotional=%s) dry_run=%s",
            "TESTNET" if self.client.testnet else "LIVE",
            self.symbol, self.trade_capital,
            self.filters.step_size, self.filters.min_qty,
            self.filters.min_notional, self.dry_run,
        )

    # -- Broker protocol ------------------------------------------------------ #

    def execute_entry(self, position: Position, note: str = "") -> tuple[Order, Fill]:
        oid = self._order_id()
        client_id = f"bc-{self.session_tag}-{oid}"
        quote_amount = Decimal(str(position.size)) * Decimal(str(self.trade_capital))
        if self.filters and quote_amount < self.filters.min_notional:
            raise ExecutionError(
                f"entry notional {quote_amount} below the exchange minimum "
                f"{self.filters.min_notional} — raise trade_capital or lot size"
            )
        key = self._position_key(position.k_entry, position.ts_entry)
        if self.dry_run:
            qty = Decimal(str(float(quote_amount) / position.entry_price))
            self._qty_by_position[key] = self.filters.snap_qty(qty) if self.filters else qty
            return self._paper_pair(
                oid, position.ts_entry, +1, position.size, position.entry_price,
                note or f"entry dry_run {client_id} quote={quote_amount}",
            )
        result = self._send_order(
            side="BUY",
            client_id=client_id,
            params={"quoteOrderQty": f"{quote_amount.normalize():f}"},
        )
        executed_qty = Decimal(str(result.get("executedQty", "0")))
        quote_spent = Decimal(str(result.get("cummulativeQuoteQty", "0")))
        if executed_qty <= 0:
            raise ExecutionError(
                f"entry order {client_id} reported zero executed quantity: {result}"
            )
        self._qty_by_position[key] = executed_qty
        fill_price = float(quote_spent / executed_qty)
        order = Order(
            order_id=oid, ts=position.ts_entry, side=+1, size=position.size,
            kind=OrderKind.MARKET,
            note=note or f"entry {client_id} qty={executed_qty} quote={quote_spent}",
        )
        fill = Fill(
            order_id=oid, ts=position.ts_entry, price=fill_price,
            size=position.size, side=+1,
        )
        return order, fill

    def execute_close(self, closed: ClosedPosition, note: str = "") -> tuple[Order, Fill]:
        oid = self._order_id()
        client_id = f"bc-{self.session_tag}-{oid}"
        key = self._position_key(closed.k_entry, closed.ts_entry)
        qty = self._qty_by_position.pop(key, None)
        if qty is None:
            # Resumed session: entry qty was recorded by a previous process.
            # Reconstruct from the strategy's own accounting.
            qty = Decimal(str(closed.size * self.trade_capital / closed.entry_price))
            logger.warning(
                "close %s: no recorded entry quantity for %s — reconstructed "
                "%s from ledger size (resume path; verify with reconcile())",
                client_id, key, qty,
            )
        qty = self.filters.snap_qty(qty) if self.filters else qty
        if self.dry_run:
            return self._paper_pair(
                oid, closed.ts_exit, -1, closed.size, closed.exit_price,
                note or f"{closed.exit_reason} dry_run {client_id} qty={qty}",
            )
        if self.filters and qty < self.filters.min_qty:
            raise ExecutionError(
                f"close quantity {qty} below exchange minQty "
                f"{self.filters.min_qty} for {key}"
            )
        result = self._send_order(
            side="SELL", client_id=client_id,
            params={"quantity": f"{qty.normalize():f}"},
        )
        executed_qty = Decimal(str(result.get("executedQty", "0")))
        quote_got = Decimal(str(result.get("cummulativeQuoteQty", "0")))
        fill_price = (
            float(quote_got / executed_qty) if executed_qty > 0 else closed.exit_price
        )
        order = Order(
            order_id=oid, ts=closed.ts_exit, side=-1, size=closed.size,
            kind=OrderKind.MARKET,
            note=note or f"{closed.exit_reason} {client_id} qty={executed_qty}",
        )
        fill = Fill(
            order_id=oid, ts=closed.ts_exit, price=fill_price,
            size=closed.size, side=-1,
        )
        return order, fill

    # -- operations ------------------------------------------------------------ #

    def reconcile(
        self, open_positions: list[Position], *, tolerance_frac: float = 0.02
    ) -> ReconcileReport:
        """Compare the ledger's open exposure to the exchange balance.

        Expected base = Σ recorded entry quantities (falling back to the
        ledger-size reconstruction); exchange base = free + locked balance
        of the base asset. In dry_run the exchange side is assumed equal
        (there is nothing on the exchange to compare against).
        """
        expected = Decimal("0")
        for pos in open_positions:
            key = self._position_key(pos.k_entry, pos.ts_entry)
            qty = self._qty_by_position.get(key)
            if qty is None:
                qty = Decimal(str(pos.size * self.trade_capital / pos.entry_price))
            expected += qty
        if self.dry_run:
            return ReconcileReport(
                expected_base_qty=float(expected),
                exchange_base_qty=float(expected),
                tolerance=float(expected) * tolerance_frac,
                details="dry_run: exchange side assumed equal",
            )
        base_asset = self.symbol.replace("USDT", "").replace("BUSD", "")
        account = self.client.request("GET", "/api/v3/account", {}, signed=True)
        exchange = Decimal("0")
        for bal in account.get("balances", []):
            if bal.get("asset") == base_asset:
                exchange = Decimal(str(bal["free"])) + Decimal(str(bal["locked"]))
                break
        tol = max(float(expected) * tolerance_frac, float(self.filters.step_size or 0))
        return ReconcileReport(
            expected_base_qty=float(expected),
            exchange_base_qty=float(exchange),
            tolerance=tol,
            details=f"base_asset={base_asset}",
        )

    # -- internals ---------------------------------------------------------- #

    @staticmethod
    def _position_key(k_entry: int, ts_entry) -> tuple:
        return (int(k_entry), int(pd.Timestamp(ts_entry).value))

    def _order_id(self) -> int:
        oid = self.next_order_id
        self.next_order_id += 1
        return oid

    def _paper_pair(
        self, oid: int, ts, side: int, size: float, price: float, note: str
    ) -> tuple[Order, Fill]:
        order = Order(
            order_id=oid, ts=ts, side=side, size=size,
            kind=OrderKind.MARKET, note=note,
        )
        fill = Fill(order_id=oid, ts=ts, price=price, size=size, side=side)
        return order, fill

    def _send_order(self, *, side: str, client_id: str, params: dict) -> dict:
        base = {
            "symbol": self.symbol,
            "side": side,
            "type": "MARKET",
            "newClientOrderId": client_id,
            "newOrderRespType": "FULL",
            **params,
        }
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.client.request("POST", "/api/v3/order", base, signed=True)
                if isinstance(result, dict):
                    return result
                raise ExchangeError(f"unexpected order response: {result!r}")
            except ExchangeError as exc:
                last = exc
                msg = str(exc)
                # Duplicate client order id => the previous attempt DID land;
                # fetch it instead of double-sending.
                if "-2010" in msg and "Duplicate" in msg or "duplicate" in msg.lower():
                    fetched = self.client.request(
                        "GET", "/api/v3/order",
                        {"symbol": self.symbol, "origClientOrderId": client_id},
                        signed=True,
                    )
                    if isinstance(fetched, dict):
                        return fetched
                logger.warning(
                    "order %s attempt %d/%d failed: %s",
                    client_id, attempt, self.max_retries, exc,
                )
        raise ExecutionError(
            f"order {client_id} ({side} {self.symbol}) failed after "
            f"{self.max_retries} attempts: {last}"
        )
