"""Binance spot order execution — the engine's exchange BROKER.

Implements the engine's :class:`~src.engine.execution.Broker` protocol
with real MARKET orders. This is EXECUTION, not data acquisition: the
market-data adapter lives in ``src.data.binance`` and feeds the engine
through the feed store; this broker independently connects to Binance to
place the orders the strategy decides. The shared signed REST client
(:class:`~src.data.binance.BinanceClient`) is imported from there.

Safety posture (unchanged): ``dry_run=True`` default (orders built +
validated against real filters, not sent); ``testnet`` first; live
requires ``dry_run=False`` AND credentials. Transport is injectable, so
every path is hermetically testable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Optional

import pandas as pd

from src.data.binance import BinanceClient
from src.engine.domain import Fill, Order, OrderKind
from src.engine.errors import ConfigError, ExchangeError, ExecutionError
from src.strategy.inventory import ClosedPosition, Position

logger = logging.getLogger("src.engine.binance")

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
