"""Binance adapter: hermetic tests (fake transport, frozen clock, no network).

Covers the signed client's retry/rejection policy, the kline timestamp
convention, bootstrap pagination, closed-candle streaming with gap
catch-up, exchange-filter arithmetic, dry-run and live order paths,
idempotent duplicate handling, reconciliation, and the alert sinks.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import src.data.binance as bz
from src.engine.alerts import NullAlerter, WebhookAlerter
from src.data.binance import BinanceClient, BinanceKlineSource, klines_to_frame
from src.engine.binance import BinanceBroker, ReconcileReport, SymbolFilters
from src.engine.errors import ConfigError, ExchangeError, ExecutionError
from src.strategy.inventory import Position, close_position

pytestmark = pytest.mark.engine

MIN_MS = 60_000


def _kline(open_ms: int, price: float = 100.0) -> list:
    return [
        open_ms, f"{price}", f"{price + 1}", f"{price - 1}", f"{price}",
        "5.0", open_ms + MIN_MS - 1, "500.0", 42, "2.5", "250.0", "0",
    ]


class FakeTransport:
    """Scripted (method, path) -> responses; records every request."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list] = {}
        self.calls: list[tuple[str, str, dict]] = []

    def route(self, method: str, path: str, *responses) -> None:
        self.routes.setdefault((method, path), []).extend(responses)

    def __call__(self, method, url, params, headers, timeout):
        path = url.split("binance.vision")[-1].split("binance.com")[-1]
        self.calls.append((method, path, dict(params)))
        queue = self.routes.get((method, path))
        if not queue:
            raise AssertionError(f"unrouted request: {method} {path}")
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, Exception):
            raise response
        return response


def _client(transport, **kw) -> BinanceClient:
    kw.setdefault("api_key", "k")
    kw.setdefault("api_secret", "s")
    kw.setdefault("max_retries", 3)
    return BinanceClient(testnet=True, transport=transport, **kw)


class TestClient:
    def test_signed_request_carries_signature_and_key(self, monkeypatch):
        monkeypatch.setattr(bz.time, "time", lambda: 1_700_000_000.0)
        ft = FakeTransport()
        ft.route("GET", "/api/v3/account", (200, {"balances": []}))
        _client(ft).request("GET", "/api/v3/account", {}, signed=True)
        method, path, params = ft.calls[0]
        assert params["timestamp"] == 1_700_000_000_000
        assert "signature" in params and len(params["signature"]) == 64

    def test_retries_5xx_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(bz.time, "sleep", lambda s: None)
        ft = FakeTransport()
        ft.route("GET", "/api/v3/ping", (500, "boom"), (200, {}), (200, {}))
        assert _client(ft).request("GET", "/api/v3/ping") == {}
        assert len(ft.calls) == 2

    def test_4xx_raises_immediately_with_payload(self):
        ft = FakeTransport()
        ft.route("POST", "/api/v3/order", (400, {"code": -1102, "msg": "bad"}))
        with pytest.raises(ExchangeError, match="-1102"):
            _client(ft).request("POST", "/api/v3/order", {}, signed=True)
        assert len(ft.calls) == 1

    def test_exhausted_retries_raise(self, monkeypatch):
        monkeypatch.setattr(bz.time, "sleep", lambda s: None)
        ft = FakeTransport()
        ft.route("GET", "/api/v3/ping", (503, "down"))
        with pytest.raises(ExchangeError, match="after 3 attempts"):
            _client(ft).request("GET", "/api/v3/ping")

    def test_signed_without_credentials_is_config_error(self):
        client = BinanceClient(
            testnet=True, api_key="", api_secret="", transport=FakeTransport()
        )
        with pytest.raises(ConfigError, match="credentials"):
            client.request("GET", "/api/v3/account", {}, signed=True)


class TestKlinesFrame:
    def test_timestamp_is_bar_complete_utc(self):
        open_ms = 1_700_000_000_000 - (1_700_000_000_000 % MIN_MS)
        frame = klines_to_frame([_kline(open_ms)])
        assert str(frame.index.tz) == "UTC"
        assert frame.index[0] == pd.Timestamp(open_ms + MIN_MS, unit="ms", tz="UTC")
        assert list(frame.columns) == [
            "open", "high", "low", "close", "volume", "quote_volume",
            "num_trades", "taker_buy_base", "taker_buy_quote",
        ]


class TestKlineSource:
    BASE_MS = (1_700_000_000_000 // MIN_MS) * MIN_MS

    def _source(self, ft, **kw) -> BinanceKlineSource:
        return BinanceKlineSource(_client(ft), symbol="BTCUSDT", **kw)

    def test_bootstrap_pages_and_orders(self, monkeypatch):
        now_s = (self.BASE_MS + 30_000) / 1000.0  # mid-bar
        monkeypatch.setattr(bz.time, "time", lambda: now_s)
        ft = FakeTransport()
        # 3 closed bars immediately before the current open bar.
        rows = [_kline(self.BASE_MS - i * MIN_MS) for i in (3, 2, 1)]
        ft.route("GET", "/api/v3/klines", (200, rows))
        frame = self._source(ft).bootstrap(3)
        assert len(frame) == 3
        assert frame.index.is_monotonic_increasing
        assert frame.index[-1] == pd.Timestamp(self.BASE_MS, unit="ms", tz="UTC")

    def test_stream_emits_only_closed_bars_in_order(self, monkeypatch):
        now_s = (self.BASE_MS + 30_000) / 1000.0
        monkeypatch.setattr(bz.time, "time", lambda: now_s)
        ft = FakeTransport()
        # Poll returns two closed + the still-open candle.
        ft.route(
            "GET", "/api/v3/klines",
            (200, [
                _kline(self.BASE_MS - 2 * MIN_MS),
                _kline(self.BASE_MS - MIN_MS),
                _kline(self.BASE_MS),  # open candle — must be excluded
            ]),
        )
        source = self._source(ft, poll_seconds=0.001)
        stream = source.stream()
        first = next(stream)
        second = next(stream)
        source.close()
        assert first.bar.ts == pd.Timestamp(self.BASE_MS - MIN_MS, unit="ms", tz="UTC")
        assert second.bar.ts == pd.Timestamp(self.BASE_MS, unit="ms", tz="UTC")
        with pytest.raises(StopIteration):
            next(stream)

    def test_gap_catch_up_requests_from_last_emitted(self, monkeypatch):
        now_s = (self.BASE_MS + 30_000) / 1000.0
        monkeypatch.setattr(bz.time, "time", lambda: now_s)
        ft = FakeTransport()
        ft.route(
            "GET", "/api/v3/klines",
            (200, [_kline(self.BASE_MS - i * MIN_MS) for i in (5, 4, 3, 2, 1)]),
        )
        source = self._source(ft, poll_seconds=0.001)
        source._last_emitted = pd.Timestamp(
            self.BASE_MS - 5 * MIN_MS, unit="ms", tz="UTC"
        )
        frame = source._fetch_since(source._last_emitted)
        assert len(frame) == 5  # all five missed bars, none duplicated
        _, _, params = ft.calls[0]
        assert params["startTime"] == self.BASE_MS - 5 * MIN_MS

    def test_stall_beyond_backfill_limit_refuses(self, monkeypatch):
        now_s = (self.BASE_MS + 30_000) / 1000.0
        monkeypatch.setattr(bz.time, "time", lambda: now_s)
        source = self._source(FakeTransport(), max_backfill_bars=10)
        stale = pd.Timestamp(self.BASE_MS - 50 * MIN_MS, unit="ms", tz="UTC")
        with pytest.raises(ExchangeError, match="stalled"):
            source._fetch_since(stale)


FILTERS = SymbolFilters(
    step_size=Decimal("0.00001"),
    min_qty=Decimal("0.00001"),
    min_notional=Decimal("5"),
)


def _position(size: float = 0.02, price: float = 50_000.0) -> Position:
    return Position(
        k_entry=7, ts_entry=pd.Timestamp("2025-06-01 12:00", tz="UTC"),
        side=1, size=size, entry_price=price, tp_price=price * 1.0025,
        sl_price=None, expiry_k=10_000,
    )


class TestSymbolFilters:
    def test_parse_from_exchange_info(self):
        payload = {"symbols": [{
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.00001000",
                 "minQty": "0.00001000", "maxQty": "9000"},
                {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
            ],
        }]}
        f = SymbolFilters.from_exchange_info(payload, "BTCUSDT")
        assert f.step_size == Decimal("0.00001000")
        assert f.min_notional == Decimal("5.00000000")

    def test_snap_rounds_down_to_grid(self):
        assert FILTERS.snap_qty(Decimal("0.000123456")) == Decimal("0.00012")

    def test_unknown_symbol_raises(self):
        with pytest.raises(ExchangeError, match="not present"):
            SymbolFilters.from_exchange_info({"symbols": []}, "NOPEUSDT")


class TestBrokerDryRun:
    def _broker(self, **kw) -> BinanceBroker:
        return BinanceBroker(
            client=_client(FakeTransport()), symbol="BTCUSDT",
            trade_capital=10_000.0, dry_run=True, filters=FILTERS, **kw,
        )

    def test_entry_fills_at_assumed_price_and_records_qty(self):
        broker = self._broker()
        pos = _position()
        order, fill = broker.execute_entry(pos)
        assert order.side == +1 and fill.price == pos.entry_price
        key = broker._position_key(pos.k_entry, pos.ts_entry)
        assert broker._qty_by_position[key] > 0

    def test_close_pops_recorded_qty(self):
        broker = self._broker()
        pos = _position()
        broker.execute_entry(pos)
        closed = close_position(
            pos, k_exit=9, ts_exit=pos.ts_entry + pd.Timedelta(minutes=2),
            exit_price=pos.entry_price * 1.001, exit_reason="tp_market",
        )
        order, fill = broker.execute_close(closed)
        assert order.side == -1 and fill.price == closed.exit_price
        assert not broker._qty_by_position

    def test_sub_notional_entry_rejected(self):
        broker = self._broker()
        with pytest.raises(ExecutionError, match="notional"):
            broker.execute_entry(_position(size=0.0001))  # 1 USDT < min 5

    def test_reconcile_dry_run_is_trivially_ok(self):
        broker = self._broker()
        pos = _position()
        broker.execute_entry(pos)
        report = broker.reconcile([pos])
        assert isinstance(report, ReconcileReport) and report.ok

    def test_live_mode_without_credentials_refused(self):
        client = BinanceClient(
            testnet=True, api_key="", api_secret="", transport=FakeTransport()
        )
        with pytest.raises(ConfigError, match="credentials"):
            BinanceBroker(
                client=client, trade_capital=1000.0, dry_run=False, filters=FILTERS
            )


class TestBrokerLive:
    def _broker(self, ft) -> BinanceBroker:
        return BinanceBroker(
            client=_client(ft), symbol="BTCUSDT", trade_capital=10_000.0,
            dry_run=False, filters=FILTERS, session_tag="t1",
        )

    def test_entry_sends_quote_order_and_uses_actual_fill(self):
        ft = FakeTransport()
        ft.route("POST", "/api/v3/order", (200, {
            "executedQty": "0.00400000", "cummulativeQuoteQty": "200.20",
            "status": "FILLED",
        }))
        broker = self._broker(ft)
        pos = _position(size=0.02, price=50_000.0)
        order, fill = broker.execute_entry(pos)
        _, _, params = ft.calls[-1]
        assert params["side"] == "BUY" and params["type"] == "MARKET"
        assert params["quoteOrderQty"] == "200"  # 0.02 * 10,000
        assert params["newClientOrderId"] == "bc-t1-1"
        assert fill.price == pytest.approx(200.20 / 0.004)

    def test_close_sells_recorded_quantity_snapped(self):
        ft = FakeTransport()
        ft.route("POST", "/api/v3/order",
                 (200, {"executedQty": "0.00400000",
                        "cummulativeQuoteQty": "200.20", "status": "FILLED"}),
                 (200, {"executedQty": "0.00400000",
                        "cummulativeQuoteQty": "201.00", "status": "FILLED"}))
        broker = self._broker(ft)
        pos = _position()
        broker.execute_entry(pos)
        closed = close_position(
            pos, k_exit=9, ts_exit=pos.ts_entry + pd.Timedelta(minutes=2),
            exit_price=50_250.0, exit_reason="tp_market",
        )
        broker.execute_close(closed)
        _, _, params = ft.calls[-1]
        assert params["side"] == "SELL"
        assert params["quantity"] == "0.004"

    def test_duplicate_client_id_fetches_existing_order(self, monkeypatch):
        monkeypatch.setattr(bz.time, "sleep", lambda s: None)
        ft = FakeTransport()
        ft.route("POST", "/api/v3/order",
                 (400, {"code": -2010, "msg": "Duplicate order sent."}))
        ft.route("GET", "/api/v3/order", (200, {
            "executedQty": "0.00400000", "cummulativeQuoteQty": "200.20",
            "status": "FILLED",
        }))
        broker = self._broker(ft)
        order, fill = broker.execute_entry(_position())
        assert fill.price == pytest.approx(200.20 / 0.004)

    def test_persistent_rejection_becomes_execution_error(self, monkeypatch):
        monkeypatch.setattr(bz.time, "sleep", lambda s: None)
        ft = FakeTransport()
        ft.route("POST", "/api/v3/order",
                 (400, {"code": -2019, "msg": "Margin is insufficient."}))
        broker = self._broker(ft)
        with pytest.raises(ExecutionError, match="failed after"):
            broker.execute_entry(_position())

    def test_reconcile_compares_exchange_balance(self):
        ft = FakeTransport()
        ft.route("POST", "/api/v3/order", (200, {
            "executedQty": "0.00400000", "cummulativeQuoteQty": "200.20",
            "status": "FILLED",
        }))
        # Both balance snapshots routed up front: matched, then diverged.
        ft.route(
            "GET", "/api/v3/account",
            (200, {"balances": [{"asset": "BTC", "free": "0.004", "locked": "0"}]}),
            (200, {"balances": [{"asset": "BTC", "free": "0.001", "locked": "0"}]}),
        )
        broker = self._broker(ft)
        pos = _position()
        broker.execute_entry(pos)
        report = broker.reconcile([pos])
        assert report.ok
        report2 = broker.reconcile([pos])
        assert not report2.ok


class TestAlerts:
    def test_webhook_posts_slack_compatible_payload(self):
        sent = []

        def transport(url, payload, timeout):
            sent.append((url, payload))
            return 200

        WebhookAlerter("https://hooks.example/x", transport=transport).send(
            level="error", event="halt", message="max_drawdown breached", dd=0.05
        )
        url, payload = sent[0]
        assert payload["text"] == "ERROR engine/halt: max_drawdown breached"
        assert payload["context"] == {"dd": 0.05}

    def test_delivery_failure_never_raises(self):
        def transport(url, payload, timeout):
            raise OSError("network down")

        WebhookAlerter("https://hooks.example/x", transport=transport).send(
            level="error", event="halt", message="x"
        )  # no exception

    def test_null_alerter_is_log_only(self, caplog):
        import logging

        with caplog.at_level(logging.ERROR, logger="src.engine.alerts"):
            NullAlerter().send(level="error", event="halt", message="boom")
        assert any("ALERT" in r.message for r in caplog.records)
