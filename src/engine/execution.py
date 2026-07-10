"""Execution port: order/fill bookkeeping and the real-broker seam.

The strategy layer (:class:`~src.engine.strategy.LiveTrader`) decides
*what* happens — entries at boundary close, exits per the spec's policy —
using the exact fill assumptions of the offline simulator (entry at
``close[t]``; ``"tp"`` at the static TP price; everything else at bar
close). A :class:`Broker` turns those intents into an order/fill audit
trail.

:class:`PaperBroker` accepts every intent at its assumed price — that IS
the paper-trading semantics, and it keeps live replays ledger-identical
to the research backtest. A real exchange adapter implements the same
protocol, submits real orders, and reports *actual* fills; slippage then
shows up as (assumed − actual) in the fills table rather than silently
biasing the strategy's own accounting.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import pandas as pd

from src.engine.domain import Fill, Order, OrderKind, Trade
from src.strategy.inventory import ClosedPosition, Position


@runtime_checkable
class Broker(Protocol):
    """Public execution interface (paper or real)."""

    def execute_entry(self, position: Position, note: str = "") -> tuple[Order, Fill]:
        """Execute a market entry the strategy just opened (fills at the
        position's recorded entry price in paper mode)."""
        ...

    def execute_close(self, closed: ClosedPosition, note: str = "") -> tuple[Order, Fill]:
        """Execute the closing order for a position the strategy resolved."""
        ...


class PaperBroker:
    """Fills every intent at the strategy's assumed price. Zero slippage,
    the researched cost model applies at the ledger layer (not here)."""

    def __init__(self) -> None:
        self._next_order_id = 1

    def _order_id(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        return oid

    def execute_entry(self, position: Position, note: str = "") -> tuple[Order, Fill]:
        oid = self._order_id()
        order = Order(
            order_id=oid, ts=position.ts_entry, side=+1, size=position.size,
            kind=OrderKind.MARKET, note=note or "entry",
        )
        fill = Fill(
            order_id=oid, ts=position.ts_entry, price=position.entry_price,
            size=position.size, side=+1,
        )
        return order, fill

    def execute_close(self, closed: ClosedPosition, note: str = "") -> tuple[Order, Fill]:
        oid = self._order_id()
        order = Order(
            order_id=oid, ts=closed.ts_exit, side=-1, size=closed.size,
            kind=OrderKind.MARKET, note=note or closed.exit_reason,
        )
        fill = Fill(
            order_id=oid, ts=closed.ts_exit, price=closed.exit_price,
            size=closed.size, side=-1,
        )
        return order, fill


def trade_from_closed(
    closed: ClosedPosition,
    *,
    trade_id: int,
    cost_per_trade: float,
    p_at_entry: float,
    model_version: str,
) -> Trade:
    """Ledger row from an inventory close.

    ``p_at_entry`` here is the *true* decision-time probability the engine
    recorded when the position opened (the offline simulator's ledger
    stores the exit-boundary p under that name — a research-code quirk;
    parity tests compare economic columns only).
    """
    return Trade(
        trade_id=trade_id,
        k_entry=int(closed.k_entry),
        ts_entry=pd.Timestamp(closed.ts_entry),
        entry_price=float(closed.entry_price),
        size=float(closed.size),
        k_exit=int(closed.k_exit),
        ts_exit=pd.Timestamp(closed.ts_exit),
        exit_price=float(closed.exit_price),
        exit_reason=str(closed.exit_reason),
        gross_log_return=float(closed.gross_log_return),
        net_log_return=float(closed.net_log_return(cost_per_trade)),
        weighted_net_log_return=float(closed.weighted_net_log_return(cost_per_trade)),
        p_at_entry=float(p_at_entry) if not math.isnan(p_at_entry) else float("nan"),
        model_version=model_version,
    )
