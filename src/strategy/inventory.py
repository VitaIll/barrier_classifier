"""Position + Portfolio data model.

A ``Position`` is the unit of commitment: open at a boundary close,
exit on TP / expiry / SL / bulk-close. ``Portfolio`` is a small bookkeeping
shell — list of open positions, list of closed positions, helpers to
mark-to-market and unwind in bulk.

Returns are stored in **log-return space** throughout (so accumulation is
additive). All P&L numbers are pre-cost; the ``cost_per_trade`` is applied
at close time via ``ClosedPosition.net_log_return(cost)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import pandas as pd


ExitReason = str  # "tp" | "sl" | "expiry" | "bulk_regime" | "bulk_unc" | "bulk_cluster_loss" | "bulk_drift"


@dataclass(frozen=True)
class Position:
    """Open position metadata. Frozen — closing produces a ``ClosedPosition``."""

    k_entry: int
    ts_entry: pd.Timestamp
    side: int                  # +1 long; -1 reserved for future short
    size: float                # sizing factor in [0, max_size]
    entry_price: float
    tp_price: float            # entry_price * exp(+phi * side)
    sl_price: Optional[float]  # None = no stop-loss (v1 default)
    expiry_k: int              # boundary index at which the position times out

    def __post_init__(self) -> None:
        # NOTE every price/size check pairs a finiteness test with the sign
        # test: ``NaN <= 0`` is False, so a bare comparison silently accepts
        # NaN — a NaN tp_price produces a position whose take-profit can
        # never trigger (``high >= NaN`` is always False) and that sits
        # corrupt until expiry. Fail at construction instead.
        if self.side not in (-1, 1):
            raise ValueError(f"side must be -1 or +1; got {self.side}")
        if not math.isfinite(self.size) or self.size < 0:
            raise ValueError(f"size must be finite and >= 0; got {self.size}")
        if not math.isfinite(self.entry_price) or self.entry_price <= 0:
            raise ValueError(f"entry_price must be finite and > 0; got {self.entry_price}")
        if not math.isfinite(self.tp_price) or self.tp_price <= 0:
            raise ValueError(f"tp_price must be finite and > 0; got {self.tp_price}")
        if self.sl_price is not None and (
            not math.isfinite(self.sl_price) or self.sl_price <= 0
        ):
            raise ValueError(f"sl_price must be finite and > 0; got {self.sl_price}")
        if self.expiry_k < self.k_entry:
            raise ValueError(f"expiry_k {self.expiry_k} must be >= k_entry {self.k_entry}")

    def mtm_log_return(self, current_price: float) -> float:
        """Unrealized log-return at ``current_price`` (pre-cost, sized by ``side``)."""
        if not math.isfinite(current_price) or current_price <= 0:
            raise ValueError(f"current_price must be finite and > 0; got {current_price}")
        return float(self.side) * math.log(current_price / self.entry_price)


@dataclass(frozen=True)
class ClosedPosition:
    """Closed position carrying both entry and exit data."""

    k_entry: int
    ts_entry: pd.Timestamp
    side: int
    size: float
    entry_price: float
    tp_price: float
    sl_price: Optional[float]
    expiry_k: int
    k_exit: int
    ts_exit: pd.Timestamp
    exit_price: float
    exit_reason: ExitReason
    gross_log_return: float    # side * log(exit/entry), unweighted by size
    # ----- extra context (filled by the simulator if available) ---------------
    p_at_entry: float = float("nan")
    knowledge_unc_at_entry: float = float("nan")
    regime_quantile_at_entry: float = float("nan")

    def net_log_return(self, cost_per_trade: float) -> float:
        """Per-unit-size net return. Multiply by ``self.size`` for portfolio P&L."""
        return float(self.gross_log_return - float(cost_per_trade))

    def weighted_net_log_return(self, cost_per_trade: float) -> float:
        """Net return scaled by position size — what the equity curve sees."""
        return float(self.size) * self.net_log_return(cost_per_trade)


def close_position(
    position: Position,
    *,
    k_exit: int,
    ts_exit: pd.Timestamp,
    exit_price: float,
    exit_reason: ExitReason,
    p_at_entry: float = float("nan"),
    knowledge_unc_at_entry: float = float("nan"),
    regime_quantile_at_entry: float = float("nan"),
) -> ClosedPosition:
    """Lift a ``Position`` to a ``ClosedPosition`` with realized P&L computed."""
    if not math.isfinite(exit_price) or exit_price <= 0:
        raise ValueError(f"exit_price must be finite and > 0; got {exit_price}")
    if k_exit < position.k_entry:
        raise ValueError(f"k_exit {k_exit} must be >= k_entry {position.k_entry}")
    gross = float(position.side) * math.log(exit_price / position.entry_price)
    return ClosedPosition(
        k_entry=position.k_entry,
        ts_entry=position.ts_entry,
        side=position.side,
        size=position.size,
        entry_price=position.entry_price,
        tp_price=position.tp_price,
        sl_price=position.sl_price,
        expiry_k=position.expiry_k,
        k_exit=k_exit,
        ts_exit=ts_exit,
        exit_price=exit_price,
        exit_reason=exit_reason,
        gross_log_return=gross,
        p_at_entry=p_at_entry,
        knowledge_unc_at_entry=knowledge_unc_at_entry,
        regime_quantile_at_entry=regime_quantile_at_entry,
    )


@dataclass
class Portfolio:
    """Bookkeeping shell: open list + closed list + helpers."""

    open_positions: list[Position] = field(default_factory=list)
    closed_positions: list[ClosedPosition] = field(default_factory=list)

    # --- mutation -----------------------------------------------------------
    def open_one(self, position: Position) -> None:
        self.open_positions.append(position)

    def close_one(
        self,
        position: Position,
        *,
        k_exit: int,
        ts_exit: pd.Timestamp,
        exit_price: float,
        exit_reason: ExitReason,
        p_at_entry: float = float("nan"),
        knowledge_unc_at_entry: float = float("nan"),
        regime_quantile_at_entry: float = float("nan"),
    ) -> ClosedPosition:
        if position not in self.open_positions:
            raise ValueError("position not in open_positions")
        closed = close_position(
            position,
            k_exit=k_exit,
            ts_exit=ts_exit,
            exit_price=exit_price,
            exit_reason=exit_reason,
            p_at_entry=p_at_entry,
            knowledge_unc_at_entry=knowledge_unc_at_entry,
            regime_quantile_at_entry=regime_quantile_at_entry,
        )
        self.open_positions.remove(position)
        self.closed_positions.append(closed)
        return closed

    def close_all(
        self,
        *,
        k_exit: int,
        ts_exit: pd.Timestamp,
        exit_price: float,
        exit_reason: ExitReason,
    ) -> list[ClosedPosition]:
        """Bulk-close every open position at the same exit price/ts/reason."""
        out: list[ClosedPosition] = []
        for pos in list(self.open_positions):
            out.append(
                self.close_one(
                    pos,
                    k_exit=k_exit,
                    ts_exit=ts_exit,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                )
            )
        return out

    # --- queries ------------------------------------------------------------
    def n_open(self) -> int:
        return len(self.open_positions)

    def gross_size(self) -> float:
        return float(sum(p.size for p in self.open_positions))

    def mtm_log_return(self, current_price: float) -> float:
        """Sum of size-weighted unrealized log returns over all open positions."""
        return float(
            sum(p.size * p.mtm_log_return(current_price) for p in self.open_positions)
        )

    def realized_log_return(self, *, cost_per_trade: float) -> float:
        """Sum of size-weighted net realized log returns over all closed positions."""
        return float(
            sum(c.weighted_net_log_return(cost_per_trade) for c in self.closed_positions)
        )

    # --- export -------------------------------------------------------------
    def closed_to_frame(self) -> pd.DataFrame:
        """One row per closed position; convenient for analytics."""
        if not self.closed_positions:
            return pd.DataFrame(
                columns=[
                    "k_entry", "ts_entry", "side", "size",
                    "entry_price", "tp_price", "sl_price", "expiry_k",
                    "k_exit", "ts_exit", "exit_price", "exit_reason",
                    "gross_log_return", "p_at_entry",
                    "knowledge_unc_at_entry", "regime_quantile_at_entry",
                ]
            )
        return pd.DataFrame([vars(c) for c in self.closed_positions])
