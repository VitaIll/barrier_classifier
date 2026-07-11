"""Pre-trade risk controls — the institutional gate in front of every entry.

A strategy signal is a REQUEST; this layer decides whether the desk is
allowed to act on it. Checked by the engine BEFORE the boundary step is
permitted to open (so a veto is a clean skip — nothing to unwind in the
ledger and nothing sent to the exchange), and enforced again inside the
broker as defense-in-depth.

Controls (all ON by default — opting OUT is the deliberate act):

- **Daily entry budget** — hard cap on entries per UTC day. A runaway
  signal (model regression, feature corruption passing the guards, feed
  loop) burns its budget and stops instead of machine-gunning orders.
- **Entry notional cap** — no single entry may exceed a fixed fraction of
  trade capital. Fat-finger/config-error guard.
- **Dislocation collar** — entries are blocked on any bar whose 1-minute
  move exceeds the collar. Market orders into a dislocated book are the
  classic way automated strategies donate money; the researched edge was
  not estimated on such bars.
- **Daily loss stop** — entries stop for the rest of the UTC day once the
  session's equity drops this much from the day's starting equity
  (log-return units). Softer than the account kill-switch (which halts
  the whole session): tomorrow starts fresh.

Vetoes are events, not errors: the engine logs them, records a guard
event, counts them, and moves on. Exits are NEVER blocked by this layer —
risk-reducing flow must always get out.

All state is event-time (bar timestamps), so replays are deterministic
and live/replay behavior is identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.engine.errors import ConfigError


@dataclass(frozen=True)
class EntryControls:
    """Pre-trade entry limits. ``None`` disables an individual control."""

    #: Max entries per UTC day (counting VETOED attempts is deliberate —
    #: a signal storm exhausts the budget whether or not orders land).
    max_daily_entries: Optional[int] = 300
    #: Max single-entry notional as a fraction of trade capital.
    max_entry_capital_frac: Optional[float] = 0.10
    #: Block entries when |1-min log move| exceeds this (log-return units).
    max_bar_move: Optional[float] = 0.05
    #: Stop entries for the day after this equity drop from the day's open
    #: (log-return units; positive number).
    max_daily_loss: Optional[float] = 0.10

    @staticmethod
    def disabled() -> "EntryControls":
        """All limits off — research-faithful (the engine reproduces the
        offline simulator exactly). The parity/replay harnesses use this;
        production leaves the defaults ON."""
        return EntryControls(
            max_daily_entries=None, max_entry_capital_frac=None,
            max_bar_move=None, max_daily_loss=None,
        )

    def __post_init__(self) -> None:
        if self.max_daily_entries is not None and self.max_daily_entries < 1:
            raise ConfigError("max_daily_entries must be >= 1 (or None)")
        for name in ("max_entry_capital_frac", "max_bar_move", "max_daily_loss"):
            v = getattr(self, name)
            if v is not None and not (math.isfinite(v) and v > 0):
                raise ConfigError(f"{name} must be finite and > 0 (or None)")


class EntryGovernor:
    """Event-time state machine applying :class:`EntryControls`.

    Owned by the engine (one per session), consulted once per bar with
    decision-time observables. Returns ``None`` (entry permitted) or a
    human-readable veto reason.
    """

    def __init__(self, controls: EntryControls) -> None:
        self.controls = controls
        self._day: Optional[pd.Timestamp] = None
        self._entries_today = 0
        self._day_open_equity = 0.0
        self.vetoes = {"daily_entries": 0, "bar_move": 0, "daily_loss": 0}

    def _roll_day(self, ts: pd.Timestamp, equity: float) -> None:
        day = ts.normalize()
        if self._day is None or day > self._day:
            self._day = day
            self._entries_today = 0
            self._day_open_equity = equity

    def check_entry(
        self,
        *,
        ts: pd.Timestamp,
        prev_close: Optional[float],
        bar_close: float,
        total_equity: float,
        entry_size: float,
    ) -> Optional[str]:
        """Permit or veto entries for this bar. Pure decision; call
        :meth:`record_entry` only when a position actually opens."""
        c = self.controls
        self._roll_day(ts, total_equity)

        if c.max_daily_entries is not None and self._entries_today >= c.max_daily_entries:
            self.vetoes["daily_entries"] += 1
            return (
                f"daily entry budget exhausted ({self._entries_today}/"
                f"{c.max_daily_entries} this UTC day)"
            )
        if (
            c.max_bar_move is not None
            and prev_close is not None
            and prev_close > 0
            and bar_close > 0
        ):
            move = abs(math.log(bar_close / prev_close))
            if move > c.max_bar_move:
                self.vetoes["bar_move"] += 1
                return (
                    f"bar dislocation {move:.4f} exceeds collar "
                    f"{c.max_bar_move:.4f} — no entries into a dislocated book"
                )
        if c.max_daily_loss is not None:
            day_loss = self._day_open_equity - total_equity
            if day_loss >= c.max_daily_loss:
                self.vetoes["daily_loss"] += 1
                return (
                    f"daily loss stop: {day_loss:.4f} >= {c.max_daily_loss:.4f} "
                    "from the day's opening equity — entries resume next UTC day"
                )
        if (
            c.max_entry_capital_frac is not None
            and entry_size > c.max_entry_capital_frac
        ):
            # Size fractions map to capital 1:1 (size × trade_capital).
            return (
                f"entry size {entry_size:.4f} exceeds the per-order cap "
                f"{c.max_entry_capital_frac:.4f} of trade capital"
            )
        return None

    def record_entry(self, ts: pd.Timestamp) -> None:
        self._roll_day(ts, self._day_open_equity)
        self._entries_today += 1

    def summary(self) -> dict:
        return {
            "entries_today": self._entries_today,
            "vetoes": dict(self.vetoes),
        }
