"""First-class boundary guards for the live data path.

Guards are cheap, explicit, and fail loud. Each guard either admits
(possibly repairing) an observation or raises an ``EngineError`` subclass;
every repair is counted and surfaced as a :class:`GuardEvent` so silent
degradation is impossible.

Repair policies follow the research spec exactly:

- **Grid gaps** (spec §4.5): missing minutes are filled with flat
  synthetic bars at the previous close, zero volume — the same rule the
  raw-data notebook applies, so live repairs and historical repairs are
  indistinguishable to the feature pipeline. Gaps wider than
  ``max_repair_gap`` abort: hours of synthetic bars would poison every
  rolling window silently.
- **OHLC sanity** (spec §4.5 validation rules): repairable violations
  (high < max(o,c), low > min(o,c), negative volume, taker > volume) are
  clamped and counted; structural violations (non-positive prices,
  non-finite fields) always raise.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import pandas as pd

from src.engine.domain import Bar, DerivSnapshot, GuardEvent, MarketUpdate
from src.engine.errors import BarSchemaError, GridError

_MINUTE = pd.Timedelta(minutes=1)

EventSink = Callable[[GuardEvent], None]


def _noop_sink(_: GuardEvent) -> None:
    return None


class GridGuard:
    """Enforces the strict 1-minute grid on the inbound stream.

    - exact next-minute bar → admitted as-is;
    - duplicate timestamp → dropped (feeds can resend), counted;
    - out-of-order timestamp → :class:`GridError` (never repairable);
    - gap of ``g`` minutes, ``g <= max_repair_gap`` → ``g-1`` synthetic
      flat bars are emitted before the real bar (deriv snapshots
      forward-fill), counted;
    - gap wider than ``max_repair_gap`` → :class:`GridError`.
    """

    def __init__(self, *, max_repair_gap: int = 120, sink: EventSink = _noop_sink) -> None:
        if max_repair_gap < 1:
            raise ValueError("max_repair_gap must be >= 1")
        self._max_gap = int(max_repair_gap)
        self._sink = sink
        self._last_ts: Optional[pd.Timestamp] = None
        self._last_close: float = float("nan")
        self._last_deriv: Optional[DerivSnapshot] = None
        self.n_repaired = 0
        self.n_duplicates = 0

    def admit(self, update: MarketUpdate) -> list[MarketUpdate]:
        """Admit one update; returns the (possibly repaired) sequence to
        process, in order. Empty list means "drop" (duplicate)."""
        ts = update.ts
        if ts.tzinfo is None:
            raise GridError(f"bar timestamp {ts} is not tz-aware")
        deriv = update.deriv.merged_over(self._last_deriv)
        update = MarketUpdate(bar=update.bar, deriv=deriv)

        if self._last_ts is None:
            self._commit(update)
            return [update]

        if ts == self._last_ts:
            self.n_duplicates += 1
            self._sink(GuardEvent(
                ts=ts, guard="grid", severity="info",
                message=f"duplicate bar at {ts} dropped",
            ))
            return []
        if ts < self._last_ts:
            raise GridError(
                f"out-of-order bar: {ts} after {self._last_ts} — refusing to "
                "rewrite history"
            )

        gap = round((ts - self._last_ts) / _MINUTE)
        if gap == 1:
            self._commit(update)
            return [update]
        if gap - 1 > self._max_gap:
            raise GridError(
                f"gap of {gap - 1} missing minutes before {ts} exceeds "
                f"max_repair_gap={self._max_gap}; upstream feed is broken"
            )
        out: list[MarketUpdate] = []
        t = self._last_ts + _MINUTE
        while t < ts:
            out.append(MarketUpdate(
                bar=Bar.flat_synthetic(t, self._last_close),
                deriv=deriv,
            ))
            t += _MINUTE
        out.append(update)
        self.n_repaired += len(out) - 1
        self._sink(GuardEvent(
            ts=ts, guard="grid", severity="warning",
            message=f"repaired {gap - 1} missing minute(s) before {ts} with flat bars",
            count=gap - 1,
        ))
        self._commit(update)
        return out

    def _commit(self, update: MarketUpdate) -> None:
        self._last_ts = update.ts
        self._last_close = update.bar.close
        self._last_deriv = update.deriv


class BarSchemaGuard:
    """Validates and (per spec rules) repairs a single bar's OHLCV sanity."""

    def __init__(self, *, sink: EventSink = _noop_sink) -> None:
        self._sink = sink
        self.n_repaired = 0

    def admit(self, bar: Bar) -> Bar:
        o, h, l, c = bar.open, bar.high, bar.low, bar.close
        for name, v in (("open", o), ("high", h), ("low", l), ("close", c)):
            if not math.isfinite(v):
                raise BarSchemaError(f"{bar.ts}: non-finite {name} price ({v})")
            if v <= 0:
                raise BarSchemaError(f"{bar.ts}: non-positive {name} price ({v})")

        repairs: list[str] = []
        body_hi, body_lo = max(o, c), min(o, c)
        if h < body_hi:
            h = body_hi
            repairs.append("high<max(o,c)")
        if l > body_lo:
            l = body_lo
            repairs.append("low>min(o,c)")
        vol = bar.volume if math.isfinite(bar.volume) else 0.0
        qvol = bar.quote_volume if math.isfinite(bar.quote_volume) else 0.0
        ntr = bar.num_trades if math.isfinite(bar.num_trades) else 0.0
        tbb = bar.taker_buy_base if math.isfinite(bar.taker_buy_base) else 0.0
        tbq = bar.taker_buy_quote if math.isfinite(bar.taker_buy_quote) else 0.0
        if vol < 0:
            vol = 0.0
            repairs.append("volume<0")
        if tbb > vol:
            tbb = vol
            repairs.append("taker>volume")

        if not repairs and vol == bar.volume and tbb == bar.taker_buy_base \
                and qvol == bar.quote_volume and ntr == bar.num_trades \
                and tbq == bar.taker_buy_quote and h == bar.high and l == bar.low:
            return bar

        self.n_repaired += 1
        self._sink(GuardEvent(
            ts=bar.ts, guard="bar_schema", severity="warning",
            message=f"repaired bar at {bar.ts}: {', '.join(repairs) or 'non-finite volume fields'}",
        ))
        return Bar(
            ts=bar.ts, open=o, high=h, low=l, close=c, volume=vol,
            quote_volume=qvol, num_trades=ntr, taker_buy_base=tbb,
            taker_buy_quote=tbq, synthetic=bar.synthetic,
        )


class WarmupGuard:
    """No features, predictions, or entries until the buffer is deep enough.

    ``min_ready_rows`` is expressed in *phase-aligned* rows — the rows the
    pipeline will actually see.
    """

    def __init__(self, min_ready_rows: int) -> None:
        if min_ready_rows < 1:
            raise ValueError("min_ready_rows must be >= 1")
        self.min_ready_rows = int(min_ready_rows)

    def ready(self, aligned_len: int) -> bool:
        return aligned_len >= self.min_ready_rows

    def deficit(self, aligned_len: int) -> int:
        return max(0, self.min_ready_rows - aligned_len)
