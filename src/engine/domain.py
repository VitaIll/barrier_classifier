"""Engine domain model.

One shared vocabulary for the whole engine::

    MarketUpdate ─▶ Bar (+ DerivSnapshot) ─▶ FeatureVector ─▶ Prediction ─▶ Decision
                                                                              │
       Trade ◀─ Fill ◀─ Order ◀──────────────────────────────────────────────┘

All types are frozen: they are *facts* observed or emitted at a specific
``ts`` and are never mutated. Position/ledger state lives in
``src.strategy.inventory`` (``Position``/``ClosedPosition``/``Portfolio``),
which the engine reuses unchanged — the offline simulator and the live
engine share one inventory truth.

Timestamps are tz-aware UTC ``pd.Timestamp`` at *bar-complete* time
(``open_time + 60s``), the research convention.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import pandas as pd

# Raw column orders — these are contracts shared with the feature pipeline
# (`src/features/pipeline.py::_RAW_COLS` and the derivatives raw inputs of
# `utils.compute_derivatives_base_series`). Order matters: the bar buffer
# stores columns positionally.
RAW_SPOT_COLS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "quote_volume",
    "num_trades", "taker_buy_base", "taker_buy_quote",
)
RAW_DERIV_COLS: tuple[str, ...] = (
    "close_fut", "volume_fut", "quote_volume_fut", "taker_buy_base_fut",
    "num_trades_fut", "funding_rate", "oi_usd", "opt_oi",
    "put_open_interest", "call_open_interest", "opt_volume", "put_volume",
    "call_volume", "bvol",
)


@dataclass(frozen=True, slots=True)
class Bar:
    """One *closed* 1-minute spot kline."""

    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    num_trades: float
    taker_buy_base: float
    taker_buy_quote: float
    synthetic: bool = False  # True when the grid guard gap-repaired this bar

    def values(self) -> tuple[float, ...]:
        """Positional values in ``RAW_SPOT_COLS`` order."""
        return (
            self.open, self.high, self.low, self.close, self.volume,
            self.quote_volume, self.num_trades, self.taker_buy_base,
            self.taker_buy_quote,
        )

    @staticmethod
    def flat_synthetic(ts: pd.Timestamp, prev_close: float) -> "Bar":
        """Deterministic gap-repair bar per spec §4.5: flat at previous
        close, zero volume/trades."""
        return Bar(
            ts=ts, open=prev_close, high=prev_close, low=prev_close,
            close=prev_close, volume=0.0, quote_volume=0.0, num_trades=0.0,
            taker_buy_base=0.0, taker_buy_quote=0.0, synthetic=True,
        )


@dataclass(frozen=True, slots=True)
class DerivSnapshot:
    """Last-known derivatives state as of a bar's close.

    Every field is optional — ``None`` means "source dark at this minute"
    and flows into the pipeline as a null, exactly like a coverage gap in
    the training data (undef flag + imputation). The engine forward-fills
    slow sources (funding: 8h, OI: 5min, options: 1h) internally, mirroring
    ``utils.align_to_1m_grid``.
    """

    close_fut: Optional[float] = None
    volume_fut: Optional[float] = None
    quote_volume_fut: Optional[float] = None
    taker_buy_base_fut: Optional[float] = None
    num_trades_fut: Optional[float] = None
    funding_rate: Optional[float] = None
    oi_usd: Optional[float] = None
    opt_oi: Optional[float] = None
    put_open_interest: Optional[float] = None
    call_open_interest: Optional[float] = None
    opt_volume: Optional[float] = None
    put_volume: Optional[float] = None
    call_volume: Optional[float] = None
    bvol: Optional[float] = None

    def values(self) -> tuple[Optional[float], ...]:
        """Positional values in ``RAW_DERIV_COLS`` order."""
        return (
            self.close_fut, self.volume_fut, self.quote_volume_fut,
            self.taker_buy_base_fut, self.num_trades_fut, self.funding_rate,
            self.oi_usd, self.opt_oi, self.put_open_interest,
            self.call_open_interest, self.opt_volume, self.put_volume,
            self.call_volume, self.bvol,
        )

    def merged_over(self, prev: "DerivSnapshot | None") -> "DerivSnapshot":
        """Forward-fill: take this snapshot's fields, falling back to
        ``prev`` for any ``None`` field (as-of semantics per source)."""
        if prev is None:
            return self
        kwargs = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kwargs[f.name] = getattr(prev, f.name) if v is None else v
        return DerivSnapshot(**kwargs)


_EMPTY_DERIV = DerivSnapshot()


@dataclass(frozen=True, slots=True)
class MarketUpdate:
    """The unit a :class:`~src.engine.sources.DataSource` emits: one closed
    spot bar plus (optionally) the derivatives state at that minute."""

    bar: Bar
    deriv: DerivSnapshot = _EMPTY_DERIV

    @property
    def ts(self) -> pd.Timestamp:
        return self.bar.ts


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """A feature row aligned to a model's ordered feature list.

    ``values`` is a float64 array whose order IS the contract — never a
    dict. Reconciliation against the contract happens before construction
    (``src.engine.features.reconcile_row``): a missing contract feature or
    a non-finite value is a hard error, so a ``FeatureVector`` is always
    complete and finite.
    """

    ts: pd.Timestamp
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class Prediction:
    ts: pd.Timestamp
    p: float
    model_version: str
    feature_ms: float = float("nan")  # feature computation latency
    predict_ms: float = float("nan")  # model inference latency


class Action(str, enum.Enum):
    ENTER = "enter"
    SKIP = "skip"      # gates passed context but no entry (gate/risk blocked)
    HALT = "halt"      # kill-switch active — entries suppressed, exits live


@dataclass(frozen=True, slots=True)
class Decision:
    """What the strategy chose at a boundary, with enough trace to audit."""

    ts: pd.Timestamp
    action: Action
    size: float = 0.0
    p: float = float("nan")
    score: float = float("nan")
    threshold: float = float("nan")
    n_open: int = 0
    gross_size: float = 0.0
    reason: str = ""


class OrderKind(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"  # reserved for real-broker TP resting orders


class OrderStatus(str, enum.Enum):
    SUBMITTED = "submitted"
    FILLED = "filled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Order:
    order_id: int
    ts: pd.Timestamp
    side: int              # +1 long (the strategy is long-only); -1 closes
    size: float
    kind: OrderKind = OrderKind.MARKET
    limit_price: Optional[float] = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: int
    ts: pd.Timestamp
    price: float
    size: float
    side: int


@dataclass(frozen=True, slots=True)
class Trade:
    """A closed round-trip — mirrors ``inventory.ClosedPosition`` plus the
    engine's provenance fields, so research reporting works unchanged."""

    trade_id: int
    k_entry: int
    ts_entry: pd.Timestamp
    entry_price: float
    size: float
    k_exit: int
    ts_exit: pd.Timestamp
    exit_price: float
    exit_reason: str
    gross_log_return: float
    net_log_return: float          # per-unit, cost-adjusted
    weighted_net_log_return: float  # size × net (what equity accrues)
    p_at_entry: float
    model_version: str


# ---------------------------------------------------------------------------
# Events (observer bus payloads — observability, never control flow)
# ---------------------------------------------------------------------------


class EventType(str, enum.Enum):
    BAR_INGESTED = "bar_ingested"
    GUARD_TRIPPED = "guard_tripped"
    PREDICTION_MADE = "prediction_made"
    DECISION_MADE = "decision_made"
    ORDER_FILLED = "order_filled"
    TRADE_CLOSED = "trade_closed"
    LABEL_MATURED = "label_matured"
    RETRAIN_STARTED = "retrain_started"
    RETRAIN_COMPLETED = "retrain_completed"
    MODEL_SWAPPED = "model_swapped"


@dataclass(frozen=True, slots=True)
class GuardEvent:
    ts: pd.Timestamp
    guard: str
    severity: str  # "info" | "warning" | "error"
    message: str
    count: int = 1
