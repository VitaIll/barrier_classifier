"""The engine: one event loop from market data to executed trades.

Per closed bar (docs/ENGINE.md §3)::

    source ─▶ GridGuard ─▶ BarSchemaGuard ─▶ BarBuffer.append
                                                 │ (warm?)
                         FeatureService.latest ◀─┘
                         ContractGuard ─▶ Model.predict ─▶ p(t)
            exits (same-bar p, simulator ordering) ─▶ entry decision
            Broker execute ─▶ Store record ─▶ label maturation
            Retrainer.on_bar / poll ─▶ hot-swap at bar boundary

Degraded bars (feature/contract failure) still resolve exits — with NaN
conviction the researched exit policies fail safe (close at market on a
TP touch) — but never open new positions. Repeated failures trip the
halt kill-switch.
"""

from __future__ import annotations

import logging
import math
import time
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.engine.buffer import BarBuffer, minutes_since
from src.engine.domain import (
    Action,
    Bar,
    Decision,
    EventType,
    GuardEvent,
    Prediction,
)
from src.engine.errors import ConfigError, FeatureContractError
from src.engine.execution import Broker, PaperBroker, trade_from_closed
from src.engine.features import (
    BatchFeatureService,
    FeatureContract,
    RollingFeatureService,
    matured_label,
)
from src.engine.guards import BarSchemaGuard, GridGuard, WarmupGuard
from src.engine.model import ModelHandle, ModelRegistry
from src.engine.retrain import Retrainer, RetrainPolicy
from src.engine.sources import DataSource, ReplaySource
from src.engine.store import SQLiteStore
from src.engine.strategy import LiveProbFeed, LiveTrader, make_live_production_spec
from src.strategy.simulator import SimConfig

logger = logging.getLogger("src.engine")


@dataclass
class EngineConfig:
    """Every knob, research-faithful defaults. See docs/ENGINE.md §9."""

    # Model registry
    model_dir: str | Path = "models"
    # Feature serving
    feature_mode: str = "rolling"          # "rolling" (true live) | "batch" (replay fast path)
    buffer_rows: int = 40_320              # 28 days: 20,160-bar windows + EWMA burn-in
    boundary_tail_rows: int = 6_000
    min_ready_rows: Optional[int] = None   # None → full buffer before predicting
    # Strategy (the researched P1+P3 spec)
    lot_size: float = 0.02
    max_concurrent: int = 50
    cost_per_trade: float = 0.0005
    exit_variant: str = "threshold"        # "threshold" | "monotonic"
    p_threshold_override: Optional[float] = None  # None → model's train-frozen threshold
    regime_col: str = "vol__rs__f__w240"
    # Persistence
    store_path: str | Path = "runtime/engine.db"
    retention_days: Optional[float] = None
    # Guards
    max_repair_gap: int = 120
    halt_after_feature_errors: int = 25
    # Retraining
    retrain: RetrainPolicy = field(default_factory=lambda: RetrainPolicy(enabled=False))
    retrain_threaded: bool = True
    # Progress logging cadence (bars)
    log_every_bars: int = 1_440

    def validate(self) -> None:
        if self.feature_mode not in ("rolling", "batch"):
            raise ConfigError(f"feature_mode must be 'rolling' or 'batch', got {self.feature_mode!r}")
        if self.exit_variant not in ("threshold", "monotonic"):
            raise ConfigError(f"exit_variant must be 'threshold' or 'monotonic'")
        if self.buffer_rows <= 0:
            raise ConfigError("buffer_rows must be positive")
        if self.lot_size <= 0 or self.max_concurrent <= 0:
            raise ConfigError("lot_size and max_concurrent must be positive")

    @classmethod
    def from_toml(cls, path: str | Path) -> "EngineConfig":
        """Load from a TOML file. Top-level keys map to fields; the
        ``[retrain]`` table maps to :class:`RetrainPolicy`."""
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        retrain_d = payload.pop("retrain", None)
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(f"unknown config keys in {path}: {sorted(unknown)}")
        cfg = cls(**payload)
        if retrain_d is not None:
            cfg.retrain = RetrainPolicy(**retrain_d)
        cfg.validate()
        return cfg


@dataclass
class SessionReport:
    """What a session did — returned by :meth:`Engine.run`."""

    started_ts: Optional[pd.Timestamp]
    ended_ts: Optional[pd.Timestamp]
    n_bars: int
    n_predictions: int
    n_trades: int
    n_open_positions: int
    realized_cum_log_return: float
    unrealized_log_return: float
    model_versions_used: list[str]
    retrain_runs: int
    guard_repairs: int
    feature_errors: int
    halted: bool
    trades: pd.DataFrame
    equity: pd.DataFrame

    def summary(self) -> str:
        span = ""
        if self.started_ts is not None and self.ended_ts is not None:
            days = (self.ended_ts - self.started_ts).total_seconds() / 86_400.0
            span = f" over {days:.1f} days"
        lines = [
            f"Session{span}: {self.n_bars:,} bars, {self.n_predictions:,} predictions, "
            f"{self.n_trades:,} closed trades, {self.n_open_positions} still open",
            f"  realized {self.realized_cum_log_return:+.4%} log-return "
            f"(unrealized {self.unrealized_log_return:+.4%})",
            f"  models: {', '.join(self.model_versions_used)}  |  retrains: {self.retrain_runs}"
            f"  |  grid repairs: {self.guard_repairs}  |  feature errors: {self.feature_errors}"
            + ("  |  HALTED" if self.halted else ""),
        ]
        return "\n".join(lines)


class Engine:
    """Ergonomic front door: construct with a config and a source, ``run()``.

    ``store`` and ``broker`` are injectable for tests; by default a
    :class:`SQLiteStore` at ``config.store_path`` and a :class:`PaperBroker`.
    """

    def __init__(
        self,
        config: EngineConfig,
        source: DataSource,
        *,
        store: Optional[SQLiteStore] = None,
        broker: Optional[Broker] = None,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.source = source
        self.registry = registry or ModelRegistry(config.model_dir)
        self.handle: ModelHandle = self.registry.active()
        self.contract: FeatureContract = self.handle.contract

        m = self.contract.m
        buffer_rows = config.buffer_rows
        if buffer_rows % m != 0:
            buffer_rows += m - (buffer_rows % m)
        min_needed = self.contract.n_warmup + m
        if buffer_rows < min_needed:
            raise ConfigError(
                f"buffer_rows={buffer_rows:,} < n_warmup+M={min_needed:,} — "
                "the deepest rolling window would never fill"
            )
        self.buffer = BarBuffer(
            buffer_rows, m=m, anchor_ts=self.contract.anchor,
            with_derivatives=self.contract.with_derivatives,
        )
        self.store = store or SQLiteStore(config.store_path)
        self.broker: Broker = broker or PaperBroker()
        self._sink = self._make_guard_sink()
        self.grid_guard = GridGuard(max_repair_gap=config.max_repair_gap, sink=self._sink)
        self.schema_guard = BarSchemaGuard(sink=self._sink)
        ready_rows = (
            int(config.min_ready_rows) if config.min_ready_rows is not None
            else buffer_rows - m + 1
        )
        if ready_rows < self.contract.n_warmup + 1:
            raise ConfigError(
                f"min_ready_rows={ready_rows:,} < n_warmup+1={self.contract.n_warmup + 1:,}; "
                "predictions would run on structurally unwarm windows"
            )
        self.warmup_guard = WarmupGuard(ready_rows)

        # Feature serving
        self.rolling_service = RollingFeatureService(
            self.contract, boundary_tail_rows=config.boundary_tail_rows
        )
        self.batch_service: Optional[BatchFeatureService] = None
        self._batch_p_cache: dict[str, np.ndarray] = {}

        # Strategy
        self._regime_idx = self._regime_index(config.regime_col)
        self.prob_feed = LiveProbFeed()
        self.trader = LiveTrader(
            self._build_spec(),
            sim_config=SimConfig(M=m, cadence_minutes=1.0),
        )

        # Retraining (reads bars via its own read-only connection)
        def _frame_provider(window_rows: Optional[int]) -> pd.DataFrame:
            conn = self.store.read_connection()
            try:
                frame = self.store.bars_frame(conn=conn)
            finally:
                if conn is not self.store._conn:  # noqa: SLF001 — :memory: shares
                    conn.close()
            if window_rows is not None and len(frame) > window_rows:
                frame = frame.tail(window_rows)
            return frame

        self.retrainer = Retrainer(
            policy=config.retrain, registry=self.registry,
            frame_provider=_frame_provider,
            incumbent_provider=lambda: self.handle,
            threaded=config.retrain_threaded,
        )

        # Session state
        self._subscribers: dict[EventType, list[Callable[[object], None]]] = {}
        self._entry_p: dict[int, float] = {}          # k_entry → decision-time p
        self._entry_version: dict[int, str] = {}      # k_entry → model version
        self._next_trade_id = 1
        self._feature_errors = 0
        self._halted = False
        self._versions_used: list[str] = [self.handle.version]
        self._retrain_completions = 0
        self._n_bars = 0
        self._n_predictions = 0
        self._started_ts: Optional[pd.Timestamp] = None
        self._last_prune_ts: Optional[pd.Timestamp] = None

        self.store.record_model_version(
            self.handle.version,
            created_ts=pd.Timestamp.now(tz="UTC"),
            path=str(Path(self.registry.root) / self.handle.version),
            metrics=self.handle.metrics,
            thresholds=self.handle.thresholds.to_dict(),
        )

    # ------------------------------------------------------------------ #
    # Wiring helpers
    # ------------------------------------------------------------------ #

    def _make_guard_sink(self):
        def sink(event: GuardEvent) -> None:
            self.store.record_guard_event(event)
            self._emit(EventType.GUARD_TRIPPED, event)
        return sink

    def _regime_index(self, regime_col: str) -> int:
        try:
            return self.contract.feature_list.index(regime_col)
        except ValueError:
            raise ConfigError(
                f"regime_col {regime_col!r} is not in the model's feature list"
            ) from None

    def _p_threshold(self) -> float:
        if self.config.p_threshold_override is not None:
            return float(self.config.p_threshold_override)
        return float(self.handle.thresholds.p_threshold)

    def _build_spec(self):
        return make_live_production_spec(
            self.prob_feed,
            p_threshold=self._p_threshold(),
            lot_size=self.config.lot_size,
            max_concurrent=self.config.max_concurrent,
            cost_per_trade=self.config.cost_per_trade,
            exit_variant=self.config.exit_variant,
        )

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #

    def on(self, event_type: EventType, callback: Callable[[object], None]) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)

    def _emit(self, event_type: EventType, payload: object) -> None:
        for cb in self._subscribers.get(event_type, ()):
            cb(payload)

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(self, *, max_bars: Optional[int] = None) -> SessionReport:
        self._bootstrap()
        try:
            for update in self.source.stream():
                for u in self.grid_guard.admit(update):
                    self._step(u.bar, u.deriv)
                    self._n_bars += 1
                    if max_bars is not None and self._n_bars >= max_bars:
                        return self._finish()
                    if self._n_bars % self.config.log_every_bars == 0:
                        eq = self.trader.realized_cum
                        logger.info(
                            "bar %s | bars=%s trades=%s open=%s realized=%+.4f",
                            u.bar.ts, self._n_bars,
                            len(self.trader.portfolio.closed_positions),
                            self.trader.portfolio.n_open(), eq,
                        )
            return self._finish()
        finally:
            self.store.flush()

    def _bootstrap(self) -> None:
        need = self.buffer.capacity
        frame = self.source.bootstrap(need)
        if frame is not None and not frame.empty:
            loaded = self.buffer.bootstrap(frame)
            self.store.record_bars_frame(frame)
            logger.info("bootstrapped %s bars (need %s for full warmup)", loaded, need)
        else:
            logger.warning(
                "no bootstrap history — engine warms up from the stream "
                "(%s bars before first prediction)", self.warmup_guard.min_ready_rows,
            )
        if self.config.feature_mode == "batch":
            if not isinstance(self.source, ReplaySource):
                raise ConfigError("feature_mode='batch' requires a ReplaySource")
            self.batch_service = BatchFeatureService(self.contract)
            t0 = time.perf_counter()
            n = self.batch_service.precompute(self.source.full_frame())
            logger.info(
                "batch feature precompute: %s rows in %.1fs",
                n, time.perf_counter() - t0,
            )

    def _batch_predictions(self) -> np.ndarray:
        assert self.batch_service is not None
        version = self.handle.version
        if version not in self._batch_p_cache:
            t0 = time.perf_counter()
            self._batch_p_cache[version] = self.handle.predict_matrix(
                self.batch_service.matrix()
            )
            logger.info(
                "vectorized predictions for %s: %.1fs", version, time.perf_counter() - t0
            )
        return self._batch_p_cache[version]

    def _predict(self, bar: Bar) -> tuple[float, float, float, float]:
        """→ (p, regime_value, feature_ms, predict_ms); raises on contract failure."""
        if self.batch_service is not None:
            i = self.batch_service.row_index(bar.ts)
            p = float(self._batch_predictions()[i])
            regime_value = float(self.batch_service.matrix()[i, self._regime_idx])
            return p, regime_value, 0.0, 0.0
        fv = self.rolling_service.latest(self.buffer)
        t0 = time.perf_counter()
        p = self.handle.predict_p(fv.values)
        predict_ms = (time.perf_counter() - t0) * 1e3
        return p, float(fv.values[self._regime_idx]), self.rolling_service.last_feature_ms, predict_ms

    def _step(self, bar: Bar, deriv) -> None:
        bar = self.schema_guard.admit(bar)
        self.buffer.append(bar, deriv)
        self.store.record_bar(bar)
        self._emit(EventType.BAR_INGESTED, bar)
        if self._started_ts is None:
            self._started_ts = bar.ts
        k = minutes_since(self.contract.anchor, bar.ts)

        # Label maturation (monitoring feed; strictly causal at t+M).
        matured = matured_label(self.buffer, m=self.contract.m, phi=self.contract.phi)
        if matured is not None:
            entry_ts, y, m_k = matured
            self.store.record_label(entry_ts, y, m_k)
            self.trader.feed_matured_label(entry_ts, y)
            self._emit(EventType.LABEL_MATURED, (entry_ts, y, m_k))

        # Warmup gate (rolling mode; batch rows embed full history).
        if self.batch_service is None and not self.warmup_guard.ready(self.buffer.aligned_len()):
            self.store.flush()
            return

        # Features + prediction (degraded bars keep exits alive with p=NaN).
        degraded = False
        p = float("nan")
        regime_value = float("nan")
        feature_ms = predict_ms = float("nan")
        try:
            p, regime_value, feature_ms, predict_ms = self._predict(bar)
        except FeatureContractError as exc:
            degraded = True
            self._feature_errors += 1
            self._sink(GuardEvent(
                ts=bar.ts, guard="feature_contract", severity="error",
                message=str(exc),
            ))
            if self._feature_errors >= self.config.halt_after_feature_errors:
                self._halted = True
        else:
            self._n_predictions += 1
            self.prob_feed.update(bar.ts, p)
            pred = Prediction(
                ts=bar.ts, p=p, model_version=self.handle.version,
                feature_ms=feature_ms, predict_ms=predict_ms,
            )
            self.store.record_prediction(pred)
            self._emit(EventType.PREDICTION_MADE, pred)

        allow_entry = not degraded and not self._halted
        result = self.trader.on_boundary(
            k=k, ts=bar.ts, p=p, bar=bar, regime_value=regime_value,
            phi=self.contract.phi, allow_entry=allow_entry,
        )

        # Execution + ledger
        for closed in result.closed:
            order, fill = self.broker.execute_close(closed)
            self.store.record_order(order, status="filled")
            self.store.record_fill(fill)
            trade = trade_from_closed(
                closed,
                trade_id=self._next_trade_id,
                cost_per_trade=self.trader.cost_per_trade,
                p_at_entry=self._entry_p.pop(closed.k_entry, float("nan")),
                model_version=self._entry_version.pop(closed.k_entry, self.handle.version),
            )
            self._next_trade_id += 1
            self.store.record_trade(trade)
            self._emit(EventType.TRADE_CLOSED, trade)
        if result.opened is not None:
            order, fill = self.broker.execute_entry(result.opened)
            self.store.record_order(order, status="filled")
            self.store.record_fill(fill)
            self._entry_p[result.opened.k_entry] = p
            self._entry_version[result.opened.k_entry] = self.handle.version
            self._emit(EventType.ORDER_FILLED, fill)

        action = (
            Action.HALT if self._halted else
            Action.ENTER if result.entered else Action.SKIP
        )
        self.store.record_decision(Decision(
            ts=bar.ts, action=action, size=result.opened.size if result.opened else 0.0,
            p=p, score=result.score, threshold=self._p_threshold(),
            n_open=result.equity.n_open, gross_size=result.equity.gross_size,
            reason=result.bulk_reason or ("degraded" if degraded else ""),
        ))
        self.store.record_equity(
            bar.ts, result.equity.realized_cum, result.equity.unrealized,
            result.equity.n_open, result.equity.gross_size,
        )
        self._emit(EventType.DECISION_MADE, result)

        # Retraining (event-time), hot-swap on completion.
        self.retrainer.on_bar(bar.ts)
        self._poll_retrain(bar.ts)

        # Retention pruning (event-time, daily cadence).
        if self.config.retention_days is not None:
            if self._last_prune_ts is None or (bar.ts - self._last_prune_ts) >= pd.Timedelta(days=1):
                cutoff = bar.ts - pd.Timedelta(days=float(self.config.retention_days))
                self.store.prune_before(cutoff)
                self._last_prune_ts = bar.ts

        self.store.flush()

    def _poll_retrain(self, ts: pd.Timestamp) -> None:
        polled = self.retrainer.poll()
        if polled is None:
            return
        trigger_ts, outcome = polled
        self._retrain_completions += 1
        run_id = self.store.open_retrain_run(trigger_ts)
        self.store.close_retrain_run(
            run_id, status=outcome.status,
            n_rows=outcome.n_rows, best_iter=outcome.best_iter,
            gate_passed=outcome.status == "published",
            new_version=outcome.new_version, notes=outcome.notes[:2000],
        )
        self._emit(EventType.RETRAIN_COMPLETED, outcome)
        logger.info("retrain completed: %s %s", outcome.status, outcome.new_version or "")
        if outcome.status != "published" or outcome.new_version is None:
            return
        # Hot-swap at the bar boundary: activate, reload, rebind strategy.
        self.registry.activate(outcome.new_version)
        new_handle = self.registry.load(outcome.new_version)
        anchor_delta = minutes_since(self.contract.anchor, new_handle.contract.anchor)
        if anchor_delta % self.contract.m != 0:
            self._sink(GuardEvent(
                ts=ts, guard="phase_alignment", severity="error",
                message=(
                    f"new model {outcome.new_version} anchor is not congruent "
                    f"with the buffer grid (Δ={anchor_delta}min) — keeping incumbent"
                ),
            ))
            return
        self.handle = new_handle
        self.contract = new_handle.contract
        self._versions_used.append(new_handle.version)
        self.store.record_model_version(
            new_handle.version, created_ts=ts,
            path=str(Path(self.registry.root) / new_handle.version),
            metrics=new_handle.metrics, thresholds=new_handle.thresholds.to_dict(),
        )
        self.store.mark_model_activated(new_handle.version, ts)
        self.trader.spec = self._build_spec()
        self.rolling_service = RollingFeatureService(
            self.contract, boundary_tail_rows=self.config.boundary_tail_rows
        )
        self._emit(EventType.MODEL_SWAPPED, new_handle.version)
        logger.info("hot-swapped to %s (p_threshold=%.4f)",
                    new_handle.version, self._p_threshold())

    def _finish(self) -> SessionReport:
        self.store.flush()
        trades = self.store.trades_frame()
        equity = self.store.equity_frame()
        last_close = self.buffer.last_close or float("nan")
        unrealized = (
            self.trader.portfolio.mtm_log_return(last_close)
            if last_close and not math.isnan(last_close) else 0.0
        )
        return SessionReport(
            started_ts=self._started_ts,
            ended_ts=self.buffer.last_ts,
            n_bars=self._n_bars,
            n_predictions=self._n_predictions,
            n_trades=len(trades),
            n_open_positions=self.trader.portfolio.n_open(),
            realized_cum_log_return=self.trader.realized_cum,
            unrealized_log_return=float(unrealized),
            model_versions_used=list(dict.fromkeys(self._versions_used)),
            retrain_runs=self._retrain_completions,
            guard_repairs=self.grid_guard.n_repaired + self.schema_guard.n_repaired,
            feature_errors=self._feature_errors,
            halted=self._halted,
            trades=trades,
            equity=equity,
        )
