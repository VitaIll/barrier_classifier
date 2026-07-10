"""Live trading engine for the 1-minute barrier-crossing strategy.

The ergonomic front door (see docs/ENGINE.md)::

    from src.engine import Engine, EngineConfig, ReplaySource

    cfg = EngineConfig(model_dir="models", store_path="runtime/engine.db",
                       feature_mode="batch")
    src = ReplaySource("data/raw_data/klines_1m.parquet",
                       start="2025-09-01", end="2025-10-01")
    report = Engine(cfg, source=src).run()
    print(report.summary())

Public surface:

- **Engine loop**: :class:`Engine`, :class:`EngineConfig`, :class:`SessionReport`
- **Data in** (public interface): :class:`DataSource` protocol,
  :class:`ReplaySource` (simulated historical stream), :class:`CallbackSource`
  (push queue for exchange adapters), :class:`MarketUpdate`, :class:`Bar`,
  :class:`DerivSnapshot`
- **Execution out**: :class:`Broker` protocol, :class:`PaperBroker`
- **Models**: :class:`ModelRegistry`, :class:`ModelHandle`,
  :class:`FeatureContract`, :class:`Thresholds`
- **Retraining**: :class:`RetrainPolicy` (schedule + gate tolerances)
- **Persistence**: :class:`SQLiteStore`
- **Errors**: :class:`EngineError` hierarchy in :mod:`src.engine.errors`

CLI: ``python -m src.engine --help`` (``import-model``, ``replay``, ``run``,
``status``).
"""

from src.engine.buffer import BarBuffer
from src.engine.domain import (
    Action,
    Bar,
    Decision,
    DerivSnapshot,
    EventType,
    Fill,
    MarketUpdate,
    Order,
    Prediction,
    Trade,
)
from src.engine.engine import Engine, EngineConfig, SessionReport
from src.engine.errors import EngineError
from src.engine.execution import Broker, PaperBroker
from src.engine.features import (
    BatchFeatureService,
    FeatureContract,
    RollingFeatureService,
)
from src.engine.model import ModelHandle, ModelRegistry, Thresholds
from src.engine.retrain import RetrainPolicy
from src.engine.sources import CallbackSource, DataSource, ReplaySource
from src.engine.store import SQLiteStore
from src.engine.strategy import LiveProbFeed, LiveTrader, make_live_production_spec

__all__ = [
    "Action",
    "Bar",
    "BarBuffer",
    "BatchFeatureService",
    "Broker",
    "CallbackSource",
    "DataSource",
    "Decision",
    "DerivSnapshot",
    "Engine",
    "EngineConfig",
    "EngineError",
    "EventType",
    "FeatureContract",
    "Fill",
    "LiveProbFeed",
    "LiveTrader",
    "MarketUpdate",
    "ModelHandle",
    "ModelRegistry",
    "Order",
    "PaperBroker",
    "Prediction",
    "ReplaySource",
    "RetrainPolicy",
    "RollingFeatureService",
    "SQLiteStore",
    "SessionReport",
    "Thresholds",
    "Trade",
    "make_live_production_spec",
]
