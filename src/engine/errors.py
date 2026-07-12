"""Engine error hierarchy.

Every failure the engine can raise deliberately derives from
``EngineError`` so callers can catch one type at the boundary. Guard
violations carry enough context to be actionable from a log line alone.
"""

from __future__ import annotations


class EngineError(RuntimeError):
    """Base class for all deliberate engine failures."""


class ConfigError(EngineError):
    """Invalid or inconsistent :class:`~src.engine.engine.EngineConfig`."""


class EnvironmentDriftError(EngineError):
    """The host's numeric stack differs from the validated environment.

    The serving contract is bit-exact only on the pinned stack
    (``src.engine.environment.VALIDATED_STACK``); a drifted host can
    compute feature values the model was never validated on without any
    crash. Raised when arming live execution on such a host.
    """


class GridError(EngineError):
    """The 1-minute time grid was violated beyond repair policy.

    Raised for out-of-order timestamps, or gaps wider than
    ``max_repair_gap`` (a wide gap means the upstream feed is broken;
    synthesizing hours of flat bars would poison every rolling window).
    """


class BarSchemaError(EngineError):
    """A bar violated OHLCV sanity in a way the repair policy won't fix."""


class PhaseAlignmentError(EngineError):
    """The buffer's M-grid phase does not match the model's grid anchor.

    Boundary-sparse kernels (quantile family) key off ``row_index % M`` of
    the frame they see, so a phase-shifted buffer silently produces a
    feature distribution the model never trained on. This is always a bug,
    never a condition to repair at runtime.
    """


class FeatureContractError(EngineError):
    """The computed feature row cannot be reconciled with the model's
    feature list (missing feature, non-finite value after imputation)."""


class ModelArtifactError(EngineError):
    """A model version directory is missing or structurally invalid."""


class StoreError(EngineError):
    """Persistence layer failure."""


class RetrainError(EngineError):
    """A retraining run failed in a way that needs operator attention.

    Gate rejection is NOT an error (it is a recorded, expected outcome);
    this is for broken inputs — empty windows, non-finite datasets, etc.
    """


class ExchangeError(EngineError):
    """The exchange API rejected a request or the transport failed
    beyond the bounded retry policy (network, 5xx, rate-limit storms)."""


class ExecutionError(EngineError):
    """An order intent could not be executed on the exchange.

    Raised only after bounded retries. The engine treats this as a
    ledger/exchange divergence: it halts new entries and alerts — the
    runbook's reconciliation procedure is the recovery path.
    """
