"""Declarative strategy definitions — specs as data, not closures.

A :class:`StrategyDefinition` describes a strategy as parameterized
component references (gate/sizer/exit/bulk kinds + params). Unlike the
runtime :class:`~src.strategy.policy.StrategySpec` (a bundle of callables,
kept unchanged as the execution contract), a definition can be printed,
compared, hashed, serialized to/from JSON, and swept over a parameter grid
without re-wiring lambdas. ``build()`` resolves it into a fresh
``StrategySpec`` through the component registries below.

Adding a component = one pure function in ``policy.py`` + one registry
entry here. Unknown kinds fail at definition time with the list of known
kinds (ConfigError), not at trade time.

Runtime-only dependencies (the live probability feed / precomputed p_map
consumed by the let-winners-run exits) are NOT part of the definition —
they are passed to ``build(prob_feed=...)``, keeping definitions pure data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional

from src.core.errors import ConfigError
from src.strategy.policy import (
    RiskConfig,
    State,
    StrategySpec,
    bulk_on_cluster_loss,
    bulk_on_regime_drop,
    bulk_on_unc_spike,
    exit_tp_or_expiry,
    exit_tp_sl_or_expiry,
    gate_knowledge_unc_cap,
    gate_regime_high,
    gate_score_above,
    gate_unc_below,
    make_exit_let_winners_run,
    make_exit_let_winners_run_monotonic,
    score_raw_p,
    score_ve_mean,
    size_clip,
    size_constant,
)

# ---------------------------------------------------------------------------
# Component registries: kind -> builder(params [, context]) -> callable
# ---------------------------------------------------------------------------

_SCORES: dict[str, Callable[[State], float]] = {
    "raw_p": score_raw_p,
    "ve_mean": score_ve_mean,
}

_GATES: dict[str, Callable[..., Callable[[State], bool]]] = {
    "score_above": lambda *, threshold: (
        lambda s, t=float(threshold): gate_score_above(s, t)
    ),
    "regime_high": lambda *, q_min: (
        lambda s, q=float(q_min): gate_regime_high(s, q)
    ),
    "unc_below": lambda *, q_max: (
        lambda s, q=float(q_max): gate_unc_below(s, q)
    ),
    "knowledge_unc_cap": lambda *, cap: (
        lambda s, c=float(cap): gate_knowledge_unc_cap(s, c)
    ),
}

_SIZERS: dict[str, Callable[..., Callable[[State], float]]] = {
    "constant_clipped": lambda *, size, max_size=1.0: (
        lambda s, sz=float(size), mx=float(max_size): size_clip(
            size_constant(s, default=sz), max_size=mx
        )
    ),
}

# Exit builders take (params, prob_feed). ``prob_feed`` is any ts->p mapping
# (pd.Series or LiveProbFeed); exits that don't need one ignore it.
_EXITS: dict[str, Callable[..., Callable]] = {
    "tp_or_expiry": lambda prob_feed: exit_tp_or_expiry,
    "tp_sl_or_expiry": lambda prob_feed: exit_tp_sl_or_expiry,
    "let_winners_run": lambda prob_feed, *, hold_threshold, sl_log_return=None: (
        make_exit_let_winners_run(
            _require_feed(prob_feed, "let_winners_run"),
            hold_threshold=float(hold_threshold),
            sl_log_return=sl_log_return,
        )
    ),
    "let_winners_run_monotonic": lambda prob_feed: (
        make_exit_let_winners_run_monotonic(
            _require_feed(prob_feed, "let_winners_run_monotonic")
        )
    ),
}

_BULKS: dict[str, Callable[..., Callable[[State], Optional[str]]]] = {
    "none": lambda: (lambda s: None),
    "on_cluster_loss": lambda *, cap_log_return: (
        lambda s, c=float(cap_log_return): bulk_on_cluster_loss(s, cap_log_return=c)
    ),
    "on_regime_drop": lambda *, exit_q=0.5: (
        lambda s, q=float(exit_q): bulk_on_regime_drop(s, exit_q=q)
    ),
    "on_unc_spike": lambda *, spike_q=0.95: (
        lambda s, q=float(spike_q): bulk_on_unc_spike(s, spike_q=q)
    ),
}


def _require_feed(prob_feed, kind: str):
    if prob_feed is None:
        raise ConfigError(
            f"exit kind {kind!r} needs a probability feed — pass "
            "build(prob_feed=<pd.Series ts->p | LiveProbFeed>)"
        )
    return prob_feed


def _resolve(registry: Mapping[str, Callable], kind: str, what: str) -> Callable:
    try:
        return registry[kind]
    except KeyError:
        raise ConfigError(
            f"unknown {what} kind {kind!r}; known: {sorted(registry)}"
        ) from None


# ---------------------------------------------------------------------------
# Component references
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentRef:
    """A named component + its parameters. Pure data."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ConfigError("ComponentRef.kind must be non-empty")

    def key(self) -> tuple:
        """Hashable identity (kind + sorted params) for dedup/sweeps."""
        return (self.kind, tuple(sorted(self.params.items())))


def gate(kind: str, **params: Any) -> ComponentRef:
    _resolve(_GATES, kind, "gate")  # validate eagerly
    return ComponentRef(kind, params)


def sizer(kind: str, **params: Any) -> ComponentRef:
    _resolve(_SIZERS, kind, "sizer")
    return ComponentRef(kind, params)


def exit_rule(kind: str, **params: Any) -> ComponentRef:
    _resolve(_EXITS, kind, "exit")
    return ComponentRef(kind, params)


def bulk(kind: str, **params: Any) -> ComponentRef:
    _resolve(_BULKS, kind, "bulk-close")
    return ComponentRef(kind, params)


# ---------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyDefinition:
    """A strategy as data: printable, serializable, sweepable.

    ``build(prob_feed=...)`` resolves the references into a fresh
    :class:`StrategySpec`. Each ``build`` call constructs NEW exit closures
    — per-position exit state never leaks between runs.
    """

    name: str
    gates: tuple[ComponentRef, ...]
    sizer: ComponentRef
    exit: ComponentRef
    bulk: ComponentRef = field(default_factory=lambda: ComponentRef("none"))
    score: str = "raw_p"
    risk: RiskConfig = field(default_factory=RiskConfig)
    requires: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("StrategyDefinition.name must be non-empty")
        _resolve(_SCORES, self.score, "score")
        for g in self.gates:
            _resolve(_GATES, g.kind, "gate")
        _resolve(_SIZERS, self.sizer.kind, "sizer")
        _resolve(_EXITS, self.exit.kind, "exit")
        _resolve(_BULKS, self.bulk.kind, "bulk-close")

    # -- resolution ---------------------------------------------------------

    def build(self, *, prob_feed=None) -> StrategySpec:
        return StrategySpec(
            name=self.name,
            requires=self.requires,
            score_fn=_SCORES[self.score],
            entry_gates=tuple(
                _GATES[g.kind](**g.params) for g in self.gates
            ),
            sizer=_SIZERS[self.sizer.kind](**self.sizer.params),
            exit_policy=_EXITS[self.exit.kind](prob_feed, **self.exit.params),
            bulk_close=_BULKS[self.bulk.kind](**self.bulk.params),
            risk=self.risk,
            description=self.description,
        )

    # -- identity / persistence ---------------------------------------------

    def key(self) -> tuple:
        """Hashable identity across all parameters (sweep dedup)."""
        return (
            self.name,
            tuple(g.key() for g in self.gates),
            self.sizer.key(),
            self.exit.key(),
            self.bulk.key(),
            self.score,
            tuple(sorted(asdict(self.risk).items())),
            self.requires,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gates": [{"kind": g.kind, "params": dict(g.params)} for g in self.gates],
            "sizer": {"kind": self.sizer.kind, "params": dict(self.sizer.params)},
            "exit": {"kind": self.exit.kind, "params": dict(self.exit.params)},
            "bulk": {"kind": self.bulk.kind, "params": dict(self.bulk.params)},
            "score": self.score,
            "risk": asdict(self.risk),
            "requires": list(self.requires),
            "description": self.description,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StrategyDefinition":
        def ref(entry: Mapping[str, Any]) -> ComponentRef:
            return ComponentRef(entry["kind"], dict(entry.get("params", {})))

        return cls(
            name=d["name"],
            gates=tuple(ref(g) for g in d.get("gates", ())),
            sizer=ref(d["sizer"]),
            exit=ref(d["exit"]),
            bulk=ref(d.get("bulk", {"kind": "none"})),
            score=d.get("score", "raw_p"),
            risk=RiskConfig(**d.get("risk", {})),
            requires=tuple(d.get("requires", ())),
            description=d.get("description", ""),
        )

    @classmethod
    def from_json(cls, s: str) -> "StrategyDefinition":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# The production definition (P1+P3) — the researched winning strategy as data
# ---------------------------------------------------------------------------


def production_definition(
    *,
    p_threshold: float,
    lot_size: float = 0.02,
    max_concurrent: int = 50,
    cost_per_trade: float = 0.0005,
    sl_log_return: Optional[float] = None,
    name: str = "production_P1_P3",
) -> StrategyDefinition:
    """The P1+P3 spec as a declarative definition.

    ``definition.build(prob_feed=...)`` produces a spec that trades
    identically to ``make_live_production_spec`` (pinned by test).
    """
    return StrategyDefinition(
        name=name,
        gates=(gate("score_above", threshold=float(p_threshold)),),
        sizer=sizer("constant_clipped", size=float(lot_size), max_size=1.0),
        exit=exit_rule(
            "let_winners_run",
            hold_threshold=float(p_threshold),
            sl_log_return=sl_log_return,
        ),
        risk=RiskConfig(
            cost_per_trade=float(cost_per_trade),
            max_open_positions=int(max_concurrent),
            max_gross_size=int(max_concurrent) * float(lot_size) + 1e-6,
            max_horizon_boundaries=1_000_000,
            position_mtm_floor_log_return=None,
        ),
        description=(
            "P1: top-q selective entry; P3: let-winners-run while conviction "
            "holds; no SL; no time expiry"
        ),
    )
