"""Feature ABC and registry.

A Feature is a class declaring metadata + the polars expression for one
column (or a window-expanded family of columns). Subclasses auto-register
on declaration unless they set __abstract__ = True.

The registry is the single source of truth for what columns the engine
emits. Naming convention: {family}__{name}__w{W}; column_name() is the
escape hatch when the strict convention does not fit (e.g. legacy parity).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Iterator, Optional

import polars as pl

from src.features.config import DEFAULT_CONFIG, FeatureConfig


# =============================================================================
# Value domains — the feature's TYPE as a value object
# =============================================================================


@dataclass(frozen=True)
class Domain:
    """The value domain a feature's outputs live in.

    One declaration drives three contracts at once:

    - ``neutral`` is the default imputation fill (the "no information"
      value of the domain — 0.5 for fractions, 1.0 for ratios, 0.0 for
      residual-like reals);
    - ``(lo, hi)`` is the default expected range for health checks;
    - ``name`` selects the default legal-operation set (see
      ``LEGAL_OPS_BY_DOMAIN``).

    Features declare ``domain = FRACTION`` instead of hand-maintaining
    three separate, silently-driftable attributes.
    """

    name: str
    lo: Optional[float]
    hi: Optional[float]
    neutral: float

    @property
    def bounded(self) -> bool:
        return self.lo is not None and self.hi is not None


#: Unbounded real — residuals, z-scores, log-returns, changes. Neutral 0.
REAL = Domain("real", None, None, 0.0)
#: Non-negative scale quantities — vols, ranges, ages. Neutral 0.
NONNEGATIVE = Domain("nonnegative", 0.0, None, 0.0)
#: Rates / ranks / shares in [0, 1]. Neutral 0.5.
FRACTION = Domain("fraction", 0.0, 1.0, 0.5)
#: Signed bounded [-1, 1] — correlations, order-flow imbalance. Neutral 0.
SIGNED_FRACTION = Domain("signed_fraction", -1.0, 1.0, 0.0)
#: Positive ratios where 1 means parity — vol ratios, put/call ratios.
RATIO = Domain("ratio", 0.0, None, 1.0)
#: {0, 1} indicator flags. Neutral 0 (event absent).
BINARY = Domain("binary", 0.0, 1.0, 0.0)
#: RSI-style [0, 100] oscillators. Neutral 50.
OSCILLATOR_0_100 = Domain("oscillator_0_100", 0.0, 100.0, 50.0)


#: Advisory taxonomy of transformations that are MEANINGFUL for each
#: domain — consumed by tooling (catalog, monitoring, preprocessing
#: helpers), not enforced at compute time. E.g. winsorization makes sense
#: for unbounded tails but is vacuous for [0,1] fractions; a logit is the
#: right link for fractions but undefined for signed values.
LEGAL_OPS_BY_DOMAIN: dict[str, frozenset[str]] = {
    "real": frozenset({"difference", "zscore", "winsorize", "rank"}),
    "nonnegative": frozenset({"difference", "zscore", "winsorize", "rank", "log1p"}),
    "fraction": frozenset({"difference", "rank", "logit"}),
    "signed_fraction": frozenset({"difference", "rank", "fisher_z"}),
    "ratio": frozenset({"difference", "rank", "winsorize", "log"}),
    "binary": frozenset({"rate"}),
    "oscillator_0_100": frozenset({"difference", "rank"}),
}


_REGISTRY: list[type["Feature"]] = []


def get_registry(
    *,
    tiers: Iterable[int | str] | None = None,
    families: Iterable[str] | None = None,
) -> list[type["Feature"]]:
    """Return registered Feature classes, optionally filtered by tier/family.

    Order matches declaration order (= import order of family modules).
    """
    out = list(_REGISTRY)
    if tiers is not None:
        tier_set = set(tiers)
        out = [c for c in out if c.tier in tier_set]
    if families is not None:
        fam_set = set(families)
        out = [c for c in out if c.family in fam_set]
    return out


def _clear_registry_for_tests() -> None:
    """Reset the global registry. Test-only — never call from production."""
    _REGISTRY.clear()


class Feature(ABC):
    """One feature column or window-expanded column family.

    Subclasses set class attributes and implement compute(w) -> pl.Expr.
    Auto-registers in _REGISTRY on declaration unless __abstract__ = True.
    """

    # --- Identity -----------------------------------------------------------
    family: ClassVar[str] = ""
    name: ClassVar[str] = ""
    inputs: ClassVar[tuple[str, ...]] = ()
    tier: ClassVar[int | str] = 1

    # --- Window binding -------------------------------------------------------
    # Two ways to declare windows:
    #   1. ``windows_field = "windows_eq"`` — resolved from the injected
    #      FeatureConfig at INSTANTIATION time (the config-driven form; two
    #      configurations can coexist in one process).
    #   2. ``windows = (5, 60)`` as a plain class attribute — a class-local
    #      constant that no experiment varies. The subclass attribute
    #      shadows the base property below, so both forms coexist.
    windows_field: ClassVar[Optional[str]] = None

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        """Bind this spec to a configuration (default: production config).

        Features hold NO data state — ``cfg`` is frozen configuration; the
        instance stays reusable and thread-safe.
        """
        self.cfg: FeatureConfig = config if config is not None else DEFAULT_CONFIG

    @property
    def windows(self) -> tuple[int, ...]:
        if self.windows_field is not None:
            return tuple(getattr(self.cfg, self.windows_field))
        return ()

    # --- Value domain (the feature's TYPE) -----------------------------------
    # One declaration drives the default imputation fill (domain.neutral),
    # the default expected range (domain bounds), and the default legal
    # operations. REAL (unbounded, neutral 0) matches the historical
    # zero-fill default, so undeclared classes behave exactly as before.
    domain: ClassVar[Domain] = REAL

    # --- Statistical contract (used by Validator) ---------------------------
    #: Explicit override; ``None`` derives from ``domain`` bounds.
    expected_range: ClassVar[tuple[float | None, float | None] | None] = None
    expected_finite: ClassVar[bool] = True
    expected_dtype: ClassVar[pl.DataType] = pl.Float64
    max_nan_rate_after_warmup: ClassVar[float] = 0.0

    # --- Categorical labels ----------------------------------------------------
    #: Class-specific enrichment; ``effective_tags`` adds the structural
    #: labels (family, tier, domain, windowed-ness) automatically.
    tags: ClassVar[frozenset[str]] = frozenset()

    # --- Imputation contract ----------------------------------------------------
    # The fill applied to missing values (warmup rows, undefined inputs)
    # AFTER the ``undef__`` flag is recorded. Declared HERE, next to the
    # feature that owns the column — previously a 140-line order-sensitive
    # regex table in utils keyed on column names, with a silent 0.0
    # catch-all. ``None`` (default) derives the fill from the value
    # domain's neutral; set explicitly ONLY for sentinel fills that are
    # deliberately NOT the domain neutral (e.g. "far from the extreme"
    # distances filled at 5.0).
    impute_default: ClassVar[Optional[float]] = None

    def impute_value(self, w: int | None = None) -> float:
        """Fill value for this feature's column at window ``w``.

        Defaults to the value domain's neutral; an explicit
        ``impute_default`` overrides (sentinel fills). Override the method
        for window-dependent policies.
        """
        if self.impute_default is not None:
            return float(self.impute_default)
        return float(self.domain.neutral)

    # --- Derived contract surfaces -------------------------------------------

    @property
    def value_range(self) -> tuple[float | None, float | None]:
        """Expected output range: explicit override or the domain bounds."""
        if self.expected_range is not None:
            return self.expected_range
        return (self.domain.lo, self.domain.hi)

    @property
    def legal_ops(self) -> frozenset[str]:
        """Transformations that are meaningful for this feature's domain."""
        return LEGAL_OPS_BY_DOMAIN.get(self.domain.name, frozenset())

    @property
    def effective_tags(self) -> frozenset[str]:
        """Structural labels + class enrichment, for selection/reporting."""
        structural = {self.family, f"tier:{self.tier}", f"domain:{self.domain.name}"}
        if self.windows:
            structural.add("windowed")
        return frozenset(structural) | self.tags

    def depends_on(self, w: int | None = None) -> tuple[str, ...]:
        """Column-level predecessors of this feature at window ``w``.

        Defaults to the declared base ``inputs``. Tier-2 features that
        read columns EMITTED BY EARLIER TIERS (today referenced via
        f-strings inside ``compute``) should override so the dependency
        is declared, not implicit — the engine validates the declared
        graph at construction and rejects same-/later-tier references
        (the polars ``with_columns`` forward-reference landmine).
        """
        return self.inputs

    def describe(self, w: int | None = None) -> dict[str, Any]:
        """Self-description of one emitted column — the metadata surface."""
        return {
            "column": self.column_name(w),
            "family": self.family,
            "name": self.name,
            "tier": self.tier,
            "window": w,
            "domain": self.domain.name,
            "range_lo": self.value_range[0],
            "range_hi": self.value_range[1],
            "impute": self.impute_value(w),
            "warmup_rows": self.warmup_for(w),
            "null_tail_rows": self.null_tail_bars,
            "depends_on": list(self.depends_on(w)),
            "tags": sorted(self.effective_tags),
            "legal_ops": sorted(self.legal_ops),
            "class": type(self).__name__,
        }

    # --- Warmup / null tail -------------------------------------------------
    null_tail_bars: ClassVar[int] = 0

    # --- Auto-registration --------------------------------------------------
    __abstract__: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("__abstract__", False):
            return
        if not cls.family:
            raise ValueError(f"{cls.__name__}: class attribute 'family' must be set")
        if not cls.inputs:
            raise ValueError(f"{cls.__name__}: class attribute 'inputs' must be set")
        if not cls.name:
            cls.name = cls.__name__.lower()
        _REGISTRY.append(cls)

    # --- API ----------------------------------------------------------------
    @abstractmethod
    def compute(self, w: int | None = None) -> pl.Expr:
        """Return the polars expression for this feature at window w.

        The expression must reference only columns named in self.inputs and
        must be causal (no future leakage).
        """

    def column_name(self, w: int | None = None) -> str:
        suffix = f"__w{w}" if w is not None else ""
        return f"{self.family}__{self.name}{suffix}"

    def warmup_for(self, w: int | None) -> int:
        """Number of leading rows expected to be null after compute(w)."""
        return (w - 1) if w else 0

    def expanded(self) -> Iterator[tuple[int | None, str]]:
        """Iterate (window, column_name) pairs.

        Single-column features yield one (None, name) pair.
        """
        if self.windows:
            for w in self.windows:
                yield w, self.column_name(w)
        else:
            yield None, self.column_name(None)
