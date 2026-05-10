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
from typing import ClassVar, Iterable, Iterator

import polars as pl


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
    windows: ClassVar[tuple[int, ...]] = ()
    tier: ClassVar[int | str] = 1

    # --- Statistical contract (used by Validator) ---------------------------
    expected_range: ClassVar[tuple[float | None, float | None] | None] = None
    expected_finite: ClassVar[bool] = True
    expected_dtype: ClassVar[pl.DataType] = pl.Float64
    max_nan_rate_after_warmup: ClassVar[float] = 0.0

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
