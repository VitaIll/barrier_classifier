"""Frame contracts: declared schemas validated at block boundaries.

LAW 3 of docs/TARGET_ARCHITECTURE.md: data crossing a block boundary is
validated against a :class:`FrameSchema` — column presence, dtype, and (at
``level="data"``) nullability / finiteness / bounds. Schemas are declared
once and imported; they are never restated as string lists at call sites.

Validation levels
-----------------
- ``"structure"`` — columns exist with the declared dtypes. O(#columns);
  cheap enough for every hand-off.
- ``"data"``      — additionally scans values: nullability, finiteness for
  float columns, and optional bounds. O(rows); intended for workflow
  boundaries (training hand-off, artifact writes), not per-bar hot paths.

Also home to :func:`snapshot` / :func:`assert_unmutated` — the mechanical
check behind LAW 6 (static operations do not mutate their arguments), used
by test suites for every transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np
import polars as pl

from src.core.errors import ConfigError, ContractError

ValidationLevel = Literal["structure", "data"]

# A dtype expectation: an exact instance (pl.Float64, pl.Datetime("us")),
# a class (pl.Datetime — any unit/tz), or a tuple of either (any-of).
DTypeLike = Union[pl.DataType, type, tuple]


def _dtype_matches(actual: pl.DataType, expected: DTypeLike) -> bool:
    if isinstance(expected, tuple):
        return any(_dtype_matches(actual, e) for e in expected)
    if isinstance(expected, type):
        # Class form: match on the base type (any parametrization).
        return actual.base_type() == expected.base_type() if isinstance(
            actual, pl.DataType
        ) else False
    return actual == expected


@dataclass(frozen=True)
class ColumnSpec:
    """Contract for one column.

    ``dtype=None`` accepts any dtype (presence-only contract). ``finite``
    applies to float columns only and is checked at ``level="data"``:
    ``True`` forbids NaN/inf among non-null values. ``bounds`` is an
    inclusive ``(lo, hi)`` range checked over non-null values; ``None`` on
    either side leaves that side open.
    """

    name: str
    dtype: Optional[DTypeLike] = None
    nullable: bool = True
    finite: bool = True
    bounds: Optional[tuple[Optional[float], Optional[float]]] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("ColumnSpec.name must be non-empty")
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo is not None and hi is not None and lo > hi:
                raise ConfigError(
                    f"ColumnSpec {self.name!r}: bounds lo={lo} > hi={hi}"
                )


@dataclass(frozen=True)
class FrameSchema:
    """An ordered set of :class:`ColumnSpec` with a name for error messages.

    Frames may carry MORE columns than the schema declares — undeclared
    columns are out of contract and pass through unvalidated (LAW 1:
    blocks state what they require and provide; the rest is not theirs to
    police).
    """

    name: str
    columns: tuple[ColumnSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for c in self.columns:
            if c.name in seen:
                raise ConfigError(
                    f"FrameSchema {self.name!r}: duplicate column {c.name!r}"
                )
            seen.add(c.name)

    # -- construction ------------------------------------------------------

    @staticmethod
    def of(name: str, *columns: ColumnSpec) -> "FrameSchema":
        return FrameSchema(name=name, columns=tuple(columns))

    def extend(self, name: str, *columns: ColumnSpec) -> "FrameSchema":
        """New schema = this schema + more columns (name must be new)."""
        return FrameSchema(name=name, columns=self.columns + tuple(columns))

    # -- introspection -------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def __contains__(self, column: str) -> bool:
        return column in self.names

    # -- validation ----------------------------------------------------------

    def validate(
        self,
        frame: pl.DataFrame,
        *,
        level: ValidationLevel = "structure",
        context: str = "",
    ) -> pl.DataFrame:
        """Validate ``frame`` against this schema; return it unchanged.

        Collects ALL violations before raising one :class:`ContractError`
        so a broken frame is diagnosed in a single round-trip.
        """
        if level not in ("structure", "data"):
            raise ConfigError(f"unknown validation level {level!r}")
        problems: list[str] = []
        schema = frame.schema

        for spec in self.columns:
            if spec.name not in schema:
                problems.append(f"missing column {spec.name!r}")
                continue
            actual = schema[spec.name]
            if spec.dtype is not None and not _dtype_matches(actual, spec.dtype):
                problems.append(
                    f"column {spec.name!r} has dtype {actual}, expected {spec.dtype}"
                )
                continue  # value checks on a wrong dtype only add noise
            if level == "data":
                problems.extend(self._data_problems(frame, spec, actual))

        if problems:
            where = f" ({context})" if context else ""
            head = (
                f"frame failed contract {self.name!r}{where}: "
                f"{len(problems)} violation(s)"
            )
            raise ContractError(head + "\n  - " + "\n  - ".join(problems))
        return frame

    @staticmethod
    def _data_problems(
        frame: pl.DataFrame, spec: ColumnSpec, dtype: pl.DataType
    ) -> list[str]:
        problems: list[str] = []
        s = frame[spec.name]
        if not spec.nullable:
            n_null = int(s.null_count())
            if n_null:
                problems.append(
                    f"column {spec.name!r} declared non-nullable but has "
                    f"{n_null} null(s)"
                )
        is_float = dtype in (pl.Float32, pl.Float64)
        if is_float and spec.finite:
            n_nan = int(s.is_nan().sum() or 0)
            if n_nan:
                problems.append(
                    f"column {spec.name!r} declared finite but has {n_nan} NaN"
                )
            if bool(s.is_infinite().any()):
                n_inf = int(s.is_infinite().sum())
                problems.append(
                    f"column {spec.name!r} declared finite but has {n_inf} inf"
                )
        if spec.bounds is not None and dtype.is_numeric():
            lo, hi = spec.bounds
            if lo is not None:
                mn = s.min()
                if mn is not None and float(mn) < lo:
                    problems.append(
                        f"column {spec.name!r} min {mn} < declared lower bound {lo}"
                    )
            if hi is not None:
                mx = s.max()
                if mx is not None and float(mx) > hi:
                    problems.append(
                        f"column {spec.name!r} max {mx} > declared upper bound {hi}"
                    )
        return problems


def require_columns(
    frame: pl.DataFrame,
    required: Iterable[str],
    *,
    context: str,
) -> None:
    """Presence-only check with a diagnostic error (no schema object needed)."""
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ContractError(
            f"{context}: missing required column(s) {missing} "
            f"(frame has {len(frame.columns)} columns)"
        )


# ---------------------------------------------------------------------------
# LAW 6 machinery — assert a "static" operation did not mutate its arguments.
# ---------------------------------------------------------------------------


def snapshot(obj):
    """Cheap defensive snapshot of a frame/array for later comparison.

    Supports ``pl.DataFrame``/``pl.Series`` (clone — copy-on-write, cheap),
    ``np.ndarray`` (materialized copy) and pandas objects (deep copy).
    """
    if isinstance(obj, (pl.DataFrame, pl.Series)):
        return obj.clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    try:  # pandas without importing it at module scope
        import pandas as pd

        if isinstance(obj, (pd.DataFrame, pd.Series, pd.Index)):
            return obj.copy(deep=True)
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"snapshot: unsupported type {type(obj).__name__}")


def _equal(before, after) -> bool:
    if isinstance(before, pl.DataFrame):
        return before.equals(after)
    if isinstance(before, pl.Series):
        return before.equals(after)
    if isinstance(before, np.ndarray):
        if before.dtype.kind in "fc":
            return bool(np.array_equal(before, after, equal_nan=True))
        return bool(np.array_equal(before, after))
    import pandas as pd

    if isinstance(before, (pd.DataFrame, pd.Series)):
        return bool(before.equals(after))
    if isinstance(before, pd.Index):
        return bool(before.equals(after))
    raise TypeError(f"assert_unmutated: unsupported type {type(before).__name__}")


class assert_unmutated:
    """Context manager asserting arguments are unchanged by a body.

    Test-suite tool for LAW 6 (data cost: one snapshot per argument)::

        with assert_unmutated(bars, weights):
            out = block.apply(bars)

    Raises :class:`ContractError` naming the mutated argument by position.
    """

    def __init__(self, *objects, context: str = "") -> None:
        self._objects = objects
        self._context = context
        self._snapshots: Sequence = ()

    def __enter__(self) -> "assert_unmutated":
        self._snapshots = tuple(snapshot(o) for o in self._objects)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            return  # body failed; do not mask its exception
        for i, (before, after) in enumerate(zip(self._snapshots, self._objects)):
            if not _equal(before, after):
                where = f" ({self._context})" if self._context else ""
                raise ContractError(
                    f"argument #{i} ({type(after).__name__}) was mutated by "
                    f"the guarded operation{where} — static operations must "
                    "not mutate their inputs (LAW 6)"
                )
