"""Block: the decomposable unit of the pipeline (docs/TARGET_ARCHITECTURE.md §4).

A Block is a stateless, configured transform over a polars frame:

- **configuration** is fixed at construction (frozen dataclass fields);
- **``requires``** names the input columns it reads; **``provides``** names
  the columns it adds. Undeclared columns pass through untouched;
- **``apply(frame)``** is the only public entry point. It validates the
  input contract, runs the subclass ``_apply``, attributes any failure to
  this block (LAW 2), verifies the promised outputs appeared and nothing
  out-of-contract was dropped (LAW 1), and logs one human-readable line
  (LAW 4);
- blocks never mutate the input frame (LAW 6) and hold no data state
  (LAW 7) — ``apply`` may be called any number of times, concurrently,
  on any frame satisfying the contract.

``Pipeline`` composes blocks sequentially with per-stage failure
attribution. Composition is intentionally minimal in Phase 1 — richer
combinators (feature unions, fitted transforms) arrive with the layers
that need them.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, Optional

import polars as pl

from src.core.contracts import FrameSchema, ValidationLevel
from src.core.errors import BlockError, ConfigError, ContractError
from src.core.log import get_logger


class Block(ABC):
    """Stateless configured transform ``pl.DataFrame -> pl.DataFrame``.

    Subclasses are encouraged to be frozen dataclasses (configuration is
    then introspectable via :meth:`params` and reprs are meaningful)::

        @dataclass(frozen=True)
        class AddReturns(Block):
            provides = ("r",)
            requires = ("close",)

            def _apply(self, frame): ...
    """

    #: Columns this block reads. Presence is enforced before ``_apply``.
    requires: ClassVar[tuple[str, ...]] = ()
    #: Columns this block adds. Presence is enforced after ``_apply``.
    provides: ClassVar[tuple[str, ...]] = ()
    #: Optional full schemas (richer than requires/provides name lists).
    input_schema: ClassVar[Optional[FrameSchema]] = None
    output_schema: ClassVar[Optional[FrameSchema]] = None
    #: Logger component; defaults to the class name at first use.
    log_component: ClassVar[str] = ""

    # -- public API ----------------------------------------------------------

    @property
    def name(self) -> str:
        return type(self).__name__

    def params(self) -> dict[str, Any]:
        """Configuration as a plain dict (dataclass fields), for logging,
        hashing, and persistence. Non-dataclass blocks return ``{}``."""
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)
        return {}

    def apply(
        self,
        frame: pl.DataFrame,
        *,
        level: ValidationLevel = "structure",
        log_level: int = logging.INFO,
    ) -> pl.DataFrame:
        """Validate -> transform -> verify -> log. The only public entry.

        ``level`` selects contract strictness for the declared schemas
        (``"data"`` scans values; use at workflow boundaries).
        """
        logger = get_logger(self.log_component or self.name)
        self._check_input(frame, level=level)

        t0 = time.perf_counter()
        try:
            out = self._apply(frame)
        except BlockError:
            raise  # already attributed by an inner block
        except Exception as exc:
            raise BlockError(
                f"{type(exc).__name__}: {exc}", block=self.name
            ) from exc
        elapsed = time.perf_counter() - t0

        self._check_output(frame, out, level=level)
        logger.log(
            log_level,
            "%s: %s rows -> %s rows, +%d column(s) in %.3fs",
            self.name,
            f"{frame.height:,}",
            f"{out.height:,}",
            len(out.columns) - len(frame.columns),
            elapsed,
        )
        return out

    # -- to implement ----------------------------------------------------------

    @abstractmethod
    def _apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Pure transform. Must not mutate ``frame``; returns a new frame."""

    # -- contract enforcement --------------------------------------------------

    def _check_input(self, frame: pl.DataFrame, *, level: ValidationLevel) -> None:
        missing = [c for c in self.requires if c not in frame.columns]
        if missing:
            raise ContractError(
                f"{self.name}: input frame is missing required column(s) "
                f"{missing}"
            )
        if self.input_schema is not None:
            self.input_schema.validate(frame, level=level, context=f"{self.name} input")

    def _check_output(
        self, frame: pl.DataFrame, out: pl.DataFrame, *, level: ValidationLevel
    ) -> None:
        if not isinstance(out, pl.DataFrame):
            raise BlockError(
                f"_apply returned {type(out).__name__}, expected pl.DataFrame",
                block=self.name,
            )
        absent = [c for c in self.provides if c not in out.columns]
        if absent:
            raise BlockError(
                f"declared provides {absent} not present on the output frame",
                block=self.name,
            )
        dropped = [
            c for c in frame.columns
            if c not in out.columns and c not in self.provides
        ]
        if dropped:
            raise BlockError(
                f"out-of-contract column(s) dropped: {dropped[:8]} — a block "
                "must pass through columns it does not provide",
                block=self.name,
            )
        if self.output_schema is not None:
            self.output_schema.validate(out, level=level, context=f"{self.name} output")


class Pipeline(Block):
    """Sequential composition with per-stage failure attribution.

    ``requires`` of the pipeline is its first-order dependency set: columns
    some stage requires that no earlier stage provides. ``provides`` is the
    union of all stage provides.
    """

    def __init__(self, *blocks: Block, name: str = "") -> None:
        if not blocks:
            raise ConfigError("Pipeline needs at least one block")
        bad = [b for b in blocks if not isinstance(b, Block)]
        if bad:
            raise ConfigError(
                f"Pipeline accepts Block instances only; got "
                f"{[type(b).__name__ for b in bad]}"
            )
        self.blocks: tuple[Block, ...] = tuple(blocks)
        self._name = name or "Pipeline"
        provided: set[str] = set()
        requires: list[str] = []
        provides: list[str] = []
        for b in self.blocks:
            for c in b.requires:
                if c not in provided and c not in requires:
                    requires.append(c)
            for c in b.provides:
                provided.add(c)
                provides.append(c)
        # Instance-level shadowing of the ClassVar declarations.
        self.requires = tuple(requires)  # type: ignore[misc]
        self.provides = tuple(provides)  # type: ignore[misc]

    @property
    def name(self) -> str:
        return self._name

    def params(self) -> dict[str, Any]:
        return {
            "stages": [
                {"block": b.name, "params": b.params()} for b in self.blocks
            ]
        }

    def _apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        out = frame
        n = len(self.blocks)
        for i, block in enumerate(self.blocks, start=1):
            try:
                out = block.apply(out)
            except BlockError as exc:
                # Re-attribute with the pipeline position, keep the cause.
                raise BlockError(
                    str(exc),
                    block=self._name,
                    stage=f"{i}/{n}:{block.name}",
                ) from (exc.__cause__ or exc)
        return out


def as_tuple(names: Iterable[str]) -> tuple[str, ...]:
    """Small helper for declaring requires/provides from any iterable."""
    return tuple(names)
