"""Error taxonomy for the block kernel.

Every failure raised by kernel code is a :class:`CoreError` subclass, so
callers can catch by meaning instead of by string. Domain packages derive
their own errors from these roots.

Design rules (docs/TARGET_ARCHITECTURE.md §2, LAW 2):

- ``ConfigError``     — invalid construction parameters. Subclasses
  ``ValueError`` so legacy call sites that catch ``ValueError`` keep working.
- ``ContractError``   — a data hand-off violated its declared schema/contract
  (missing column, wrong dtype, non-finite where finiteness was promised,
  misaligned frames). Also a ``ValueError``.
- ``StateError``      — object-lifecycle misuse (using a closed resource,
  applying an unfitted artifact). Subclasses ``RuntimeError``.
- ``BlockError``      — any failure *inside* a block, wrapped with the block's
  name and pipeline position so the responsible component is always named.
  The original exception is chained as ``__cause__``.
"""

from __future__ import annotations


class CoreError(Exception):
    """Root of the kernel error taxonomy."""


class ConfigError(CoreError, ValueError):
    """A block/spec was constructed with invalid parameters."""


class ContractError(CoreError, ValueError):
    """A data hand-off violated its declared contract."""


class StateError(CoreError, RuntimeError):
    """An object was used outside its declared lifecycle."""


class BlockError(CoreError, RuntimeError):
    """A block failed; carries attribution to the failing component.

    Always raised ``from`` the original exception, so ``__cause__`` holds
    the root failure and the traceback is preserved.
    """

    def __init__(
        self,
        message: str,
        *,
        block: str,
        stage: str | None = None,
    ) -> None:
        self.block = block
        self.stage = stage
        prefix = f"[{block}]" if stage is None else f"[{block} @ {stage}]"
        super().__init__(f"{prefix} {message}")
