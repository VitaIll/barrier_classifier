"""Kernel: contracts, blocks, errors, logging, numerics, RNG policy.

The foundation layer of the target architecture
(docs/TARGET_ARCHITECTURE.md). Domain packages (labels, weights, features,
strategy, ...) build on these primitives; nothing in ``src/core`` imports
from any domain package.
"""

from src.core.block import Block, Pipeline
from src.core.contracts import (
    ColumnSpec,
    FrameSchema,
    assert_unmutated,
    require_columns,
    snapshot,
)
from src.core.errors import (
    BlockError,
    ConfigError,
    ContractError,
    CoreError,
    StateError,
)
from src.core.log import configure_logging, get_logger, timed
from src.core.num import (
    EPS,
    assert_all_finite,
    clip_exp,
    require_finite_scalar,
    safe_div,
    stable_sigmoid,
)
from src.core.rng import RngLike, resolve_rng

__all__ = [
    "Block",
    "Pipeline",
    "ColumnSpec",
    "FrameSchema",
    "assert_unmutated",
    "require_columns",
    "snapshot",
    "BlockError",
    "ConfigError",
    "ContractError",
    "CoreError",
    "StateError",
    "configure_logging",
    "get_logger",
    "timed",
    "EPS",
    "assert_all_finite",
    "clip_exp",
    "require_finite_scalar",
    "safe_div",
    "stable_sigmoid",
    "RngLike",
    "resolve_rng",
]
