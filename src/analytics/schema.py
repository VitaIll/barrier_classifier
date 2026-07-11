"""Decision-cache schema: one declaration, validated at consumer entry.

The prediction cache's column contract used to live as hardcoded string
literals scattered across ``edge``/``degradation``/``cohorts`` (a missing
column surfaced as a bare ``KeyError`` deep inside pandas). The canonical
column set is declared once here (mirroring
``fast_train.CACHE_REQUIRED_COLS``, which stays for the writer side);
consumers call :func:`require_cache_columns` up front and fail with a
diagnostic that names the caller, the missing columns, and the likely fix.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.core.errors import ContractError

#: Written by fast_train.compute_predictions / nb03; read by every
#: analytics consumer.
CACHE_CORE_COLS: tuple[str, ...] = (
    "k", "ts", "y", "m_k", "tau_k", "phi", "regime", "p", "split",
)

#: Added by src/strategy/cache.py augmenters when needed.
CACHE_REALIZED_COL = "r_realized"
CACHE_OHLC_COLS: tuple[str, ...] = ("open", "high", "low", "close")


def require_cache_columns(
    cache: pd.DataFrame,
    required: Iterable[str],
    *,
    context: str,
) -> None:
    """Fail loudly (and helpfully) when the cache misses required columns."""
    missing = [c for c in required if c not in cache.columns]
    if not missing:
        return
    hints = []
    if CACHE_REALIZED_COL in missing:
        hints.append(
            "add r_realized via src.strategy.cache.augment_cache_with_r_realized"
        )
    if any(c in missing for c in CACHE_OHLC_COLS):
        hints.append(
            "add boundary OHLC via "
            "src.strategy.cache.augment_cache_with_boundary_ohlc"
        )
    hint = f" ({'; '.join(hints)})" if hints else ""
    raise ContractError(
        f"{context}: prediction cache is missing column(s) {missing}"
        f"{hint}. Cache frames carry {sorted(CACHE_CORE_COLS)} at minimum "
        "(fast_train.CACHE_REQUIRED_COLS)."
    )
