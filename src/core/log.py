"""Logging conventions for the kernel (docs/TARGET_ARCHITECTURE.md §2, LAW 4).

Every component gets a namespaced logger under the ``bc`` root
(``bc.labels``, ``bc.features``, ...). Key actions log ONE human-readable
line at INFO: what ran, over how much data, with which key parameters, in
how long, and what was dropped/trimmed/imputed (with counts).

``configure_logging`` is the opt-in for scripts and notebooks; libraries
never install handlers themselves.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

_ROOT = "bc"


def get_logger(component: str) -> logging.Logger:
    """Namespaced logger, e.g. ``get_logger('labels')`` -> ``bc.labels``."""
    if not component:
        return logging.getLogger(_ROOT)
    return logging.getLogger(f"{_ROOT}.{component}")


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a stream handler to the ``bc`` root (idempotent).

    For scripts/notebooks only — importing library code never calls this.
    """
    root = logging.getLogger(_ROOT)
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        )
        root.addHandler(handler)


@contextmanager
def timed(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
) -> Iterator[dict]:
    """Log ``message ... done in Xs`` around a body.

    Yields a dict the body may fill with extra ``key=value`` context that is
    appended to the completion line::

        with timed(log, "labeling 527,040 rows") as info:
            info["positives"] = n_pos
        # -> "labeling 527,040 rows: done in 0.05s (positives=36412)"

    On exception the failure is logged at ERROR with the elapsed time and
    re-raised unchanged (attribution/wrapping is the Block layer's job).
    """
    t0 = time.perf_counter()
    info: dict = {}
    try:
        yield info
    except BaseException:
        elapsed = time.perf_counter() - t0
        logger.error("%s: FAILED after %.2fs", message, elapsed)
        raise
    elapsed = time.perf_counter() - t0
    if info:
        extras = ", ".join(f"{k}={v}" for k, v in info.items())
        logger.log(level, "%s: done in %.2fs (%s)", message, elapsed, extras)
    else:
        logger.log(level, "%s: done in %.2fs", message, elapsed)
