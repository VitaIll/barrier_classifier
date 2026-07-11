"""RNG injection policy (docs/TARGET_ARCHITECTURE.md §2, LAW 7).

The only sanctioned way randomness enters kernel code:

- callers pass either a seed (reproducible) or a ``numpy.random.Generator``
  (caller-owned stream, thread it through a whole workflow);
- ``None`` means fresh OS entropy — allowed only at true entry points
  (scripts), never as a library default for anything that lands in a
  persisted artifact.

Global ``np.random.*`` state is never read or written.
"""

from __future__ import annotations

from typing import Union

import numpy as np

RngLike = Union[int, np.random.Generator, None]


def resolve_rng(rng: RngLike) -> np.random.Generator:
    """Normalize ``seed | Generator | None`` to a ``Generator``.

    An explicit ``Generator`` is returned as-is (same object — the caller
    owns the stream and its state advances with use). An int seeds a fresh
    generator. ``None`` draws fresh OS entropy.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        if isinstance(rng, bool):  # bool is an int subclass; reject it
            raise TypeError("rng must be an int seed, a Generator, or None; got bool")
        return np.random.default_rng(int(rng))
    raise TypeError(
        f"rng must be an int seed, a numpy Generator, or None; got {type(rng).__name__}"
    )
