"""Bootstrap primitives: iid resampling with optional class stratification,
and **block bootstrap** for autocorrelated label streams.

Public API:
- ``iid_indices(n, B, rng, *, stratify=None)`` -> (B, n) int array
- ``block_indices(n, B, rng, *, block_size)`` -> (B, n) int array
- ``bootstrap_metric(fn, y, p, *, B, ci, stratify, seed, block_size=None)`` -> BootstrapResult

**When to use which**

IID bootstrap (default) is correct when labels are independent — that's true
for the legacy *boundary-cadence* dataset, where consecutive labels look at
non-overlapping M-bar windows. Stratification resamples within each class
so the resampled empirical class counts equal the original; it stabilizes
base-rate-sensitive metrics (PR-AUC, ECE) and is the right default at
boundary cadence.

Block bootstrap (``block_size > 1``) is required when labels are
autocorrelated — that's the case for the *1-min-cadence* dataset, where
adjacent labels share M−1 of their M future bars. IID bootstrap on that
data dramatically underestimates CI width (by a factor of roughly √M);
block bootstrap samples contiguous blocks so the within-block correlation
structure is preserved across replicates. Block bootstrap is incompatible
with class stratification (blocks are sampled without regard to class), so
``stratify`` is ignored when ``block_size`` is set.

Pick ``block_size`` ≈ M (the label horizon). At BTCUSDT M=20 that means
``block_size=20``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

DEFAULT_B = 1000
DEFAULT_CI = 0.95


@dataclass
class BootstrapResult:
    point: float
    median: float
    ci_low: float
    ci_high: float
    ci: float
    B: int
    samples: np.ndarray = field(repr=False)

    def to_dict(self, *, include_samples: bool = False) -> dict:
        d = {
            "point": float(self.point),
            "median": float(self.median),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "ci": float(self.ci),
            "B": int(self.B),
        }
        if include_samples:
            d["samples"] = self.samples.tolist()
        return d


def block_indices(
    n: int,
    B: int,
    rng: np.random.Generator,
    *,
    block_size: int,
) -> np.ndarray:
    """Moving-block bootstrap: ``(B, n)`` index matrix built by concatenating
    random contiguous blocks of length ``block_size``.

    The block start index is uniformly distributed in ``[0, n - block_size]``,
    so each block contains ``block_size`` consecutive original indices. We
    concatenate ``ceil(n / block_size)`` blocks per replicate, then truncate
    to length ``n``.

    Preserves the within-block correlation structure of the original
    sequence. Required for autocorrelated label streams (e.g. 1-min cadence
    barrier labels where adjacent labels share M−1 future bars).
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if block_size > n:
        raise ValueError(f"block_size ({block_size}) > n ({n})")
    n_blocks = int(np.ceil(n / float(block_size)))
    starts = rng.integers(0, n - block_size + 1, size=(B, n_blocks), dtype=np.int64)
    offsets = np.arange(block_size, dtype=np.int64)
    # (B, n_blocks, 1) + (1, 1, block_size) -> (B, n_blocks, block_size)
    blocks = starts[..., None] + offsets[None, None, :]
    out = blocks.reshape(B, n_blocks * block_size)
    return out[:, :n]


def iid_indices(
    n: int,
    B: int,
    rng: np.random.Generator,
    *,
    stratify: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a ``(B, n)`` integer index array for iid bootstrap with replacement.

    If ``stratify`` is a binary 0/1 array of length ``n``, resample within each
    class so each row has the same per-class counts as ``stratify``. Falls back
    to plain iid when ``stratify`` is single-class (degenerate).
    """
    if stratify is None:
        return rng.integers(0, n, size=(B, n))
    s = np.asarray(stratify).astype(int)
    pos = np.flatnonzero(s == 1)
    neg = np.flatnonzero(s == 0)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return rng.integers(0, n, size=(B, n))
    pos_idx = pos[rng.integers(0, n_pos, size=(B, n_pos))]
    neg_idx = neg[rng.integers(0, n_neg, size=(B, n_neg))]
    return np.concatenate([pos_idx, neg_idx], axis=1)


def bootstrap_metric(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    p: np.ndarray,
    *,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
    block_size: Optional[int] = None,
) -> BootstrapResult:
    """Bootstrap an arbitrary ``(y_true, y_pred_proba) -> float`` metric.

    Class-stratified IID bootstrap by default (the right call for
    independent samples — e.g. the legacy boundary-cadence dataset).

    For autocorrelated label streams (e.g. 1-min-cadence barrier labels,
    where adjacent samples share M−1 of their M future bars), pass
    ``block_size`` ≈ M (the label horizon). This swaps to a moving-block
    bootstrap so the within-block correlation structure is preserved
    across replicates, producing honest (wider) CIs. Stratification is
    incompatible with block bootstrap and is ignored when ``block_size`` is set.

    Returns ``BootstrapResult`` with ``point`` (metric on the full data),
    ``median``, ``ci_low``, ``ci_high`` (percentile CI at level ``ci``), and
    raw ``samples`` of shape ``(B,)`` for downstream use.
    """
    y_arr = np.asarray(y)
    p_arr = np.asarray(p)
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"y and p must have the same shape, got {y_arr.shape} vs {p_arr.shape}")
    if y_arr.ndim != 1:
        raise ValueError(f"y and p must be 1D, got {y_arr.ndim}D")
    rng = np.random.default_rng(seed)
    if block_size is not None and int(block_size) > 1:
        idx = block_indices(len(y_arr), B, rng, block_size=int(block_size))
    else:
        idx = iid_indices(len(y_arr), B, rng, stratify=y_arr if stratify else None)
    samples = np.empty(B, dtype=float)
    for b in range(B):
        i = idx[b]
        samples[b] = float(fn(y_arr[i], p_arr[i]))
    point = float(fn(y_arr, p_arr))
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.quantile(samples, alpha))
    ci_high = float(np.quantile(samples, 1.0 - alpha))
    median = float(np.quantile(samples, 0.5))
    return BootstrapResult(
        point=point,
        median=median,
        ci_low=ci_low,
        ci_high=ci_high,
        ci=ci,
        B=B,
        samples=samples,
    )
