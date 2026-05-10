"""Bootstrap primitives: iid resampling with optional class stratification.

Public API:
- ``iid_indices(n, B, rng, *, stratify=None)`` -> (B, n) int array
- ``bootstrap_metric(fn, y, p, *, B, ci, stratify, seed)`` -> BootstrapResult

Stratification resamples within each class so the resampled empirical class
counts equal the original. This stabilizes base-rate-sensitive metrics
(PR-AUC, ECE) and is the right default for production-trading evaluation
where prevalence is a property of the eval window, not a property of the
bootstrap procedure.
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
) -> BootstrapResult:
    """Bootstrap an arbitrary ``(y_true, y_pred_proba) -> float`` metric.

    Class-stratified iid bootstrap by default. Set ``stratify=False`` to let
    prevalence vary across replicates.

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
