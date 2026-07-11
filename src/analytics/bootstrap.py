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
    # B_effective: count of bootstrap replicates that produced a finite metric
    # value. Block bootstrap on small / single-class datasets can yield
    # resamples on which roc_auc_score / precision_recall_curve raise
    # ValueError; those replicates contribute NaN to ``samples``, are
    # excluded from ``median`` / ``ci_low`` / ``ci_high`` (computed with
    # ``np.nanquantile`` / ``np.nanmedian``), and are counted out of
    # B_effective. Defaults to ``B`` so existing call sites that pre-date
    # the field continue to work — :py:meth:`__post_init__` fills it.
    B_effective: int = field(default=-1)

    def __post_init__(self) -> None:
        if self.B_effective < 0:
            self.B_effective = int(self.B)

    def to_dict(self, *, include_samples: bool = False) -> dict:
        d = {
            "point": float(self.point),
            "median": float(self.median),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "ci": float(self.ci),
            "B": int(self.B),
            "B_effective": int(self.B_effective),
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

    Tail-truncation behavior
    ------------------------
    The ``ceil(n / block_size) * block_size`` matrix may slightly exceed
    ``n`` rows; we **truncate** the last (partial) block to length ``n``
    rather than wrapping the indices to the head of the sequence. The
    alternative — *circular wrap*, where indices ``>= n`` fold back to
    the start — is deliberately rejected: in a chronological label stream
    the wrap would splice the sequence tail to its head, manufacturing a
    cross-boundary "neighbor" relationship that does not exist in the
    underlying data and leaking the test period's distribution back into
    earlier resamples. Truncation introduces a mild edge bias (the last
    ``block_size - 1`` indices are slightly under-represented near the
    tail) but preserves the no-cross-boundary invariant that matters for
    time-series CIs.
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


def choose_indices(
    n: int,
    B: int,
    rng: np.random.Generator,
    *,
    stratify_y: Optional[np.ndarray],
    stratify: bool,
    block_size: Optional[int],
) -> np.ndarray:
    """Pick block or iid (optionally stratified) resampling indices.

    THE precedence rule, shared by every bootstrap surface in this package
    (this helper used to exist as four private copies): ``block_size > 1``
    -> moving-block bootstrap (stratification is incompatible with blocks
    and is ignored), else stratified-iid (if ``stratify=True`` and labels
    are available), else plain iid.
    """
    if block_size is not None and int(block_size) > 1:
        return block_indices(n, B, rng, block_size=int(block_size))
    return iid_indices(
        n, B, rng, stratify=stratify_y if stratify and stratify_y is not None else None
    )


def bootstrap_apply(
    fn: Callable[[np.ndarray], float],
    idx: np.ndarray,
    *,
    B: Optional[int] = None,
) -> np.ndarray:
    """Evaluate ``fn(row_indices)`` per replicate; NaN-tolerant.

    The canonical resample loop: a replicate whose metric raises
    ``ValueError`` (single-class slice under block bootstrap) records NaN
    and drops out of ``nanquantile`` aggregation — the package-wide NaN
    contract. Returns the ``(B,)`` sample vector; callers aggregate.
    """
    n_reps = int(B) if B is not None else int(idx.shape[0])
    samples = np.full(n_reps, np.nan, dtype=float)
    for b in range(n_reps):
        try:
            samples[b] = float(fn(idx[b]))
        except ValueError:
            samples[b] = np.nan
    return samples


def wilson_interval(k: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Robust at small n where the normal approximation breaks down. Returns
    ``(0.0, 1.0)`` for ``n=0``. (Moved here from ``degradation`` — it is a
    generic statistic consumed across the package, not a drift concept;
    ``degradation.wilson_interval`` remains as a re-export.)
    """
    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import norm

    z = norm.ppf(1.0 - alpha / 2.0)
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half_width = z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    lo = center - half_width
    hi = center + half_width
    # Math says lo=0 exactly at k=0 and hi=1 exactly at k=n; floating-point
    # leaves micro-positive/-negative dust. Hardcode the boundary cases.
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return float(max(0.0, lo)), float(min(1.0, hi))


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
    idx = choose_indices(
        len(y_arr), B, rng,
        stratify_y=y_arr if stratify else None,
        stratify=stratify,
        block_size=block_size,
    )
    # NaN-tolerant loop: block resamples can produce single-class slices
    # that crash roc_auc_score / precision_recall_curve; those replicates
    # record NaN, drop from CI aggregation, and are counted out of
    # B_effective.
    samples = bootstrap_apply(lambda i: fn(y_arr[i], p_arr[i]), idx, B=B)
    point = float(fn(y_arr, p_arr))
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.nanquantile(samples, alpha))
    ci_high = float(np.nanquantile(samples, 1.0 - alpha))
    median = float(np.nanmedian(samples))
    b_effective = int(np.count_nonzero(~np.isnan(samples)))
    return BootstrapResult(
        point=point,
        median=median,
        ci_low=ci_low,
        ci_high=ci_high,
        ci=ci,
        B=B,
        samples=samples,
        B_effective=b_effective,
    )
