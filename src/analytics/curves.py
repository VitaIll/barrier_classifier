"""Curves with bootstrap quantile bands: ROC, PR, calibration.

All three share the same bootstrap primitive (``bootstrap.iid_indices``,
class-stratified by default) and the same dataclass (``CurveBootstrapResult``).
For each curve:

- the **point** is the curve computed on the full data, interpolated onto a
  fixed x-grid;
- the **band** is the per-grid-point percentile interval over ``B`` resamples;
- the **auc_*** scalars are the curve's summary metric (ROC-AUC, PR-AP, ECE)
  with their bootstrap CI.

Interpolation choices:

- **ROC** uses linear interpolation on a fixed FPR grid. TPR is monotone
  non-decreasing in FPR, so linear interp is exact for the underlying curve
  between knots.
- **PR** uses **max-envelope interpolation** on a fixed recall grid:
  ``precision_at_recall(r) = max{precision[i] : recall[i] >= r}``. This is
  the Pascal VOC convention and matches the AP integral that
  ``average_precision_score`` computes; using linear interp here would
  inflate the band in regions where the raw curve is jagged.
- **Calibration** uses **fixed equal-width bins** (default 10) over
  predicted probability. The point estimate per bin is
  ``(mean_p_full, empirical_y_full)``; the bootstrap CI is on the
  empirical fraction y holding bins fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils import expected_calibration_error

from .bootstrap import DEFAULT_B, DEFAULT_CI, block_indices, iid_indices


def _choose_indices(
    n: int,
    B: int,
    rng: np.random.Generator,
    *,
    stratify_y: Optional[np.ndarray],
    stratify: bool,
    block_size: Optional[int],
) -> np.ndarray:
    """Pick block or iid (optionally stratified) resampling indices.

    Centralized helper so every curve / metric bootstrap in this module
    uses the same precedence rule: ``block_size > 1`` -> moving-block
    bootstrap (stratification is incompatible with blocks and is
    ignored), else stratified-iid (if ``stratify=True`` and labels are
    available), else plain iid.
    """
    if block_size is not None and int(block_size) > 1:
        return block_indices(n, B, rng, block_size=int(block_size))
    return iid_indices(
        n, B, rng, stratify=stratify_y if stratify and stratify_y is not None else None
    )


@dataclass
class CurveBootstrapResult:
    """Outcome of bootstrapping a curve onto a fixed x-grid.

    Attributes
    ----------
    name : str
        ``'roc'``, ``'pr'``, or ``'calibration'``.
    x_grid : np.ndarray, shape (G,)
        Grid the curve is interpolated onto (FPR / recall / per-bin mean_p).
    point : np.ndarray, shape (G,)
        Curve on the full data interpolated to ``x_grid``.
    median, ci_low, ci_high : np.ndarray, shape (G,)
        Per-grid-point bootstrap quantiles.
    samples : np.ndarray, shape (B, G)
        Raw bootstrap samples (each row = one resample's curve).
    auc_point, auc_median, auc_ci_low, auc_ci_high : float
        Summary metric of the curve (ROC-AUC / AP / ECE) with bootstrap CI.
    """

    name: str
    x_grid: np.ndarray
    point: np.ndarray
    median: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    ci: float
    B: int
    samples: np.ndarray = field(repr=False)
    auc_point: float = float("nan")
    auc_median: float = float("nan")
    auc_ci_low: float = float("nan")
    auc_ci_high: float = float("nan")

    def to_summary_dict(self) -> dict:
        """JSON-serializable summary excluding raw ``(B, G)`` samples."""
        return {
            "name": self.name,
            "B": int(self.B),
            "ci": float(self.ci),
            "auc_point": float(self.auc_point),
            "auc_median": float(self.auc_median),
            "auc_ci_low": float(self.auc_ci_low),
            "auc_ci_high": float(self.auc_ci_high),
        }


# ---------------------------------------------------------------------------
# Curve interpolators
# ---------------------------------------------------------------------------


def _interp_roc(y: np.ndarray, p: np.ndarray, fpr_grid: np.ndarray) -> np.ndarray:
    """TPR at fixed FPR grid via the upper-envelope step curve.

    sklearn's ``roc_curve`` produces (FPR, TPR) pairs that can have duplicate
    FPRs (consecutive positives encountered as threshold decreases — TPR rises
    while FPR stays). For interpolation we take the maximum TPR per unique
    FPR (the standard ROC operating-point convention: "best TPR achievable at
    this FPR"). With both arrays monotone non-decreasing, the max per group
    is the last occurrence in the group.

    ``drop_intermediate=False`` is forced so the curve has every operating
    point — necessary for the upper-envelope to be exact.
    """
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    last_in_group = np.r_[np.diff(fpr) > 0, True]
    return np.interp(fpr_grid, fpr[last_in_group], tpr[last_in_group])


def _interp_pr(y: np.ndarray, p: np.ndarray, recall_grid: np.ndarray) -> np.ndarray:
    """Precision at fixed recall grid (max-envelope, Pascal VOC convention)."""
    precision, recall, _ = precision_recall_curve(y, p)
    order = np.argsort(recall)
    rec_sorted = recall[order]
    prec_sorted = precision[order]
    # rev_cummax[i] = max(prec_sorted[i:]) — max precision over recall >= rec_sorted[i]
    rev_cummax = np.maximum.accumulate(prec_sorted[::-1])[::-1]
    # For each r in grid, find first i with rec_sorted[i] >= r
    idx = np.searchsorted(rec_sorted, recall_grid, side="left")
    out = np.where(
        idx < len(rec_sorted),
        rev_cummax[np.minimum(idx, len(rec_sorted) - 1)],
        0.0,
    )
    # If the requested recall exceeds the empirical maximum, precision is undefined; use 0.
    out = np.where(recall_grid > rec_sorted[-1], 0.0, out)
    return out


def _calibration_per_bin(
    y: np.ndarray, p: np.ndarray, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean_p_per_bin, empirical_y_per_bin)`` for fixed bin edges.

    Bins where no sample falls produce ``nan`` for both — the caller decides
    how to render empties.
    """
    n_bins = len(bin_edges) - 1
    bin_idx = np.clip(np.digitize(p, bin_edges, right=False) - 1, 0, n_bins - 1)
    mean_p = np.full(n_bins, np.nan)
    emp_y = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            mean_p[b] = float(p[mask].mean())
            emp_y[b] = float(y[mask].mean())
    return mean_p, emp_y


# ---------------------------------------------------------------------------
# Public bootstrap-curve API
# ---------------------------------------------------------------------------


def bootstrap_roc_curve(
    y: np.ndarray,
    p: np.ndarray,
    *,
    fpr_grid: Optional[np.ndarray] = None,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
    block_size: Optional[int] = None,
) -> CurveBootstrapResult:
    """Bootstrap an ROC curve onto a fixed FPR grid.

    Returns the curve (TPR per FPR), per-grid-point CI band, and the AUC
    bootstrap distribution. Uses the same ``iid_indices(seed)`` as
    ``bootstrap_metric``, so the AUC samples are identical between the two
    functions when called with matching ``B``, ``stratify``, and ``seed``.

    Pass ``block_size`` ≈ M for autocorrelated label streams (1-min
    cadence). Stratification is incompatible with block bootstrap and is
    ignored when ``block_size`` is set.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    if fpr_grid is None:
        fpr_grid = np.linspace(0.0, 1.0, 201)

    point_tpr = _interp_roc(y, p, fpr_grid)
    point_auc = float(roc_auc_score(y, p))

    rng = np.random.default_rng(seed)
    idx = _choose_indices(
        len(y), B, rng, stratify_y=y, stratify=stratify, block_size=block_size
    )
    samples = np.full((B, len(fpr_grid)), np.nan, dtype=float)
    auc_samples = np.full(B, np.nan, dtype=float)
    for b in range(B):
        i = idx[b]
        try:
            samples[b] = _interp_roc(y[i], p[i], fpr_grid)
            auc_samples[b] = float(roc_auc_score(y[i], p[i]))
        except ValueError:
            # Single-class block resample — roc_auc_score raises. Leave NaN.
            pass

    alpha = (1.0 - ci) / 2.0
    return CurveBootstrapResult(
        name="roc",
        x_grid=fpr_grid,
        point=point_tpr,
        median=np.nanquantile(samples, 0.5, axis=0),
        ci_low=np.nanquantile(samples, alpha, axis=0),
        ci_high=np.nanquantile(samples, 1.0 - alpha, axis=0),
        ci=ci,
        B=B,
        samples=samples,
        auc_point=point_auc,
        auc_median=float(np.nanquantile(auc_samples, 0.5)),
        auc_ci_low=float(np.nanquantile(auc_samples, alpha)),
        auc_ci_high=float(np.nanquantile(auc_samples, 1.0 - alpha)),
    )


def bootstrap_pr_curve(
    y: np.ndarray,
    p: np.ndarray,
    *,
    recall_grid: Optional[np.ndarray] = None,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
    block_size: Optional[int] = None,
) -> CurveBootstrapResult:
    """Bootstrap a PR curve onto a fixed recall grid (max-envelope precision).

    AP samples are identical to ``bootstrap_metric(average_precision_score, ...)``
    with matching ``B``, ``stratify``, ``seed`` — see consistency test.

    Pass ``block_size`` ≈ M to switch to block bootstrap for autocorrelated
    label streams (1-min cadence).
    """
    y = np.asarray(y)
    p = np.asarray(p)
    if recall_grid is None:
        recall_grid = np.linspace(0.0, 1.0, 201)

    point_prec = _interp_pr(y, p, recall_grid)
    point_ap = float(average_precision_score(y, p))

    rng = np.random.default_rng(seed)
    idx = _choose_indices(
        len(y), B, rng, stratify_y=y, stratify=stratify, block_size=block_size
    )
    samples = np.full((B, len(recall_grid)), np.nan, dtype=float)
    ap_samples = np.full(B, np.nan, dtype=float)
    for b in range(B):
        i = idx[b]
        try:
            samples[b] = _interp_pr(y[i], p[i], recall_grid)
            ap_samples[b] = float(average_precision_score(y[i], p[i]))
        except ValueError:
            pass

    alpha = (1.0 - ci) / 2.0
    return CurveBootstrapResult(
        name="pr",
        x_grid=recall_grid,
        point=point_prec,
        median=np.nanquantile(samples, 0.5, axis=0),
        ci_low=np.nanquantile(samples, alpha, axis=0),
        ci_high=np.nanquantile(samples, 1.0 - alpha, axis=0),
        ci=ci,
        B=B,
        samples=samples,
        auc_point=point_ap,
        auc_median=float(np.nanquantile(ap_samples, 0.5)),
        auc_ci_low=float(np.nanquantile(ap_samples, alpha)),
        auc_ci_high=float(np.nanquantile(ap_samples, 1.0 - alpha)),
    )


def bootstrap_calibration_curve(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
    block_size: Optional[int] = None,
) -> CurveBootstrapResult:
    """Bootstrap a reliability diagram with per-bin CIs on empirical fraction.

    Bins are equal-width over [0, 1]. The point ``(x_grid, point)`` is the
    full-data ``(mean_p, empirical_y)`` per bin; ``ci_low``/``ci_high`` is
    the per-bin bootstrap percentile interval on empirical_y. ``auc_*``
    fields hold ECE (10-bin), the curve's summary metric.

    Empty bins (no predictions in [edge_i, edge_{i+1})) carry ``nan`` for
    both x and y so they can be skipped at plot time. Pass ``block_size``
    ≈ M to switch to block bootstrap for autocorrelated label streams.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    point_x, point_y = _calibration_per_bin(y, p, bin_edges)
    point_ece = float(expected_calibration_error(y, p, n_bins=n_bins))

    rng = np.random.default_rng(seed)
    idx = _choose_indices(
        len(y), B, rng, stratify_y=y, stratify=stratify, block_size=block_size
    )
    samples = np.full((B, n_bins), np.nan, dtype=float)
    ece_samples = np.full(B, np.nan, dtype=float)
    for b in range(B):
        i = idx[b]
        try:
            _, emp_y_b = _calibration_per_bin(y[i], p[i], bin_edges)
            samples[b] = emp_y_b
            ece_samples[b] = float(expected_calibration_error(y[i], p[i], n_bins=n_bins))
        except ValueError:
            pass

    alpha = (1.0 - ci) / 2.0
    return CurveBootstrapResult(
        name="calibration",
        x_grid=point_x,
        point=point_y,
        median=np.nanquantile(samples, 0.5, axis=0),
        ci_low=np.nanquantile(samples, alpha, axis=0),
        ci_high=np.nanquantile(samples, 1.0 - alpha, axis=0),
        ci=ci,
        B=B,
        samples=samples,
        auc_point=point_ece,
        auc_median=float(np.nanmedian(ece_samples)),
        auc_ci_low=float(np.nanquantile(ece_samples, alpha)),
        auc_ci_high=float(np.nanquantile(ece_samples, 1.0 - alpha)),
    )


# ---------------------------------------------------------------------------
# Plot helpers (lazy matplotlib imports keep module load cheap)
# ---------------------------------------------------------------------------


def plot_roc_with_band(
    result: CurveBootstrapResult,
    *,
    ax=None,
    color: str = "C0",
    label: str = "Model",
    alpha_band: float = 0.25,
    plot_diagonal: bool = True,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if plot_diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6, label="Random")
    ax.fill_between(
        result.x_grid, result.ci_low, result.ci_high, color=color, alpha=alpha_band
    )
    auc_str = (
        f"AUC={result.auc_point:.3f} [{result.auc_ci_low:.3f}, {result.auc_ci_high:.3f}]"
    )
    ax.plot(result.x_grid, result.point, color=color, label=f"{label} {auc_str}")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC ({int(result.ci * 100)}% bootstrap band, B={result.B})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return ax


def plot_pr_with_band(
    result: CurveBootstrapResult,
    *,
    base_rate: Optional[float] = None,
    ax=None,
    color: str = "C0",
    label: str = "Model",
    alpha_band: float = 0.25,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if base_rate is not None:
        ax.axhline(
            base_rate,
            linestyle="--",
            color="gray",
            alpha=0.6,
            label=f"Base rate ({base_rate:.3f})",
        )
    ax.fill_between(
        result.x_grid, result.ci_low, result.ci_high, color=color, alpha=alpha_band
    )
    ap_str = (
        f"AP={result.auc_point:.3f} [{result.auc_ci_low:.3f}, {result.auc_ci_high:.3f}]"
    )
    ax.plot(result.x_grid, result.point, color=color, label=f"{label} {ap_str}")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision-Recall ({int(result.ci * 100)}% bootstrap band, B={result.B})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return ax


def plot_calibration_with_ci(
    result: CurveBootstrapResult,
    *,
    ax=None,
    color: str = "C0",
    label: str = "Model",
    plot_perfect: bool = True,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if plot_perfect:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="Perfect")
    valid = ~np.isnan(result.x_grid) & ~np.isnan(result.point)
    yerr_low = np.where(valid, result.point - result.ci_low, 0.0)
    yerr_high = np.where(valid, result.ci_high - result.point, 0.0)
    ece_str = (
        f"ECE={result.auc_point:.3f} [{result.auc_ci_low:.3f}, {result.auc_ci_high:.3f}]"
    )
    ax.errorbar(
        result.x_grid[valid],
        result.point[valid],
        yerr=[yerr_low[valid], yerr_high[valid]],
        marker="o",
        color=color,
        capsize=3,
        label=f"{label} {ece_str}",
    )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Empirical frequency",
        title=f"Calibration ({int(result.ci * 100)}% bootstrap CI on empirical y, B={result.B})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return ax
