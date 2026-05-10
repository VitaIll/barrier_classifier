"""Core point metrics with bootstrap CIs.

Drop-in replacement for ``utils.compute_all_metrics`` that returns
``BootstrapResult`` per metric. Same metric definitions; CIs added.

Available metrics: ROC-AUC, PR-AUC, log-loss, Brier, ECE (10-bin).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.utils import expected_calibration_error

from .bootstrap import DEFAULT_B, DEFAULT_CI, BootstrapResult, bootstrap_metric


_METRIC_FUNCS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": lambda y, p: float(roc_auc_score(y, p)),
    "pr_auc": lambda y, p: float(average_precision_score(y, p)),
    "log_loss": lambda y, p: float(log_loss(y, p, labels=[0, 1])),
    "brier_score": lambda y, p: float(brier_score_loss(y, p)),
    "ece_10bin": lambda y, p: float(expected_calibration_error(y, p, n_bins=10)),
}


def bootstrap_all_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
) -> Dict[str, BootstrapResult]:
    """Bootstrap the headline metric bundle on ``(y, p)``.

    Returns ``{metric_name: BootstrapResult}``. Point estimates match
    ``utils.compute_all_metrics`` (with ``ece_10bin`` mapping to ``ece``).
    """
    out: Dict[str, BootstrapResult] = {}
    for name, fn in _METRIC_FUNCS.items():
        out[name] = bootstrap_metric(fn, y, p, B=B, ci=ci, stratify=stratify, seed=seed)
    return out


def bootstrap_metrics_by_regime(
    y: np.ndarray,
    p: np.ndarray,
    regime: np.ndarray,
    *,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
    metrics: Optional[Sequence[str]] = None,
    min_samples: int = 50,
) -> Dict[str, Dict[str, BootstrapResult]]:
    """Bootstrap a subset of metrics within each tercile of ``regime``.

    Terciles are computed via ``pd.qcut(regime, 3) -> {low, med, high}``.
    Regimes with fewer than ``min_samples`` rows are skipped. Per-regime
    bootstrap is class-stratified within the regime by default.

    Returns ``{regime_label: {metric_name: BootstrapResult}}``.
    """
    metric_names = list(metrics) if metrics is not None else list(_METRIC_FUNCS.keys())
    unknown = set(metric_names) - set(_METRIC_FUNCS.keys())
    if unknown:
        raise ValueError(f"Unknown metric names: {sorted(unknown)}")
    terciles = pd.qcut(regime, 3, labels=["low", "med", "high"])
    out: Dict[str, Dict[str, BootstrapResult]] = {}
    for label in ["low", "med", "high"]:
        mask = np.asarray(terciles == label)
        if mask.sum() < min_samples:
            continue
        y_r = y[mask]
        p_r = p[mask]
        # Some regimes may be single-class for very small slices; skip those.
        if y_r.sum() == 0 or y_r.sum() == len(y_r):
            continue
        out[label] = {}
        for name in metric_names:
            fn = _METRIC_FUNCS[name]
            out[label][name] = bootstrap_metric(
                fn, y_r, p_r, B=B, ci=ci, stratify=stratify, seed=seed
            )
    return out


def to_summary_dict(results: Dict[str, BootstrapResult]) -> dict:
    """Compact JSON-serializable summary excluding raw bootstrap samples."""
    return {name: r.to_dict(include_samples=False) for name, r in results.items()}


def by_regime_to_summary_dict(
    results: Dict[str, Dict[str, BootstrapResult]],
) -> dict:
    """JSON-serializable summary for ``bootstrap_metrics_by_regime`` output."""
    return {regime: to_summary_dict(metrics) for regime, metrics in results.items()}
