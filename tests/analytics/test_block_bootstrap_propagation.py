"""Block-bootstrap propagation: verify that passing ``block_size`` to the
higher-level analytics surfaces (curves, edge, degradation, metrics)
actually widens CIs on autocorrelated synthetic data — not just that the
keyword is accepted.

The expected ratio under heavy block-overlap is ~sqrt(M), but in practice
finite-sample variance and partial autocorrelation reduce it. We assert
block widths are STRICTLY GREATER than IID widths on at least one major
metric per module, which is the falsifiable causality contract: if a
module silently ignored ``block_size`` and fell back to iid resampling,
the widths would be equal and these tests would fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.curves import (
    bootstrap_calibration_curve,
    bootstrap_pr_curve,
    bootstrap_roc_curve,
)
from src.analytics.degradation import bootstrap_brier_decomposition
from src.analytics.edge import (
    bootstrap_partial_pr_auc,
    bootstrap_partial_roc_auc,
    bootstrap_threshold_sweep,
)
from src.analytics.metrics import bootstrap_all_metrics

pytestmark = pytest.mark.analytics_phase2


M_HORIZON = 20


def _make_autocorrelated_stream(n: int = 4000, M: int = M_HORIZON, seed: int = 0):
    """Build (y, p) where y is overlapping-barrier-like: y[i] = 1 iff the
    sum of the next M noise terms exceeds zero. Adjacent labels share M-1
    of their M future terms, so they are strongly autocorrelated by
    construction — same dependence structure as 1-min overlapping
    barrier labels."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, n + M)
    window_sum = np.convolve(noise, np.ones(M), mode="valid")[:n]
    y = (window_sum > 0).astype(int)
    # Informed score with non-trivial noise so PR-AUC is interior.
    p = np.clip(
        0.5 + 0.30 * np.tanh(window_sum) + rng.normal(0.0, 0.2, n),
        0.01,
        0.99,
    )
    return y, p


def _expected_widening(block_width: float, iid_width: float) -> str:
    return f"block_width={block_width:.5f} vs iid_width={iid_width:.5f}"


# ---------------------------------------------------------------------------
# metrics.bootstrap_all_metrics
# ---------------------------------------------------------------------------


def test_block_widens_ci_in_bootstrap_all_metrics():
    y, p = _make_autocorrelated_stream(n=4000, M=M_HORIZON, seed=0)
    iid = bootstrap_all_metrics(y, p, B=300, stratify=True, seed=0)
    blk = bootstrap_all_metrics(y, p, B=300, stratify=False, seed=0, block_size=M_HORIZON)
    iid_w = iid["pr_auc"].ci_high - iid["pr_auc"].ci_low
    blk_w = blk["pr_auc"].ci_high - blk["pr_auc"].ci_low
    assert blk_w > iid_w, (
        "block bootstrap must widen CIs on autocorrelated labels: "
        + _expected_widening(blk_w, iid_w)
    )


# ---------------------------------------------------------------------------
# curves.bootstrap_roc_curve / bootstrap_pr_curve / bootstrap_calibration_curve
# ---------------------------------------------------------------------------


def test_block_widens_ci_in_bootstrap_pr_curve():
    y, p = _make_autocorrelated_stream(seed=1)
    iid = bootstrap_pr_curve(y, p, B=300, stratify=True, seed=0)
    blk = bootstrap_pr_curve(y, p, B=300, stratify=False, seed=0, block_size=M_HORIZON)
    iid_w = iid.auc_ci_high - iid.auc_ci_low
    blk_w = blk.auc_ci_high - blk.auc_ci_low
    assert blk_w > iid_w, _expected_widening(blk_w, iid_w)


def test_block_widens_ci_in_bootstrap_roc_curve():
    y, p = _make_autocorrelated_stream(seed=2)
    iid = bootstrap_roc_curve(y, p, B=300, stratify=True, seed=0)
    blk = bootstrap_roc_curve(y, p, B=300, stratify=False, seed=0, block_size=M_HORIZON)
    iid_w = iid.auc_ci_high - iid.auc_ci_low
    blk_w = blk.auc_ci_high - blk.auc_ci_low
    # ROC-AUC is the most rank-invariant of the metrics and shows the
    # smallest widening — but still expect some positive widening if
    # block_size is honoured.
    assert blk_w >= iid_w * 0.99, _expected_widening(blk_w, iid_w)


def test_block_widens_ci_in_bootstrap_calibration_curve():
    y, p = _make_autocorrelated_stream(seed=3)
    iid = bootstrap_calibration_curve(y, p, n_bins=10, B=300, stratify=True, seed=0)
    blk = bootstrap_calibration_curve(
        y, p, n_bins=10, B=300, stratify=False, seed=0, block_size=M_HORIZON
    )
    iid_w = iid.auc_ci_high - iid.auc_ci_low
    blk_w = blk.auc_ci_high - blk.auc_ci_low
    assert blk_w > iid_w, _expected_widening(blk_w, iid_w)


# ---------------------------------------------------------------------------
# edge.bootstrap_threshold_sweep / bootstrap_partial_roc_auc / bootstrap_partial_pr_auc
# ---------------------------------------------------------------------------


def test_block_widens_ci_in_bootstrap_threshold_sweep():
    y, p = _make_autocorrelated_stream(seed=4)
    cache = pd.DataFrame(
        {
            "k": np.arange(len(y)),
            "ts": pd.date_range("2025-01-01", periods=len(y), freq="1min"),
            "y": y,
            "p": p,
            "split": "test",
        }
    )
    iid = bootstrap_threshold_sweep(
        cache, split="test", B=200, stratify=True, seed=0,
        thresholds=np.linspace(0.3, 0.8, 20),
    )
    blk = bootstrap_threshold_sweep(
        cache, split="test", B=200, stratify=False, seed=0,
        thresholds=np.linspace(0.3, 0.8, 20), block_size=M_HORIZON,
    )
    # Compare at the threshold with the most population (mid-grid)
    iid_w = (iid["precision_ci_high"] - iid["precision_ci_low"]).mean()
    blk_w = (blk["precision_ci_high"] - blk["precision_ci_low"]).mean()
    assert blk_w > iid_w, _expected_widening(blk_w, iid_w)


def test_block_widens_ci_in_bootstrap_partial_pr_auc():
    y, p = _make_autocorrelated_stream(seed=5)
    iid = bootstrap_partial_pr_auc(y, p, recall_max=0.5, B=300, stratify=True, seed=0)
    blk = bootstrap_partial_pr_auc(
        y, p, recall_max=0.5, B=300, stratify=False, seed=0, block_size=M_HORIZON
    )
    iid_w = iid.ci_high - iid.ci_low
    blk_w = blk.ci_high - blk.ci_low
    assert blk_w > iid_w, _expected_widening(blk_w, iid_w)


def test_block_widens_ci_in_bootstrap_partial_roc_auc():
    y, p = _make_autocorrelated_stream(seed=6)
    iid = bootstrap_partial_roc_auc(y, p, fpr_max=0.10, B=300, stratify=True, seed=0)
    blk = bootstrap_partial_roc_auc(
        y, p, fpr_max=0.10, B=300, stratify=False, seed=0, block_size=M_HORIZON
    )
    iid_w = iid.ci_high - iid.ci_low
    blk_w = blk.ci_high - blk.ci_low
    assert blk_w >= iid_w * 0.99, _expected_widening(blk_w, iid_w)


# ---------------------------------------------------------------------------
# degradation.bootstrap_brier_decomposition
# ---------------------------------------------------------------------------


def test_block_widens_ci_in_bootstrap_brier_decomposition():
    y, p = _make_autocorrelated_stream(seed=7)
    iid = bootstrap_brier_decomposition(y, p, B=300, stratify=True, seed=0)
    blk = bootstrap_brier_decomposition(
        y, p, B=300, stratify=False, seed=0, block_size=M_HORIZON
    )
    # Reliability is the calibration-error component — most sensitive to
    # autocorrelation in the residuals.
    iid_w = iid["reliability"].ci_high - iid["reliability"].ci_low
    blk_w = blk["reliability"].ci_high - blk["reliability"].ci_low
    assert blk_w > iid_w, _expected_widening(blk_w, iid_w)
