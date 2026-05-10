"""Tests for src.analytics.metrics.

Strategy: every public function is checked by computing the metric a second,
independent way (sklearn directly, or manual masked subset) and asserting
exact equality at the point estimate. Bootstrap CI properties are inherited
from test_bootstrap.py — here we only verify the wrapping is correct.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src import utils
from src.analytics.metrics import (
    bootstrap_all_metrics,
    bootstrap_metrics_by_regime,
    by_regime_to_summary_dict,
    to_summary_dict,
)

pytestmark = pytest.mark.analytics_phase0


def _make_synthetic(n: int = 2000, base_rate: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < base_rate).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
    return y, p


def test_bootstrap_all_metrics_keys():
    y, p = _make_synthetic()
    res = bootstrap_all_metrics(y, p, B=20, seed=0)
    assert set(res.keys()) == {"roc_auc", "pr_auc", "log_loss", "brier_score", "ece_10bin"}


def test_bootstrap_all_metrics_point_matches_independent_recompute():
    """Point estimates from bootstrap_all_metrics must equal each metric
    recomputed directly via sklearn (and utils for ECE), to numerical
    precision. This catches metric-name/function mis-pairing."""
    y, p = _make_synthetic()
    res = bootstrap_all_metrics(y, p, B=20, seed=0)
    assert res["roc_auc"].point == pytest.approx(float(roc_auc_score(y, p)), rel=1e-12)
    assert res["pr_auc"].point == pytest.approx(
        float(average_precision_score(y, p)), rel=1e-12
    )
    assert res["log_loss"].point == pytest.approx(
        float(log_loss(y, p, labels=[0, 1])), rel=1e-12
    )
    assert res["brier_score"].point == pytest.approx(
        float(brier_score_loss(y, p)), rel=1e-12
    )
    assert res["ece_10bin"].point == pytest.approx(
        float(utils.expected_calibration_error(y, p, n_bins=10)), rel=1e-12
    )


def test_bootstrap_all_metrics_legacy_compute_all_metrics_compatible():
    """Drop-in compatibility: each metric's point matches utils.compute_all_metrics
    with the ECE name renamed (`ece` -> `ece_10bin`)."""
    y, p = _make_synthetic()
    res = bootstrap_all_metrics(y, p, B=20, seed=0)
    legacy = utils.compute_all_metrics(y.astype(float), p)
    rename = {"roc_auc": "roc_auc", "pr_auc": "pr_auc", "log_loss": "log_loss",
              "brier_score": "brier_score", "ece_10bin": "ece"}
    for new_name, legacy_name in rename.items():
        assert res[new_name].point == pytest.approx(legacy[legacy_name], rel=1e-12)


def test_bootstrap_all_metrics_ci_brackets_point_for_well_behaved_data():
    """Sanity regression: with B=200 stratified bootstrap on 2k synthetic
    samples with real signal, every CI brackets its point. (For pathological
    inputs this is not guaranteed by percentile bootstrap, but on well-behaved
    data it should hold.)"""
    y, p = _make_synthetic()
    res = bootstrap_all_metrics(y, p, B=200, seed=0)
    for name, r in res.items():
        assert r.ci_low <= r.point <= r.ci_high, (
            f"{name}: CI [{r.ci_low}, {r.ci_high}] does not bracket point {r.point}"
        )


def test_bootstrap_metrics_by_regime_point_equals_manual_mask_computation():
    """Per-regime point estimates must equal each metric recomputed on the
    masked tercile subset using the same qcut. This catches mask alignment
    or stratification-within-regime bugs."""
    y, p = _make_synthetic(n=3000, seed=0)
    rng = np.random.default_rng(0)
    regime = rng.normal(size=len(y))

    out = bootstrap_metrics_by_regime(y, p, regime, B=20, seed=0)
    terciles = pd.qcut(regime, 3, labels=["low", "med", "high"])
    for label in ["low", "med", "high"]:
        if label not in out:
            continue
        mask = np.asarray(terciles == label)
        # Each regime must reproduce sklearn metrics on the masked subset
        if y[mask].sum() > 0 and y[mask].sum() < mask.sum():
            assert out[label]["roc_auc"].point == pytest.approx(
                float(roc_auc_score(y[mask], p[mask])), rel=1e-12
            )
            assert out[label]["pr_auc"].point == pytest.approx(
                float(average_precision_score(y[mask], p[mask])), rel=1e-12
            )
        assert out[label]["brier_score"].point == pytest.approx(
            float(brier_score_loss(y[mask], p[mask])), rel=1e-12
        )


def test_bootstrap_metrics_by_regime_returns_three_buckets():
    y, p = _make_synthetic(n=3000)
    rng = np.random.default_rng(0)
    regime = rng.normal(size=len(y))
    out = bootstrap_metrics_by_regime(y, p, regime, B=20, seed=0)
    assert set(out.keys()) == {"low", "med", "high"}


def test_bootstrap_metrics_by_regime_metric_subset():
    y, p = _make_synthetic(n=3000)
    rng = np.random.default_rng(0)
    regime = rng.normal(size=len(y))
    out = bootstrap_metrics_by_regime(
        y, p, regime, B=20, seed=0, metrics=["roc_auc", "ece_10bin"]
    )
    for label in out:
        assert set(out[label].keys()) == {"roc_auc", "ece_10bin"}


def test_bootstrap_metrics_by_regime_unknown_metric_raises():
    y, p = _make_synthetic(n=200)
    regime = np.random.default_rng(0).normal(size=len(y))
    with pytest.raises(ValueError, match="Unknown metric"):
        bootstrap_metrics_by_regime(y, p, regime, B=10, metrics=["not_a_metric"])


def test_bootstrap_metrics_by_regime_min_samples_filter():
    """Regimes with fewer than min_samples rows are skipped silently."""
    rng = np.random.default_rng(0)
    n = 60  # 60/3 = 20 per regime, well below default min_samples=50
    y = (rng.random(n) < 0.3).astype(int)
    p = rng.random(n)
    regime = rng.normal(size=n)
    out = bootstrap_metrics_by_regime(y, p, regime, B=10, seed=0)
    assert out == {}


def test_bootstrap_metrics_by_regime_skips_single_class_regime():
    """A regime tercile that ends up single-class must be skipped, not raise.

    Construct a deterministic case: positives correlate perfectly with regime,
    so the lowest tercile contains only negatives.
    """
    n = 600
    rng = np.random.default_rng(0)
    regime = np.linspace(0.0, 1.0, n)
    # All positives are in the upper half of regime
    y = (regime > 0.5).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.05, n), 0.001, 0.999)
    out = bootstrap_metrics_by_regime(y, p, regime, B=10, seed=0)
    # Low regime is all-negative -> must be skipped
    assert "low" not in out
    # High regime is all-positive (since regime>0.5 perfectly defines y) -> skipped
    # Med regime straddles the boundary -> should be kept
    assert "med" in out


def test_to_summary_dict_is_json_safe_and_excludes_samples():
    y, p = _make_synthetic()
    res = bootstrap_all_metrics(y, p, B=10, seed=0)
    d = to_summary_dict(res)
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["roc_auc"]["B"] == 10
    assert d2["roc_auc"]["ci"] == 0.95
    assert "samples" not in d2["roc_auc"]


def test_by_regime_to_summary_dict_is_json_safe():
    y, p = _make_synthetic(n=3000)
    rng = np.random.default_rng(0)
    regime = rng.normal(size=len(y))
    res = bootstrap_metrics_by_regime(y, p, regime, B=10, seed=0)
    d = by_regime_to_summary_dict(res)
    s = json.dumps(d)
    d2 = json.loads(s)
    for label in d2:
        assert "roc_auc" in d2[label]
        assert "samples" not in d2[label]["roc_auc"]
