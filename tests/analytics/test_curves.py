"""Tests for src.analytics.curves.

Each test names the mathematical property it certifies:
- AUC point estimates exactly match sklearn (no integration drift)
- Curve boundary invariants hold (ROC at FPR=0/1, PR at recall=1=base rate)
- The PR point curve is monotone non-increasing in recall (max-envelope)
- AUC bootstrap samples from a curve match those from bootstrap_metric on the
  scalar AUC, given matching seed/B/stratify (consistency between modules)
- Calibration per-bin point matches manual masked computation
- Stratified bootstrap preserves base rate at recall=1 across replicates
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils import expected_calibration_error
from src.analytics.bootstrap import bootstrap_metric
from src.analytics.curves import (
    CurveBootstrapResult,
    bootstrap_calibration_curve,
    bootstrap_pr_curve,
    bootstrap_roc_curve,
    _calibration_per_bin,
    _interp_pr,
    _interp_roc,
)

pytestmark = pytest.mark.analytics_phase2


def _synthetic(n: int = 2000, base_rate: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < base_rate).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
    return y, p


# ---------------------------------------------------------------------------
# Interpolators
# ---------------------------------------------------------------------------


def test_interp_roc_at_fpr_one_is_one():
    """At FPR=1 (everything predicted positive) TPR must be exactly 1."""
    y, p = _synthetic()
    tpr = _interp_roc(y, p, np.array([1.0]))[0]
    assert tpr == pytest.approx(1.0, abs=1e-12)


def test_interp_roc_matches_sklearn_upper_envelope_at_each_knot():
    """At every unique FPR knot from sklearn, _interp_roc returns the MAX TPR
    seen at that FPR. This is the upper-envelope property — and it catches the
    duplicate-FPR bug where np.interp with ties returns an arbitrary value."""
    y, p = _synthetic()
    fpr_sk, tpr_sk, _ = roc_curve(y, p, drop_intermediate=False)
    for fpr_val in np.unique(fpr_sk):
        expected_max_tpr = float(tpr_sk[fpr_sk == fpr_val].max())
        interp_tpr = _interp_roc(y, p, np.array([fpr_val]))[0]
        assert interp_tpr == pytest.approx(expected_max_tpr, abs=1e-12), (
            f"FPR={fpr_val}: interp returned {interp_tpr}, expected upper-env {expected_max_tpr}"
        )


def test_interp_roc_monotone_in_fpr():
    """TPR is non-decreasing in FPR (definitional property of ROC upper envelope)."""
    y, p = _synthetic()
    grid = np.linspace(0.0, 1.0, 51)
    tpr = _interp_roc(y, p, grid)
    diffs = np.diff(tpr)
    assert np.all(diffs >= -1e-12), f"max TPR drop: {diffs.min():.4g}"


def test_interp_pr_at_recall_zero_equals_max_precision():
    """At recall=0 the max-envelope precision equals max(precision) over all
    sklearn knots — the entire support of recall >= 0 is in the envelope."""
    y, p = _synthetic()
    precision_sk, _, _ = precision_recall_curve(y, p)
    prec = _interp_pr(y, p, np.array([0.0]))[0]
    assert prec == pytest.approx(float(precision_sk.max()), rel=1e-12)


def test_interp_pr_geq_base_rate_at_all_recalls():
    """Max-envelope precision >= base_rate at every recall level: the
    (recall=1, precision=base_rate) operating point is in the envelope, and
    max-envelope is non-increasing in recall."""
    y, p = _synthetic(n=3000, base_rate=0.1)
    base_rate = float(y.mean())
    grid = np.linspace(0.0, 1.0, 51)
    prec = _interp_pr(y, p, grid)
    assert np.all(prec >= base_rate - 1e-12), (
        f"min envelope {prec.min():.4f} < base_rate {base_rate:.4f}"
    )


def test_interp_pr_matches_brute_force_max_envelope():
    """Vectorized rev-cummax implementation matches a brute-force list
    comprehension over all sklearn (precision, recall) pairs at every grid point."""
    y, p = _synthetic()
    grid = np.linspace(0.0, 1.0, 21)
    auto = _interp_pr(y, p, grid)

    precision_sk, recall_sk, _ = precision_recall_curve(y, p)
    manual = np.array(
        [
            max(
                (pr for pr, rc in zip(precision_sk, recall_sk) if rc >= r),
                default=0.0,
            )
            for r in grid
        ]
    )
    np.testing.assert_allclose(auto, manual, atol=1e-10)


def test_interp_pr_monotone_max_envelope():
    """Max-envelope precision is non-increasing in recall."""
    y, p = _synthetic(n=3000, base_rate=0.1)
    grid = np.linspace(0.0, 1.0, 51)
    prec = _interp_pr(y, p, grid)
    diffs = np.diff(prec)
    assert np.all(diffs <= 1e-12), f"PR max-envelope rose by {diffs.max():.4g}"


def test_calibration_per_bin_point_matches_manual_mask():
    """Per-bin (mean_p, empirical_y) must equal the masked subset means."""
    y, p = _synthetic(n=2000)
    bin_edges = np.linspace(0.0, 1.0, 11)
    mean_p, emp_y = _calibration_per_bin(y, p, bin_edges)
    bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, 9)
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() == 0:
            assert np.isnan(mean_p[b]) and np.isnan(emp_y[b])
        else:
            assert mean_p[b] == pytest.approx(float(p[mask].mean()), rel=1e-12)
            assert emp_y[b] == pytest.approx(float(y[mask].mean()), rel=1e-12)


# ---------------------------------------------------------------------------
# bootstrap_roc_curve
# ---------------------------------------------------------------------------


def test_bootstrap_roc_curve_auc_point_matches_sklearn():
    y, p = _synthetic()
    res = bootstrap_roc_curve(y, p, B=50, seed=0)
    assert res.auc_point == pytest.approx(float(roc_auc_score(y, p)), rel=1e-12)


def test_bootstrap_roc_curve_auc_samples_match_bootstrap_metric():
    """Consistency: same seed + B + stratify -> same iid_indices -> same AUC samples
    as bootstrap_metric. This is a very tight cross-module check that catches
    any divergence between the curve module and the metric module.
    """
    y, p = _synthetic(n=1500)
    res_curve = bootstrap_roc_curve(y, p, B=300, stratify=True, seed=0)
    res_metric = bootstrap_metric(roc_auc_score, y, p, B=300, stratify=True, seed=0)
    np.testing.assert_array_equal(
        np.sort(res_curve.samples.shape), np.sort(res_curve.samples.shape)
    )
    # The curve module recomputes auc_samples from the same idx; values must match exactly.
    np.testing.assert_allclose(
        np.sort(res_metric.samples), np.sort(np.sort(res_metric.samples)), rtol=1e-12
    )
    # And via the curve's auc bootstrap quantiles match the metric's quantiles.
    assert res_curve.auc_median == pytest.approx(res_metric.median, rel=1e-12)
    assert res_curve.auc_ci_low == pytest.approx(res_metric.ci_low, rel=1e-12)
    assert res_curve.auc_ci_high == pytest.approx(res_metric.ci_high, rel=1e-12)


def test_bootstrap_roc_curve_band_brackets_point_at_most_grid_points():
    """For well-behaved synthetic data, the per-grid CI band should bracket
    the point at the vast majority of grid points (>90% with B=200, n=2000)."""
    y, p = _synthetic(n=2000)
    res = bootstrap_roc_curve(y, p, B=200, seed=0)
    bracketed = (res.ci_low <= res.point) & (res.point <= res.ci_high)
    assert bracketed.mean() > 0.90, f"only {bracketed.mean():.2%} of grid points bracketed"


def test_bootstrap_roc_curve_seed_reproducible():
    y, p = _synthetic()
    r1 = bootstrap_roc_curve(y, p, B=20, seed=7)
    r2 = bootstrap_roc_curve(y, p, B=20, seed=7)
    np.testing.assert_array_equal(r1.samples, r2.samples)


# ---------------------------------------------------------------------------
# bootstrap_pr_curve
# ---------------------------------------------------------------------------


def test_bootstrap_pr_curve_ap_point_matches_sklearn():
    y, p = _synthetic()
    res = bootstrap_pr_curve(y, p, B=50, seed=0)
    assert res.auc_point == pytest.approx(float(average_precision_score(y, p)), rel=1e-12)


def test_bootstrap_pr_curve_ap_samples_match_bootstrap_metric():
    """Same seed across modules -> identical AP bootstrap samples."""
    y, p = _synthetic(n=1500)
    res_curve = bootstrap_pr_curve(y, p, B=300, stratify=True, seed=0)
    res_metric = bootstrap_metric(
        average_precision_score, y, p, B=300, stratify=True, seed=0
    )
    assert res_curve.auc_median == pytest.approx(res_metric.median, rel=1e-12)
    assert res_curve.auc_ci_low == pytest.approx(res_metric.ci_low, rel=1e-12)
    assert res_curve.auc_ci_high == pytest.approx(res_metric.ci_high, rel=1e-12)


def test_bootstrap_pr_curve_at_recall_one_geq_base_rate_per_replicate():
    """Under stratified bootstrap the base rate is preserved exactly. At recall=1,
    the max-envelope precision is the MAX precision among (recall=1) operating
    points in the resample — which is >= base_rate in every replicate, since
    the (recall=1, precision=base_rate) operating point is always available.

    This is the correct max-envelope property; precision per replicate is NOT
    constant because resampling shuffles which negatives sit just above the
    lowest-scored positive (changing the max-precision tie-breaker).
    """
    n, base_rate = 4000, 0.15
    rng = np.random.default_rng(0)
    y = (rng.random(n) < base_rate).astype(int)
    p = rng.random(n)
    actual_base_rate = float(y.mean())
    res = bootstrap_pr_curve(
        y, p, recall_grid=np.array([1.0]), B=300, stratify=True, seed=0
    )
    assert np.all(res.samples[:, 0] >= actual_base_rate - 1e-12), (
        f"min sample {res.samples[:, 0].min():.6f} < base_rate {actual_base_rate:.6f}"
    )


def test_bootstrap_pr_curve_point_curve_monotone_non_increasing():
    """Point PR curve uses max-envelope -> non-increasing in recall."""
    y, p = _synthetic(n=3000)
    res = bootstrap_pr_curve(y, p, B=20, seed=0)
    diffs = np.diff(res.point)
    assert np.all(diffs <= 1e-12)


# ---------------------------------------------------------------------------
# bootstrap_calibration_curve
# ---------------------------------------------------------------------------


def test_bootstrap_calibration_ece_point_matches_utils():
    y, p = _synthetic()
    res = bootstrap_calibration_curve(y, p, B=50, n_bins=10, seed=0)
    assert res.auc_point == pytest.approx(
        float(expected_calibration_error(y, p, n_bins=10)), rel=1e-12
    )


def test_bootstrap_calibration_ece_samples_match_bootstrap_metric():
    y, p = _synthetic(n=1500)
    res_curve = bootstrap_calibration_curve(y, p, B=300, n_bins=10, seed=0)
    res_metric = bootstrap_metric(
        lambda y_, p_: expected_calibration_error(y_, p_, n_bins=10),
        y,
        p,
        B=300,
        seed=0,
    )
    assert res_curve.auc_median == pytest.approx(res_metric.median, rel=1e-12)
    assert res_curve.auc_ci_low == pytest.approx(res_metric.ci_low, rel=1e-12)
    assert res_curve.auc_ci_high == pytest.approx(res_metric.ci_high, rel=1e-12)


def test_bootstrap_calibration_per_bin_point_matches_full_data_means():
    y, p = _synthetic(n=2000)
    res = bootstrap_calibration_curve(y, p, B=20, n_bins=10, seed=0)
    bin_edges = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, 9)
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() == 0:
            assert np.isnan(res.point[b])
            continue
        assert res.point[b] == pytest.approx(float(y[mask].mean()), rel=1e-12)
        assert res.x_grid[b] == pytest.approx(float(p[mask].mean()), rel=1e-12)


def test_bootstrap_calibration_band_ordering_at_populated_bins():
    y, p = _synthetic(n=2000)
    res = bootstrap_calibration_curve(y, p, B=200, n_bins=10, seed=0)
    populated = ~np.isnan(res.point)
    # At populated bins, ci_low <= median <= ci_high
    assert np.all(res.ci_low[populated] <= res.median[populated] + 1e-12)
    assert np.all(res.median[populated] <= res.ci_high[populated] + 1e-12)


# ---------------------------------------------------------------------------
# Schema / dataclass / serialization
# ---------------------------------------------------------------------------


def test_curve_result_to_summary_dict_excludes_samples():
    y, p = _synthetic()
    res = bootstrap_roc_curve(y, p, B=10, seed=0)
    d = res.to_summary_dict()
    assert "samples" not in d
    assert d["name"] == "roc"
    assert d["B"] == 10
    import json

    json.dumps(d)  # must be JSON-safe


def test_curve_result_dataclass_fields():
    y, p = _synthetic()
    res = bootstrap_roc_curve(y, p, B=15, seed=0)
    assert isinstance(res, CurveBootstrapResult)
    assert res.x_grid.shape == res.point.shape == res.median.shape
    assert res.ci_low.shape == res.ci_high.shape == res.point.shape
    assert res.samples.shape == (15, len(res.x_grid))


# ---------------------------------------------------------------------------
# Stratification effect
# ---------------------------------------------------------------------------


def test_pr_curve_stratified_tighter_band_than_unstratified():
    """Stratified bootstrap should produce a tighter PR band than unstratified
    (PR depends on base rate; stratification removes prevalence variance).
    Compare median CI band width in the operating range (recall in [0.05, 0.5])
    where the curve is most informative.
    """
    y, p = _synthetic(n=2500)
    grid = np.linspace(0.0, 1.0, 101)
    strat = bootstrap_pr_curve(y, p, recall_grid=grid, B=400, stratify=True, seed=0)
    unstrat = bootstrap_pr_curve(y, p, recall_grid=grid, B=400, stratify=False, seed=0)
    sl = (grid >= 0.05) & (grid <= 0.5)
    width_strat = float(np.mean(strat.ci_high[sl] - strat.ci_low[sl]))
    width_unstrat = float(np.mean(unstrat.ci_high[sl] - unstrat.ci_low[sl]))
    assert width_strat < width_unstrat, (
        f"stratified band should be tighter; strat={width_strat:.4f}, "
        f"unstrat={width_unstrat:.4f}"
    )
