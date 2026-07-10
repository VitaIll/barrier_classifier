"""Tests for src.analytics.bootstrap.

Each test names the mathematical property it certifies. The suite is structured
to challenge the failure modes that matter: non-uniform sampling, broken
stratification, miscalibrated CIs, mismatched variance scaling, and bias.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analytics.bootstrap import (
    BootstrapResult,
    block_indices,
    bootstrap_metric,
    iid_indices,
)

pytestmark = pytest.mark.analytics_phase0


# ---------------------------------------------------------------------------
# iid_indices: distributional + stratification properties
# ---------------------------------------------------------------------------


def test_iid_indices_shape_and_range():
    """Output shape is (B, n) and every value is a valid index in [0, n)."""
    rng = np.random.default_rng(0)
    idx = iid_indices(100, 50, rng)
    assert idx.shape == (50, 100)
    assert idx.dtype.kind == "i"
    assert idx.min() >= 0 and idx.max() < 100


def test_iid_indices_distributional_uniformity():
    """Each base index appears uniformly across all (B*n) draws.

    With iid uniform sampling over [0, n), the count of any specific base
    index across B*n total draws is Binomial(B*n, 1/n) with mean B and
    variance B*(1 - 1/n). A 6-sigma band catches non-uniform sampling
    (off-by-one, biased PRNG misuse, wrong upper bound) at ~10^-9 false-positive
    rate. Seed is fixed so this is deterministic.
    """
    rng = np.random.default_rng(0)
    n, B = 100, 2000
    idx = iid_indices(n, B, rng)
    counts = np.bincount(idx.flatten(), minlength=n)
    # By construction sum(counts) == B*n exactly; mean is B exactly.
    assert int(counts.sum()) == B * n
    assert counts.mean() == pytest.approx(B, abs=1e-9)
    # Variance band
    se = math.sqrt(B * (1.0 - 1.0 / n))
    z = (counts - B) / se
    assert np.all(np.abs(z) < 6.0), f"max |z|={np.abs(z).max():.2f} (non-uniform sampling)"


def test_iid_indices_stratified_preserves_class_counts():
    """Class-stratified resampling preserves the original positive count
    in every bootstrap row exactly. This is the defining property."""
    n_pos, n_neg = 40, 160
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    rng = np.random.default_rng(0)
    B = 30
    idx = iid_indices(len(y), B, rng, stratify=y)
    pos_per_row = (y[idx] == 1).sum(axis=1)
    assert np.all(pos_per_row == n_pos)
    neg_per_row = (y[idx] == 0).sum(axis=1)
    assert np.all(neg_per_row == n_neg)


def test_iid_indices_stratified_within_class_uniform():
    """Within each class, resampled draws are iid uniform over that class.

    This catches stratification implementations that accidentally bias
    within-class sampling (e.g., fixed permutation, draws-without-replacement
    bug). Same 6-sigma band as the unstratified test.
    """
    n_pos, n_neg = 50, 200
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    rng = np.random.default_rng(0)
    B = 2000
    idx = iid_indices(len(y), B, rng, stratify=y)
    # Pull out the positive draws across all rows (implementation-agnostic).
    pos_resamples = idx[y[idx] == 1]
    assert pos_resamples.size == B * n_pos
    pos_counts = np.bincount(pos_resamples, minlength=len(y))[:n_pos]
    se = math.sqrt(B * (1.0 - 1.0 / n_pos))
    z = (pos_counts - B) / se
    assert np.all(np.abs(z) < 6.0), (
        f"within-class non-uniformity, max |z|={np.abs(z).max():.2f}"
    )


def test_iid_indices_extreme_imbalance_stratified():
    """Stratification works under extreme imbalance (n_pos=2 in n=500).

    Resampling 2 positives with replacement gives 4 possible patterns:
    {(0,0), (0,1), (1,0), (1,1)} (in order-aware view). Probability mass
    over the unordered support {0,0}, {0,1}, {1,1} is {1/4, 1/2, 1/4}.
    Test that across many resamples we hit roughly that mix.
    """
    n_pos, n_neg = 2, 498
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    rng = np.random.default_rng(0)
    B = 4000
    idx = iid_indices(len(y), B, rng, stratify=y)
    # Per row: exactly 2 positives, drawn iid uniform over {0, 1}.
    # Count rows by (count of pos index 0, count of pos index 1)
    pos_mask = y[idx] == 1
    counts_of_zero = np.array([np.sum(idx[b][pos_mask[b]] == 0) for b in range(B)])
    # counts_of_zero takes values in {0, 1, 2} with probabilities {1/4, 1/2, 1/4}
    freq = np.bincount(counts_of_zero, minlength=3) / B
    assert abs(freq[0] - 0.25) < 0.03
    assert abs(freq[1] - 0.50) < 0.03
    assert abs(freq[2] - 0.25) < 0.03


def test_iid_indices_seed_reproducible():
    """Same seed produces identical output (determinism)."""
    a = iid_indices(50, 10, np.random.default_rng(42))
    b = iid_indices(50, 10, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_iid_indices_seeds_meaningfully_differ():
    """Different seeds produce >50% different cells (not just one bit different).

    A weak '!= ' check passes even if the two outputs differ in a single
    cell. This requires substantial divergence, the property we actually want.
    """
    a = iid_indices(200, 50, np.random.default_rng(1))
    b = iid_indices(200, 50, np.random.default_rng(2))
    differ_frac = float(np.mean(a != b))
    # Two iid uniform samples from [0, n) match with probability 1/n=0.005
    # so the differ frac should be ~99.5% almost surely.
    assert differ_frac > 0.95


def test_iid_indices_single_class_falls_back():
    """All-positive y must fall back to plain iid (no division by zero)."""
    y = np.ones(100, dtype=int)
    idx = iid_indices(100, 5, np.random.default_rng(0), stratify=y)
    assert idx.shape == (5, 100)


# ---------------------------------------------------------------------------
# bootstrap_metric: point estimate, SE, bias, coverage
# ---------------------------------------------------------------------------


def test_bootstrap_metric_point_matches_raw_fn():
    """Point estimate must equal fn(y, p) on the full data, exactly."""
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) > 0.5).astype(int)
    p = rng.random(n)
    res = bootstrap_metric(roc_auc_score, y, p, B=50, seed=0)
    assert res.point == pytest.approx(float(roc_auc_score(y, p)))


def test_bootstrap_distribution_concentrates_around_point():
    """For symmetric statistics (sample mean of normal data), the median of
    the bootstrap distribution lies within 0.5 SE of the point estimate.

    This is a stronger property than 'CI brackets point' — it says the
    bootstrap is well-calibrated, not just non-degenerate.
    """
    rng = np.random.default_rng(0)
    n = 1000
    sigma = 1.5
    x = rng.normal(0.0, sigma, n)
    y = np.zeros(n, dtype=int)

    def mean_fn(y_, p_):
        return float(p_.mean())

    res = bootstrap_metric(mean_fn, y, x, B=2000, stratify=False, seed=0)
    se_analytic = sigma / math.sqrt(n)
    assert abs(res.median - res.point) < 0.5 * se_analytic
    # Bootstrap samples are also approximately symmetric around the point.
    assert abs(res.samples.mean() - res.point) < 0.5 * se_analytic


def test_bootstrap_se_matches_analytical_for_sample_mean():
    """Bootstrap SE of the sample mean must match s/sqrt(n) within 10%.

    This is THE diagnostic that the resample size is correct. If we
    accidentally resample n+1 or n-1 elements per replicate the SE shifts
    in a way this test catches.
    """
    rng = np.random.default_rng(42)
    n = 500
    sigma_true = 2.0
    x = rng.normal(0.0, sigma_true, n)
    y = np.zeros(n, dtype=int)
    s_emp = float(np.std(x, ddof=1))
    se_analytic = s_emp / math.sqrt(n)

    def mean_fn(y_, p_):
        return float(p_.mean())

    res = bootstrap_metric(mean_fn, y, x, B=3000, stratify=False, seed=0)
    se_bootstrap = float(res.samples.std(ddof=1))
    rel_error = abs(se_bootstrap - se_analytic) / se_analytic
    assert rel_error < 0.10, (
        f"bootstrap SE {se_bootstrap:.5f} vs analytical {se_analytic:.5f} "
        f"(rel error {rel_error:.3f})"
    )


def test_bootstrap_distribution_unbiased_for_sample_mean():
    """The mean of the bootstrap sampling distribution is an unbiased estimator
    of the original sample mean (for the sample mean statistic). Bias should
    be on the order of the bootstrap MC error, not systematic.
    """
    rng = np.random.default_rng(7)
    n = 500
    x = rng.normal(0.5, 1.0, n)
    y = np.zeros(n, dtype=int)

    def mean_fn(y_, p_):
        return float(p_.mean())

    res = bootstrap_metric(mean_fn, y, x, B=4000, stratify=False, seed=0)
    point = float(np.mean(x))
    # MC error of bootstrap mean estimate = sigma/sqrt(n)/sqrt(B)
    mc_se = float(np.std(x, ddof=1)) / math.sqrt(n) / math.sqrt(4000)
    assert abs(res.samples.mean() - point) < 4.0 * mc_se


def test_bootstrap_metric_metadata():
    """B, ci, samples shape, and result type are all reported correctly."""
    rng = np.random.default_rng(0)
    y = (rng.random(500) > 0.5).astype(int)
    p = rng.random(500)
    res = bootstrap_metric(roc_auc_score, y, p, B=50, ci=0.9, seed=0)
    assert res.B == 50
    assert res.ci == 0.9
    assert res.samples.shape == (50,)
    assert isinstance(res, BootstrapResult)
    # ci_low <= median <= ci_high (percentile order)
    assert res.ci_low <= res.median <= res.ci_high


def test_bootstrap_metric_seed_reproducible():
    rng = np.random.default_rng(0)
    y = (rng.random(500) > 0.5).astype(int)
    p = rng.random(500)
    r1 = bootstrap_metric(roc_auc_score, y, p, B=50, seed=7)
    r2 = bootstrap_metric(roc_auc_score, y, p, B=50, seed=7)
    np.testing.assert_array_equal(r1.samples, r2.samples)


def test_bootstrap_metric_to_dict_roundtrip():
    rng = np.random.default_rng(0)
    y = (rng.random(500) > 0.5).astype(int)
    p = rng.random(500)
    res = bootstrap_metric(roc_auc_score, y, p, B=20, seed=0)
    d = res.to_dict(include_samples=False)
    assert "samples" not in d
    assert set(d.keys()) == {"point", "median", "ci_low", "ci_high", "ci", "B", "B_effective"}
    # On clean two-class iid data, every replicate produces a finite metric,
    # so B_effective equals B.
    assert d["B_effective"] == d["B"]
    d2 = res.to_dict(include_samples=True)
    assert "samples" in d2 and len(d2["samples"]) == 20


def test_bootstrap_metric_shape_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_metric(roc_auc_score, np.zeros(10), np.zeros(11), B=10, seed=0)


def test_bootstrap_metric_2d_raises():
    with pytest.raises(ValueError):
        bootstrap_metric(
            roc_auc_score, np.zeros((10, 2)), np.zeros((10, 2)), B=10, seed=0
        )


def test_bootstrap_metric_propagates_metric_errors():
    """If the metric fn raises, bootstrap_metric must propagate the error
    rather than swallow it.

    Uses an explicitly raising metric: sklearn changed ``roc_auc_score``'s
    single-class behavior from raising ValueError to returning NaN with an
    ``UndefinedMetricWarning`` (>= 1.8), so the old sklearn-based premise
    no longer exercises the propagation contract."""
    y = np.zeros(100, dtype=int)
    p = np.random.default_rng(0).random(100)

    def exploding_metric(y_, p_):
        raise ValueError("metric is undefined on this resample")

    with pytest.raises(ValueError, match="undefined on this resample"):
        bootstrap_metric(exploding_metric, y, p, B=20, seed=0, stratify=False)


def test_bootstrap_metric_propagates_nan_metrics():
    """A metric that *returns* NaN must surface as a NaN point estimate, not be
    silently dropped.

    Uses a metric that returns ``float('nan')`` directly instead of relying on
    single-class ``roc_auc_score``: sklearn's single-class behavior is
    version-specific (raises ``ValueError`` before 1.8, returns NaN with an
    ``UndefinedMetricWarning`` from 1.8 on), so the old premise only exercised
    this contract on sklearn >= 1.8. ``bootstrap_metric`` computes ``point`` on
    the full data unguarded, so a NaN-returning metric deterministically yields
    a NaN point estimate on any supported sklearn (>= 1.3)."""
    y = np.zeros(100, dtype=int)
    p = np.random.default_rng(0).random(100)
    res = bootstrap_metric(
        lambda y_, p_: float("nan"),
        y, p, B=10, seed=0, stratify=False,
    )
    assert math.isnan(res.point)


def test_bootstrap_stratified_preserves_prevalence_unstratified_does_not():
    """Stratified resampling produces ZERO variance in the empirical positive
    rate; unstratified produces binomial-scale variance. This is the
    signature of stratification actually working."""
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) > 0.5).astype(int)
    p = rng.random(n)

    def pos_rate(y_, p_):
        return float(np.mean(y_))

    strat = bootstrap_metric(pos_rate, y, p, B=200, stratify=True, seed=0)
    unstrat = bootstrap_metric(pos_rate, y, p, B=200, stratify=False, seed=0)
    assert strat.samples.std() < 1e-12
    # Theoretical SD of empirical pos rate from binomial: sqrt(p*(1-p)/n)
    p_hat = float(y.mean())
    expected_unstrat_sd = math.sqrt(p_hat * (1.0 - p_hat) / n)
    assert 0.5 * expected_unstrat_sd < unstrat.samples.std() < 1.5 * expected_unstrat_sd


def test_bootstrap_stratified_reduces_variance_for_pr_auc():
    """For a base-rate-sensitive metric (PR-AUC), stratified bootstrap should
    produce *lower* variance than unstratified at fixed B. (PR-AUC depends
    on prevalence; removing prevalence variance tightens the distribution.)
    ROC-AUC is invariant to prevalence so this property would not hold for
    ROC-AUC -- testing PR-AUC specifically.
    """
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.10).astype(int)
    # signal: positives have higher p
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.15, n), 0.001, 0.999)

    strat = bootstrap_metric(
        average_precision_score, y, p, B=800, stratify=True, seed=0
    )
    unstrat = bootstrap_metric(
        average_precision_score, y, p, B=800, stratify=False, seed=0
    )
    var_strat = float(strat.samples.var(ddof=1))
    var_unstrat = float(unstrat.samples.var(ddof=1))
    assert var_strat < var_unstrat, (
        f"stratification should tighten PR-AUC bootstrap; "
        f"got var_strat={var_strat:.6f}, var_unstrat={var_unstrat:.6f}"
    )


@pytest.mark.parametrize("ci_level,tol", [(0.90, 0.05), (0.95, 0.05), (0.99, 0.04)])
def test_bootstrap_coverage_at_nominal_levels(ci_level: float, tol: float):
    """Empirical coverage of the percentile CI tracks nominal level on a
    well-behaved problem (sample mean of normal data, n=200).

    M=200 outer trials gives binomial SE on coverage ~0.015, so a tolerance
    of 0.04-0.05 (~3 SE) is robust. The test catches mis-located percentile
    cuts, swapped low/high, or wrong alpha calculation.
    """
    M = 200
    n = 200
    B = 300
    mu = 0.0
    rng = np.random.default_rng(2024)

    def mean_of_p(y_, p_):
        return float(p_.mean())

    in_ci = 0
    for trial in range(M):
        x = rng.normal(0.0, 1.0, n)
        y = np.zeros(n, dtype=int)
        res = bootstrap_metric(
            mean_of_p, y, x, B=B, ci=ci_level, stratify=False, seed=trial
        )
        if res.ci_low <= mu <= res.ci_high:
            in_ci += 1
    coverage = in_ci / M
    assert abs(coverage - ci_level) < tol, (
        f"coverage {coverage:.3f} for nominal {ci_level} (tol {tol})"
    )


# ---------------------------------------------------------------------------
# block_indices + block-bootstrap properties (for autocorrelated streams)
# ---------------------------------------------------------------------------


def test_block_indices_shape_and_range():
    rng = np.random.default_rng(0)
    idx = block_indices(200, B=30, rng=rng, block_size=20)
    assert idx.shape == (30, 200)
    assert idx.min() >= 0 and idx.max() < 200


def test_block_indices_blocks_are_contiguous():
    """Every block of size ``block_size`` is a contiguous run of integers.

    Within each replicate, consecutive cells in each block must differ by 1.
    """
    rng = np.random.default_rng(0)
    n, B, bs = 200, 50, 10
    idx = block_indices(n, B, rng, block_size=bs)
    # Reshape to (B, n_blocks, block_size) — only the leading n cells, but n is multiple of bs here.
    reshaped = idx.reshape(B, n // bs, bs)
    diffs_within_block = np.diff(reshaped, axis=2)
    assert np.all(diffs_within_block == 1), "blocks must be contiguous runs"


def test_block_indices_rejects_bad_block_size():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="block_size"):
        block_indices(100, 5, rng, block_size=0)
    with pytest.raises(ValueError, match="block_size"):
        block_indices(100, 5, rng, block_size=200)


def test_block_indices_blocks_uniformly_start():
    """Block starts are uniform on [0, n - block_size].

    Each valid start (n - block_size + 1 possible values) should be hit
    approximately equally across all blocks in all replicates.
    """
    rng = np.random.default_rng(0)
    n, B, bs = 500, 200, 10
    idx = block_indices(n, B, rng, block_size=bs)
    # First element of each block is the start
    starts = idx[:, ::bs].flatten()
    valid_range = n - bs + 1
    counts = np.bincount(starts, minlength=valid_range)
    expected = B * (n // bs) / valid_range
    se = math.sqrt(expected * (1.0 - 1.0 / valid_range))
    z = (counts[:valid_range] - expected) / se
    assert np.all(np.abs(z) < 6.0), f"non-uniform block starts (max |z|={np.abs(z).max():.2f})"


def test_block_bootstrap_ci_wider_than_iid_on_autocorrelated_stream():
    """Block bootstrap should produce *materially wider* CIs than IID bootstrap
    when the underlying series is autocorrelated.

    Build an AR(1) sequence with strong correlation (rho=0.95); the IID
    bootstrap will underestimate the variance of the sample mean because
    it ignores the within-block dependence. Block bootstrap captures it.
    """
    rng = np.random.default_rng(0)
    n = 2000
    rho = 0.95
    eps = rng.normal(0, 1.0, n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + math.sqrt(1 - rho * rho) * eps[t]
    y = np.zeros(n, dtype=int)

    def mean_fn(_y, p_):
        return float(p_.mean())

    iid_res = bootstrap_metric(
        mean_fn, y, x, B=1000, stratify=False, seed=0,
    )
    block_res = bootstrap_metric(
        mean_fn, y, x, B=1000, stratify=False, seed=0, block_size=40,
    )
    iid_width = iid_res.ci_high - iid_res.ci_low
    block_width = block_res.ci_high - block_res.ci_low
    # For AR(1) with rho=0.95, effective n is much smaller than n, so block
    # bootstrap SE should be substantially larger. Require >= 2x widening.
    assert block_width > 2.0 * iid_width, (
        f"block CI width {block_width:.5f} not wider than iid {iid_width:.5f} (ratio {block_width/iid_width:.2f})"
    )


def test_block_bootstrap_ci_similar_to_iid_on_independent_stream():
    """On truly iid data, block bootstrap should *not* be much wider than IID
    — block size > 1 wastes information when there's no correlation to preserve.

    A factor-of-2 widening on iid data would be too much; require < 1.6x.
    """
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(0, 1.0, n)
    y = np.zeros(n, dtype=int)

    def mean_fn(_y, p_):
        return float(p_.mean())

    iid_res = bootstrap_metric(
        mean_fn, y, x, B=1000, stratify=False, seed=0,
    )
    block_res = bootstrap_metric(
        mean_fn, y, x, B=1000, stratify=False, seed=0, block_size=20,
    )
    iid_width = iid_res.ci_high - iid_res.ci_low
    block_width = block_res.ci_high - block_res.ci_low
    assert block_width < 1.6 * iid_width, (
        f"block widening too aggressive on iid data: "
        f"block {block_width:.5f} vs iid {iid_width:.5f}"
    )


def test_block_bootstrap_ignores_stratify_flag():
    """``block_size > 1`` overrides stratify because blocks are class-agnostic.

    Verify the function doesn't error and produces a non-degenerate result
    when stratify=True and block_size is set simultaneously."""
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) > 0.5).astype(int)
    p = rng.random(n)
    res = bootstrap_metric(
        roc_auc_score, y, p, B=200, stratify=True, seed=0, block_size=20,
    )
    assert res.samples.std() > 0  # non-degenerate


def test_bootstrap_ci_low_le_high_under_skewed_statistic():
    """ECE on noisy data is highly skewed (lower-bounded at 0). The percentile
    CI must still be ordered (ci_low <= median <= ci_high) — this catches
    quantile() argument-order bugs."""
    from src.utils import expected_calibration_error
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.1).astype(int)
    p = np.clip(0.1 + rng.normal(0, 0.1, n), 0.001, 0.999)

    def ece_fn(y_, p_):
        return expected_calibration_error(y_, p_, n_bins=10)

    res = bootstrap_metric(ece_fn, y, p, B=300, seed=0)
    assert res.ci_low <= res.median <= res.ci_high
    assert 0.0 <= res.ci_low  # ECE is non-negative


# ---------------------------------------------------------------------------
# B_effective: block bootstrap on pathological data drops crashed replicates.
# ---------------------------------------------------------------------------


def test_bootstrap_metric_B_effective_drops_single_class_replicates():
    """A block bootstrap on a stream where positives sit in a single small
    contiguous run will produce some all-zero block resamples — roc_auc_score
    raises on those, and the new try/except in bootstrap_metric records them
    as NaN. Verify that:
        (a) B_effective < B (at least one replicate was dropped);
        (b) median / CI bounds are computed only over surviving replicates
            (finite, despite NaN samples).
    """
    n = 1000
    # Positives only in rows 50..70 — block_size=20 will sometimes miss them.
    y = np.zeros(n, dtype=int)
    y[50:70] = 1
    rng = np.random.default_rng(0)
    p = np.clip(0.1 + 0.4 * y + rng.normal(0, 0.05, n), 0.001, 0.999)
    res = bootstrap_metric(
        roc_auc_score, y, p, B=300, stratify=False, seed=0, block_size=20
    )
    # Some replicates produced NaN -> B_effective strictly < B.
    assert 0 < res.B_effective < res.B, (
        f"expected 0 < B_effective={res.B_effective} < B={res.B}"
    )
    # Sample array has exactly B_effective non-NaN entries.
    finite_count = int(np.count_nonzero(~np.isnan(res.samples)))
    assert finite_count == res.B_effective
    # Median / CI are finite — computed from the surviving replicates.
    assert np.isfinite(res.median)
    assert np.isfinite(res.ci_low)
    assert np.isfinite(res.ci_high)


def test_bootstrap_result_B_effective_defaults_to_B_for_legacy_constructors():
    """The dataclass post_init fills B_effective with B when not supplied,
    so any existing code that constructs ``BootstrapResult(..., samples=...)``
    without the new field keeps working."""
    samples = np.array([0.5, 0.6, 0.7])
    res = BootstrapResult(
        point=0.6,
        median=0.6,
        ci_low=0.5,
        ci_high=0.7,
        ci=0.95,
        B=3,
        samples=samples,
    )
    assert res.B_effective == 3
    assert res.to_dict()["B_effective"] == 3
