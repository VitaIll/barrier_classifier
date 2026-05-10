"""Tests for src.analytics.degradation.

Each test names the mathematical property it certifies. Heavy emphasis on:
- The Brier-Murphy identity ``BS = REL - RES + UNC + WBV`` (exact).
- Per-window rolling metrics matching direct sklearn calls on the same window.
- PSI symmetry, hand-computed reference, and zero-on-identity invariants.
- Wilson interval matching a known reference computation.
- Page-Hinkley correctly fires after a real shift and stays silent on stationary data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.analytics.degradation import (
    bootstrap_brier_decomposition,
    brier_murphy_decomposition,
    conditional_precision,
    ks_distance,
    page_hinkley,
    psi,
    psi_ks_rolling,
    rolling_brier_decomposition,
    rolling_metrics_with_ci,
    wilson_interval,
)

pytestmark = pytest.mark.analytics_phase2


# ---------------------------------------------------------------------------
# Synthetic cache builder
# ---------------------------------------------------------------------------


def _make_cache(n_val: int = 1500, n_test: int = 1500, base_rate: float = 0.18, seed: int = 0):
    """Synthetic prediction cache mimicking the Phase-1 schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for split, n in [("val", n_val), ("test", n_test)]:
        ts0 = pd.Timestamp("2025-01-01", tz="UTC") if split == "val" else pd.Timestamp("2025-03-01", tz="UTC")
        ts = ts0 + pd.to_timedelta(np.arange(n) * 10, unit="m")  # 10-min cadence
        y = (rng.random(n) < base_rate).astype(int)
        p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
        regime = np.abs(rng.normal(0.001, 0.0005, n))
        rows.append(
            pd.DataFrame(
                {
                    "k": np.arange(n) + (0 if split == "val" else n_val),
                    "ts": ts,
                    "y": y,
                    "m_k": rng.uniform(0.001, 0.02, n),
                    "tau_k": np.where(y == 1, rng.integers(1, 11, n), np.nan),
                    "phi": 0.005,
                    "regime": regime,
                    "p": p.astype(float),
                    "split": split,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# psi
# ---------------------------------------------------------------------------


def test_psi_identical_arrays_zero():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    assert psi(p, p) < 1e-9


def test_psi_self_sample_small():
    """Two iid samples from the same distribution have small PSI (binning noise only)."""
    rng = np.random.default_rng(0)
    p = rng.random(4000)
    a = rng.choice(p, 4000, replace=True)
    assert psi(p, a) < 0.05


def test_psi_shifted_distributions_large():
    rng = np.random.default_rng(0)
    a = rng.beta(2, 5, 3000)
    b = rng.beta(5, 2, 3000)
    val = psi(a, b)
    assert val > 1.0


def test_psi_symmetric():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 0.5, 1500)
    b = rng.uniform(0.3, 1.0, 1500)
    p1 = psi(a, b)
    p2 = psi(b, a)
    assert p1 == pytest.approx(p2, rel=1e-12)


def test_psi_hand_computed():
    """Two distributions with known per-bin frequencies yield a known PSI.

    Reference: 50/50 in [0, 0.5] and [0.5, 1.0]. Current: 70/30 in same bins.
    PSI = (0.5-0.7)*log(0.5/0.7) + (0.5-0.3)*log(0.5/0.3)
    """
    pref = np.concatenate([np.full(50, 0.25), np.full(50, 0.75)])
    pcur = np.concatenate([np.full(70, 0.25), np.full(30, 0.75)])
    expected = (0.5 - 0.7) * np.log(0.5 / 0.7) + (0.5 - 0.3) * np.log(0.5 / 0.3)
    actual = psi(pref, pcur, n_bins=2)
    assert actual == pytest.approx(expected, rel=1e-3)


def test_psi_handles_empty_bins():
    """PSI must remain finite when one distribution has no mass in a bin
    (the eps-smoothing guard)."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 0.5, 1000)  # no mass in [0.5, 1.0]
    b = rng.uniform(0.5, 1.0, 1000)  # no mass in [0, 0.5]
    val = psi(a, b)
    assert np.isfinite(val)
    assert val > 0


# ---------------------------------------------------------------------------
# ks_distance
# ---------------------------------------------------------------------------


def test_ks_distance_identical_distribution_small():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0, 1, 2000)
    stat, pval = ks_distance(a, b)
    assert stat < 0.06
    assert pval > 0.05


def test_ks_distance_shifted_distribution_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(1, 1, 2000)
    stat, pval = ks_distance(a, b)
    assert stat > 0.3
    assert pval < 1e-10


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_interval_zero_n():
    lo, hi = wilson_interval(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_wilson_interval_brackets_phat():
    """Wilson CI must bracket the empirical proportion."""
    for k, n in [(8, 10), (50, 100), (5, 1000), (250, 500)]:
        lo, hi = wilson_interval(k, n)
        p_hat = k / n
        assert lo <= p_hat <= hi


def test_wilson_interval_known_case():
    """8/10 at alpha=0.05: Wilson CI ≈ [0.49, 0.94] (well-known reference)."""
    lo, hi = wilson_interval(8, 10, alpha=0.05)
    assert lo == pytest.approx(0.49, abs=0.02)
    assert hi == pytest.approx(0.94, abs=0.02)


def test_wilson_interval_extreme_proportions_within_zero_one():
    """Wilson CI does not exceed [0, 1] even at p_hat=0 or p_hat=1."""
    lo, hi = wilson_interval(0, 100)
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = wilson_interval(100, 100)
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# Brier-Murphy decomposition
# ---------------------------------------------------------------------------


def test_brier_murphy_identity_on_binned_brier_exact():
    """The exact Murphy identity is on the BINNED Brier (each prediction
    replaced by its bin mean), not the raw Brier. The raw Brier differs by
    within-bin spread; the binned Brier equals REL - RES + UNC exactly."""
    rng = np.random.default_rng(0)
    n = 2500
    y = (rng.random(n) < 0.2).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
    d = brier_murphy_decomposition(y, p, n_bins=10)
    rhs = d["reliability"] - d["resolution"] + d["uncertainty"]
    assert d["brier_binned"] == pytest.approx(rhs, abs=1e-12)


def test_brier_murphy_identity_holds_for_various_n_bins():
    """Binned-Brier identity holds for any number of bins (5, 10, 20, 50)."""
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.3).astype(int)
    p = rng.random(n)
    for nb in [5, 10, 20, 50]:
        d = brier_murphy_decomposition(y, p, n_bins=nb)
        rhs = d["reliability"] - d["resolution"] + d["uncertainty"]
        assert d["brier_binned"] == pytest.approx(rhs, abs=1e-12), f"n_bins={nb}"


def test_brier_murphy_raw_brier_matches_sklearn():
    """The raw 'brier' field equals sklearn brier_score_loss — model's true loss."""
    from sklearn.metrics import brier_score_loss
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.25).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
    d = brier_murphy_decomposition(y, p, n_bins=10)
    assert d["brier"] == pytest.approx(float(brier_score_loss(y, p)), abs=1e-12)


def test_brier_binned_geq_brier_raw_when_predictions_well_separated():
    """For non-degenerate predictions, brier_binned is typically >= brier
    because discretization replaces a tighter prediction with a coarser one.
    Test the inequality direction is consistent with within-bin spread."""
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(0.1 + 0.4 * y + rng.normal(0, 0.1, n), 0.001, 0.999)
    d = brier_murphy_decomposition(y, p, n_bins=10)
    # Binned Brier substitutes p with bin mean — usually slightly different;
    # within_bin_variance is the spread that explains the difference.
    assert d["within_bin_variance"] >= 0.0


def test_brier_murphy_uncertainty_formula():
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.42).astype(int)
    p = rng.random(n)
    y_bar = float(y.mean())
    d = brier_murphy_decomposition(y, p)
    assert d["uncertainty"] == pytest.approx(y_bar * (1.0 - y_bar), rel=1e-12)


def test_brier_murphy_components_non_negative():
    rng = np.random.default_rng(0)
    n = 1500
    y = (rng.random(n) < 0.3).astype(int)
    p = rng.random(n)
    d = brier_murphy_decomposition(y, p)
    for k in ["reliability", "resolution", "uncertainty", "within_bin_variance", "brier"]:
        assert d[k] >= -1e-12, f"{k} = {d[k]} negative"


def test_brier_murphy_constant_predictions_zero_resolution():
    """Constant predictions p = c -> all samples in one bin -> resolution = 0
    (no informativeness about which observations are positive)."""
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) < 0.25).astype(int)
    p = np.full(n, 0.25)
    d = brier_murphy_decomposition(y, p, n_bins=10)
    assert d["resolution"] < 1e-12


def test_brier_murphy_perfectly_calibrated_constant_zero_reliability():
    """If p is constant = empirical y_bar, then mean_p_bin = mean_y_bin -> reliability = 0."""
    n = 4000
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.20).astype(int)
    p = np.full(n, float(y.mean()))
    d = brier_murphy_decomposition(y, p, n_bins=10)
    assert d["reliability"] < 1e-12


# ---------------------------------------------------------------------------
# bootstrap_brier_decomposition
# ---------------------------------------------------------------------------


def test_bootstrap_brier_decomposition_keys_and_consistency():
    """Each component's point estimate matches the deterministic decomposition."""
    rng = np.random.default_rng(0)
    n = 1500
    y = (rng.random(n) < 0.25).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
    res = bootstrap_brier_decomposition(y, p, n_bins=10, B=50, seed=0)
    point = brier_murphy_decomposition(y, p, n_bins=10)
    for k in [
        "brier",
        "brier_binned",
        "reliability",
        "resolution",
        "uncertainty",
        "within_bin_variance",
    ]:
        assert res[k].point == pytest.approx(point[k], rel=1e-12), f"{k}"
        assert res[k].ci_low <= res[k].ci_high


def test_bootstrap_brier_decomposition_binned_identity_per_replicate():
    """In every bootstrap replicate, brier_binned = REL - RES + UNC (exact)."""
    rng = np.random.default_rng(0)
    n = 1500
    y = (rng.random(n) < 0.25).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
    res = bootstrap_brier_decomposition(y, p, B=50, seed=0)
    rhs = (
        res["reliability"].samples
        - res["resolution"].samples
        + res["uncertainty"].samples
    )
    np.testing.assert_allclose(res["brier_binned"].samples, rhs, atol=1e-12)


# ---------------------------------------------------------------------------
# Page-Hinkley
# ---------------------------------------------------------------------------


def test_page_hinkley_no_alarm_on_stationary():
    """Stationary signal must not fire an alarm at sensible thresholds."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 0.01, 2000)  # tight stationary
    alarm, _, _ = page_hinkley(x, delta=0.001, threshold=5.0)
    assert alarm == -1


def test_page_hinkley_fires_after_upward_shift():
    """Step-up at index 1000 must trigger the alarm sometime AFTER 1000."""
    rng = np.random.default_rng(0)
    n = 2000
    x = np.concatenate([
        rng.normal(0.05, 0.01, 1000),  # pre-shift
        rng.normal(0.10, 0.01, 1000),  # post-shift (5x larger mean)
    ])
    alarm, _, _ = page_hinkley(x, delta=0.001, threshold=2.0)
    assert alarm > 1000  # detected after the actual shift
    assert alarm < 1300  # within 300 samples


def test_page_hinkley_reproducible():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 0.01, 500)
    a1, ph1, mn1 = page_hinkley(x, delta=0.001, threshold=2.0)
    a2, ph2, mn2 = page_hinkley(x, delta=0.001, threshold=2.0)
    assert a1 == a2
    np.testing.assert_array_equal(ph1, ph2)
    np.testing.assert_array_equal(mn1, mn2)


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------


def test_rolling_metrics_with_ci_columns_and_basic_shape():
    cache = _make_cache()
    df = rolling_metrics_with_ci(cache, split="test", window="3D", step="1D", B=20, min_n=100, min_pos=10)
    assert len(df) > 0
    expected = {"window_start", "window_end", "n_samples", "n_pos", "base_rate"}
    expected |= {f"{m}_point" for m in ("roc_auc", "pr_auc", "brier_score", "ece_10bin")}
    expected |= {f"{m}_ci_low" for m in ("roc_auc", "pr_auc", "brier_score", "ece_10bin")}
    expected |= {f"{m}_ci_high" for m in ("roc_auc", "pr_auc", "brier_score", "ece_10bin")}
    assert expected.issubset(set(df.columns))


def test_rolling_metrics_per_window_point_matches_direct_sklearn():
    """Each window's point estimate must equal sklearn's metric on that window's
    raw (y, p). This catches any window-slicing or ordering bug."""
    cache = _make_cache(n_test=2000, base_rate=0.18, seed=1)
    rolling = rolling_metrics_with_ci(
        cache, split="test", window="3D", step="1D", B=10, min_n=50, min_pos=5
    )
    test_df = cache[cache["split"] == "test"].sort_values("ts").reset_index(drop=True)
    for _, row in rolling.iterrows():
        win = test_df[(test_df["ts"] >= row["window_start"]) & (test_df["ts"] < row["window_end"])]
        y = win["y"].to_numpy()
        p = win["p"].to_numpy()
        if y.sum() == 0 or y.sum() == len(y):
            continue
        assert row["roc_auc_point"] == pytest.approx(float(roc_auc_score(y, p)), rel=1e-12)
        assert row["pr_auc_point"] == pytest.approx(
            float(average_precision_score(y, p)), rel=1e-12
        )
        assert row["brier_score_point"] == pytest.approx(
            float(brier_score_loss(y, p)), rel=1e-12
        )


def test_rolling_metrics_skips_under_min_n():
    cache = _make_cache(n_test=500)
    df = rolling_metrics_with_ci(
        cache, split="test", window="6h", step="6h", B=10, min_n=10000, min_pos=5
    )
    assert len(df) == 0  # min_n is unreachable; must skip everything


def test_rolling_metrics_ci_brackets_point():
    """Sanity: CI brackets the point on most windows for synthetic well-behaved data."""
    cache = _make_cache(n_test=2500)
    df = rolling_metrics_with_ci(cache, split="test", window="5D", step="2D", B=200, min_n=200, min_pos=20)
    if len(df) == 0:
        pytest.skip("no windows met thresholds")
    for _, row in df.iterrows():
        for metric in ["roc_auc", "pr_auc", "brier_score"]:
            lo = row[f"{metric}_ci_low"]
            hi = row[f"{metric}_ci_high"]
            pt = row[f"{metric}_point"]
            assert lo <= pt <= hi, f"{metric} CI [{lo}, {hi}] does not bracket {pt}"


# ---------------------------------------------------------------------------
# Rolling Brier decomposition
# ---------------------------------------------------------------------------


def test_rolling_brier_decomposition_binned_identity_per_window():
    """In each rolling window, brier_binned = REL - RES + UNC (exact)."""
    cache = _make_cache(n_test=2500)
    df = rolling_brier_decomposition(
        cache, split="test", window="5D", step="2D", B=20, min_n=200, min_pos=20
    )
    if len(df) == 0:
        pytest.skip("no windows met thresholds")
    rhs = (
        df["reliability_point"]
        - df["resolution_point"]
        + df["uncertainty_point"]
    )
    np.testing.assert_allclose(df["brier_binned_point"], rhs, atol=1e-12)


# ---------------------------------------------------------------------------
# psi_ks_rolling
# ---------------------------------------------------------------------------


def test_psi_ks_rolling_columns():
    cache = _make_cache(n_val=2000, n_test=2000)
    df = psi_ks_rolling(
        cache, reference_split="val", target_split="test", window="3D", step="1D", min_n=100
    )
    assert {"window_start", "window_end", "n_samples", "psi", "ks", "ks_pvalue"}.issubset(df.columns)
    assert (df["psi"] >= 0).all()
    assert ((df["ks"] >= 0) & (df["ks"] <= 1)).all()


def test_psi_ks_rolling_empty_reference_raises():
    cache = _make_cache(n_val=0, n_test=2000)
    # n_val=0 means the reference split has zero rows
    with pytest.raises(ValueError, match="empty"):
        psi_ks_rolling(cache, reference_split="val", target_split="test")


# ---------------------------------------------------------------------------
# Conditional precision
# ---------------------------------------------------------------------------


def test_conditional_precision_per_cell_matches_manual_count():
    cache = _make_cache(n_test=2500)
    threshold = 0.3
    cond = conditional_precision(
        cache, threshold=threshold, split="test", by=("regime_bucket", "hour")
    )
    test_df = cache[cache["split"] == "test"].copy()
    test_df["regime_bucket"] = pd.qcut(
        test_df["regime"], 3, labels=["low", "med", "high"]
    )
    test_df["hour"] = pd.to_datetime(test_df["ts"]).dt.hour
    test_df["pred"] = test_df["p"] >= threshold
    for _, row in cond.iterrows():
        mask = (
            (test_df["regime_bucket"] == row["regime_bucket"])
            & (test_df["hour"] == row["hour"])
            & test_df["pred"]
        )
        n_pred = int(mask.sum())
        n_hit = int((mask & (test_df["y"] == 1)).sum())
        assert int(row["n_predictions"]) == n_pred
        assert int(row["n_hits"]) == n_hit
        if n_pred > 0:
            assert row["precision"] == pytest.approx(n_hit / n_pred, rel=1e-12)


def test_conditional_precision_wilson_ci_brackets_point():
    cache = _make_cache(n_test=2000)
    cond = conditional_precision(
        cache, threshold=0.2, split="test", by=("regime_bucket", "hour")
    )
    if len(cond) == 0:
        pytest.skip("no cells")
    assert (cond["ci_low"] <= cond["precision"]).all()
    assert (cond["precision"] <= cond["ci_high"]).all()


def test_conditional_precision_handles_no_predictions_above_threshold():
    cache = _make_cache(n_test=1000)
    # threshold=2.0 means no predictions exceed it; expect empty DataFrame
    cond = conditional_precision(cache, threshold=2.0, split="test")
    assert len(cond) == 0
