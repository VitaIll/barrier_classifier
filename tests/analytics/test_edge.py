"""Tests for src.analytics.edge.

Each test names the property it certifies. Heavy emphasis on:
- Threshold-sweep precision/recall match sklearn's confusion matrix at every threshold
- EV formula correctness under the binary outcome model
- Partial AUC reduces to full AUC at fpr_max=1.0 / recall_max=1.0
- Kelly formula correctness and Wilson-CI propagation
- Lift curve invariants (precision_at_k=n equals base rate; lift_at_k=n=1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.analytics.edge import (
    OutcomeModel,
    bootstrap_partial_pr_auc,
    bootstrap_partial_roc_auc,
    bootstrap_threshold_sweep,
    kelly_by_bin,
    lift_curve,
)

pytestmark = pytest.mark.analytics_phase3


def _make_cache(n: int = 2000, base_rate: float = 0.18, seed: int = 0):
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(np.arange(n) * 10, unit="m")
    y = (rng.random(n) < base_rate).astype(int)
    p = np.clip(0.05 + 0.4 * y + rng.normal(0, 0.15, n), 0.001, 0.999)
    return pd.DataFrame(
        {
            "k": np.arange(n),
            "ts": ts,
            "y": y,
            "m_k": rng.uniform(0.001, 0.02, n),
            "tau_k": np.where(y == 1, rng.integers(1, 11, n), np.nan),
            "phi": 0.005,
            "regime": np.abs(rng.normal(0.001, 0.0005, n)),
            "p": p.astype(float),
            "split": "test",
        }
    )


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def test_threshold_sweep_per_threshold_precision_recall_match_sklearn():
    """At every threshold, the sweep's precision and recall match sklearn's
    precision_score / recall_score on the same threshold-binarized predictions."""
    cache = _make_cache(seed=0)
    sweep = bootstrap_threshold_sweep(
        cache, split="test", thresholds=np.array([0.10, 0.20, 0.30, 0.40, 0.50]), B=5
    )
    y = cache["y"].to_numpy()
    p = cache["p"].to_numpy()
    for _, row in sweep.iterrows():
        thr = row["threshold"]
        pred = (p >= thr).astype(int)
        if pred.sum() == 0:
            assert np.isnan(row["precision"])
            continue
        expected_prec = float(precision_score(y, pred, zero_division=0))
        expected_rec = float(recall_score(y, pred, zero_division=0))
        assert row["precision"] == pytest.approx(expected_prec, rel=1e-12)
        assert row["recall"] == pytest.approx(expected_rec, rel=1e-12)


def test_threshold_sweep_trade_rate_equals_n_pred_over_n():
    cache = _make_cache(n=1500, seed=1)
    sweep = bootstrap_threshold_sweep(
        cache, split="test", thresholds=np.linspace(0.05, 0.55, 11), B=5
    )
    n = len(cache)
    assert (
        sweep["n_trades"].astype(float) / n - sweep["trade_rate"]
    ).abs().max() < 1e-9


def test_threshold_sweep_ev_formula_under_binary_outcome():
    """EV per trade = precision*gain - (1-precision)*loss - cost. Verify exact
    correspondence at every threshold under the symmetric default outcome."""
    cache = _make_cache(n=1500, seed=2)
    om = OutcomeModel(gain_per_hit=0.005, loss_per_miss=0.005, cost_per_trade=0.0005)
    sweep = bootstrap_threshold_sweep(
        cache, split="test", thresholds=np.linspace(0.10, 0.50, 5),
        outcome_model=om, B=5,
    )
    for _, row in sweep.iterrows():
        if np.isnan(row["precision"]):
            continue
        expected_ev = (
            row["precision"] * om.gain_per_hit
            - (1.0 - row["precision"]) * abs(om.loss_per_miss)
            - om.cost_per_trade
        )
        assert row["ev_per_trade"] == pytest.approx(expected_ev, abs=1e-12)


def test_threshold_sweep_lift_equals_precision_over_base_rate():
    cache = _make_cache(n=2000, seed=3)
    sweep = bootstrap_threshold_sweep(cache, split="test", thresholds=np.linspace(0.1, 0.6, 11), B=5)
    base = float(cache["y"].mean())
    expected = sweep["precision"] / base
    np.testing.assert_allclose(sweep["lift"].dropna(), expected.dropna(), rtol=1e-12)


def test_threshold_sweep_ci_brackets_point_for_well_behaved_data():
    cache = _make_cache(n=2500, seed=4)
    sweep = bootstrap_threshold_sweep(
        cache, split="test", thresholds=np.linspace(0.05, 0.5, 10), B=200
    )
    sweep_clean = sweep.dropna(subset=["precision"])
    assert len(sweep_clean) > 0
    bracketed = (
        (sweep_clean["precision_ci_low"] <= sweep_clean["precision"])
        & (sweep_clean["precision"] <= sweep_clean["precision_ci_high"])
    )
    assert bracketed.mean() > 0.85


def test_threshold_sweep_use_realized_return_missing_column_raises():
    cache = _make_cache()
    om = OutcomeModel(use_realized_return=True)
    with pytest.raises(ValueError, match="r_realized"):
        bootstrap_threshold_sweep(cache, split="test", outcome_model=om, B=5)


def test_threshold_sweep_use_realized_return_with_column():
    """When r_realized is provided, EV equals the mean realized return on
    selected trades minus cost (and Sharpe is computed from realized variance)."""
    cache = _make_cache(n=1500, seed=5)
    rng = np.random.default_rng(99)
    cache = cache.copy()
    cache["r_realized"] = np.where(cache["y"] == 1, 0.005, rng.normal(-0.0005, 0.002, len(cache)))
    om = OutcomeModel(cost_per_trade=0.0001, use_realized_return=True)
    sweep = bootstrap_threshold_sweep(
        cache, split="test", thresholds=np.array([0.20, 0.30]), outcome_model=om, B=5
    )
    y = cache["y"].to_numpy()
    p = cache["p"].to_numpy()
    r = cache["r_realized"].to_numpy()
    for _, row in sweep.iterrows():
        thr = row["threshold"]
        mask = p >= thr
        if mask.sum() == 0:
            continue
        expected_ev = float(r[mask].mean()) - om.cost_per_trade
        assert row["ev_per_trade"] == pytest.approx(expected_ev, abs=1e-12)


def test_threshold_sweep_handles_zero_predictions_above_threshold():
    cache = _make_cache(n=500, seed=6)
    sweep = bootstrap_threshold_sweep(
        cache, split="test", thresholds=np.array([1.5]), B=5
    )
    row = sweep.iloc[0]
    assert row["n_trades"] == 0
    assert np.isnan(row["precision"])


# ---------------------------------------------------------------------------
# Partial AUC
# ---------------------------------------------------------------------------


def test_partial_roc_auc_at_fpr_max_one_equals_full():
    cache = _make_cache(n=1500, seed=0)
    y = cache["y"].to_numpy()
    p = cache["p"].to_numpy()
    res = bootstrap_partial_roc_auc(y, p, fpr_max=1.0, B=20)
    full = float(roc_auc_score(y, p))
    # Numerical equality up to trapezoidal-grid resolution
    assert res.point == pytest.approx(full, abs=2e-3)


def test_partial_pr_auc_at_recall_max_one_close_to_full():
    cache = _make_cache(n=1500, seed=0)
    y = cache["y"].to_numpy()
    p = cache["p"].to_numpy()
    res = bootstrap_partial_pr_auc(y, p, recall_max=1.0, B=20)
    full = float(average_precision_score(y, p))
    # Max-envelope partial AP differs slightly from sklearn's rectangular AP.
    # 5% absolute tolerance is plenty.
    assert res.point == pytest.approx(full, abs=0.05)


def test_partial_roc_auc_monotone_in_fpr_max():
    """Larger fpr_max grows the integration window so the *un-normalized* area
    is non-decreasing. The normalized partial-AUC isn't monotone — but the
    cumulative TPR up to fpr_max IS, which is what we test."""
    cache = _make_cache(n=2000, seed=0)
    y = cache["y"].to_numpy()
    p = cache["p"].to_numpy()
    auc_05 = bootstrap_partial_roc_auc(y, p, fpr_max=0.05, B=10).point * 0.05
    auc_10 = bootstrap_partial_roc_auc(y, p, fpr_max=0.10, B=10).point * 0.10
    auc_20 = bootstrap_partial_roc_auc(y, p, fpr_max=0.20, B=10).point * 0.20
    assert auc_05 <= auc_10 + 1e-9
    assert auc_10 <= auc_20 + 1e-9


def test_partial_roc_auc_ci_brackets_point():
    cache = _make_cache(n=2500, seed=1)
    y = cache["y"].to_numpy()
    p = cache["p"].to_numpy()
    res = bootstrap_partial_roc_auc(y, p, fpr_max=0.05, B=200)
    assert res.ci_low <= res.point <= res.ci_high


def test_partial_roc_auc_zero_fpr_max_raises():
    """fpr_max <= 0 is undefined (division by zero in normalization)."""
    cache = _make_cache(n=500, seed=0)
    y = cache["y"].to_numpy()
    p = cache["p"].to_numpy()
    with pytest.raises(ValueError, match="fpr_max"):
        bootstrap_partial_roc_auc(y, p, fpr_max=0.0, B=5)
    with pytest.raises(ValueError, match="fpr_max"):
        bootstrap_partial_roc_auc(y, p, fpr_max=-0.1, B=5)


def test_partial_pr_auc_zero_recall_max_raises():
    """recall_max <= 0 is undefined (division by zero in normalization)."""
    cache = _make_cache(n=500, seed=0)
    y = cache["y"].to_numpy()
    p = cache["p"].to_numpy()
    with pytest.raises(ValueError, match="recall_max"):
        bootstrap_partial_pr_auc(y, p, recall_max=0.0, B=5)
    with pytest.raises(ValueError, match="recall_max"):
        bootstrap_partial_pr_auc(y, p, recall_max=-0.05, B=5)


# ---------------------------------------------------------------------------
# Kelly by bin
# ---------------------------------------------------------------------------


def test_kelly_by_bin_hit_rate_matches_manual_count():
    cache = _make_cache(n=2000, seed=0)
    df = kelly_by_bin(cache, split="test", n_bins=10)
    p = cache["p"].to_numpy()
    y = cache["y"].to_numpy()
    bin_edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 11)))
    bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, len(bin_edges) - 2)
    for _, row in df.iterrows():
        b = int(row["bin"])
        mask = bin_idx == b
        n_b = int(mask.sum())
        n_hit = int(y[mask].sum())
        assert int(row["n"]) == n_b
        assert int(row["n_hits"]) == n_hit
        if n_b > 0:
            assert row["hit_rate"] == pytest.approx(n_hit / n_b, rel=1e-12)


def test_kelly_by_bin_wilson_ci_brackets_hit_rate():
    cache = _make_cache(n=2000, seed=0)
    df = kelly_by_bin(cache, split="test", n_bins=10)
    assert (df["hit_rate_ci_low"] <= df["hit_rate"]).all()
    assert (df["hit_rate"] <= df["hit_rate_ci_high"]).all()


def test_kelly_formula_symmetric_b_one():
    """At b=gain/loss=1, Kelly = 2*hit_rate - 1; verify against formula in every bin."""
    cache = _make_cache(n=2000, seed=0)
    om = OutcomeModel(gain_per_hit=0.005, loss_per_miss=0.005, cost_per_trade=0.0)
    df = kelly_by_bin(cache, split="test", n_bins=10, outcome_model=om)
    expected = 2.0 * df["hit_rate"] - 1.0
    np.testing.assert_allclose(df["kelly"], expected, atol=1e-12)


def test_kelly_monotone_in_hit_rate():
    """Kelly is monotonically increasing in hit_rate at fixed b > 0; therefore
    Wilson-CI on hit_rate should map to a Wilson-equivalent CI on Kelly."""
    cache = _make_cache(n=2000, seed=0)
    df = kelly_by_bin(cache, split="test", n_bins=10)
    assert (df["kelly_ci_low"] <= df["kelly"]).all()
    assert (df["kelly"] <= df["kelly_ci_high"]).all()


def test_kelly_by_bin_higher_p_bins_have_higher_hit_rates():
    """For a non-degenerate model, the highest-p bin should have a higher
    hit rate than the lowest-p bin."""
    cache = _make_cache(n=2500, seed=0)
    df = kelly_by_bin(cache, split="test", n_bins=10)
    assert df["hit_rate"].iloc[-1] > df["hit_rate"].iloc[0]


# ---------------------------------------------------------------------------
# Lift / gain curve
# ---------------------------------------------------------------------------


def test_lift_curve_at_full_population_equals_base_rate():
    """precision_at_k=N (everyone selected) = base rate; lift = 1.

    Also assert the last-row k is N — i.e. the curve really did include the
    full population (catches off-by-one truncation in the cumulative sum).
    """
    cache = _make_cache(n=1500, seed=0)
    df = lift_curve(cache, split="test")
    base = float(cache["y"].mean())
    assert df.iloc[-1]["precision_at_k"] == pytest.approx(base, rel=1e-12)
    assert df.iloc[-1]["lift_at_k"] == pytest.approx(1.0, rel=1e-12)
    # The cumulative population reaches all rows in the cache.
    assert int(df.iloc[-1]["k"]) == len(cache)


def test_lift_curve_recall_at_full_equals_one():
    cache = _make_cache(n=1500, seed=0)
    df = lift_curve(cache, split="test")
    assert df.iloc[-1]["recall_at_k"] == pytest.approx(1.0, rel=1e-12)


def test_lift_curve_top_decile_lift_matches_manual():
    cache = _make_cache(n=2000, seed=0)
    df = lift_curve(cache, split="test")
    base = float(cache["y"].mean())
    # k = 200 (top 10%)
    k_target = 200
    row = df[df["k"] == k_target].iloc[0]
    p = cache["p"].to_numpy()
    y = cache["y"].to_numpy()
    order = np.argsort(p, kind="mergesort")[::-1]
    top_k_y = y[order[:k_target]]
    expected_precision = float(top_k_y.mean())
    expected_lift = expected_precision / base
    assert row["precision_at_k"] == pytest.approx(expected_precision, rel=1e-12)
    assert row["lift_at_k"] == pytest.approx(expected_lift, rel=1e-12)


def test_lift_curve_columns():
    cache = _make_cache(n=500, seed=0)
    df = lift_curve(cache, split="test")
    assert {"k", "k_frac", "cum_tp", "precision_at_k", "lift_at_k", "recall_at_k"}.issubset(df.columns)
