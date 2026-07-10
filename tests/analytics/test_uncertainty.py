"""Tests for src.analytics.uncertainty.

Heavy emphasis on the math identities (total = data + knowledge, MI >= 0,
Jensen's inequality limit cases) plus a CatBoost integration smoke test that
exercises the posterior-sampling -> virtual-ensemble path end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.uncertainty import (
    _binary_entropy,
    hit_rate_heatmap,
    joint_gate_sweep,
    predictive_uncertainty,
    variance_reliability,
    virtual_ensemble_predictions,
)

pytestmark = pytest.mark.analytics_phase5


# ---------------------------------------------------------------------------
# _binary_entropy
# ---------------------------------------------------------------------------


def test_binary_entropy_at_half_equals_log2():
    """H(0.5) = log(2) for binary entropy in nats."""
    h = float(_binary_entropy(np.array([0.5]))[0])
    assert h == pytest.approx(np.log(2.0), rel=1e-12)


def test_binary_entropy_at_extremes_zero():
    """H(0) = H(1) = 0 (with the 0 * log 0 = 0 convention)."""
    h = _binary_entropy(np.array([0.0, 1.0]))
    assert h[0] == pytest.approx(0.0, abs=1e-9)
    assert h[1] == pytest.approx(0.0, abs=1e-9)


def test_binary_entropy_symmetric_around_half():
    h = _binary_entropy(np.array([0.2, 0.8]))
    assert h[0] == pytest.approx(h[1], rel=1e-12)


def test_binary_entropy_non_negative():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200)
    assert (_binary_entropy(p) >= 0).all()


# ---------------------------------------------------------------------------
# predictive_uncertainty: identities and limit cases
# ---------------------------------------------------------------------------


def test_predictive_uncertainty_identity_total_eq_data_plus_knowledge():
    """Exact: total_uncertainty == data_uncertainty + knowledge_uncertainty."""
    rng = np.random.default_rng(0)
    p_ve = rng.uniform(0.01, 0.99, size=(200, 8))
    res = predictive_uncertainty(p_ve)
    rhs = res["data_uncertainty"] + res["knowledge_uncertainty"]
    np.testing.assert_allclose(res["total_uncertainty"], rhs, atol=1e-12)


def test_predictive_uncertainty_mi_non_negative_jensen():
    """MI = H(mean p) - mean H(p) >= 0 by Jensen's inequality (entropy is concave)."""
    rng = np.random.default_rng(1)
    p_ve = rng.uniform(0.01, 0.99, size=(500, 12))
    res = predictive_uncertainty(p_ve)
    assert (res["knowledge_uncertainty"] >= -1e-12).all()


def test_predictive_uncertainty_k_one_zero_mi():
    """Single replicate: data = total -> MI = 0 exactly."""
    rng = np.random.default_rng(2)
    p_ve = rng.uniform(0.01, 0.99, size=(100, 1))
    res = predictive_uncertainty(p_ve)
    np.testing.assert_allclose(res["knowledge_uncertainty"], 0.0, atol=1e-12)
    np.testing.assert_allclose(res["data_uncertainty"], res["total_uncertainty"], atol=1e-12)


def test_predictive_uncertainty_all_replicates_equal_zero_mi():
    """If p_b is the same across all replicates, MI = 0."""
    p = np.array([[0.3, 0.3, 0.3, 0.3], [0.7, 0.7, 0.7, 0.7]])
    res = predictive_uncertainty(p)
    np.testing.assert_allclose(res["knowledge_uncertainty"], 0.0, atol=1e-12)


def test_predictive_uncertainty_disagreement_increases_mi():
    """Replicates that span [0, 1] have higher MI than concentrated replicates,
    at matched mean_p."""
    # Both have mean_p ~ 0.5; the second has wider spread
    tight = np.array([[0.45, 0.5, 0.55]])
    wide = np.array([[0.05, 0.5, 0.95]])
    mi_tight = predictive_uncertainty(tight)["knowledge_uncertainty"][0]
    mi_wide = predictive_uncertainty(wide)["knowledge_uncertainty"][0]
    assert mi_wide > mi_tight


def test_predictive_uncertainty_mean_p_correct():
    rng = np.random.default_rng(3)
    p_ve = rng.uniform(0.01, 0.99, size=(50, 8))
    res = predictive_uncertainty(p_ve)
    np.testing.assert_allclose(res["mean_p"], p_ve.mean(axis=1), rtol=1e-12)


def test_predictive_uncertainty_2d_required():
    with pytest.raises(ValueError, match="2D"):
        predictive_uncertainty(np.array([0.5, 0.5, 0.5]))


# ---------------------------------------------------------------------------
# hit_rate_heatmap
# ---------------------------------------------------------------------------


def test_hit_rate_heatmap_per_cell_matches_manual():
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) < 0.2).astype(int)
    mean_p = rng.uniform(0, 0.5, n)
    unc = rng.uniform(0, 0.3, n)
    df = hit_rate_heatmap(y, mean_p, unc, n_bins_p=5, n_bins_unc=5)

    p_edges = np.unique(np.quantile(mean_p, np.linspace(0, 1, 6)))
    u_edges = np.unique(np.quantile(unc, np.linspace(0, 1, 6)))
    p_idx = np.clip(np.digitize(mean_p, p_edges) - 1, 0, len(p_edges) - 2)
    u_idx = np.clip(np.digitize(unc, u_edges) - 1, 0, len(u_edges) - 2)

    for _, row in df.iterrows():
        mask = (p_idx == int(row["p_bin"])) & (u_idx == int(row["unc_bin"]))
        n_b = int(mask.sum())
        n_hit = int(y[mask].sum())
        assert int(row["n"]) == n_b
        assert int(row["n_hits"]) == n_hit
        if n_b > 0:
            assert row["hit_rate"] == pytest.approx(n_hit / n_b, rel=1e-12)


def test_hit_rate_heatmap_wilson_ci_brackets_hit_rate():
    rng = np.random.default_rng(0)
    n = 800
    y = (rng.random(n) < 0.25).astype(int)
    mean_p = rng.uniform(0, 0.5, n)
    unc = rng.uniform(0, 0.3, n)
    df = hit_rate_heatmap(y, mean_p, unc, n_bins_p=4, n_bins_unc=4)
    assert (df["ci_low"] <= df["hit_rate"]).all()
    assert (df["hit_rate"] <= df["ci_high"]).all()


def test_hit_rate_heatmap_uncertainty_signal_drops_hit_rate():
    """Synthetic case: hit rate is a known monotonically decreasing function
    of uncertainty at fixed mean_p. The heatmap should reflect this trend."""
    rng = np.random.default_rng(0)
    n = 4000
    mean_p = rng.uniform(0.4, 0.6, n)  # narrow p band
    unc = rng.uniform(0, 1, n)
    # Hit rate decreases linearly in unc: from 0.7 at unc=0 to 0.1 at unc=1
    p_hit = 0.7 - 0.6 * unc
    y = (rng.random(n) < p_hit).astype(int)
    df = hit_rate_heatmap(y, mean_p, unc, n_bins_p=2, n_bins_unc=5)
    # Within each p_bin, hit rate should fall as unc_bin rises
    for p_bin in df["p_bin"].unique():
        sub = df[df["p_bin"] == p_bin].sort_values("unc_bin")
        # Check first hit rate > last hit rate (the planted trend)
        assert sub["hit_rate"].iloc[0] > sub["hit_rate"].iloc[-1]


# ---------------------------------------------------------------------------
# joint_gate_sweep
# ---------------------------------------------------------------------------


def test_joint_gate_sweep_per_cell_matches_manual():
    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) < 0.2).astype(int)
    mean_p = rng.uniform(0, 0.7, n)
    unc = rng.uniform(0, 0.4, n)
    sweep = joint_gate_sweep(
        y, mean_p, unc,
        p_thresholds=np.array([0.2, 0.3]),
        unc_thresholds=np.array([0.1, 0.3]),
    )
    for _, row in sweep.iterrows():
        sel = (mean_p >= row["p_threshold"]) & (unc <= row["unc_threshold"])
        n_sel = int(sel.sum())
        n_hit = int(y[sel].sum())
        assert int(row["n_selected"]) == n_sel
        assert int(row["n_hits"]) == n_hit
        if n_sel > 0:
            assert row["precision"] == pytest.approx(n_hit / n_sel, rel=1e-12)


def test_joint_gate_sweep_higher_unc_threshold_admits_more_trades():
    """For fixed p_threshold, increasing unc_threshold (loosening the gate)
    can only INCREASE n_selected — never decrease."""
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) < 0.2).astype(int)
    mean_p = rng.uniform(0, 0.7, n)
    unc = rng.uniform(0, 0.4, n)
    sweep = joint_gate_sweep(
        y, mean_p, unc,
        p_thresholds=np.array([0.3]),
        unc_thresholds=np.array([0.05, 0.1, 0.2, 0.4]),  # ascending
    )
    sweep = sweep.sort_values("unc_threshold")
    counts = sweep["n_selected"].to_numpy()
    assert (np.diff(counts) >= 0).all(), "n_selected must be non-decreasing in unc_threshold"


def test_joint_gate_sweep_higher_p_threshold_reduces_trades():
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) < 0.2).astype(int)
    mean_p = rng.uniform(0, 0.7, n)
    unc = rng.uniform(0, 0.4, n)
    sweep = joint_gate_sweep(
        y, mean_p, unc,
        p_thresholds=np.array([0.1, 0.2, 0.3, 0.5]),
        unc_thresholds=np.array([1.0]),  # very loose unc gate
    )
    sweep = sweep.sort_values("p_threshold")
    counts = sweep["n_selected"].to_numpy()
    assert (np.diff(counts) <= 0).all(), "n_selected must be non-increasing in p_threshold"


# ---------------------------------------------------------------------------
# variance_reliability
# ---------------------------------------------------------------------------


def test_variance_reliability_perfect_calibration_ratio_one():
    """If predicted_var equals observed_sq_error per row, the ratio is 1 in every bin."""
    rng = np.random.default_rng(0)
    n = 500
    pv = rng.uniform(0.01, 0.5, n)
    obs = pv.copy()
    df = variance_reliability(pv, obs, n_bins=5)
    assert np.allclose(df["ratio_obs_over_pred"], 1.0, atol=1e-12)


def test_variance_reliability_overconfident_ratio_above_one():
    """If predicted_var underestimates observed_sq_error, ratio > 1."""
    rng = np.random.default_rng(0)
    n = 500
    pv = rng.uniform(0.01, 0.5, n)
    obs = 2.0 * pv  # observed is double predicted -> ratio = 2
    df = variance_reliability(pv, obs, n_bins=5)
    assert (df["ratio_obs_over_pred"] > 1.5).all()


def test_variance_reliability_returns_per_bin_sample_count():
    rng = np.random.default_rng(0)
    n = 200
    pv = rng.uniform(0.01, 0.5, n)
    obs = pv * 2 + rng.normal(0, 0.05, n).clip(0, None)
    df = variance_reliability(pv, obs, n_bins=10)
    assert df["n"].sum() == n


# ---------------------------------------------------------------------------
# CatBoost virtual-ensemble integration smoke
# ---------------------------------------------------------------------------


def _fit_smoke_model(n=400, n_features=5, seed=0):
    """Tiny CatBoost classifier with posterior_sampling enabled."""
    from catboost import CatBoostClassifier

    rng = np.random.default_rng(seed)
    feats = [f"feat_{i:02d}" for i in range(n_features)]
    X = rng.normal(size=(n, n_features))
    y = ((X[:, 0] * 1.5 + X[:, 1] * 0.7 - X[:, 2] * 0.4) > 0).astype(int)
    model = CatBoostClassifier(
        iterations=80,
        depth=4,
        learning_rate=0.1,
        loss_function="Logloss",
        posterior_sampling=True,
        verbose=0,
        allow_writing_files=False,
        random_seed=0,
    )
    model.fit(X, y)
    return model, X, y, feats


def test_virtual_ensemble_predictions_shape():
    model, X, _, feats = _fit_smoke_model()
    K = 5
    p_ve = virtual_ensemble_predictions(model, X, virtual_ensembles_count=K, feature_list=feats)
    assert p_ve.shape == (len(X), K)
    assert (p_ve >= 0).all() and (p_ve <= 1).all()
    # The K ensemble columns are NOT all identical — different sub-ensembles
    # produce different per-row probabilities (the whole point of the virtual
    # ensemble). At least one row should show non-zero std across columns.
    assert p_ve.std(axis=1).max() > 0, (
        "all K virtual-ensemble columns are identical — the ensemble "
        "decomposition is degenerate"
    )
    # Probabilities must be in [0, 1] — explicit check (not logits).
    assert p_ve.min() >= 0.0
    assert p_ve.max() <= 1.0


def test_virtual_ensemble_predictions_rejects_invalid_prediction_type():
    model, X, _, feats = _fit_smoke_model()
    with pytest.raises(ValueError, match="prediction_type"):
        virtual_ensemble_predictions(
            model, X, virtual_ensembles_count=5, feature_list=feats,
            prediction_type="bogus",  # type: ignore[arg-type]
        )


def test_virtual_ensemble_predictions_dataframe_input():
    model, X, _, feats = _fit_smoke_model()
    df = pd.DataFrame(X, columns=feats)
    p_ve = virtual_ensemble_predictions(model, df, virtual_ensembles_count=5)
    assert p_ve.shape == (len(df), 5)


def test_virtual_ensemble_decomposition_runs_end_to_end():
    """Smoke: VE prediction + decomposition gives finite, non-negative MI everywhere."""
    model, X, _, feats = _fit_smoke_model(n=500)
    p_ve = virtual_ensemble_predictions(model, X, virtual_ensembles_count=8, feature_list=feats)
    res = predictive_uncertainty(p_ve)
    assert (np.isfinite(res["mean_p"])).all()
    assert (np.isfinite(res["total_uncertainty"])).all()
    assert (np.isfinite(res["data_uncertainty"])).all()
    assert (np.isfinite(res["knowledge_uncertainty"])).all()
    assert (res["knowledge_uncertainty"] >= -1e-9).all()
    np.testing.assert_allclose(
        res["total_uncertainty"],
        res["data_uncertainty"] + res["knowledge_uncertainty"],
        atol=1e-12,
    )
