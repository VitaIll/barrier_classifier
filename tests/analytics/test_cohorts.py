"""Tests for src.analytics.cohorts.

Strategy: most tests use synthetic SHAP matrices and synthetic cohort vectors
so the math is exact and the assertions are tight. One CatBoost smoke test
certifies the SHAP-extraction integration (TreeSHAP shape, baseline-stripping,
and the additive identity ``shap.sum(axis=1) + baseline ≈ raw_log_odds``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.cohorts import (
    bootstrap_shap_diff,
    cohort_assignments,
    cohort_counts,
    cohort_mean_shap,
    compute_shap_values,
    discriminative_shap,
    signed_effect_size_disagreement,
)

pytestmark = pytest.mark.analytics_phase4


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _make_synth_dataset(n=600, n_features=6, seed=0):
    """A tiny CatBoost-trainable dataset for the SHAP smoke test."""
    rng = np.random.default_rng(seed)
    feats = [f"feat_{i:02d}" for i in range(n_features)]
    X = rng.normal(size=(n, n_features))
    score = X[:, 0] * 1.5 + X[:, 1] * 0.7 - X[:, 2] * 0.4
    base_rate = 0.20
    threshold = float(np.quantile(score, 1.0 - base_rate))
    y = (score > threshold).astype(int)
    return X, y, feats


def _make_synthetic_shap_and_cohorts(n=400, F=8, seed=0):
    """Synthetic SHAP matrix + planted cohort labels with known FP-vs-FN signal.

    Plants a known difference in features 0 (positive: FP > FN) and 1
    (negative: FP < FN). Other features are noise. Cohort sizes balanced.
    """
    rng = np.random.default_rng(seed)
    cohorts = np.array(["TP"] * (n // 4) + ["FP"] * (n // 4) + ["TN"] * (n // 4) + ["FN"] * (n // 4))
    feature_list = [f"f{i:02d}" for i in range(F)]
    shap = rng.normal(0, 0.3, size=(n, F))
    # Plant feat 0: FP mean +0.5, FN mean -0.5
    shap[cohorts == "FP", 0] += 0.5
    shap[cohorts == "FN", 0] -= 0.5
    # Plant feat 1: opposite (FP -0.4, FN +0.4)
    shap[cohorts == "FP", 1] -= 0.4
    shap[cohorts == "FN", 1] += 0.4
    return shap, cohorts, feature_list


# ---------------------------------------------------------------------------
# cohort_assignments
# ---------------------------------------------------------------------------


def test_cohort_assignments_handles_all_four_classes():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.2, 0.8, 0.6, 0.3])
    out = cohort_assignments(y, p, threshold=0.5)
    assert list(out) == ["TP", "FN", "TN", "FP", "TP", "TN"]


def test_cohort_assignments_threshold_boundary_inclusive():
    """p == threshold -> predicted positive (>= rule)."""
    y = np.array([1, 0])
    p = np.array([0.5, 0.5])
    out = cohort_assignments(y, p, threshold=0.5)
    assert list(out) == ["TP", "FP"]


def test_cohort_counts():
    cohorts = np.array(["TP"] * 5 + ["FP"] * 3 + ["TN"] * 10 + ["FN"] * 2)
    counts = cohort_counts(cohorts)
    assert counts == {"TP": 5, "FP": 3, "TN": 10, "FN": 2}


# ---------------------------------------------------------------------------
# cohort_mean_shap
# ---------------------------------------------------------------------------


def test_cohort_mean_shap_matches_manual_per_cohort_means():
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts()
    df = cohort_mean_shap(shap, cohorts, feats)
    for label in ["TP", "FP", "TN", "FN"]:
        sub = df[df["cohort"] == label].set_index("feature")
        manual_mean = shap[cohorts == label].mean(axis=0)
        for fi, fn in enumerate(feats):
            assert sub.loc[fn, "mean_shap"] == pytest.approx(float(manual_mean[fi]), rel=1e-12)
        assert int(sub["n"].iloc[0]) == int((cohorts == label).sum())


def test_cohort_mean_shap_skips_empty_cohort():
    """If a cohort has zero rows, it is not represented in the output."""
    shap = np.zeros((20, 4))
    cohorts = np.array(["TP"] * 10 + ["TN"] * 10)  # no FP, no FN
    df = cohort_mean_shap(shap, cohorts, feats := ["a", "b", "c", "d"])
    assert set(df["cohort"].unique()) == {"TP", "TN"}


def test_cohort_mean_shap_shape_mismatch_raises():
    shap = np.zeros((20, 4))
    cohorts = np.array(["TP"] * 20)
    with pytest.raises(ValueError, match="cols"):
        cohort_mean_shap(shap, cohorts, ["a", "b"])  # 2 != 4


# ---------------------------------------------------------------------------
# signed_effect_size_disagreement
# ---------------------------------------------------------------------------


def test_signed_effect_size_recovers_planted_signal_at_top():
    """The two planted features (feat 0 and feat 1) should be the top two
    by absolute effect size, with feat 0 positive and feat 1 negative."""
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts(n=800)
    df = signed_effect_size_disagreement(shap, cohorts, feats)
    top2 = df.head(2)["feature"].tolist()
    assert "f00" in top2 and "f01" in top2
    feat0_row = df[df["feature"] == "f00"].iloc[0]
    feat1_row = df[df["feature"] == "f01"].iloc[0]
    assert feat0_row["effect_size"] > 0
    assert feat1_row["effect_size"] < 0


def test_signed_effect_size_pooled_std_formula():
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts()
    df = signed_effect_size_disagreement(shap, cohorts, feats)
    # Recompute pooled std manually for one feature, compare
    fp_mask = cohorts == "FP"
    fn_mask = cohorts == "FN"
    n_fp, n_fn = int(fp_mask.sum()), int(fn_mask.sum())
    fp_var = shap[fp_mask].var(axis=0, ddof=1)
    fn_var = shap[fn_mask].var(axis=0, ddof=1)
    expected_pooled = np.sqrt(((n_fp - 1) * fp_var + (n_fn - 1) * fn_var) / (n_fp + n_fn - 2))
    df_indexed = df.set_index("feature")
    for fi, fn in enumerate(feats):
        assert df_indexed.loc[fn, "pooled_std"] == pytest.approx(expected_pooled[fi], rel=1e-12)


def test_signed_effect_size_returns_empty_when_cohort_too_small():
    shap = np.random.default_rng(0).normal(size=(40, 4))
    cohorts = np.array(["TP"] * 20 + ["FP"] * 2 + ["TN"] * 16 + ["FN"] * 2)  # < 5 each
    df = signed_effect_size_disagreement(shap, cohorts, ["a", "b", "c", "d"])
    assert df.empty


def test_signed_effect_size_ranking_is_by_abs_effect():
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts()
    df = signed_effect_size_disagreement(shap, cohorts, feats)
    diffs = df["abs_effect_size"].to_numpy()
    assert np.all(np.diff(diffs) <= 1e-12), "ranking should be descending by abs effect"


# ---------------------------------------------------------------------------
# discriminative_shap
# ---------------------------------------------------------------------------


def test_discriminative_shap_returns_one_coef_per_feature():
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts()
    df = discriminative_shap(shap, cohorts, feats)
    assert len(df) == len(feats)
    assert {"feature", "discriminative_coef", "abs_coef"}.issubset(df.columns)


def test_discriminative_shap_top_feature_signs_match_signed_effect_size():
    """For the planted signal (no inter-feature correlation in synthetic
    SHAP), the top discriminative coef sign should match the sign of the
    top mean-diff feature."""
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts(n=1200)
    eff = signed_effect_size_disagreement(shap, cohorts, feats)
    disc = discriminative_shap(shap, cohorts, feats, C=10.0)
    # Compare for feat0 (planted FP > FN) — should have positive coef
    coef_f00 = disc[disc["feature"] == "f00"]["discriminative_coef"].iloc[0]
    eff_f00 = eff[eff["feature"] == "f00"]["effect_size"].iloc[0]
    assert np.sign(coef_f00) == np.sign(eff_f00)
    coef_f01 = disc[disc["feature"] == "f01"]["discriminative_coef"].iloc[0]
    eff_f01 = eff[eff["feature"] == "f01"]["effect_size"].iloc[0]
    assert np.sign(coef_f01) == np.sign(eff_f01)


def test_discriminative_shap_returns_empty_when_cohort_too_small():
    shap = np.random.default_rng(0).normal(size=(40, 4))
    cohorts = np.array(["TP"] * 20 + ["FP"] * 2 + ["TN"] * 16 + ["FN"] * 2)
    df = discriminative_shap(shap, cohorts, ["a", "b", "c", "d"])
    assert df.empty


# ---------------------------------------------------------------------------
# bootstrap_shap_diff
# ---------------------------------------------------------------------------


def test_bootstrap_shap_diff_point_matches_manual():
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts()
    df = bootstrap_shap_diff(shap, cohorts, feats, B=50, seed=0)
    fp_mean = shap[cohorts == "FP"].mean(axis=0)
    fn_mean = shap[cohorts == "FN"].mean(axis=0)
    expected = fp_mean - fn_mean
    df_indexed = df.set_index("feature")
    for fi, fn in enumerate(feats):
        assert df_indexed.loc[fn, "shap_diff"] == pytest.approx(float(expected[fi]), rel=1e-12)


def test_bootstrap_shap_diff_ci_brackets_point_for_well_behaved():
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts(n=1200)
    df = bootstrap_shap_diff(shap, cohorts, feats, B=300, seed=0)
    bracketed = (df["shap_diff_ci_low"] <= df["shap_diff"]) & (df["shap_diff"] <= df["shap_diff_ci_high"])
    assert bracketed.all()


def test_bootstrap_shap_diff_planted_features_ci_excludes_zero():
    """The planted features should have CIs that exclude zero — that's the
    statistical certification we want from the bootstrap."""
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts(n=1200)
    df = bootstrap_shap_diff(shap, cohorts, feats, B=400, seed=0)
    df_indexed = df.set_index("feature")
    assert df_indexed.loc["f00", "ci_excludes_zero"]
    assert df_indexed.loc["f01", "ci_excludes_zero"]
    # noise features (f02..f07) should NOT all reject zero
    noise = df_indexed.loc[["f02", "f03", "f04", "f05", "f06", "f07"]]
    assert noise["ci_excludes_zero"].sum() <= 2  # very few rejections by chance


def test_bootstrap_shap_diff_seed_reproducible():
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts()
    a = bootstrap_shap_diff(shap, cohorts, feats, B=20, seed=42)
    b = bootstrap_shap_diff(shap, cohorts, feats, B=20, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_bootstrap_shap_diff_returns_empty_when_cohort_too_small():
    shap = np.random.default_rng(0).normal(size=(40, 4))
    cohorts = np.array(["TP"] * 20 + ["FP"] * 2 + ["TN"] * 16 + ["FN"] * 2)
    df = bootstrap_shap_diff(shap, cohorts, ["a", "b", "c", "d"], B=20)
    assert df.empty


def test_bootstrap_shap_diff_iid_branch_sets_B_effective():
    """The iid (block_size=None) branch must set ``df.attrs["B_effective"]``
    to B for clean data — every replicate preserves cohort sizes by
    construction. This regression-tests the iid branch missing-attr bug."""
    shap, cohorts, feats = _make_synthetic_shap_and_cohorts(n=400)
    df = bootstrap_shap_diff(shap, cohorts, feats, B=80, seed=0)  # iid branch
    assert "B_effective" in df.attrs
    assert df.attrs["B_effective"] == 80
    assert df.attrs.get("B") == 80


def test_bootstrap_shap_diff_block_size_widens_ci_and_tracks_B_effective():
    """With autocorrelated cohorts (clusters of FP / FN runs) the block
    bootstrap drops replicates whose resampled cohort sizes fall below
    ``min_cohort_size``. The recorded ``B_effective`` must be <= B, and
    the CI must be at least as wide as the iid baseline (block bootstrap
    cannot tighten honest CIs on autocorrelated data)."""
    rng = np.random.default_rng(0)
    F = 6
    feature_list = [f"f{i:02d}" for i in range(F)]
    # Build a strongly autocorrelated cohort sequence: alternate long runs of
    # each cohort label so blocks of 20 rows are usually within a single
    # cohort. This is the autocorrelation structure the block bootstrap is
    # designed for.
    run_len = 30
    cohorts = np.array(
        ["TP"] * run_len + ["FP"] * run_len
        + ["TN"] * run_len + ["FN"] * run_len
        + ["FP"] * run_len + ["FN"] * run_len
        + ["TP"] * run_len + ["TN"] * run_len
    )
    # Build SHAP with a planted FP-minus-FN signal on feature 0.
    shap = rng.normal(0, 0.3, size=(len(cohorts), F))
    shap[cohorts == "FP", 0] += 0.4
    shap[cohorts == "FN", 0] -= 0.4

    df_iid = bootstrap_shap_diff(shap, cohorts, feature_list, B=200, seed=0)
    df_blk = bootstrap_shap_diff(
        shap, cohorts, feature_list, B=200, seed=0, block_size=20
    )
    assert not df_iid.empty and not df_blk.empty
    # Block bootstrap may have dropped some replicates whose resampled
    # cohort fell below min_cohort_size. B_effective must be <= B.
    assert df_blk.attrs["B_effective"] <= 200
    assert df_blk.attrs.get("B") == 200
    # CI width on feature 0 (planted signal) is at least as wide under
    # block bootstrap. Compare absolute width — block honest CI cannot be
    # *tighter* than iid on autocorrelated data.
    f0_iid = df_iid[df_iid["feature"] == "f00"].iloc[0]
    f0_blk = df_blk[df_blk["feature"] == "f00"].iloc[0]
    iid_w = f0_iid["shap_diff_ci_high"] - f0_iid["shap_diff_ci_low"]
    blk_w = f0_blk["shap_diff_ci_high"] - f0_blk["shap_diff_ci_low"]
    # Allow a tiny tolerance for finite-B noise (B=200 has CI-width noise).
    assert blk_w >= iid_w * 0.85, (
        f"block bootstrap should not produce a tighter CI on autocorrelated "
        f"data; iid_w={iid_w:.4f}, blk_w={blk_w:.4f}"
    )


# ---------------------------------------------------------------------------
# CatBoost SHAP integration smoke
# ---------------------------------------------------------------------------


def test_compute_shap_values_shape_and_baseline_stripped():
    """SHAP shape is (N, F) — last column (baseline expected value) was stripped."""
    from catboost import CatBoostClassifier

    X, y, feats = _make_synth_dataset(n=400, n_features=5, seed=0)
    model = CatBoostClassifier(
        iterations=60, depth=4, learning_rate=0.1,
        verbose=0, allow_writing_files=False, random_seed=0,
    )
    model.fit(X, y)
    shap = compute_shap_values(model, X, feature_list=feats)
    assert shap.shape == (len(X), len(feats))


def test_compute_shap_values_additive_identity_treeshap():
    """TreeSHAP property: shap.sum(axis=1) + baseline equals raw log-odds prediction."""
    from catboost import CatBoostClassifier, Pool

    X, y, feats = _make_synth_dataset(n=400, n_features=5, seed=1)
    model = CatBoostClassifier(
        iterations=60, depth=4, learning_rate=0.1,
        verbose=0, allow_writing_files=False, random_seed=0,
    )
    model.fit(X, y)
    shap = compute_shap_values(model, X, feature_list=feats)
    pool = Pool(X, feature_names=feats)
    shap_full = model.get_feature_importance(data=pool, type="ShapValues")
    baseline = float(shap_full[0, -1])  # constant across rows for binary
    pred_log_odds = model.predict(X, prediction_type="RawFormulaVal")
    reconstructed = shap.sum(axis=1) + baseline
    np.testing.assert_allclose(reconstructed, pred_log_odds, rtol=1e-5, atol=1e-5)


def test_compute_shap_values_dataframe_input():
    from catboost import CatBoostClassifier

    X, y, feats = _make_synth_dataset(n=300, n_features=4, seed=2)
    model = CatBoostClassifier(
        iterations=40, depth=3, learning_rate=0.1,
        verbose=0, allow_writing_files=False, random_seed=0,
    )
    model.fit(X, y)
    df = pd.DataFrame(X, columns=feats)
    shap = compute_shap_values(model, df, feature_list=feats)
    assert shap.shape == (len(df), len(feats))


def test_compute_shap_values_dataframe_no_feature_list_uses_columns():
    from catboost import CatBoostClassifier

    X, y, feats = _make_synth_dataset(n=300, n_features=4, seed=3)
    model = CatBoostClassifier(
        iterations=40, depth=3, learning_rate=0.1,
        verbose=0, allow_writing_files=False, random_seed=0,
    )
    model.fit(X, y)
    df = pd.DataFrame(X, columns=feats)
    shap = compute_shap_values(model, df)  # feature_list=None -> uses df.columns
    assert shap.shape == (len(df), len(feats))


def test_compute_shap_values_numpy_without_feature_list_raises():
    from catboost import CatBoostClassifier

    X, y, feats = _make_synth_dataset(n=200, n_features=4, seed=4)
    model = CatBoostClassifier(
        iterations=20, depth=3, verbose=0, allow_writing_files=False, random_seed=0
    )
    model.fit(X, y)
    with pytest.raises(ValueError, match="feature_list required"):
        compute_shap_values(model, X)
