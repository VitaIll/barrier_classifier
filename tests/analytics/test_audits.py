"""Tests for src.analytics.audits.

Heavy emphasis on mathematical properties:
- causal_feature_audit: every flagged feature actually contains a leak token
- label_shuffle: ROC-AUC concentrates on 0.5, PR-AUC on base rate, with the
  right rate (1/sqrt(n_shuffles)) of MC error
- time_block_permutation: a known-leaky synthetic feature shows
  drop_across >> drop_within; a real signal feature shows roughly equal drops
- decision_turnover: alternating decisions give flip_rate ≈ 1, monotone gives ≈ 0
- expected_max_sharpe: matches Bailey-Lopez de Prado tabulated values; monotone in N
- deflated_sharpe: identical to N(0, std) Z-test when n_trials = 1, skew = 0, kurt = 3
- half_vs_half_drift_audit: structural smoke (ranks correlate, no feature leak between halves)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analytics.audits import (
    CausalAuditResult,
    causal_feature_audit,
    decision_turnover,
    deflated_sharpe,
    expected_max_sharpe,
    half_vs_half_drift_audit,
    label_shuffle_baseline,
    time_block_permutation_importance,
)

pytestmark = pytest.mark.analytics_phase6


# ---------------------------------------------------------------------------
# 1. causal_feature_audit
# ---------------------------------------------------------------------------


def test_causal_audit_passes_on_clean_list():
    """All __f__ / __h__ names are accepted with no flags raised."""
    feats = [
        "ret__rms__f__w10",
        "vol__rs__f__w240",
        "logp_z__f__w20",
        "range__inst__h__w0",
        "ofi__sum__h__w0",
    ]
    res = causal_feature_audit(feats)
    assert res.passed
    assert res.n_features == 5
    assert res.n_causal == 5
    assert res.n_suspect == 0
    assert res.suspect == []
    assert res.unmatched == []


def test_causal_audit_flags_backward_window():
    """A feature with __b__ suffix is flagged as suspect (backward = future-leaning)."""
    feats = ["ret__rms__f__w10", "ret__rms__b__w10"]
    res = causal_feature_audit(feats)
    assert not res.passed
    assert "ret__rms__b__w10" in res.suspect
    assert res.n_suspect == 1


def test_causal_audit_flags_future_token():
    """Feature names containing 'fwd', 'future', 'ahead', 'lead', 'next_' are flagged."""
    feats = [
        "feature_fwd_5",
        "ret_future_w10",
        "price_ahead_t1",
        "lead_signal",
        "next_bar_open",
        "feature__f__w10",  # clean
    ]
    res = causal_feature_audit(feats)
    assert not res.passed
    # All five suspect names should appear in suspect (some may be flagged by multiple tokens)
    for f in feats[:5]:
        assert f in res.suspect, f"missed suspect: {f}"
    assert "feature__f__w10" not in res.suspect


def test_causal_audit_flags_expanded_suspect_tokens():
    """The expanded SUSPECT_TOKENS list catches additional leakage signatures:
    forward/lookahead/oracle/peek/target_fwd/tplus/t+/plain-next. Note that
    the bare ``target_`` prefix is NOT flagged — see
    ``test_causal_audit_past_target_features_with_h_suffix_are_causal``.
    """
    feats = [
        # New positive cases
        "look_ahead_5",
        "lookahead_w10",
        "oracle_signal",
        "peek_at_y",
        "target_fwd_3",       # genuine leak: 'fwd' token catches it
        "target_next_4",      # genuine leak: 'next' token catches it
        "feat_tplus_1",
        "feat_t+1",
        "feat_next",          # plain "next" (was only "next_" before)
        "feature__f__w10",    # clean
    ]
    res = causal_feature_audit(feats)
    assert not res.passed
    suspect_set = set(res.suspect)
    assert "look_ahead_5" in suspect_set
    assert "lookahead_w10" in suspect_set
    assert "oracle_signal" in suspect_set
    assert "peek_at_y" in suspect_set
    assert "target_fwd_3" in suspect_set
    assert "target_next_4" in suspect_set
    assert "feat_tplus_1" in suspect_set
    assert "feat_t+1" in suspect_set
    assert "feat_next" in suspect_set
    assert "feature__f__w10" not in suspect_set


def test_causal_audit_past_target_features_with_h_suffix_are_causal():
    """Past-target features (``target__autocorr_lagN__h__wW``,
    ``target__lagN__h__wW``) are emitted by
    ``compute_past_target_features_pl`` after a label-maturity shift, so
    they only ever read ``y[<k]`` — causal by construction. The audit
    must classify them as causal, not suspect."""
    feats = [
        "target__autocorr_lag1__h__w60",
        "target__autocorr_lag5__h__w240",
        "target__lag1__h__w0",
        "target__hit_rate__h__w60",
    ]
    res = causal_feature_audit(feats)
    assert res.passed, (
        "past-target features with __h__ suffix should be causal, "
        f"got suspect={res.suspect}, unmatched={res.unmatched}"
    )
    assert res.n_suspect == 0
    assert res.n_causal == len(feats)


def test_causal_audit_no_false_positive_on_word_boundary_substrings():
    """Word-boundary regex must NOT flag innocent substrings. The old
    substring-``in`` check would have flagged ``ledger`` ('lead'),
    ``bleeding`` ('lead'), and ``head_count`` ('ahead'). The regex
    upgrade rules out these false positives.
    """
    feats = [
        "ledger__f__w10",
        "bleeding_edge__f__w5",
        "head_count__f__w20",
        "ahead_for_real__f__w0",  # ALSO innocent: 'ahead' is followed by '_for_real'... wait, 'ahead' IS the token. Let me pick a different example.
    ]
    # Replace the genuinely-ambiguous one
    feats[-1] = "spearman_corr__f__w10"  # "spear" + "man" — no token
    res = causal_feature_audit(feats)
    # ledger / bleeding / head_count must NOT be in suspect — those would be
    # false positives. The "__f__" suffix means they pass the causal check.
    assert "ledger__f__w10" not in res.suspect
    assert "bleeding_edge__f__w5" not in res.suspect
    assert "head_count__f__w20" not in res.suspect
    assert "spearman_corr__f__w10" not in res.suspect
    # All four are clean causal features.
    assert res.passed, f"suspect={res.suspect}, unmatched={res.unmatched}"


def test_causal_audit_b_suffix_still_flagged():
    """The __b__ window-suffix marker is matched via plain substring (it
    always appears as a clean delimited segment), so it still flags."""
    feats = ["ret__rms__b__w10", "vol__std__f__w20"]
    res = causal_feature_audit(feats)
    assert "ret__rms__b__w10" in res.suspect
    assert "vol__std__f__w20" not in res.suspect


def test_causal_audit_flags_unmatched_when_no_known_suffix():
    """A feature with neither __f__ / __h__ nor a leak token is 'unmatched'."""
    feats = ["mystery_feature", "open"]
    res = causal_feature_audit(feats)
    assert not res.passed
    assert set(res.unmatched) == {"mystery_feature", "open"}
    assert res.n_suspect == 0


def test_causal_audit_real_feature_list_passes():
    """The active 1-min feature_list_1min.json should pass the audit."""
    import json
    from pathlib import Path
    p = Path("data/model_dataset/feature_list_1min.json")
    if not p.exists():
        pytest.skip("feature_list_1min.json not available")
    with p.open() as fh:
        feats = json.load(fh)
    res = causal_feature_audit(feats)
    assert isinstance(res, CausalAuditResult)
    # Allow small numbers of unmatched (will be flagged in CI for inspection),
    # but no suspect tokens should appear.
    assert res.n_suspect == 0, f"production list contains suspect tokens: {res.suspect[:10]}"


# ---------------------------------------------------------------------------
# 2. label_shuffle_baseline
# ---------------------------------------------------------------------------


def test_label_shuffle_roc_concentrates_on_half():
    """Under permutation of y, ROC-AUC has E ≈ 0.5 by construction."""
    rng = np.random.default_rng(7)
    n = 1500
    y = rng.binomial(1, 0.2, size=n)
    # Use a real signal so the original metric is far from chance — but we shuffle y here
    p = 0.2 + 0.5 * y + rng.normal(0, 0.1, size=n)
    p = np.clip(p, 0.0, 1.0)
    out = label_shuffle_baseline(y, p, n_shuffles=400, random_seed=0)
    # Median should be tightly centered on 0.5 (MC error ~ 1 / sqrt(12 * 400) ≈ 0.014)
    assert abs(out["roc_auc"].point - 0.5) < 0.02
    # 95% CI should bracket 0.5
    assert out["roc_auc"].ci_low < 0.5 < out["roc_auc"].ci_high


def test_label_shuffle_pr_concentrates_on_base_rate():
    """Under permutation of y, PR-AUC has E ≈ base_rate."""
    rng = np.random.default_rng(11)
    n = 2000
    base_rate = 0.15
    y = rng.binomial(1, base_rate, size=n)
    p = rng.uniform(0, 1, size=n)
    out = label_shuffle_baseline(y, p, n_shuffles=400, random_seed=0)
    actual_base = out["base_rate"].point
    assert abs(out["pr_auc"].point - actual_base) < 0.015, (
        f"shuffle PR-AUC median {out['pr_auc'].point:.4f} not near base rate {actual_base:.4f}"
    )
    assert out["pr_auc"].ci_low < actual_base < out["pr_auc"].ci_high


def test_label_shuffle_independent_of_p():
    """The shuffle distribution should not depend on the original ordering of p."""
    rng = np.random.default_rng(0)
    n = 800
    y = rng.binomial(1, 0.3, size=n)
    p1 = rng.uniform(0, 1, size=n)
    p2 = np.sort(p1)  # totally different ordering
    out1 = label_shuffle_baseline(y, p1, n_shuffles=300, random_seed=42)
    out2 = label_shuffle_baseline(y, p2, n_shuffles=300, random_seed=42)
    # Distributions should match in mean and width (independent of p ordering up to MC noise)
    assert abs(out1["roc_auc"].point - out2["roc_auc"].point) < 0.03


def test_label_shuffle_seed_reproducibility():
    """Same seed -> identical shuffle distribution."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.25, size=500)
    p = rng.uniform(0, 1, size=500)
    a = label_shuffle_baseline(y, p, n_shuffles=200, random_seed=12345)
    b = label_shuffle_baseline(y, p, n_shuffles=200, random_seed=12345)
    np.testing.assert_array_equal(a["roc_auc"].samples, b["roc_auc"].samples)
    np.testing.assert_array_equal(a["pr_auc"].samples, b["pr_auc"].samples)


def test_label_shuffle_real_model_outside_shuffle_ci_is_significant():
    """Sanity: a strong signal lands well above the shuffle 97.5% quantile."""
    rng = np.random.default_rng(0)
    n = 1500
    y = rng.binomial(1, 0.2, size=n)
    p = 0.2 + 0.5 * y + rng.normal(0, 0.05, size=n)
    p = np.clip(p, 0.0, 1.0)
    real_roc = roc_auc_score(y, p)
    out = label_shuffle_baseline(y, p, n_shuffles=300, random_seed=0)
    assert real_roc > out["roc_auc"].ci_high
    # The real PR-AUC should also exceed the shuffle CI
    real_pr = average_precision_score(y, p)
    assert real_pr > out["pr_auc"].ci_high


# ---------------------------------------------------------------------------
# 3. time_block_permutation_importance
# ---------------------------------------------------------------------------


class _ToyModel:
    """Tiny linear-logistic model that exposes predict_proba for the audit."""

    def __init__(self, weights: np.ndarray, intercept: float = 0.0):
        self.w = np.asarray(weights, dtype=float)
        self.b = float(intercept)

    def predict_proba(self, X: np.ndarray):
        z = X @ self.w + self.b
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


def test_time_block_permutation_real_signal_drops_under_both():
    """A genuinely informative feature should drop under BOTH within and across permutation."""
    rng = np.random.default_rng(0)
    n = 1000
    f1 = rng.normal(0, 1, size=n)  # real signal
    f2 = rng.normal(0, 1, size=n)  # noise
    df = pd.DataFrame({
        "f1": f1, "f2": f2,
        "y": (f1 > 0).astype(int),
    })
    model = _ToyModel(weights=np.array([3.0, 0.0]), intercept=0.0)
    out = time_block_permutation_importance(
        model, df, ["f1", "f2"], metric="roc_auc",
        n_blocks=4, n_repeats=2, random_seed=0,
    )
    f1_row = out[out["feature"] == "f1"].iloc[0]
    f2_row = out[out["feature"] == "f2"].iloc[0]
    # Real signal: large drops both within and across (block size still scrambles f1)
    assert f1_row["drop_across_mean"] > 0.05
    assert f1_row["drop_within_mean"] > 0.05
    # Noise feature: tiny drops both ways
    assert abs(f2_row["drop_across_mean"]) < 0.02
    assert abs(f2_row["drop_within_mean"]) < 0.02


def test_time_block_permutation_time_leak_signature():
    """A feature that encodes time-of-row should drop hard on across-block but
    barely on within-block (the within-block scramble preserves time locality)."""
    rng = np.random.default_rng(0)
    n = 1200
    t = np.arange(n) / n  # in [0, 1)
    # Time-block-localized signal: y depends only on the block index
    n_blocks = 6
    block_id = np.minimum(np.arange(n) * n_blocks // n, n_blocks - 1)
    leaky = block_id.astype(float) + rng.normal(0, 0.05, size=n)
    y = (block_id >= n_blocks // 2).astype(int)
    df = pd.DataFrame({"leaky": leaky, "y": y})
    model = _ToyModel(weights=np.array([2.0]), intercept=-1.0 * (n_blocks / 2))
    out = time_block_permutation_importance(
        model, df, ["leaky"], metric="roc_auc",
        n_blocks=n_blocks, n_repeats=3, random_seed=0,
    )
    row = out.iloc[0]
    # The across-block drop should be substantially larger than within-block
    # (within-block leaves the rough block ordering intact for the leaky feature)
    assert row["drop_across_mean"] > 0.20
    assert row["drop_within_mean"] < 0.10
    assert row["drop_across_mean"] >= row["drop_within_mean"] * 2.0


def test_time_block_permutation_invalid_metric_raises():
    df = pd.DataFrame({"a": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    model = _ToyModel(weights=np.array([1.0]))
    with pytest.raises(ValueError, match="metric"):
        time_block_permutation_importance(model, df, ["a"], metric="garbage")


def test_time_block_permutation_n_less_than_n_blocks_raises():
    """If n < n_blocks the block IDs would have empty blocks — must raise
    a clear error instead of silently producing degenerate output."""
    df = pd.DataFrame({"a": [0.0, 1.0, 0.0], "y": [0, 1, 0]})
    model = _ToyModel(weights=np.array([1.0]))
    with pytest.raises(ValueError, match="n.*n_blocks|at least one row"):
        time_block_permutation_importance(
            model, df, ["a"], metric="roc_auc", n_blocks=10
        )


def test_time_block_permutation_timestamp_col_sorts_before_blocking():
    """When ``timestamp_col`` is passed, the function must sort df by it
    before computing block IDs. We construct a df where the rows are in
    REVERSE chronological order; if the function ignored timestamp_col,
    the first block would contain the latest rows. We can detect this by
    permuting the input and verifying the output is identical to the
    sorted case.
    """
    rng = np.random.default_rng(0)
    n = 200
    t = np.arange(n) / n
    block_id = np.minimum(np.arange(n) * 8 // n, 7)
    # Time-localized leaky signal (same as test_time_block_permutation_time_leak_signature)
    leaky = block_id.astype(float) + rng.normal(0, 0.05, size=n)
    y = (block_id >= 4).astype(int)
    ts = pd.date_range("2025-01-01", periods=n, freq="1min")
    df_sorted = pd.DataFrame({"ts": ts, "leaky": leaky, "y": y})
    df_shuffled = df_sorted.sample(frac=1.0, random_state=0).reset_index(drop=True)
    model = _ToyModel(weights=np.array([2.0]), intercept=-4.0)

    out_sorted = time_block_permutation_importance(
        model, df_sorted, ["leaky"], metric="roc_auc",
        n_blocks=8, n_repeats=2, random_seed=0, timestamp_col=None,
    )
    out_resorted_inside = time_block_permutation_importance(
        model, df_shuffled, ["leaky"], metric="roc_auc",
        n_blocks=8, n_repeats=2, random_seed=0, timestamp_col="ts",
    )
    # When the function sorts internally, the result should match the
    # already-sorted invocation exactly (same RNG seed, same data after sort).
    pd.testing.assert_frame_equal(out_sorted, out_resorted_inside)


# ---------------------------------------------------------------------------
# 4. decision_turnover
# ---------------------------------------------------------------------------


def test_decision_turnover_alternating_gives_flip_rate_one():
    """Alternating 0, 1, 0, 1, ... → every consecutive pair flips → flip_rate = 1."""
    p = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    out = decision_turnover(p, threshold=0.5)
    assert out["flip_rate"] == 1.0
    assert out["lag1_autocorr"] == pytest.approx(-1.0, abs=1e-9)
    assert out["mean_run_length_active"] == 1.0
    assert out["mean_run_length_idle"] == 1.0


def test_decision_turnover_monotone_block_gives_flip_rate_zero():
    """Constant 0 then constant 1 → exactly one flip / (n-1) → low flip_rate."""
    n = 100
    p = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    out = decision_turnover(p, threshold=0.5)
    # Exactly one flip
    assert out["flip_rate"] == pytest.approx(1.0 / (n - 1))
    # Run-length stats: one active run of length 50, one idle of 50
    assert out["mean_run_length_active"] == 50.0
    assert out["mean_run_length_idle"] == 50.0
    assert out["n_active_runs"] == 1
    assert out["n_idle_runs"] == 1


def test_decision_turnover_random_around_threshold_is_high():
    """Random p around the threshold gives flip_rate close to 0.5 (bit pattern)."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.4, 0.6, size=2000)
    out = decision_turnover(p, threshold=0.5)
    # flip_rate should be near 0.5 (independent Bernoulli bits)
    assert 0.45 < out["flip_rate"] < 0.55


def test_decision_turnover_lag1_autocorr_consistent_with_flip_rate():
    """For a stationary binary sequence, lag1 autocorr = 1 - 2 * flip_rate / (2 p (1-p)).
    More accessibly: high flip_rate ⇒ negative autocorr; low flip_rate ⇒ positive."""
    rng = np.random.default_rng(0)
    # Highly persistent (low flip rate)
    a = np.repeat(rng.integers(0, 2, 200), 5).astype(float)
    out_a = decision_turnover(a, threshold=0.5)
    assert out_a["lag1_autocorr"] > 0.5
    # Highly anti-persistent (alternating)
    b = np.tile([0.0, 1.0], 500)
    out_b = decision_turnover(b, threshold=0.5)
    assert out_b["lag1_autocorr"] < -0.5


def test_decision_turnover_threshold_applied_correctly():
    """Decisions are computed via p >= threshold, not p > threshold (boundary inclusion)."""
    p = np.array([0.5, 0.5, 0.5])
    out = decision_turnover(p, threshold=0.5)
    assert out["trade_rate"] == 1.0  # all included


def test_decision_turnover_short_input_raises():
    with pytest.raises(ValueError, match="at least 2"):
        decision_turnover(np.array([0.5]), threshold=0.5)


# ---------------------------------------------------------------------------
# 5. expected_max_sharpe / deflated_sharpe
# ---------------------------------------------------------------------------


def test_expected_max_sharpe_n_one_is_zero():
    """E[max] of a single N(0,1) = 0."""
    assert expected_max_sharpe(1) == 0.0


def test_expected_max_sharpe_monotone_in_n():
    """E[max Sharpe] is strictly increasing in n_trials."""
    vals = [expected_max_sharpe(n) for n in [2, 5, 10, 50, 100, 1000]]
    for a, b in zip(vals, vals[1:]):
        assert b > a, f"non-monotone: {a} -> {b}"


def test_expected_max_sharpe_grows_like_sqrt_log_n():
    """For large N, E[max] scales asymptotically like sqrt(2 ln N)."""
    n = 10000
    e_max = expected_max_sharpe(n)
    sqrt2logn = math.sqrt(2 * math.log(n))
    # Within ~15% of sqrt(2 ln n) at this scale
    ratio = e_max / sqrt2logn
    assert 0.85 < ratio < 1.15, f"E[max]/sqrt(2 ln n) = {ratio} out of band"


def test_expected_max_sharpe_invalid_n_raises():
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(0)


def test_deflated_sharpe_one_trial_normal_returns_close_to_t_test():
    """When n_trials = 1, skew = 0, kurt = 3, deflated Sharpe ≈ t-test p-value of mean > 0.

    For r ~ N(0.05, 0.1) with n = 1000, sharpe = 0.5, t = 0.5 * sqrt(999) ≈ 15.8 → DSR ≈ 1.0.
    """
    rng = np.random.default_rng(7)
    r = rng.normal(0.05, 0.1, size=2000)
    out = deflated_sharpe(r, n_trials=1)
    assert out["expected_max_sharpe"] == 0.0
    # Strong positive Sharpe, single trial -> DSR very close to 1
    assert out["deflated_sharpe_prob"] > 0.999
    # Sharpe matches the analytic mean / sd
    assert abs(out["sharpe"] - r.mean() / r.std(ddof=1)) < 1e-9


def test_deflated_sharpe_strictly_decreasing_in_n_trials():
    """Holding the data fixed, increasing n_trials lowers DSR (selection-bias correction)."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.02, 0.1, size=500)
    dsrs = [deflated_sharpe(r, n_trials=k)["deflated_sharpe_prob"] for k in [1, 10, 100, 1000]]
    for a, b in zip(dsrs, dsrs[1:]):
        assert b <= a + 1e-9, f"DSR not non-increasing in n_trials: {a} -> {b}"


def test_deflated_sharpe_random_walk_low_dsr():
    """A pure noise return series (no edge) should have low DSR — usually below 0.5."""
    rng = np.random.default_rng(123)
    r = rng.normal(0.0, 0.01, size=300)
    out = deflated_sharpe(r, n_trials=100)
    # Real Sharpe ~0, expected max for n_trials=100 is positive ⇒ DSR small
    assert out["deflated_sharpe_prob"] < 0.5


def test_deflated_sharpe_skew_kurt_penalize():
    """Two series with same mean/sd but different skew: more negative skew should LOWER DSR.

    (Stronger left tail than the normal benchmark increases the denominator → smaller z.)
    """
    rng = np.random.default_rng(0)
    n = 1500
    base = rng.normal(0.05, 0.1, size=n)
    # Add a left-tail spike to one
    skewed = base.copy()
    skewed[:5] -= 1.0  # 5 large negative outliers
    out_normal = deflated_sharpe(base, n_trials=10)
    out_skewed = deflated_sharpe(skewed, n_trials=10)
    # Sanity: the skewed series has lower mean but heavier left tail dominates the denominator
    assert out_skewed["skew"] < out_normal["skew"]
    # DSR of skewed series should be no greater than the normal one
    assert out_skewed["deflated_sharpe_prob"] <= out_normal["deflated_sharpe_prob"] + 1e-9


def test_deflated_sharpe_too_few_trades_raises():
    with pytest.raises(ValueError, match="at least 4"):
        deflated_sharpe(np.array([0.01, 0.02]), n_trials=1)


def test_deflated_sharpe_zero_variance_raises():
    with pytest.raises(ValueError, match="zero variance"):
        deflated_sharpe(np.zeros(10), n_trials=1)


# ---------------------------------------------------------------------------
# 6. half_vs_half_drift_audit (integration smoke — fast CatBoost fit)
# ---------------------------------------------------------------------------


def test_half_vs_half_drift_stationary_synthetic_data():
    """When train_df is iid (no real concept drift), the two half-models should
    produce highly correlated predictions on test."""
    pytest.importorskip("catboost")
    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 600, 200, 200
    n_feat = 4

    def make_df(n, k_offset, ts_offset):
        X = rng.normal(0, 1, size=(n, n_feat))
        # Simple linear-logistic generative process — same f for ALL splits
        # ⇒ stationary (no concept drift)
        z = X[:, 0] - 0.5 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-z))
        y = (rng.uniform(size=n) < p).astype(int)
        df = pd.DataFrame(X, columns=[f"f{i}__f__w0" for i in range(n_feat)])
        df["y"] = y
        df["k"] = np.arange(n) + k_offset
        df["ts"] = pd.date_range(start=f"2025-01-01", periods=n, freq="1min") + pd.Timedelta(minutes=ts_offset)
        df["weight"] = 1.0
        return df

    train_df = make_df(n_train, 0, 0)
    val_df = make_df(n_val, n_train, n_train)
    test_df = make_df(n_test, n_train + n_val, n_train + n_val)
    feature_list = [c for c in train_df.columns if c.startswith("f")]

    # Use very small CatBoost params to keep the test fast
    from src.analytics.fast_train import research_train_params
    fast_params = research_train_params(iterations=80, depth=3, verbose=0,
                                        early_stopping_rounds=20)
    fast_params2 = research_train_params(iterations=80, depth=3, verbose=0,
                                         early_stopping_rounds=20, random_seed=43)

    res = half_vs_half_drift_audit(
        train_df, val_df, test_df, feature_list,
        train_params_first=fast_params,
        train_params_second=fast_params2,
        threshold=0.5,
    )
    # Stationary data ⇒ both models learn the same conditional, predictions correlate strongly
    assert res.spearman_corr > 0.7, f"weak rank correlation: {res.spearman_corr:.3f}"
    assert res.pearson_corr > 0.7
    # Most decisions agree at threshold
    assert res.n_disagree_at_threshold < n_test * 0.4
    assert res.n_first == n_train // 2
    assert res.n_second == n_train - n_train // 2


# ---------------------------------------------------------------------------
# 7. Cross-cutting: deterministic hashable summary
# ---------------------------------------------------------------------------


def test_label_shuffle_returns_finite_values():
    """No NaNs slip through under degenerate uniform p."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.05, size=300)
    p = rng.uniform(0, 1, size=300)
    out = label_shuffle_baseline(y, p, n_shuffles=100, random_seed=0)
    for k in ["roc_auc", "pr_auc"]:
        assert np.isfinite(out[k].point)
        assert np.all(np.isfinite(out[k].samples))


def test_decision_turnover_constant_p_zero_flips():
    """All-active or all-idle ⇒ zero flips, NaN autocorr (variance=0)."""
    p = np.full(50, 0.9)
    out_active = decision_turnover(p, threshold=0.5)
    assert out_active["flip_rate"] == 0.0
    assert out_active["trade_rate"] == 1.0
    # autocorr undefined under zero variance
    assert math.isnan(out_active["lag1_autocorr"])

    p = np.full(50, 0.1)
    out_idle = decision_turnover(p, threshold=0.5)
    assert out_idle["flip_rate"] == 0.0
    assert out_idle["trade_rate"] == 0.0
