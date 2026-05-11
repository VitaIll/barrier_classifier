"""Per-instance SHAP cohort decomposition (TP / FP / TN / FN).

Operating idea: aggregate per-row TreeSHAP attributions within each error
cohort, then compare. The most actionable view is the **signed effect-size
disagreement** — features whose mean SHAP differs most between FPs and FNs.
Those features are systematically pushing the model in the wrong direction
in different ways for the two error modes.

Public API:

- ``compute_shap_values(model, X, feature_list)`` -> (N, F) per-row SHAP
  matrix in log-odds space. Strips CatBoost's trailing baseline column.
- ``cohort_assignments(y, p, threshold)`` -> length-N array of cohort
  labels in {"TP", "FP", "TN", "FN"}.
- ``cohort_mean_shap(shap, cohorts, feature_list)`` -> long-form DataFrame
  of per-cohort, per-feature mean SHAP and cohort sample sizes.
- ``signed_effect_size_disagreement(shap, cohorts, feature_list)`` -> DataFrame
  with ``(mean_FP - mean_FN) / pooled_std`` and ranked by absolute effect.
  Standardized so features at different SHAP scales are comparable.
- ``discriminative_shap(shap, cohorts, feature_list)`` -> DataFrame of
  L2-regularized logistic-regression coefficients of {FP=1, FN=0} on
  per-row SHAP. Handles cross-feature correlations that mean-SHAP alone
  cannot.
- ``bootstrap_shap_diff(shap, cohorts, feature_list, B, ci, seed)`` ->
  bootstrap CIs on the per-feature mean(FP) - mean(FN) difference. Shows
  which "inconsistent direction" features survive sampling noise.

All modules consume a SHAP matrix and a cohort vector — they are
model-agnostic once SHAP is computed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .bootstrap import DEFAULT_B, DEFAULT_CI, block_indices

COHORT_LABELS = ("TP", "FP", "TN", "FN")


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------


def compute_shap_values(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    feature_list: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """TreeSHAP per-row, per-feature contribution matrix in log-odds space.

    CatBoost's ``get_feature_importance(type='ShapValues')`` returns
    ``(N, F+1)`` for binary classifiers — the last column is the baseline
    expected log-odds. This function strips it and returns ``(N, F)``.

    Identity: ``shap.sum(axis=1) + baseline ≈ model.predict(X, prediction_type='RawFormulaVal')``.
    """
    from catboost import Pool

    if isinstance(X, pd.DataFrame):
        if feature_list is None:
            feature_list = list(X.columns)
        X_arr = X[list(feature_list)].to_numpy()
    else:
        X_arr = np.asarray(X)
        if feature_list is None:
            raise ValueError("feature_list required when X is not a DataFrame")

    pool = Pool(X_arr, feature_names=list(feature_list))
    shap_full = model.get_feature_importance(data=pool, type="ShapValues")
    if shap_full.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape {shap_full.shape}; expected 2D")
    if shap_full.shape[1] == len(feature_list) + 1:
        return shap_full[:, :-1].astype(float)
    if shap_full.shape[1] == len(feature_list):
        return shap_full.astype(float)
    raise ValueError(
        f"SHAP shape {shap_full.shape} doesn't match feature count {len(feature_list)}"
    )


# ---------------------------------------------------------------------------
# Cohort assignment
# ---------------------------------------------------------------------------


def cohort_assignments(
    y: np.ndarray, p: np.ndarray, *, threshold: float
) -> np.ndarray:
    """Assign each row to ``{TP, FP, TN, FN}`` at the given operating threshold."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= threshold).astype(int)
    out = np.empty(len(y), dtype="<U2")
    out[(pred == 1) & (y == 1)] = "TP"
    out[(pred == 1) & (y == 0)] = "FP"
    out[(pred == 0) & (y == 0)] = "TN"
    out[(pred == 0) & (y == 1)] = "FN"
    return out


def cohort_counts(cohorts: np.ndarray) -> dict:
    """Count of rows per cohort label."""
    out = {label: 0 for label in COHORT_LABELS}
    unique, counts = np.unique(cohorts, return_counts=True)
    for u, c in zip(unique, counts):
        out[str(u)] = int(c)
    return out


# ---------------------------------------------------------------------------
# Cohort-mean SHAP (long form)
# ---------------------------------------------------------------------------


def cohort_mean_shap(
    shap: np.ndarray, cohorts: np.ndarray, feature_list: Sequence[str]
) -> pd.DataFrame:
    """Long-form per-cohort, per-feature mean SHAP.

    Columns: ``cohort, feature, mean_shap, n``. Cohorts with zero rows are
    omitted (they contribute no row to the result).
    """
    if shap.shape[1] != len(feature_list):
        raise ValueError(
            f"SHAP cols {shap.shape[1]} != feature_list len {len(feature_list)}"
        )
    rows = []
    for label in COHORT_LABELS:
        mask = cohorts == label
        n = int(mask.sum())
        if n == 0:
            continue
        means = shap[mask].mean(axis=0)
        for fi, fname in enumerate(feature_list):
            rows.append(
                {"cohort": label, "feature": fname, "mean_shap": float(means[fi]), "n": n}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Signed effect-size disagreement (the headline cohort metric)
# ---------------------------------------------------------------------------


def signed_effect_size_disagreement(
    shap: np.ndarray,
    cohorts: np.ndarray,
    feature_list: Sequence[str],
    *,
    min_cohort_size: int = 5,
) -> pd.DataFrame:
    """Per-feature ``(mean_SHAP_FP - mean_SHAP_FN) / pooled_std``.

    Features at the top of this ranking are pushing toward FPs and away from
    FNs (positive effect size) or the opposite (negative effect size). The
    standardization makes features at different SHAP scales comparable.

    Returns an empty DataFrame if either cohort has fewer than
    ``min_cohort_size`` samples.

    Ranked by absolute effect size descending.
    """
    if shap.shape[1] != len(feature_list):
        raise ValueError("SHAP cols / feature_list length mismatch")
    fp_mask = cohorts == "FP"
    fn_mask = cohorts == "FN"
    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())
    if n_fp < min_cohort_size or n_fn < min_cohort_size:
        return pd.DataFrame()

    fp_shap = shap[fp_mask]
    fn_shap = shap[fn_mask]
    fp_mean = fp_shap.mean(axis=0)
    fn_mean = fn_shap.mean(axis=0)
    fp_var = fp_shap.var(axis=0, ddof=1)
    fn_var = fn_shap.var(axis=0, ddof=1)
    pooled_var = ((n_fp - 1) * fp_var + (n_fn - 1) * fn_var) / max(n_fp + n_fn - 2, 1)
    pooled_std = np.sqrt(np.maximum(pooled_var, 1e-18))
    effect = (fp_mean - fn_mean) / pooled_std
    df = pd.DataFrame(
        {
            "feature": list(feature_list),
            "mean_shap_fp": fp_mean,
            "mean_shap_fn": fn_mean,
            "shap_diff_fp_minus_fn": fp_mean - fn_mean,
            "pooled_std": pooled_std,
            "effect_size": effect,
            "n_fp": n_fp,
            "n_fn": n_fn,
        }
    )
    df["abs_effect_size"] = df["effect_size"].abs()
    return df.sort_values("abs_effect_size", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Discriminative SHAP (logistic regression on FP vs FN)
# ---------------------------------------------------------------------------


def discriminative_shap(
    shap: np.ndarray,
    cohorts: np.ndarray,
    feature_list: Sequence[str],
    *,
    C: float = 1.0,
    max_iter: int = 2000,
    min_cohort_size: int = 5,
) -> pd.DataFrame:
    """L2-regularized logistic regression of ``{FP=1, FN=0}`` on per-row SHAP.

    Coefficients reveal which features, *jointly*, distinguish FPs from FNs
    after controlling for correlations with other features. This is more
    robust than mean-SHAP-per-cohort when SHAP values are correlated across
    features (which they always are in tree models).

    Returns an empty DataFrame if either cohort has fewer than
    ``min_cohort_size`` rows.
    """
    if shap.shape[1] != len(feature_list):
        raise ValueError("SHAP cols / feature_list length mismatch")
    from sklearn.linear_model import LogisticRegression

    fp_mask = cohorts == "FP"
    fn_mask = cohorts == "FN"
    if int(fp_mask.sum()) < min_cohort_size or int(fn_mask.sum()) < min_cohort_size:
        return pd.DataFrame()

    error_mask = fp_mask | fn_mask
    X = shap[error_mask]
    y_err = fp_mask[error_mask].astype(int)  # 1 if FP, 0 if FN
    if len(np.unique(y_err)) < 2:
        return pd.DataFrame()

    lr = LogisticRegression(
        C=C, max_iter=max_iter, penalty="l2", solver="lbfgs", n_jobs=-1
    )
    lr.fit(X, y_err)
    coefs = lr.coef_[0]
    df = pd.DataFrame(
        {
            "feature": list(feature_list),
            "discriminative_coef": coefs,
            "abs_coef": np.abs(coefs),
        }
    )
    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bootstrap CI on (mean_FP - mean_FN)
# ---------------------------------------------------------------------------


def bootstrap_shap_diff(
    shap: np.ndarray,
    cohorts: np.ndarray,
    feature_list: Sequence[str],
    *,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    seed: int = 0,
    min_cohort_size: int = 5,
    block_size: Optional[int] = None,
) -> pd.DataFrame:
    """Bootstrap CI on per-feature ``mean(SHAP_FP) - mean(SHAP_FN)``.

    Returns ranked DataFrame: ``feature, shap_diff, shap_diff_ci_low,
    shap_diff_ci_high, abs_diff``. Use this to filter the
    ``signed_effect_size_disagreement`` shortlist by survival under
    sampling noise.

    Sampling mode is cadence-aware:

    - ``block_size=None`` (default, correct for boundary cadence): resample
      WITHIN each cohort independently, preserving the FP and FN cohort
      sizes. Right when cohort members are independent samples.
    - ``block_size > 1`` (required for 1-min cadence overlapping labels):
      resample the ORIGINAL row indices as contiguous blocks of length
      ``block_size`` (≈ M). For each replicate, cohort membership is
      re-derived from which resampled rows were originally FP / FN, then
      the SHAP-mean difference is computed on those subsets. Replicates
      whose resampled FP or FN count falls below ``min_cohort_size`` are
      skipped — the recorded ``B_effective`` is the number that survived.
    """
    if shap.shape[1] != len(feature_list):
        raise ValueError("SHAP cols / feature_list length mismatch")
    fp_idx = np.flatnonzero(cohorts == "FP")
    fn_idx = np.flatnonzero(cohorts == "FN")
    if len(fp_idx) < min_cohort_size or len(fn_idx) < min_cohort_size:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    F = shap.shape[1]

    if block_size is not None and int(block_size) > 1:
        idx_matrix = block_indices(len(cohorts), B, rng, block_size=int(block_size))
        replicate_samples: list[np.ndarray] = []
        is_fp = cohorts == "FP"
        is_fn = cohorts == "FN"
        for b in range(B):
            row_idx = idx_matrix[b]
            fp_sel = row_idx[is_fp[row_idx]]
            fn_sel = row_idx[is_fn[row_idx]]
            if len(fp_sel) < min_cohort_size or len(fn_sel) < min_cohort_size:
                continue
            replicate_samples.append(
                shap[fp_sel].mean(axis=0) - shap[fn_sel].mean(axis=0)
            )
        if not replicate_samples:
            return pd.DataFrame()
        samples = np.stack(replicate_samples, axis=0)
    else:
        fp_resamples = fp_idx[rng.integers(0, len(fp_idx), size=(B, len(fp_idx)))]
        fn_resamples = fn_idx[rng.integers(0, len(fn_idx), size=(B, len(fn_idx)))]
        samples = np.empty((B, F), dtype=float)
        for b in range(B):
            samples[b] = shap[fp_resamples[b]].mean(axis=0) - shap[fn_resamples[b]].mean(axis=0)

    point = shap[fp_idx].mean(axis=0) - shap[fn_idx].mean(axis=0)
    alpha = (1.0 - ci) / 2.0
    df = pd.DataFrame(
        {
            "feature": list(feature_list),
            "shap_diff": point,
            "shap_diff_ci_low": np.quantile(samples, alpha, axis=0),
            "shap_diff_ci_high": np.quantile(samples, 1.0 - alpha, axis=0),
        }
    )
    df["abs_diff"] = df["shap_diff"].abs()
    df["ci_excludes_zero"] = (df["shap_diff_ci_low"] > 0) | (df["shap_diff_ci_high"] < 0)
    df.attrs["B_effective"] = int(samples.shape[0])
    return df.sort_values("abs_diff", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_top_features_grouped_by_cohort(
    cohort_means: pd.DataFrame,
    feature_subset: Sequence[str],
    *,
    ax=None,
):
    """Grouped barplot: each top feature gets one bar per cohort (TP/FP/TN/FN)
    showing mean SHAP. Best read alongside the effect-size ranking."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5.5))
    sub = cohort_means[cohort_means["feature"].isin(feature_subset)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no rows in subset", ha="center", va="center", transform=ax.transAxes)
        return ax
    pivot = sub.pivot(index="feature", columns="cohort", values="mean_shap").reindex(
        list(feature_subset)
    )
    available = [c for c in COHORT_LABELS if c in pivot.columns]
    pivot = pivot[available]
    n_feat = len(pivot)
    width = 0.8 / max(len(available), 1)
    x = np.arange(n_feat)
    palette = {"TP": "#2ecc71", "FP": "#e74c3c", "TN": "#95a5a6", "FN": "#f1c40f"}
    for i, c in enumerate(available):
        ax.bar(x + (i - len(available) / 2 + 0.5) * width, pivot[c], width=width, label=c, color=palette[c])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("mean SHAP (log-odds)")
    ax.set_title("Mean SHAP per cohort (top features)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_signed_effect_size(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
    ax=None,
):
    """Horizontal bar of effect-size ranking: positive = pushes toward FPs."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    if df.empty:
        ax.text(0.5, 0.5, "no rows (cohort too small)", ha="center", va="center", transform=ax.transAxes)
        return ax
    sub = df.head(top_n).iloc[::-1]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in sub["effect_size"]]
    ax.barh(sub["feature"], sub["effect_size"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Standardized effect size: (mean SHAP FP - FN) / pooled std")
    ax.set_title(
        f"Top {top_n} features by signed effect-size disagreement\n"
        "red = pushes toward FPs (away from FNs); blue = the opposite"
    )
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_bootstrap_shap_diff_with_ci(
    df: pd.DataFrame, *, top_n: int = 20, ax=None,
):
    """Top-N features by |mean(SHAP FP) - mean(SHAP FN)| with bootstrap CIs."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    if df.empty:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes)
        return ax
    sub = df.head(top_n).iloc[::-1]
    y_pos = np.arange(len(sub))
    err = np.abs(np.vstack([
        sub["shap_diff"] - sub["shap_diff_ci_low"],
        sub["shap_diff_ci_high"] - sub["shap_diff"],
    ]))
    colors = [
        "#e74c3c" if (lo > 0) else "#3498db" if (hi < 0) else "#7f8c8d"
        for lo, hi in zip(sub["shap_diff_ci_low"], sub["shap_diff_ci_high"])
    ]
    ax.barh(y_pos, sub["shap_diff"], xerr=err, color=colors, alpha=0.85, capsize=3)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["feature"])
    ax.set_xlabel("mean SHAP (FP) - mean SHAP (FN), with 95% bootstrap CI")
    ax.set_title(
        f"Top {top_n} features by |mean SHAP FP - FN|  "
        "(red bars = CI excludes 0 toward FP; blue = toward FN; grey = ambiguous)"
    )
    ax.grid(axis="x", alpha=0.3)
    return ax
