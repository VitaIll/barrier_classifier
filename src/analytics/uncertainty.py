"""CatBoost virtual-ensemble uncertainty decomposition.

Requires the model to have been trained with ``posterior_sampling=True``
(set by default in ``fast_train.research_train_params``). The virtual
ensemble slices the trained tree sequence into K overlapping sub-ensembles
and returns per-replicate class probabilities; we decompose them into:

- **Total uncertainty** (predictive entropy): H[E_b[p_b]] — what the model
  is unsure about overall.
- **Data uncertainty** (expected entropy): E_b[H(p_b)] — irreducible
  aleatoric noise the data carries even with a perfect model.
- **Knowledge uncertainty** (mutual information): MI = Total - Data —
  epistemic uncertainty the model itself carries; high values flag inputs
  far from training distribution.

Reference: Malinin, Prokhorenkova, Ustimenko, "Uncertainty in Gradient
Boosting via Ensembles" (ICLR 2021). For binary classification, the
correct path is ``Logloss + posterior_sampling=True`` and the
``VirtEnsembles`` prediction type, NOT the regression-loss
``RMSEWithUncertainty`` route.

Public API:

- ``virtual_ensemble_predictions(model, X, virtual_ensembles_count, feature_list)``
  -> (N, K) per-row, per-replicate class-1 probabilities.
- ``predictive_uncertainty(p_ve)`` -> dict with mean_p, total_uncertainty,
  data_uncertainty, knowledge_uncertainty (all length N).
- ``hit_rate_heatmap(y, mean_p, knowledge_unc, n_bins_p, n_bins_unc)`` ->
  long-form per-cell empirical hit rate with Wilson CIs. **Critical
  validation**: hit rate should DROP as knowledge_uncertainty rises at
  fixed p. If it doesn't, the uncertainty signal is dead.
- ``joint_gate_sweep(y, mean_p, knowledge_unc, p_thresholds, unc_thresholds)``
  -> precision / recall / trade rate at every (τ_p, τ_unc) pair, for the
  two-dimensional gate ``predict iff mean_p >= τ_p AND knowledge_unc <= τ_unc``.
- ``variance_reliability(predicted_var, observed_sq_error, n_bins)`` ->
  per-bin diagnostic of whether predicted variance matches realized error.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .degradation import wilson_interval

EPS_ENTROPY = 1e-12  # safe clip for binary entropy at 0 / 1

PredictionType = Literal["probability", "logit"]


# ---------------------------------------------------------------------------
# CatBoost VE prediction
# ---------------------------------------------------------------------------


def virtual_ensemble_predictions(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    *,
    virtual_ensembles_count: int = 10,
    feature_list: Optional[Sequence[str]] = None,
    thread_count: int = -1,
    prediction_type: PredictionType = "probability",
) -> np.ndarray:
    """Per-row, per-virtual-ensemble class-1 probabilities, shape ``(N, K)``.

    The model must have been trained with ``posterior_sampling=True``.
    CatBoost's ``virtual_ensembles_predict(prediction_type='VirtEnsembles')``
    returns per-ensemble outputs in either probability space or log-odds
    space depending on the CatBoost build. ``prediction_type`` tells this
    function which space to expect:

    - ``"probability"`` (default): values are class-1 probabilities in
      ``[0, 1]``. The function clips numerical dust at the endpoints and
      returns them unchanged.
    - ``"logit"``: values are raw log-odds. The function applies the
      sigmoid ``1 / (1 + exp(-x))`` to convert to probabilities.

    The previous behavior auto-detected logits by checking whether any value
    sat outside ``[0, 1]``; that heuristic was brittle (a CatBoost build
    that returns probabilities-with-tiny-negative-noise would have been
    silently sigmoid-mapped, doubly transforming them). The explicit param
    makes the caller's expectation contract-level.
    """
    from catboost import Pool

    if prediction_type not in ("probability", "logit"):
        raise ValueError(
            f"prediction_type must be 'probability' or 'logit', got {prediction_type!r}"
        )

    if isinstance(X, pd.DataFrame):
        if feature_list is None:
            feature_list = list(X.columns)
        X_arr = X[list(feature_list)].to_numpy()
    else:
        X_arr = np.asarray(X)
        if feature_list is None:
            raise ValueError("feature_list required when X is not a DataFrame")
    pool = Pool(X_arr, feature_names=list(feature_list))
    out = model.virtual_ensembles_predict(
        pool,
        prediction_type="VirtEnsembles",
        virtual_ensembles_count=int(virtual_ensembles_count),
        thread_count=int(thread_count),
        verbose=False,
    )
    out = np.asarray(out, dtype=float)
    # CatBoost returns (N, K, 1) or (N, K, 2) for binary classifiers under
    # prediction_type='VirtEnsembles'. Handle both shapes plus bare 2D.
    if out.ndim == 3:
        if out.shape[2] == 2:
            return out[:, :, 1].astype(float)
        if out.shape[2] == 1:
            out = out.squeeze(axis=2)
        else:
            raise ValueError(f"Unexpected last dim {out.shape[2]}; expected 1 or 2")
    if out.ndim != 2:
        raise ValueError(f"Unexpected VE shape after squeeze {out.shape}; expected 2D")
    if prediction_type == "logit":
        # expit is the overflow-safe logistic: 1/(1+exp(-x)) overflows
        # (RuntimeWarning, then 0/inf artifacts) for large negative logits.
        from scipy.special import expit

        out = expit(out)
    # Numerical safety
    return np.clip(out, 0.0, 1.0).astype(float)


# ---------------------------------------------------------------------------
# Uncertainty decomposition
# ---------------------------------------------------------------------------


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Element-wise binary entropy ``H(p) = -p log p - (1-p) log(1-p)``,
    safe at the 0/1 endpoints (using ``0 * log 0 = 0``)."""
    p = np.asarray(p, dtype=float)
    p_safe = np.clip(p, EPS_ENTROPY, 1.0 - EPS_ENTROPY)
    return -(p_safe * np.log(p_safe) + (1.0 - p_safe) * np.log(1.0 - p_safe))


def predictive_uncertainty(p_ve: np.ndarray) -> Dict[str, np.ndarray]:
    """Decompose virtual-ensemble probabilities into total / data / knowledge.

    Identity (Jensen's inequality, exact):
        ``total_uncertainty = data_uncertainty + knowledge_uncertainty``
    where total = H[mean_b p_b], data = mean_b H[p_b], knowledge = MI.

    All four returned arrays have shape ``(N,)``.
    """
    p_ve = np.asarray(p_ve, dtype=float)
    if p_ve.ndim != 2:
        raise ValueError(f"p_ve must be 2D (N, K); got {p_ve.shape}")
    K = p_ve.shape[1]
    if K < 1:
        raise ValueError("virtual_ensembles_count must be >= 1")
    mean_p = p_ve.mean(axis=1)
    total = _binary_entropy(mean_p)
    # Per-replicate entropy then mean over K (axis=1)
    data = _binary_entropy(p_ve).mean(axis=1)
    knowledge = total - data
    # Numerical noise: clip to non-negative
    knowledge = np.maximum(knowledge, 0.0)
    return {
        "mean_p": mean_p,
        "total_uncertainty": total,
        "data_uncertainty": data,
        "knowledge_uncertainty": knowledge,
    }


# ---------------------------------------------------------------------------
# 2D hit-rate validation: does uncertainty separate hits from misses?
# ---------------------------------------------------------------------------


def hit_rate_heatmap(
    y: np.ndarray,
    mean_p: np.ndarray,
    knowledge_unc: np.ndarray,
    *,
    n_bins_p: int = 10,
    n_bins_unc: int = 10,
    alpha_wilson: float = 0.05,
) -> pd.DataFrame:
    """Per-cell empirical hit rate over equal-frequency (p, knowledge_unc) bins.

    Wilson CI on hit rate per cell. Plot expected pattern: at fixed p decile,
    hit rate should DROP as knowledge_unc rises (uncertainty signal informative).
    """
    y = np.asarray(y).astype(int)
    mean_p = np.asarray(mean_p, dtype=float)
    knowledge_unc = np.asarray(knowledge_unc, dtype=float)
    p_edges = np.unique(np.quantile(mean_p, np.linspace(0.0, 1.0, n_bins_p + 1)))
    u_edges = np.unique(np.quantile(knowledge_unc, np.linspace(0.0, 1.0, n_bins_unc + 1)))
    # If many ties, some quantile edges collapse; clip indices accordingly
    p_idx = np.clip(np.digitize(mean_p, p_edges) - 1, 0, len(p_edges) - 2)
    u_idx = np.clip(np.digitize(knowledge_unc, u_edges) - 1, 0, len(u_edges) - 2)

    rows = []
    for i in range(len(p_edges) - 1):
        for j in range(len(u_edges) - 1):
            mask = (p_idx == i) & (u_idx == j)
            n = int(mask.sum())
            if n == 0:
                continue
            n_hit = int(y[mask].sum())
            hit_rate = n_hit / n
            ci_lo, ci_hi = wilson_interval(n_hit, n, alpha=alpha_wilson)
            rows.append(
                {
                    "p_bin": i,
                    "unc_bin": j,
                    "p_lo": float(p_edges[i]),
                    "p_hi": float(p_edges[i + 1]),
                    "unc_lo": float(u_edges[j]),
                    "unc_hi": float(u_edges[j + 1]),
                    "n": n,
                    "n_hits": n_hit,
                    "hit_rate": hit_rate,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Joint gate sweep: (mean_p, knowledge_unc) -> precision / trade rate
# ---------------------------------------------------------------------------


def joint_gate_sweep(
    y: np.ndarray,
    mean_p: np.ndarray,
    knowledge_unc: np.ndarray,
    *,
    p_thresholds: Optional[np.ndarray] = None,
    unc_thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Precision / recall / trade-rate at every ``(τ_p, τ_unc)`` combination.

    Joint gate: predict positive iff ``mean_p >= τ_p AND knowledge_unc <= τ_unc``.
    Comparing the joint gate to the scalar-p gate (i.e. ``τ_unc = ∞``) tells
    you whether uncertainty filtering improves precision at fixed trade rate.
    """
    y = np.asarray(y).astype(int)
    mean_p = np.asarray(mean_p, dtype=float)
    knowledge_unc = np.asarray(knowledge_unc, dtype=float)
    n = len(y)
    n_pos = int(y.sum())

    if p_thresholds is None:
        p_thresholds = np.linspace(0.05, float(mean_p.max()), 20)
    if unc_thresholds is None:
        # decile sweep (high to low)
        unc_thresholds = np.quantile(knowledge_unc, np.linspace(1.0, 0.1, 10))
    rows = []
    for tp in p_thresholds:
        for tu in unc_thresholds:
            sel = (mean_p >= tp) & (knowledge_unc <= tu)
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue
            n_hit = int(y[sel].sum())
            precision = n_hit / n_sel
            recall = n_hit / n_pos if n_pos > 0 else np.nan
            ci_lo, ci_hi = wilson_interval(n_hit, n_sel)
            rows.append(
                {
                    "p_threshold": float(tp),
                    "unc_threshold": float(tu),
                    "n_selected": n_sel,
                    "trade_rate": n_sel / n,
                    "n_hits": n_hit,
                    "precision": precision,
                    "precision_ci_low": ci_lo,
                    "precision_ci_high": ci_hi,
                    "recall": recall,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Variance reliability (calibration of the uncertainty signal itself)
# ---------------------------------------------------------------------------


def variance_reliability(
    predicted_var: np.ndarray,
    observed_sq_error: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin samples by predicted variance, compare mean predicted to mean observed.

    Per equal-frequency bin of predicted variance:
    - mean predicted variance
    - mean observed squared error
    - ratio (1 = perfectly calibrated; > 1 = under-confident; < 1 = over-confident)

    For binary classification, ``predicted_var`` is typically
    ``data_uncertainty`` or the variance ``Var_b[p_b]`` from the virtual
    ensemble; ``observed_sq_error = (y - mean_p) ** 2``.
    """
    predicted_var = np.asarray(predicted_var, dtype=float)
    observed_sq_error = np.asarray(observed_sq_error, dtype=float)
    edges = np.unique(np.quantile(predicted_var, np.linspace(0.0, 1.0, n_bins + 1)))
    bin_idx = np.clip(np.digitize(predicted_var, edges) - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(predicted_var[mask].mean())
        mean_obs = float(observed_sq_error[mask].mean())
        rows.append(
            {
                "bin": b,
                "var_lo": float(edges[b]),
                "var_hi": float(edges[b + 1]),
                "n": n,
                "mean_predicted_var": mean_pred,
                "mean_observed_sq_error": mean_obs,
                "ratio_obs_over_pred": mean_obs / mean_pred if mean_pred > 1e-18 else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_uncertainty_distributions_by_cohort(
    cohorts: np.ndarray,
    unc: Dict[str, np.ndarray],
    *,
    component: str = "knowledge_uncertainty",
    ax=None,
):
    """Box / violin of one uncertainty component (default: knowledge) per cohort."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    arr = np.asarray(unc[component])
    data = []
    labels = []
    palette = {"TP": "#2ecc71", "FP": "#e74c3c", "TN": "#95a5a6", "FN": "#f1c40f"}
    colors = []
    for c in ["TP", "FP", "TN", "FN"]:
        mask = cohorts == c
        if mask.sum() > 0:
            data.append(arr[mask])
            labels.append(f"{c} (n={int(mask.sum())})")
            colors.append(palette[c])
    if not data:
        ax.text(0.5, 0.5, "no cohort rows", ha="center", va="center", transform=ax.transAxes)
        return ax
    bplot = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(component)
    ax.set_title(f"{component} per cohort")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_hit_rate_heatmap_2d(
    df: pd.DataFrame,
    *,
    ax=None,
    annotate: bool = True,
):
    """Heatmap of empirical hit rate over (p_bin, unc_bin)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if df.empty:
        ax.text(0.5, 0.5, "no cells", ha="center", va="center", transform=ax.transAxes)
        return ax
    pivot = df.pivot(index="p_bin", columns="unc_bin", values="hit_rate")
    n_pivot = df.pivot(index="p_bin", columns="unc_bin", values="n")
    sns.heatmap(
        pivot,
        annot=annotate,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=min(1.0, float(pivot.max().max() * 1.05)) if not pivot.empty else 1.0,
        ax=ax,
        cbar_kws={"label": "empirical hit rate"},
    )
    ax.set(
        xlabel="knowledge_uncertainty decile (low -> high, left -> right)",
        ylabel="mean_p decile (low -> high, top -> bottom)",
        title="Empirical hit rate by (p, knowledge_unc) decile  "
        "[expect monotone decrease left-to-right at fixed row]",
    )
    return ax


def plot_joint_gate_vs_scalar_p(
    sweep: pd.DataFrame,
    *,
    ax=None,
):
    """Precision vs trade-rate scatter, scalar-p gate (highest unc threshold) vs joint."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5))
    if sweep.empty:
        ax.text(0.5, 0.5, "empty sweep", ha="center", va="center", transform=ax.transAxes)
        return ax
    # Scalar-p baseline = the highest unc_threshold (i.e. no uncertainty filtering)
    max_unc = sweep["unc_threshold"].max()
    scalar = sweep[sweep["unc_threshold"] == max_unc].sort_values("trade_rate")
    joint = sweep[sweep["unc_threshold"] < max_unc]

    ax.scatter(
        joint["trade_rate"], joint["precision"], c=joint["unc_threshold"],
        cmap="viridis", alpha=0.7, label=None, s=22,
    )
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=joint["unc_threshold"].min() if len(joint) else 0,
                           vmax=joint["unc_threshold"].max() if len(joint) else 1),
    )
    plt.colorbar(sm, ax=ax, label="τ_unc (lower = stricter gate)")
    ax.plot(scalar["trade_rate"], scalar["precision"], color="black", marker="o",
            linewidth=2, label="scalar-p only (no unc filter)")
    ax.set_xscale("log")
    ax.set(
        xlabel="trade rate (log)",
        ylabel="precision",
        title="Joint gate (p AND knowledge_unc) vs scalar-p only\n"
        "Points above the black line = uncertainty filtering improves precision at fixed trade rate",
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return ax


def plot_variance_reliability(
    df: pd.DataFrame,
    *,
    ax=None,
):
    """Mean observed sq error vs mean predicted variance per bin (perfect = identity)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if df.empty:
        ax.text(0.5, 0.5, "no bins", ha="center", va="center", transform=ax.transAxes)
        return ax
    ax.plot(df["mean_predicted_var"], df["mean_observed_sq_error"], marker="o", color="C0")
    lim = max(df["mean_predicted_var"].max(), df["mean_observed_sq_error"].max()) * 1.05
    ax.plot([0, lim], [0, lim], linestyle="--", color="gray", alpha=0.7, label="perfect (y = x)")
    ax.set(
        xlabel="mean predicted variance per bin",
        ylabel="mean observed squared error per bin",
        title="Variance reliability (calibration of the uncertainty signal)",
        xlim=(0, lim), ylim=(0, lim),
    )
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
