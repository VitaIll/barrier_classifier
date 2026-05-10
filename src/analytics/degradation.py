"""Time-degradation diagnostics on the prediction cache.

Operates purely on the cached predictions (k, ts, y, p, regime, split) — no
model retraining required. The cache is the canonical input.

Public surface:

- ``psi(p_ref, p_cur, n_bins)`` -> float — Population Stability Index between
  two probability distributions.
- ``ks_distance(p_ref, p_cur)`` -> (stat, pvalue) — two-sample KS distance.
- ``brier_murphy_decomposition(y, p, n_bins)`` -> dict — exact decomposition
  ``BS = REL - RES + UNC + WBV`` (reliability, resolution, uncertainty,
  within-bin variance).
- ``bootstrap_brier_decomposition(y, p, ...)`` -> dict[str, BootstrapResult] —
  bootstrap CIs on all four components in one pass.
- ``page_hinkley(values, delta, threshold)`` -> (alarm, ph, min) — sequential
  changepoint detector for upward shifts in mean (drift on residuals).
- ``rolling_metrics_with_ci(cache, split, window, step, metric_funcs, B, ...)`` ->
  DataFrame — per-time-window bootstrap CI bands on each metric.
- ``rolling_brier_decomposition(cache, split, window, step, B, ...)`` ->
  DataFrame — per-window Brier-Murphy components with CIs.
- ``psi_ks_rolling(cache, ref_split, target_split, window, step, n_bins)`` ->
  DataFrame — per-window PSI and KS of target probability distribution vs the
  full reference split.
- ``conditional_precision(cache, threshold, split, by)`` -> DataFrame —
  long-form per-cell precision with Wilson CIs (heatmap-ready).
- ``wilson_interval(k, n, alpha)`` -> (low, high) — exact Wilson score CI for
  binomial proportion (used by conditional_precision).

Plot helpers are lazy on matplotlib import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils import expected_calibration_error
from .bootstrap import DEFAULT_B, DEFAULT_CI, BootstrapResult, bootstrap_metric, iid_indices

PSI_EPS = 1e-6  # smoothing for empty bins (so log(a/b) is finite)


# ---------------------------------------------------------------------------
# Distribution-shift primitives
# ---------------------------------------------------------------------------


def psi(
    p_reference: np.ndarray,
    p_current: np.ndarray,
    *,
    n_bins: int = 10,
    eps: float = PSI_EPS,
) -> float:
    """Population Stability Index between two probability distributions.

    ``PSI = sum_b (a_b - b_b) * log(a_b / b_b)`` where ``a_b``, ``b_b`` are
    bin frequencies (proportions of samples per bin). Symmetric in its
    arguments. Empty bins are smoothed with ``eps`` so the log is finite.

    Conventions: PSI < 0.1 stable; 0.1-0.2 moderate shift; > 0.2 significant.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    a = (
        np.bincount(
            np.clip(np.digitize(p_reference, bin_edges) - 1, 0, n_bins - 1),
            minlength=n_bins,
        )
        / len(p_reference)
    )
    b = (
        np.bincount(
            np.clip(np.digitize(p_current, bin_edges) - 1, 0, n_bins - 1),
            minlength=n_bins,
        )
        / len(p_current)
    )
    a = np.where(a == 0, eps, a)
    b = np.where(b == 0, eps, b)
    return float(np.sum((a - b) * np.log(a / b)))


def ks_distance(p_reference: np.ndarray, p_current: np.ndarray) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov distance and p-value."""
    from scipy.stats import ks_2samp

    res = ks_2samp(p_reference, p_current)
    return float(res.statistic), float(res.pvalue)


def wilson_interval(k: int, n: int, *, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Robust at small n where the normal approximation breaks down. Returns
    ``(0.0, 1.0)`` for ``n=0``.
    """
    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import norm

    z = norm.ppf(1.0 - alpha / 2.0)
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half_width = z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    lo = center - half_width
    hi = center + half_width
    # Math says lo=0 exactly at k=0 and hi=1 exactly at k=n; floating-point
    # leaves micro-positive/-negative dust. Hardcode the boundary cases.
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return float(max(0.0, lo)), float(min(1.0, hi))


# ---------------------------------------------------------------------------
# Brier-Murphy decomposition
# ---------------------------------------------------------------------------


def brier_murphy_decomposition(
    y: np.ndarray, p: np.ndarray, *, n_bins: int = 10
) -> Dict[str, float]:
    """Murphy decomposition of the binned Brier score.

    Exact identity (after replacing each prediction with its bin's mean
    probability — the canonical reliability-diagram convention):

        ``brier_binned = reliability - resolution + uncertainty``

    Components, all bin-weighted by ``n_b / N``:

    - ``reliability``  = sum_b (mean_p_b - mean_y_b)^2   — miscalibration squared
    - ``resolution``   = sum_b (mean_y_b - y_bar)^2      — informativeness vs marginal
    - ``uncertainty``  = y_bar * (1 - y_bar)             — irreducible Bernoulli variance

    Diagnostics also returned:

    - ``brier``        : the model's raw Brier score (without binning) — its
      actual squared-error loss. Differs from ``brier_binned`` by within-bin
      prediction spread minus within-bin Cov(p, y); equal when predictions
      are well-discretized within each bin.
    - ``within_bin_variance`` = sum_b (n_b/N) * Var(p|b) — how much raw
      predictions disperse around their bin mean.
    """
    from sklearn.metrics import brier_score_loss

    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, bin_edges, right=False) - 1, 0, n_bins - 1)

    bs = float(brier_score_loss(y, p))
    y_bar = float(y.mean())
    unc = y_bar * (1.0 - y_bar)

    n = len(y)
    rel = 0.0
    res = 0.0
    wbv = 0.0
    p_binned = np.empty_like(p)
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        w = n_b / n
        p_b = float(p[mask].mean())
        y_b = float(y[mask].mean())
        rel += w * (p_b - y_b) ** 2
        res += w * (y_b - y_bar) ** 2
        wbv += w * float(p[mask].var())  # ddof=0
        p_binned[mask] = p_b
    bs_binned = float(brier_score_loss(y, p_binned))
    return {
        "brier": bs,
        "brier_binned": bs_binned,
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "within_bin_variance": wbv,
    }


def bootstrap_brier_decomposition(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
) -> Dict[str, BootstrapResult]:
    """Bootstrap all four Brier-Murphy components in one pass.

    Returns ``{component_name: BootstrapResult}`` for ``brier``, ``reliability``,
    ``resolution``, ``uncertainty``, ``within_bin_variance``.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    keys = [
        "brier",
        "brier_binned",
        "reliability",
        "resolution",
        "uncertainty",
        "within_bin_variance",
    ]
    rng = np.random.default_rng(seed)
    idx = iid_indices(len(y), B, rng, stratify=y if stratify else None)
    samples: Dict[str, np.ndarray] = {k: np.empty(B, dtype=float) for k in keys}
    for b in range(B):
        i = idx[b]
        d = brier_murphy_decomposition(y[i], p[i], n_bins=n_bins)
        for k in keys:
            samples[k][b] = d[k]
    point = brier_murphy_decomposition(y, p, n_bins=n_bins)
    alpha = (1.0 - ci) / 2.0
    return {
        k: BootstrapResult(
            point=point[k],
            median=float(np.quantile(samples[k], 0.5)),
            ci_low=float(np.quantile(samples[k], alpha)),
            ci_high=float(np.quantile(samples[k], 1.0 - alpha)),
            ci=ci,
            B=B,
            samples=samples[k],
        )
        for k in keys
    }


# ---------------------------------------------------------------------------
# Page-Hinkley sequential changepoint detector
# ---------------------------------------------------------------------------


def page_hinkley(
    values: np.ndarray,
    *,
    delta: float = 0.005,
    threshold: float = 50.0,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Page-Hinkley test for an upward shift in the mean.

    ``ph[t] = sum_{i<=t} (x_i - running_mean(x[0:t]) - delta)``
    ``min[t] = min_{s<=t} ph[s]``
    Alarm at the first t where ``ph[t] - min[t] > threshold``.

    Returns ``(alarm_index, ph_series, min_series)`` with ``alarm_index = -1``
    if no alarm fires. Useful on Brier residuals (per-sample squared error)
    to flag when calibration starts drifting.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    ph = np.zeros(n, dtype=float)
    mn = np.zeros(n, dtype=float)
    cum = 0.0
    min_cum = 0.0
    running_mean = 0.0
    alarm = -1
    for t in range(n):
        running_mean = running_mean + (x[t] - running_mean) / (t + 1)
        cum += x[t] - running_mean - delta
        if cum < min_cum:
            min_cum = cum
        ph[t] = cum
        mn[t] = min_cum
        if alarm < 0 and (cum - min_cum) > threshold:
            alarm = t
    return alarm, ph, mn


# ---------------------------------------------------------------------------
# Rolling-window metrics with bootstrap CIs
# ---------------------------------------------------------------------------


def _iter_time_windows(
    df: pd.DataFrame, ts_col: str, window: pd.Timedelta, step: pd.Timedelta
):
    """Yield ``(window_start, window_end, win_df)`` for a time-anchored
    rolling window. Windows are right-open: ``[start, end)``."""
    if len(df) == 0:
        return
    start_ts = df[ts_col].min()
    end_ts = df[ts_col].max()
    t = start_ts + window
    while t <= end_ts + step:
        win_start = t - window
        win = df[(df[ts_col] >= win_start) & (df[ts_col] < t)]
        yield win_start, t, win
        t += step


def rolling_metrics_with_ci(
    cache: pd.DataFrame,
    split: str = "test",
    *,
    window: str = "3D",
    step: str = "1D",
    metric_funcs: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
    B: int = 500,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
    min_n: int = 150,
    min_pos: int = 15,
) -> pd.DataFrame:
    """Per-window bootstrap CI bands on each metric for a given split.

    Returns long-form DataFrame with columns:
    ``window_start, window_end, n_samples, n_pos, base_rate``,
    plus ``<metric>_point, <metric>_ci_low, <metric>_ci_high`` per metric.
    Skips windows with fewer than ``min_n`` samples or ``min_pos`` positives,
    or those that are single-class.
    """
    if metric_funcs is None:
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            roc_auc_score,
        )

        metric_funcs = {
            "roc_auc": lambda y, p: float(roc_auc_score(y, p)),
            "pr_auc": lambda y, p: float(average_precision_score(y, p)),
            "brier_score": lambda y, p: float(brier_score_loss(y, p)),
            "ece_10bin": lambda y, p: float(expected_calibration_error(y, p, n_bins=10)),
        }

    df = cache[cache["split"] == split].sort_values("ts").reset_index(drop=True)
    rows = []
    for win_start, win_end, win in _iter_time_windows(
        df, "ts", pd.Timedelta(window), pd.Timedelta(step)
    ):
        n = len(win)
        n_pos = int(win["y"].sum()) if n > 0 else 0
        if n < min_n or n_pos < min_pos or n_pos == n:
            continue
        y = win["y"].to_numpy()
        p = win["p"].to_numpy()
        row: Dict[str, Any] = {
            "window_start": win_start,
            "window_end": win_end,
            "n_samples": n,
            "n_pos": n_pos,
            "base_rate": float(y.mean()),
        }
        for name, fn in metric_funcs.items():
            res = bootstrap_metric(fn, y, p, B=B, ci=ci, stratify=stratify, seed=seed)
            row[f"{name}_point"] = res.point
            row[f"{name}_ci_low"] = res.ci_low
            row[f"{name}_ci_high"] = res.ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_brier_decomposition(
    cache: pd.DataFrame,
    split: str = "test",
    *,
    window: str = "3D",
    step: str = "1D",
    n_bins: int = 10,
    B: int = 300,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
    min_n: int = 150,
    min_pos: int = 15,
) -> pd.DataFrame:
    """Per-window Brier-Murphy decomposition with bootstrap CIs.

    Each row carries point + CI for ``brier``, ``reliability``, ``resolution``,
    ``uncertainty``, ``within_bin_variance``.
    """
    df = cache[cache["split"] == split].sort_values("ts").reset_index(drop=True)
    rows = []
    keys = [
        "brier",
        "brier_binned",
        "reliability",
        "resolution",
        "uncertainty",
        "within_bin_variance",
    ]
    for win_start, win_end, win in _iter_time_windows(
        df, "ts", pd.Timedelta(window), pd.Timedelta(step)
    ):
        n = len(win)
        n_pos = int(win["y"].sum()) if n > 0 else 0
        if n < min_n or n_pos < min_pos or n_pos == n:
            continue
        y = win["y"].to_numpy()
        p = win["p"].to_numpy()
        decomp = bootstrap_brier_decomposition(
            y, p, n_bins=n_bins, B=B, ci=ci, stratify=stratify, seed=seed
        )
        row: Dict[str, Any] = {
            "window_start": win_start,
            "window_end": win_end,
            "n_samples": n,
            "n_pos": n_pos,
            "base_rate": float(y.mean()),
        }
        for k in keys:
            row[f"{k}_point"] = decomp[k].point
            row[f"{k}_ci_low"] = decomp[k].ci_low
            row[f"{k}_ci_high"] = decomp[k].ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def psi_ks_rolling(
    cache: pd.DataFrame,
    *,
    reference_split: str = "val",
    target_split: str = "test",
    window: str = "3D",
    step: str = "1D",
    n_bins: int = 10,
    min_n: int = 150,
) -> pd.DataFrame:
    """Per-window PSI and KS distance of target probabilities vs full reference split.

    The reference is the entire ``reference_split``'s probability distribution
    (a static benchmark); each rolling window of the target split is compared
    against it. Use to detect prediction-distribution drift over time.
    """
    ref_p = cache[cache["split"] == reference_split]["p"].to_numpy()
    if len(ref_p) == 0:
        raise ValueError(f"reference split '{reference_split}' is empty")
    tgt = cache[cache["split"] == target_split].sort_values("ts").reset_index(drop=True)

    rows = []
    for win_start, win_end, win in _iter_time_windows(
        tgt, "ts", pd.Timedelta(window), pd.Timedelta(step)
    ):
        n = len(win)
        if n < min_n:
            continue
        cur_p = win["p"].to_numpy()
        psi_val = psi(ref_p, cur_p, n_bins=n_bins)
        ks_stat, ks_pval = ks_distance(ref_p, cur_p)
        rows.append(
            {
                "window_start": win_start,
                "window_end": win_end,
                "n_samples": n,
                "psi": psi_val,
                "ks": ks_stat,
                "ks_pvalue": ks_pval,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Conditional precision (heatmap-ready)
# ---------------------------------------------------------------------------


def conditional_precision(
    cache: pd.DataFrame,
    *,
    threshold: float,
    split: str = "test",
    by: Sequence[str] = ("regime_bucket", "hour"),
    n_regime_buckets: int = 3,
    regime_labels: Sequence[str] = ("low", "med", "high"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Long-form per-cell precision at a fixed threshold, with Wilson 95% CIs.

    Cells are formed by the combination of ``by`` columns. Special pseudo-columns
    derived from the cache:

    - ``regime_bucket`` : tercile (or n_regime_buckets quantiles) of ``regime``
    - ``hour`` : hour-of-day from ``ts`` (0..23)
    - ``weekday`` : 0=Monday..6=Sunday from ``ts``

    Returns a DataFrame with one row per cell:
    ``[*by, n_predictions, n_hits, precision, ci_low, ci_high]``.
    Cells with ``n_predictions == 0`` are excluded.
    """
    df = cache[cache["split"] == split].copy()
    if len(df) == 0:
        return pd.DataFrame(
            columns=list(by) + ["n_predictions", "n_hits", "precision", "ci_low", "ci_high"]
        )
    derived = set(by) - set(df.columns)
    if "regime_bucket" in derived:
        df["regime_bucket"] = pd.qcut(
            df["regime"], n_regime_buckets, labels=list(regime_labels)
        )
    if "hour" in derived:
        df["hour"] = pd.to_datetime(df["ts"]).dt.hour
    if "weekday" in derived:
        df["weekday"] = pd.to_datetime(df["ts"]).dt.weekday

    df["pred"] = df["p"] >= float(threshold)
    if not df["pred"].any():
        return pd.DataFrame(
            columns=list(by) + ["n_predictions", "n_hits", "precision", "ci_low", "ci_high"]
        )

    df["hit"] = (df["pred"]) & (df["y"] == 1)

    grouped = (
        df[df["pred"]]
        .groupby(list(by), observed=False)
        .agg(n_predictions=("pred", "sum"), n_hits=("hit", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["n_predictions"] > 0].copy()
    grouped["precision"] = grouped["n_hits"] / grouped["n_predictions"]
    cis = grouped.apply(
        lambda r: pd.Series(
            wilson_interval(int(r["n_hits"]), int(r["n_predictions"]), alpha=alpha),
            index=["ci_low", "ci_high"],
        ),
        axis=1,
    )
    grouped[["ci_low", "ci_high"]] = cis
    return grouped


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_rolling_metric_with_band(
    rolling_df: pd.DataFrame,
    metric: str,
    *,
    ax=None,
    color: str = "C0",
    label: Optional[str] = None,
    alpha_band: float = 0.25,
):
    """Line + shaded CI band over time. ``rolling_df`` from rolling_metrics_with_ci."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    if rolling_df.empty:
        ax.text(0.5, 0.5, f"no rolling windows met min_n / min_pos for {metric}",
                ha="center", va="center", transform=ax.transAxes)
        return ax
    x = rolling_df["window_end"]
    ax.fill_between(
        x,
        rolling_df[f"{metric}_ci_low"],
        rolling_df[f"{metric}_ci_high"],
        color=color,
        alpha=alpha_band,
    )
    ax.plot(x, rolling_df[f"{metric}_point"], color=color, label=label or metric)
    ax.set(xlabel="window end", ylabel=metric)
    ax.grid(alpha=0.3)
    if label is not None:
        ax.legend()
    return ax


def plot_brier_decomposition_over_time(
    rolling_df: pd.DataFrame,
    *,
    ax=None,
):
    """Stacked-line view of REL, RES, UNC, WBV with the total Brier overlaid.

    With the identity ``BS = REL - RES + UNC + WBV``, plotting all components
    side by side reveals which one moves when calibration drifts.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    if rolling_df.empty:
        ax.text(0.5, 0.5, "no rolling windows met min_n / min_pos",
                ha="center", va="center", transform=ax.transAxes)
        return ax
    x = rolling_df["window_end"]
    components = [
        ("brier", "Brier (raw)", "k", "-"),
        ("brier_binned", "Brier (binned, = REL-RES+UNC)", "k", ":"),
        ("reliability", "Reliability (lower=better)", "C3", "--"),
        ("resolution", "Resolution (higher=better)", "C2", "--"),
        ("uncertainty", "Uncertainty", "C7", ":"),
        ("within_bin_variance", "Within-bin var", "C9", ":"),
    ]
    for k, label, color, ls in components:
        if f"{k}_point" not in rolling_df.columns:
            continue
        ax.plot(x, rolling_df[f"{k}_point"], color=color, linestyle=ls, label=label)
        if f"{k}_ci_low" in rolling_df.columns:
            ax.fill_between(
                x,
                rolling_df[f"{k}_ci_low"],
                rolling_df[f"{k}_ci_high"],
                color=color,
                alpha=0.15,
            )
    ax.set(
        xlabel="window end",
        ylabel="component value",
        title="Brier decomposition over time (95% bootstrap bands)",
    )
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_psi_ks_over_time(
    drift_df: pd.DataFrame,
    *,
    ax=None,
):
    """PSI and KS time series with conventional reference levels.

    PSI threshold lines at 0.1 and 0.2 (industry conventions for moderate /
    significant drift). KS plotted on a twin axis since its scale differs.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    if drift_df.empty:
        ax.text(0.5, 0.5, "no rolling windows met min_n",
                ha="center", va="center", transform=ax.transAxes)
        return ax
    x = drift_df["window_end"]
    ax.plot(x, drift_df["psi"], color="C0", label="PSI")
    ax.axhline(0.1, color="C1", linestyle="--", alpha=0.7, label="moderate (0.1)")
    ax.axhline(0.2, color="C3", linestyle="--", alpha=0.7, label="significant (0.2)")
    ax.set(xlabel="window end", ylabel="PSI")
    ax.grid(alpha=0.3)

    ax_r = ax.twinx()
    ax_r.plot(x, drift_df["ks"], color="C4", linestyle=":", label="KS distance")
    ax_r.set_ylabel("KS distance", color="C4")
    ax_r.tick_params(axis="y", labelcolor="C4")
    ax.legend(loc="upper left")
    return ax


def plot_conditional_precision_heatmap(
    cond_df: pd.DataFrame,
    *,
    index_col: str,
    column_col: str,
    value_col: str = "precision",
    annotate: str = "value",  # "value" or "value_n" or "none"
    ax=None,
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """Heatmap of per-cell precision (or other value) over (index_col, column_col).

    Empty cells render as NaN (greyed out). Annotation can show the value
    or "value (n=..)" for the prediction count.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))
    if cond_df.empty:
        ax.text(0.5, 0.5, "no cells (no predictions above threshold)",
                ha="center", va="center", transform=ax.transAxes)
        return ax
    pivot = cond_df.pivot(index=index_col, columns=column_col, values=value_col)
    n_pivot = cond_df.pivot(index=index_col, columns=column_col, values="n_predictions")

    if annotate == "none":
        annot = False
    elif annotate == "value_n":
        annot = pivot.applymap(lambda v: "" if np.isnan(v) else f"{v:.2f}").combine(
            n_pivot.applymap(lambda v: "" if pd.isna(v) else f"\\n(n={int(v)})"),
            lambda a, b: a + b,
        )
        annot = annot.where(~pivot.isna(), "")
    else:  # "value"
        annot = pivot.applymap(lambda v: "" if np.isnan(v) else f"{v:.2f}")

    sns.heatmap(
        pivot,
        annot=annot,
        fmt="" if isinstance(annot, pd.DataFrame) else ".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": value_col},
    )
    ax.set(title=f"{value_col} by ({index_col} x {column_col})")
    return ax


def plot_page_hinkley(
    values: np.ndarray,
    *,
    delta: float = 0.005,
    threshold: float = 50.0,
    ax=None,
    label: str = "Page-Hinkley",
):
    """Run page_hinkley and overlay the alarm on the value series."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    alarm, ph, mn = page_hinkley(values, delta=delta, threshold=threshold)
    ax.plot(ph - mn, label="PH(t) - min(PH)", color="C0")
    ax.axhline(threshold, linestyle="--", color="C3", label=f"threshold ({threshold:.0f})")
    if alarm >= 0:
        ax.axvline(alarm, linestyle=":", color="k", label=f"alarm @ idx={alarm}")
    ax.set(xlabel="sample index", ylabel="cumulative deviation")
    ax.set_title(label)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
