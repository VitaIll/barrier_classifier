"""Tail / edge analysis: operating-point analytics for the trading entry decision.

Operates on the prediction cache (k, ts, y, m_k, tau_k, phi, regime, p, split).

Public API:

- ``bootstrap_threshold_sweep`` -> DataFrame with bootstrap CI bands on
  precision, recall, trade rate, EV per trade, and per-trade Sharpe at every
  threshold in the grid.
- ``bootstrap_partial_roc_auc`` / ``bootstrap_partial_pr_auc`` -> ``BootstrapResult``
  for the partial-AUC over an operating band (e.g. FPR <= 0.05 or recall <= 0.10).
- ``kelly_by_bin`` -> DataFrame with empirical hit rate per probability decile
  and the implied Kelly fraction under a configurable outcome model. Uses
  Wilson intervals on hit rate for small-bin robustness.
- ``lift_curve`` -> DataFrame with cumulative precision / lift over base rate
  as the population is taken in p-descending order.

Outcome model (parameterized — see ``OutcomeModel``):
- ``gain_per_hit``: realized log-return when y=1 (default = ``phi`` from the
  cache, i.e. exit at barrier)
- ``loss_per_miss``: realized log-return magnitude when y=0 (default = ``phi``,
  i.e. symmetric outcome). The TRUE realized end-of-horizon return is dataset-
  specific; ``augment_cache_with_r_realized`` (``src/strategy/cache.py``) adds
  the column from raw 1-min bars — pass ``cache_with_realized_return=True``
  to compute EV from it.
- ``cost_per_trade``: fees + half-spread + slippage (default 5 bps = 0.0005)

The default symmetric-outcome assumption is conservative for entry-only
strategies; override with the realized-return column for accurate EV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .bootstrap import DEFAULT_B, DEFAULT_CI, BootstrapResult, block_indices, iid_indices
from .curves import _interp_pr, _interp_roc
from .degradation import wilson_interval


def _choose_indices(
    n: int,
    B: int,
    rng: np.random.Generator,
    *,
    stratify_y: Optional[np.ndarray],
    stratify: bool,
    block_size: Optional[int],
) -> np.ndarray:
    """Block bootstrap if ``block_size > 1``, else stratified/iid."""
    if block_size is not None and int(block_size) > 1:
        return block_indices(n, B, rng, block_size=int(block_size))
    return iid_indices(
        n, B, rng, stratify=stratify_y if stratify and stratify_y is not None else None
    )


# ---------------------------------------------------------------------------
# Outcome model
# ---------------------------------------------------------------------------


@dataclass
class OutcomeModel:
    """Parameterized binary-outcome model for EV / Sharpe / Kelly.

    Defaults: TP earns ``+gain_per_hit`` (= phi at exit); FP loses
    ``loss_per_miss`` (default symmetric to gain — conservative); ``cost_per_trade``
    accounts for fees + half-spread + slippage. With ``use_realized_return=True``
    EV is computed from the cache's ``r_realized`` column instead.
    """

    gain_per_hit: float = 0.005       # default barrier (phi) magnitude
    loss_per_miss: float = 0.005      # symmetric default (override per strategy)
    cost_per_trade: float = 0.0005    # 5 bps fees + half-spread placeholder
    use_realized_return: bool = False # if True, requires r_realized in cache

    @property
    def b_ratio(self) -> float:
        """Gain/loss ratio used by Kelly."""
        return self.gain_per_hit / max(self.loss_per_miss, 1e-12)


def _trades_per_day(cache_split: pd.DataFrame, n_pred: np.ndarray) -> np.ndarray:
    """Convert per-threshold trade count to per-day rate based on ts span."""
    if len(cache_split) == 0:
        return np.zeros_like(n_pred, dtype=float)
    ts_span = (cache_split["ts"].max() - cache_split["ts"].min()).total_seconds()
    days = max(ts_span / 86400.0, 1e-9)
    return n_pred.astype(float) / days


# ---------------------------------------------------------------------------
# Threshold sweep with bootstrap bands
# ---------------------------------------------------------------------------


def _threshold_metrics(
    y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, om: OutcomeModel,
    r_realized: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-threshold metric matrix [precision, recall, trade_rate, ev, sharpe].

    Vectorized: uses cumulative TP / FP counts on the score-sorted array,
    then maps each threshold to the corresponding cumulative position. Avoids
    the O(T x N) inner loop.
    """
    n = len(y)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == n:
        return np.full((len(thresholds), 5), np.nan)

    order = np.argsort(p, kind="mergesort")[::-1]
    p_sorted = p[order]
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)
    cum_n = np.arange(1, n + 1)
    if r_realized is not None:
        r_sorted = r_realized[order]
        cum_realized_sum = np.cumsum(r_sorted)
        cum_realized_sq_sum = np.cumsum(r_sorted ** 2)

    out = np.full((len(thresholds), 5), np.nan)
    for ti, thr in enumerate(thresholds):
        # n_pred = count of p >= thr
        n_pred = int(np.searchsorted(-p_sorted, -thr, side="right"))
        if n_pred == 0:
            out[ti, :] = [np.nan, 0.0, 0.0, np.nan, np.nan]
            continue
        tp = int(cum_tp[n_pred - 1])
        precision = tp / n_pred
        recall = tp / n_pos
        trade_rate = n_pred / n

        if r_realized is not None and om.use_realized_return:
            mean_r = cum_realized_sum[n_pred - 1] / n_pred - om.cost_per_trade
            var_r = (
                cum_realized_sq_sum[n_pred - 1] / n_pred - (cum_realized_sum[n_pred - 1] / n_pred) ** 2
            )
            sharpe = mean_r / np.sqrt(var_r) if var_r > 1e-18 else np.nan
            ev = mean_r
        else:
            # Binary outcome model
            ev = (
                precision * om.gain_per_hit
                - (1 - precision) * abs(om.loss_per_miss)
                - om.cost_per_trade
            )
            spread = om.gain_per_hit + abs(om.loss_per_miss)
            var_per_trade = precision * (1 - precision) * spread * spread
            sharpe = ev / np.sqrt(var_per_trade) if var_per_trade > 1e-18 else np.nan

        out[ti, :] = [precision, recall, trade_rate, ev, sharpe]
    return out


def bootstrap_threshold_sweep(
    cache: pd.DataFrame,
    *,
    split: str = "test",
    thresholds: Optional[np.ndarray] = None,
    outcome_model: Optional[OutcomeModel] = None,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
    block_size: Optional[int] = None,
) -> pd.DataFrame:
    """Bootstrap CI bands at every threshold in the grid.

    Returns a DataFrame with one row per threshold and columns:
    ``threshold, n_trades, trades_per_day, precision[/_ci_low/_ci_high],
    recall[/_ci_low/_ci_high], trade_rate[/_ci_low/_ci_high],
    ev_per_trade[/_ci_low/_ci_high], sharpe_per_trade[/_ci_low/_ci_high]``.

    By default the bootstrap uses class-stratified iid (same as
    ``bootstrap_metric``); pass ``block_size`` ≈ M to switch to a moving-
    block bootstrap for autocorrelated label streams (1-min cadence).
    Every replicate evaluates ALL thresholds in one vectorized pass, so
    cost scales as O(B * (N log N + T)) rather than O(B * T * N).
    """
    if outcome_model is None:
        outcome_model = OutcomeModel()
    cs = cache[cache["split"] == split].reset_index(drop=True)
    if len(cs) == 0:
        return pd.DataFrame()
    y = cs["y"].astype(int).to_numpy()
    p = cs["p"].astype(float).to_numpy()
    if thresholds is None:
        thresholds = np.linspace(0.01, float(p.max()), 100)
    thresholds = np.asarray(thresholds, dtype=float)

    r_realized = None
    if outcome_model.use_realized_return:
        if "r_realized" not in cs.columns:
            raise ValueError(
                "outcome_model.use_realized_return=True but 'r_realized' "
                "column not in cache. Run augment_cache_with_r_realized "
                "(src/strategy/cache.py) first, or set use_realized_return=False."
            )
        r_realized = cs["r_realized"].astype(float).to_numpy()

    point = _threshold_metrics(y, p, thresholds, outcome_model, r_realized)

    rng = np.random.default_rng(seed)
    idx = _choose_indices(
        len(y), B, rng, stratify_y=y, stratify=stratify, block_size=block_size
    )
    T = len(thresholds)
    samples = np.empty((B, T, 5), dtype=float)
    for b in range(B):
        i = idx[b]
        r_b = r_realized[i] if r_realized is not None else None
        samples[b] = _threshold_metrics(y[i], p[i], thresholds, outcome_model, r_b)

    alpha = (1.0 - ci) / 2.0
    metric_names = ["precision", "recall", "trade_rate", "ev_per_trade", "sharpe_per_trade"]
    df = pd.DataFrame({"threshold": thresholds})
    df["n_trades"] = (point[:, 2] * len(y)).round().astype(int)
    df["trades_per_day"] = _trades_per_day(cs, df["n_trades"].to_numpy())
    for mi, name in enumerate(metric_names):
        df[name] = point[:, mi]
        df[f"{name}_ci_low"] = np.nanquantile(samples[:, :, mi], alpha, axis=0)
        df[f"{name}_ci_high"] = np.nanquantile(samples[:, :, mi], 1.0 - alpha, axis=0)
    df["base_rate"] = float(y.mean())
    df["lift"] = df["precision"] / df["base_rate"]
    return df


# ---------------------------------------------------------------------------
# Partial AUC (operating-band)
# ---------------------------------------------------------------------------


def _trapezoid(values: np.ndarray, xs: np.ndarray) -> float:
    """Wrapper that prefers ``np.trapezoid`` (NumPy >= 2.0), falling back to
    the legacy ``np.trapz`` on older NumPy. Identical semantics; only the
    name changed in NumPy 2."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, xs))
    return float(np.trapz(values, xs))  # type: ignore[attr-defined]


def _partial_roc_auc_from(y: np.ndarray, p: np.ndarray, fpr_max: float) -> float:
    """Partial ROC-AUC normalized to [0, 1] over FPR in [0, fpr_max]."""
    if fpr_max <= 0:
        raise ValueError(f"fpr_max must be > 0, got {fpr_max}")
    fpr_grid = np.linspace(0.0, float(fpr_max), 201)
    tpr_grid = _interp_roc(y, p, fpr_grid)
    return _trapezoid(tpr_grid, fpr_grid) / float(fpr_max)


def _partial_pr_auc_from(y: np.ndarray, p: np.ndarray, recall_max: float) -> float:
    """Partial PR-AUC (max-envelope) normalized to [0, 1] over recall [0, recall_max]."""
    if recall_max <= 0:
        raise ValueError(f"recall_max must be > 0, got {recall_max}")
    recall_grid = np.linspace(0.0, float(recall_max), 201)
    prec_grid = _interp_pr(y, p, recall_grid)
    return _trapezoid(prec_grid, recall_grid) / float(recall_max)


def bootstrap_partial_roc_auc(
    y: np.ndarray,
    p: np.ndarray,
    *,
    fpr_max: float = 0.05,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
    block_size: Optional[int] = None,
) -> BootstrapResult:
    """Partial ROC-AUC over the low-FPR operating band, with bootstrap CI.

    For an entry-gate model that operates at low false-positive rate
    (sub-percent in trading), the full ROC-AUC averages over irrelevant
    high-FPR regions. Partial AUC isolates the operating band. Pass
    ``block_size`` ≈ M for autocorrelated label streams (1-min cadence).
    """
    y = np.asarray(y)
    p = np.asarray(p)
    point = _partial_roc_auc_from(y, p, fpr_max)
    rng = np.random.default_rng(seed)
    idx = _choose_indices(
        len(y), B, rng, stratify_y=y, stratify=stratify, block_size=block_size
    )
    samples = np.full(B, np.nan, dtype=float)
    for b in range(B):
        i = idx[b]
        try:
            samples[b] = _partial_roc_auc_from(y[i], p[i], fpr_max)
        except ValueError:
            pass
    alpha = (1.0 - ci) / 2.0
    b_effective = int(np.count_nonzero(~np.isnan(samples)))
    return BootstrapResult(
        point=point,
        median=float(np.nanmedian(samples)),
        ci_low=float(np.nanquantile(samples, alpha)),
        ci_high=float(np.nanquantile(samples, 1.0 - alpha)),
        ci=ci,
        B=B,
        samples=samples,
        B_effective=b_effective,
    )


def bootstrap_partial_pr_auc(
    y: np.ndarray,
    p: np.ndarray,
    *,
    recall_max: float = 0.10,
    B: int = DEFAULT_B,
    ci: float = DEFAULT_CI,
    stratify: bool = True,
    seed: int = 0,
    block_size: Optional[int] = None,
) -> BootstrapResult:
    """Partial PR-AUC (max-envelope) over the operating recall band.

    Pass ``block_size`` ≈ M for autocorrelated label streams (1-min cadence).
    """
    y = np.asarray(y)
    p = np.asarray(p)
    point = _partial_pr_auc_from(y, p, recall_max)
    rng = np.random.default_rng(seed)
    idx = _choose_indices(
        len(y), B, rng, stratify_y=y, stratify=stratify, block_size=block_size
    )
    samples = np.full(B, np.nan, dtype=float)
    for b in range(B):
        i = idx[b]
        try:
            samples[b] = _partial_pr_auc_from(y[i], p[i], recall_max)
        except ValueError:
            pass
    alpha = (1.0 - ci) / 2.0
    b_effective = int(np.count_nonzero(~np.isnan(samples)))
    return BootstrapResult(
        point=point,
        median=float(np.nanmedian(samples)),
        ci_low=float(np.nanquantile(samples, alpha)),
        ci_high=float(np.nanquantile(samples, 1.0 - alpha)),
        ci=ci,
        B=B,
        samples=samples,
        B_effective=b_effective,
    )


# ---------------------------------------------------------------------------
# Kelly fraction by probability bin
# ---------------------------------------------------------------------------


def kelly_by_bin(
    cache: pd.DataFrame,
    *,
    split: str = "test",
    n_bins: int = 10,
    outcome_model: Optional[OutcomeModel] = None,
    alpha_wilson: float = 0.05,
) -> pd.DataFrame:
    """Per-decile empirical hit rate + Kelly fraction under outcome model.

    Bins are equal-frequency (quantile-based) on the predicted probability.
    Kelly: ``f* = (b * hit_rate - (1 - hit_rate)) / b`` where ``b = gain/loss``.
    Negative or zero Kelly = don't take that bin's bets at the configured
    outcome model.

    Wilson interval on hit_rate is propagated to a Kelly CI by mapping the
    endpoints through the Kelly formula (monotone increasing in hit_rate).
    """
    om = outcome_model or OutcomeModel()
    cs = cache[cache["split"] == split].reset_index(drop=True)
    if len(cs) == 0:
        return pd.DataFrame()
    p = cs["p"].astype(float).to_numpy()
    y = cs["y"].astype(int).to_numpy()
    bin_edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    # Make edges strictly increasing (deduplicate degenerate ties)
    bin_edges = np.unique(bin_edges)
    bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, len(bin_edges) - 2)

    rows = []
    b_ratio = om.b_ratio
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        n_hit = int(y[mask].sum())
        hit_rate = n_hit / n_b
        ci_lo, ci_hi = wilson_interval(n_hit, n_b, alpha=alpha_wilson)
        kelly_pt = (b_ratio * hit_rate - (1.0 - hit_rate)) / b_ratio
        kelly_lo = (b_ratio * ci_lo - (1.0 - ci_lo)) / b_ratio
        kelly_hi = (b_ratio * ci_hi - (1.0 - ci_hi)) / b_ratio
        rows.append(
            {
                "bin": b,
                "p_lo": float(bin_edges[b]),
                "p_hi": float(bin_edges[b + 1]),
                "n": n_b,
                "n_hits": n_hit,
                "mean_p": float(p[mask].mean()),
                "hit_rate": hit_rate,
                "hit_rate_ci_low": ci_lo,
                "hit_rate_ci_high": ci_hi,
                "kelly": kelly_pt,
                "kelly_ci_low": kelly_lo,
                "kelly_ci_high": kelly_hi,
                "half_kelly": kelly_pt / 2.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lift / gain curve
# ---------------------------------------------------------------------------


def lift_curve(cache: pd.DataFrame, *, split: str = "test") -> pd.DataFrame:
    """Cumulative precision and lift as the population is taken p-descending.

    For each top-k (k=1..n) prediction, returns the cumulative TP count, the
    cumulative precision (TP / k), and the lift over base rate (precision /
    base_rate). The "gain curve" is precision_at_k vs k_frac.
    """
    cs = cache[cache["split"] == split].reset_index(drop=True)
    if len(cs) == 0:
        return pd.DataFrame()
    y = cs["y"].astype(int).to_numpy()
    p = cs["p"].astype(float).to_numpy()
    n = len(y)
    base_rate = float(y.mean())
    if base_rate == 0:
        return pd.DataFrame()
    order = np.argsort(p, kind="mergesort")[::-1]
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)
    cum_n = np.arange(1, n + 1)
    return pd.DataFrame(
        {
            "k": cum_n,
            "k_frac": cum_n / n,
            "cum_tp": cum_tp,
            "precision_at_k": cum_tp / cum_n,
            "lift_at_k": (cum_tp / cum_n) / base_rate,
            "recall_at_k": cum_tp / max(int(y.sum()), 1),
        }
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_net_ev_vs_trades_per_day(
    sweep: pd.DataFrame,
    *,
    ax=None,
    color: str = "C0",
    label: str = "Test",
    base_rate: Optional[float] = None,
    show_zero_line: bool = True,
):
    """The canonical chart for an entry-gate model.

    X-axis: trades-per-day on a log scale (so the high-precision low-frequency
    region is visible). Y-axis: net EV per trade in log-return units, with
    bootstrap band.

    Skips threshold rows where the trade count was zero or EV is undefined.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.2))
    df = sweep.dropna(subset=["ev_per_trade", "trades_per_day"]).copy()
    df = df[df["trades_per_day"] > 0]
    df = df.sort_values("trades_per_day")
    if len(df) == 0:
        ax.text(0.5, 0.5, "no trades above any threshold", ha="center", va="center", transform=ax.transAxes)
        return ax
    ax.fill_between(
        df["trades_per_day"], df["ev_per_trade_ci_low"], df["ev_per_trade_ci_high"],
        color=color, alpha=0.25,
    )
    ax.plot(df["trades_per_day"], df["ev_per_trade"], color=color, label=label, marker="o", markersize=3)
    if show_zero_line:
        ax.axhline(0.0, linestyle="--", color="gray", alpha=0.7, label="break-even")
    ax.set_xscale("log")
    ax.set_xlabel("Trades per day (log)")
    ax.set_ylabel("Net EV per trade (log-return units)")
    ax.set_title(
        "Net EV per trade vs trades-per-day  (95% bootstrap band)"
        + (f"  (base rate {base_rate:.3f})" if base_rate is not None else "")
    )
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_threshold_sweep_with_bands(
    sweep: pd.DataFrame,
    *,
    ax=None,
    metrics=("precision", "recall", "trade_rate", "lift"),
):
    """Multi-metric overlay of the threshold sweep with bootstrap CIs."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    if sweep.empty:
        ax.text(0.5, 0.5, "empty sweep", ha="center", va="center", transform=ax.transAxes)
        return ax
    palette = {"precision": "C0", "recall": "C1", "trade_rate": "C2", "lift": "C3"}
    for m in metrics:
        if m not in sweep.columns:
            continue
        ax.fill_between(
            sweep["threshold"],
            sweep[f"{m}_ci_low"] if f"{m}_ci_low" in sweep.columns else sweep[m],
            sweep[f"{m}_ci_high"] if f"{m}_ci_high" in sweep.columns else sweep[m],
            color=palette.get(m, "k"), alpha=0.18,
        )
        ax.plot(sweep["threshold"], sweep[m], color=palette.get(m, "k"), label=m)
    ax.set(xlabel="threshold", ylabel="metric", title="Threshold sweep with bootstrap bands")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return ax


def plot_kelly_by_bin(
    df: pd.DataFrame,
    *,
    base_rate: Optional[float] = None,
    ax=None,
):
    """Hit rate per probability decile + Kelly fraction overlay.

    Hit rate shown with Wilson 95% CI error bars; Kelly fraction on twin
    axis. Bin width on x-axis encodes p_lo..p_hi range.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    if df.empty:
        ax.text(0.5, 0.5, "no bins", ha="center", va="center", transform=ax.transAxes)
        return ax
    x = df["mean_p"]
    lo_err = (df["hit_rate"] - df["hit_rate_ci_low"]).clip(lower=0)
    hi_err = (df["hit_rate_ci_high"] - df["hit_rate"]).clip(lower=0)
    ax.errorbar(
        x, df["hit_rate"],
        yerr=[lo_err, hi_err],
        marker="o", color="C0", capsize=3, linestyle="-", label="Empirical hit rate",
    )
    if base_rate is not None:
        ax.axhline(base_rate, linestyle=":", color="gray", alpha=0.7, label=f"Base rate ({base_rate:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="y = mean(p)")
    ax.set(xlabel="mean predicted probability (per bin)", ylabel="empirical hit rate",
           xlim=(0, max(1.0, df["mean_p"].max() * 1.05)))

    ax_r = ax.twinx()
    ax_r.bar(x, df["kelly"], width=0.025, alpha=0.25, color="C2", label="Kelly fraction")
    ax_r.axhline(0.0, color="C3", linestyle="--", alpha=0.7)
    ax_r.set_ylabel("Kelly fraction (negative = abstain)", color="C2")
    ax_r.tick_params(axis="y", labelcolor="C2")

    ax.set_title("Hit rate + Kelly fraction by probability decile")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return ax


def plot_lift_curve(
    df: pd.DataFrame, *, ax=None, color: str = "C0", label: str = "Test"
):
    """Gain curve: precision_at_k as the population is taken p-descending."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if df.empty:
        ax.text(0.5, 0.5, "empty cache", ha="center", va="center", transform=ax.transAxes)
        return ax
    base_rate = (df["cum_tp"].iloc[-1]) / df["k"].iloc[-1] if len(df) else 0.0
    ax.plot(df["k_frac"], df["precision_at_k"], color=color, label=label)
    ax.axhline(base_rate, linestyle="--", color="gray", alpha=0.7,
               label=f"Random (base rate {base_rate:.3f})")
    ax.set(xlabel="Fraction of population taken (top-k by p)",
           ylabel="Cumulative precision",
           title="Gain curve: precision at top-k")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
