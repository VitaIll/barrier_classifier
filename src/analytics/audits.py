"""Production-trading audits — sanity checks the model has to pass before it is
allowed near a live execution path.

Public API:

- ``causal_feature_audit(feature_list)`` -> ``CausalAuditResult`` flagging any
  feature whose name does not match the project's causal convention
  (``__f__`` frozen-window-up-to-now or ``__h__`` instantaneous current-bar).
  Names containing ``__b__``, ``fwd``, ``future``, ``ahead``, ``lead`` or a
  negative-window suffix are treated as suspect. CI-runnable.
- ``label_shuffle_baseline(y_true, p)`` -> metrics + 95% CI under permutation
  of ``y_true``. With a finite shuffle PR-AUC is ``base_rate + O(1/sqrt(n))``
  and ROC-AUC is ``0.5 + O(1/sqrt(n))``; the function reports the empirical CI
  so a downstream test can refuse to ship a model whose real metrics are
  inside the shuffle CI (i.e. the model is not significantly better than
  random at this sample size).
- ``time_block_permutation_importance(model, df, feature_list, n_blocks)`` ->
  per-feature drop in PR-AUC under (a) within-block permutation and
  (b) across-block permutation. A feature whose importance only shows up
  under across-block permutation is leaking through time — its ranking
  comes from where in time the row sits, not from local feature value.
- ``decision_turnover(p, threshold)`` -> per-bar binary decision rate, lag-1
  autocorrelation, and run-length statistics. Frequent flips destroy
  net-of-cost performance; the audit motivates a hysteresis band.
- ``deflated_sharpe(per_trade_returns, n_trials)`` -> Bailey & Lopez de Prado
  deflated Sharpe ratio: probability that the observed Sharpe is significantly
  greater than the expected maximum across ``n_trials`` HPO attempts.
- ``half_vs_half_drift_audit(train_df, val_df, test_df, feature_list)`` ->
  fits two research models on the first and second halves of train_df,
  predicts on test_df, returns rank correlation and EV-disagreement.
  Same predictions on test ⇒ covariate-shift only; disagreement ⇒ concept
  drift in the gap between halves.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from .bootstrap import DEFAULT_B, DEFAULT_CI, BootstrapResult, iid_indices


# ---------------------------------------------------------------------------
# 1. Causal feature audit
# ---------------------------------------------------------------------------


CAUSAL_SUFFIX_PATTERN = re.compile(r"__(f|h)__")  # frozen-up-to-now or instantaneous

# Word-boundary regex avoids substring false positives like "ledger" / "bleeding"
# (which contain "lead") or "head_count" (which contains "ahead"). Tokens are
# matched only when not surrounded by alphanumerics, so they sit on their own
# underscore/dash/start/end-delimited segment.
SUSPECT_TOKENS = (
    "fwd",
    "future",
    "ahead",
    "lead",
    "lookahead",
    "look_ahead",
    "peek",
    "oracle",
    "tplus",
    "t+",
    "next",
    "_t1",
    "_t2",
)
# The word-boundary regex catches every word-like token (fwd, future, ahead,
# lead, lookahead, peek, oracle, tplus, next) — non-word lookarounds keep
# "ledger" / "bleeding" / "head_count" from matching.
SUSPECT_TOKEN_PATTERN = re.compile(
    r"(?<![a-z0-9])(fwd|future|ahead|lookahead|look_ahead|lead|peek|oracle|tplus|next)(?![a-z0-9])",
    re.IGNORECASE,
)
# ``t+`` (and ``T+``) is the t+N notation for future-offset features
# (e.g. ``feat_t+1``). The trailing digit IS part of the leak signature, so we
# only need a left-boundary check (preceded by a non-alphanumeric).
SUSPECT_TPLUS_PATTERN = re.compile(r"(?<![a-z0-9])t\+", re.IGNORECASE)
# Note: we intentionally do NOT flag the ``target_`` prefix as a leak
# signal in this codebase. Past-target features (``target__autocorr_lagN``,
# ``target__lagN``, etc.) are causal by construction: they're emitted by
# ``compute_past_target_features_pl`` after a ``_label_maturity_shift``
# that only exposes ``y[<k]``. The ``__h__`` causal suffix on every such
# column already proves this. Genuine target-leak names like
# ``target_fwd_3`` or ``target_next_4`` are still caught by the word-
# boundary regex above (``fwd``/``next`` tokens are preceded by ``_``,
# which is not alphanumeric, so the lookarounds match).
# The "__b__" window-suffix convention is a clean delimited segment, so
# plain substring is sufficient.
SUSPECT_SUFFIX_LITERAL = "__b__"


@dataclass
class CausalAuditResult:
    n_features: int
    n_causal: int
    n_suspect: int
    suspect: List[str]
    unmatched: List[str]  # neither __f__ nor __h__ but no obvious leak token

    @property
    def passed(self) -> bool:
        return len(self.suspect) == 0 and len(self.unmatched) == 0


def causal_feature_audit(feature_list: Sequence[str]) -> CausalAuditResult:
    """Static naming-convention check on every feature.

    Pass condition: every feature contains ``__f__`` (frozen window ending at
    bar ``t``) or ``__h__`` (instantaneous, computed from bar ``t`` only). No
    suspect tokens (``__b__``, ``fwd``, ``future``, ``ahead``, ``lead``,
    ``lookahead``, ``look_ahead``, ``peek``, ``oracle``, ``tplus``, ``t+``,
    ``next``, ``_t1``, ``_t2``).

    Tokens are matched on word boundaries (regex with ``(?<![a-z0-9])`` /
    ``(?![a-z0-9])`` lookarounds), so ``ledger``, ``bleeding``, and
    ``head_count`` do NOT trigger false positives on ``lead`` / ``ahead``.
    The ``target__`` prefix used by past-target features is causal by
    construction (`compute_past_target_features_pl` enforces maturity
    shift) — not flagged here; genuine ``target_fwd_*`` / ``target_next_*``
    variants are still caught by the word-boundary regex.
    """
    feats = list(feature_list)
    suspect: List[str] = []
    causal: List[str] = []
    unmatched: List[str] = []
    for f in feats:
        lower = f.lower()
        is_suspect = (
            SUSPECT_SUFFIX_LITERAL in f
            or SUSPECT_TOKEN_PATTERN.search(f) is not None
            or SUSPECT_TPLUS_PATTERN.search(f) is not None
            or "_t1" in f
            or "_t2" in f
        )
        if is_suspect:
            suspect.append(f)
            continue
        if CAUSAL_SUFFIX_PATTERN.search(f) is not None:
            causal.append(f)
        else:
            unmatched.append(f)
    return CausalAuditResult(
        n_features=len(feats),
        n_causal=len(causal),
        n_suspect=len(suspect),
        suspect=suspect,
        unmatched=unmatched,
    )


# ---------------------------------------------------------------------------
# 2. Label-shuffle baseline
# ---------------------------------------------------------------------------


def label_shuffle_baseline(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    n_shuffles: int = 1000,
    random_seed: int = 42,
    ci: float = DEFAULT_CI,
) -> Dict[str, BootstrapResult]:
    """Permute ``y_true`` ``n_shuffles`` times, recompute ROC-AUC / PR-AUC.

    Both metrics should sit at chance under permutation:
    - ROC-AUC concentrates on 0.5
    - PR-AUC concentrates on the base rate ``y_true.mean()``

    Returns a ``BootstrapResult`` per metric. A real-model metric whose point
    estimate falls inside the shuffle CI is statistically indistinguishable
    from random at this ``n``.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(random_seed)
    base_rate = float(y.mean())
    rocs = np.empty(n_shuffles, dtype=float)
    prs = np.empty(n_shuffles, dtype=float)
    for i in range(n_shuffles):
        y_perm = rng.permutation(y)
        # Single-class permutation guard (extremely rare for non-degenerate y)
        if y_perm.sum() == 0 or y_perm.sum() == len(y_perm):
            rocs[i] = 0.5
            prs[i] = base_rate
            continue
        rocs[i] = roc_auc_score(y_perm, p)
        prs[i] = average_precision_score(y_perm, p)
    alpha = 1.0 - ci
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "roc_auc": BootstrapResult(
            point=float(np.median(rocs)),
            median=float(np.median(rocs)),
            ci_low=float(np.percentile(rocs, lo)),
            ci_high=float(np.percentile(rocs, hi)),
            ci=ci,
            B=int(n_shuffles),
            samples=rocs,
        ),
        "pr_auc": BootstrapResult(
            point=float(np.median(prs)),
            median=float(np.median(prs)),
            ci_low=float(np.percentile(prs, lo)),
            ci_high=float(np.percentile(prs, hi)),
            ci=ci,
            B=int(n_shuffles),
            samples=prs,
        ),
        "base_rate": BootstrapResult(
            point=base_rate,
            median=base_rate,
            ci_low=base_rate,
            ci_high=base_rate,
            ci=ci,
            B=1,
            samples=np.full(1, base_rate),
        ),
    }


# ---------------------------------------------------------------------------
# 3. Time-block permutation importance
# ---------------------------------------------------------------------------


def time_block_permutation_importance(
    model,
    df: pd.DataFrame,
    feature_list: Sequence[str],
    *,
    label_col: str = "y",
    metric: str = "pr_auc",
    n_blocks: int = 8,
    features_to_test: Optional[Sequence[str]] = None,
    n_repeats: int = 3,
    random_seed: int = 42,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    """For each feature, compare metric drop under within-block vs across-block
    permutation.

    Within-block permutation preserves temporal context (the feature stays in
    its time neighborhood); across-block permutation destroys it. A feature
    whose drop is concentrated in across-block (drop_across >> drop_within)
    derives most of its value from time-of-row rather than local value — a
    leakage smell.

    ``metric`` ∈ {pr_auc, roc_auc}.

    ``timestamp_col`` (optional): if given, ``df`` is sorted by this column
    before block IDs are computed. If ``None``, the caller is responsible
    for ensuring ``df`` is already chronologically sorted — block IDs are
    otherwise meaningless.
    """
    if metric not in ("pr_auc", "roc_auc"):
        raise ValueError(f"metric must be 'pr_auc' or 'roc_auc', got '{metric}'")
    score_fn = average_precision_score if metric == "pr_auc" else roc_auc_score

    feature_list = list(feature_list)
    if features_to_test is None:
        features_to_test = feature_list
    features_to_test = [f for f in features_to_test if f in feature_list]

    if timestamp_col is not None:
        if timestamp_col not in df.columns:
            raise ValueError(
                f"timestamp_col {timestamp_col!r} not in df.columns"
            )
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    rng = np.random.default_rng(random_seed)
    n = len(df)
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
    if n < n_blocks:
        raise ValueError(
            f"n={n} rows < n_blocks={n_blocks}; need at least one row per block"
        )
    block_id = np.minimum(np.arange(n) * n_blocks // n, n_blocks - 1)
    y = df[label_col].astype(int).to_numpy()
    X = df[feature_list].to_numpy(dtype=float).copy()

    base_p = model.predict_proba(X)[:, 1]
    base_score = float(score_fn(y, base_p))

    rows = []
    for feat in features_to_test:
        col_idx = feature_list.index(feat)
        col_orig = X[:, col_idx].copy()

        # within-block
        within_drops = np.empty(n_repeats)
        for r in range(n_repeats):
            permuted = col_orig.copy()
            for b in range(n_blocks):
                m = block_id == b
                idx = np.where(m)[0]
                permuted[idx] = rng.permutation(col_orig[idx])
            X[:, col_idx] = permuted
            p = model.predict_proba(X)[:, 1]
            within_drops[r] = base_score - float(score_fn(y, p))

        # across-block
        across_drops = np.empty(n_repeats)
        for r in range(n_repeats):
            X[:, col_idx] = rng.permutation(col_orig)
            p = model.predict_proba(X)[:, 1]
            across_drops[r] = base_score - float(score_fn(y, p))

        # restore
        X[:, col_idx] = col_orig

        rows.append(
            {
                "feature": feat,
                "drop_within_mean": float(within_drops.mean()),
                "drop_within_std": float(within_drops.std(ddof=0)),
                "drop_across_mean": float(across_drops.mean()),
                "drop_across_std": float(across_drops.std(ddof=0)),
                "ratio_across_to_within": float(across_drops.mean() / within_drops.mean())
                if abs(within_drops.mean()) > 1e-12 else float("inf"),
            }
        )
    out = pd.DataFrame(rows).sort_values("drop_across_mean", ascending=False).reset_index(drop=True)
    out.attrs["base_score"] = base_score
    out.attrs["metric"] = metric
    out.attrs["n_blocks"] = n_blocks
    return out


# ---------------------------------------------------------------------------
# 4. Decision turnover
# ---------------------------------------------------------------------------


def decision_turnover(
    p: np.ndarray,
    threshold: float,
    *,
    ts: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Stats on the binary entry decision ``act_t = (p_t >= threshold)``.

    - ``trade_rate``: fraction of bars where act = 1
    - ``flip_rate``: fraction of consecutive pairs (act_{t-1}, act_t) where act flips
    - ``lag1_autocorr``: Pearson autocorr of the act sequence at lag 1
    - ``mean_run_length_active``: mean consecutive run of act = 1
    - ``mean_run_length_idle``:   mean consecutive run of act = 0

    High flip_rate with low autocorr ⇒ trade churn; motivates a hysteresis
    band (enter at p > τ_in, exit at p < τ_out < τ_in).
    """
    p = np.asarray(p, dtype=float)
    act = (p >= float(threshold)).astype(int)
    n = len(act)
    if n < 2:
        raise ValueError("decision_turnover requires at least 2 observations")

    if ts is not None:
        order = np.argsort(np.asarray(ts))
        act = act[order]

    flips = int((act[1:] != act[:-1]).sum())
    flip_rate = flips / (n - 1)
    if act.std(ddof=0) > 0:
        lag1 = float(np.corrcoef(act[:-1], act[1:])[0, 1])
    else:
        lag1 = float("nan")

    # run lengths
    boundaries = np.where(np.diff(act) != 0)[0]
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries, [n - 1]))
    lengths = ends - starts + 1
    states = act[starts]
    active_lengths = lengths[states == 1]
    idle_lengths = lengths[states == 0]

    return {
        "trade_rate": float(act.mean()),
        "flip_rate": float(flip_rate),
        "lag1_autocorr": lag1,
        "mean_run_length_active": float(active_lengths.mean()) if len(active_lengths) else 0.0,
        "mean_run_length_idle": float(idle_lengths.mean()) if len(idle_lengths) else 0.0,
        "n_active_runs": int(len(active_lengths)),
        "n_idle_runs": int(len(idle_lengths)),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# 5. Deflated Sharpe (Bailey & Lopez de Prado)
# ---------------------------------------------------------------------------


_EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int) -> float:
    """E[max Sharpe across N iid N(0,1) trials], from extreme value theory.

    Bailey & Lopez de Prado approximation (their eq. 7):
        E[max] ≈ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N e))
    where γ is Euler-Mascheroni and Φ⁻¹ is the inverse standard normal CDF.
    """
    if not isinstance(n_trials, (int, np.integer)) or isinstance(n_trials, bool):
        raise ValueError(f"n_trials must be an int, got {type(n_trials).__name__}")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    gamma = _EULER_MASCHERONI
    e = math.e
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float((1 - gamma) * z1 + gamma * z2)


def deflated_sharpe(
    per_trade_returns: np.ndarray,
    *,
    n_trials: int = 1,
) -> Dict[str, float]:
    """Deflated Sharpe ratio (probability that observed Sharpe > expected max).

    Returns observed Sharpe, expected-max Sharpe under ``n_trials`` HPO attempts,
    and the deflated Sharpe probability ``DSR``. ``DSR > 0.95`` is the usual
    threshold for "significantly better than the best of n_trials random
    backtests after correcting for non-normality" (Bailey & Lopez de Prado).

    Reference: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    """
    r = np.asarray(per_trade_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4:
        raise ValueError("deflated_sharpe requires at least 4 trades")

    mean = float(r.mean())
    std = float(r.std(ddof=1))
    if std <= 1e-18:
        raise ValueError("zero variance — Sharpe undefined")
    sr = mean / std
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # standard, kurt of N(0,1) = 3

    e_max = expected_max_sharpe(int(n_trials))

    # Test statistic (Bailey & Lopez de Prado eq. 8)
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0:
        return {
            "sharpe": sr,
            "expected_max_sharpe": e_max,
            "deflated_sharpe_z": float("nan"),
            "deflated_sharpe_prob": float("nan"),
            "n": n,
            "skew": skew,
            "kurtosis": kurt,
            "n_trials": int(n_trials),
            "warning": "non-positive denominator (skew/kurt extreme)",
        }
    z = (sr - e_max) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    dsr = float(stats.norm.cdf(z))

    return {
        "sharpe": sr,
        "expected_max_sharpe": e_max,
        "deflated_sharpe_z": z,
        "deflated_sharpe_prob": dsr,
        "n": n,
        "skew": skew,
        "kurtosis": kurt,
        "n_trials": int(n_trials),
    }


# ---------------------------------------------------------------------------
# 6. Half-vs-half drift audit
# ---------------------------------------------------------------------------


@dataclass
class HalfVsHalfResult:
    n_first: int
    n_second: int
    spearman_corr: float
    pearson_corr: float
    mean_abs_diff: float
    median_abs_diff: float
    n_disagree_at_threshold: int
    threshold: float
    metrics_first: Dict[str, float]
    metrics_second: Dict[str, float]
    p_first: np.ndarray
    p_second: np.ndarray


def half_vs_half_drift_audit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_list: Sequence[str],
    *,
    label_col: str = "y",
    timestamp_col: str = "k",
    weight_col: Optional[str] = "weight",
    threshold: float = 0.30,
    train_params_first: Optional[dict] = None,
    train_params_second: Optional[dict] = None,
) -> HalfVsHalfResult:
    """Train two models on the first/second chronological half of ``train_df``;
    compare their predictions on ``test_df``.

    - High rank-correlation, low disagreement at op threshold ⇒ covariate
      shift only (model is stationary; test data lives in train's covariate
      space mixture).
    - Low correlation or high disagreement ⇒ concept drift in the gap
      between halves: ``P(y | x)`` itself moved.
    """
    from .fast_train import fit_research_model, research_train_params

    df_sorted = train_df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df_sorted)
    cut = n // 2
    first = df_sorted.iloc[:cut].copy()
    second = df_sorted.iloc[cut:].copy()
    if len(first) < 100 or len(second) < 100:
        raise ValueError(f"half too small (first={len(first)}, second={len(second)})")

    p1 = train_params_first or research_train_params(verbose=0)
    p2 = train_params_second or research_train_params(verbose=0, random_seed=43)

    m1 = fit_research_model(
        first, val_df, list(feature_list),
        label_col=label_col, timestamp_col=timestamp_col, weight_col=weight_col, params=p1,
    )
    m2 = fit_research_model(
        second, val_df, list(feature_list),
        label_col=label_col, timestamp_col=timestamp_col, weight_col=weight_col, params=p2,
    )

    X_test = test_df[list(feature_list)].to_numpy(dtype=float)
    y_test = test_df[label_col].astype(int).to_numpy()
    p_first = m1.predict_proba(X_test)[:, 1]
    p_second = m2.predict_proba(X_test)[:, 1]

    spear = float(stats.spearmanr(p_first, p_second).statistic)
    pear = float(stats.pearsonr(p_first, p_second).statistic)
    abs_diff = np.abs(p_first - p_second)
    a1 = (p_first >= threshold).astype(int)
    a2 = (p_second >= threshold).astype(int)

    metrics_first = {
        "roc_auc": float(roc_auc_score(y_test, p_first)),
        "pr_auc": float(average_precision_score(y_test, p_first)),
    }
    metrics_second = {
        "roc_auc": float(roc_auc_score(y_test, p_second)),
        "pr_auc": float(average_precision_score(y_test, p_second)),
    }

    return HalfVsHalfResult(
        n_first=len(first),
        n_second=len(second),
        spearman_corr=spear,
        pearson_corr=pear,
        mean_abs_diff=float(abs_diff.mean()),
        median_abs_diff=float(np.median(abs_diff)),
        n_disagree_at_threshold=int((a1 != a2).sum()),
        threshold=float(threshold),
        metrics_first=metrics_first,
        metrics_second=metrics_second,
        p_first=p_first,
        p_second=p_second,
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_label_shuffle_baseline(
    real_metric_value: float,
    shuffle_result: BootstrapResult,
    *,
    metric_name: str = "PR-AUC",
    base_rate: Optional[float] = None,
    ax=None,
):
    """Histogram of shuffle distribution with the real metric overlaid."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(shuffle_result.samples, bins=40, color="#cccccc", alpha=0.85,
            label=f"shuffled {metric_name} (n={len(shuffle_result.samples)})")
    ax.axvline(real_metric_value, color="C3", linewidth=2, label=f"real model: {real_metric_value:.3f}")
    ax.axvline(shuffle_result.point, color="black", linestyle="--",
               label=f"shuffle median: {shuffle_result.point:.3f}")
    ax.axvline(shuffle_result.ci_low, color="black", linestyle=":", alpha=0.5)
    ax.axvline(shuffle_result.ci_high, color="black", linestyle=":", alpha=0.5)
    if base_rate is not None and metric_name.lower().startswith("pr"):
        ax.axvline(base_rate, color="C1", linestyle="-.",
                   label=f"base rate: {base_rate:.3f}")
    ax.set(
        xlabel=metric_name,
        ylabel="count of shuffles",
        title=f"Label-shuffle baseline ({metric_name})",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_time_block_permutation(
    df: pd.DataFrame,
    *,
    top_n: int = 25,
    ax=None,
):
    """Bar plot: top-N features by drop_across, with within-block bars overlaid."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7))
    if df.empty:
        ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes)
        return ax
    sub = df.head(top_n).iloc[::-1]
    y = np.arange(len(sub))
    ax.barh(y - 0.2, sub["drop_across_mean"], height=0.4, color="C3", alpha=0.85,
            label="across-block (random shuffle)", xerr=sub["drop_across_std"])
    ax.barh(y + 0.2, sub["drop_within_mean"], height=0.4, color="C0", alpha=0.85,
            label="within-block (preserves time)", xerr=sub["drop_within_std"])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"], fontsize=8)
    metric = df.attrs.get("metric", "metric")
    ax.set(
        xlabel=f"{metric} drop (positive = important)",
        title=f"Time-block permutation importance (top {top_n})\n"
        "Drop only on across-block ⇒ feature works through time, not through local value",
    )
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_half_vs_half_scatter(result: HalfVsHalfResult, *, ax=None, sample_n: int = 5000):
    """Scatter of p_first vs p_second on test, with y=x reference."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6.5))
    n = len(result.p_first)
    idx = np.arange(n) if n <= sample_n else np.random.default_rng(0).choice(n, sample_n, replace=False)
    ax.scatter(result.p_first[idx], result.p_second[idx], s=4, alpha=0.4)
    lim = max(result.p_first.max(), result.p_second.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", alpha=0.6, label="y = x")
    ax.set(
        xlabel="p (model trained on FIRST half of train)",
        ylabel="p (model trained on SECOND half of train)",
        title=f"Half-vs-half drift on test  Spearman={result.spearman_corr:.3f}  "
        f"mean|Δp|={result.mean_abs_diff:.3f}  "
        f"disagree@τ={result.threshold}: {result.n_disagree_at_threshold}",
        xlim=(0, lim), ylim=(0, lim),
    )
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_decision_turnover_runs(
    p: np.ndarray,
    threshold: float,
    *,
    ax=None,
):
    """Strip-plot of run lengths (active vs idle) at a given threshold."""
    import matplotlib.pyplot as plt

    p = np.asarray(p, dtype=float)
    act = (p >= float(threshold)).astype(int)
    n = len(act)
    if n < 2:
        raise ValueError("plot_decision_turnover_runs requires >= 2 obs")
    boundaries = np.where(np.diff(act) != 0)[0]
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries, [n - 1]))
    lengths = ends - starts + 1
    states = act[starts]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    if (states == 1).any():
        ax.hist(lengths[states == 1], bins=30, alpha=0.7, color="C2", label="active runs")
    if (states == 0).any():
        ax.hist(lengths[states == 0], bins=30, alpha=0.45, color="C7", label="idle runs")
    ax.set(
        xlabel="run length (bars)",
        ylabel="count",
        title=f"Decision-state runs at τ={threshold}  "
        f"flip rate={float((act[1:]!=act[:-1]).mean()):.3f}",
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
