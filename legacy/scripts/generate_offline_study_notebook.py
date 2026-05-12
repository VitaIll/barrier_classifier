"""Generate notebooks/04_offline_study.ipynb.

Run this from the repo root: ``python scripts/generate_offline_study_notebook.py``.
The notebook is the analytical companion to ``03_model_training.ipynb`` and is
extended in place as later analytics phases land. Re-running this script
regenerates the Phase 0 + Phase 1 baseline.
"""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = []

CELLS.append(md("""# 04 - Offline Model Study (Bootstrap-Certified Analytics)

Research / analytical companion to `03_model_training.ipynb` (production training).

Trains a single CatBoost classifier with `posterior_sampling=True` (so the virtual-ensemble uncertainty pipeline in Phase 5 has what it needs) on a recent slice of the training window, caches predictions to disk, then runs a sequence of certified analytics with bootstrap CIs.

**Inputs**

- `data/model_dataset/dataset.parquet`
- `data/model_dataset/feature_list.json`

**Outputs**

- `data/model_dataset/research_predictions.parquet` (cache: k, ts, y, m_k, tau_k, phi, regime, p, split)
- `data/model_dataset/analytics/research_metrics_with_ci.json` (certified bundle)
- `data/model_dataset/analytics/research_metrics_by_regime.json`

Every analytics module lives under `src/analytics/`; the notebook is thin glue.
"""))

CELLS.append(code("""from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
if not (ROOT / "docs" / "MINIMAL_PROJECT_SPEC_v2.md").exists():
    if (ROOT.parent / "docs" / "MINIMAL_PROJECT_SPEC_v2.md").exists():
        ROOT = ROOT.parent
    else:
        raise RuntimeError("Could not locate repo root")
sys.path.insert(0, str(ROOT))

from src import utils
from src.analytics.bootstrap import BootstrapResult
from src.analytics.metrics import (
    bootstrap_all_metrics,
    bootstrap_metrics_by_regime,
    by_regime_to_summary_dict,
    to_summary_dict,
)
from src.analytics.fast_train import (
    TrainSliceConfig,
    compute_predictions,
    fit_research_model,
    research_train_params,
    save_predictions_cache,
    select_recent_train_slice,
)

DATASET_DIR = ROOT / "data" / "model_dataset"
DATASET_PATH = DATASET_DIR / "dataset.parquet"
FEATURE_LIST_PATH = DATASET_DIR / "feature_list.json"
ANALYTICS_DIR = DATASET_DIR / "analytics"
PLOTS_DIR = DATASET_DIR / "plots"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
warnings.filterwarnings("ignore", category=UserWarning)
print("ROOT       :", ROOT)
print("DATASET    :", DATASET_PATH)
"""))

CELLS.append(md("""## 1. Configuration

All knobs for the study live in this cell. Change defaults here, re-run from this point.
"""))

CELLS.append(code("""# Train-slice config: keep only the most recent N months of train for fast iteration.
# Set months_back=None and frac_back=None to use the full training window.
TRAIN_SLICE = TrainSliceConfig(months_back=6.0)

# Bootstrap configuration for certified metrics.
BOOTSTRAP_B = 1000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_STRATIFY = True   # preserve per-class counts across resamples
BOOTSTRAP_SEED = 0

# Regime signal (spec Section 11.4)
REGIME_SIGNAL = "vol__rs__f__w240"

# Output paths
RESEARCH_CACHE_PATH = DATASET_DIR / "research_predictions.parquet"
RESEARCH_METRICS_PATH = ANALYTICS_DIR / "research_metrics_with_ci.json"
RESEARCH_REGIME_PATH = ANALYTICS_DIR / "research_metrics_by_regime.json"

print(f"Train slice : months_back={TRAIN_SLICE.months_back}  frac_back={TRAIN_SLICE.frac_back}")
print(f"Bootstrap   : B={BOOTSTRAP_B}, CI={BOOTSTRAP_CI}, stratify={BOOTSTRAP_STRATIFY}, seed={BOOTSTRAP_SEED}")
print(f"Regime sig  : {REGIME_SIGNAL}")
"""))

CELLS.append(md("""## 2. Load dataset and chronological split

Same chronological split + embargo as production (`utils.chronological_split_with_embargo`).
The full training window is shown for reference; the research model below trains on only a recent slice.
"""))

CELLS.append(code("""df = pd.read_parquet(DATASET_PATH)
feature_list = utils.load_json(FEATURE_LIST_PATH)
df = df.sort_values("k").reset_index(drop=True)

missing = [c for c in feature_list if c not in df.columns]
if missing:
    raise ValueError(f"feature_list missing in dataset: {missing[:10]} (total {len(missing)})")

# Honor the feature_list.json contract — drop raw base columns the dataset
# carries alongside engineered features (open/high/low/funding_rate/oi/etc.).
# This is *not* data-quality work (that's owned by 02_feature_building); it's
# just enforcing the model-input contract before the strict NaN checkpoint.
non_feature_cols = ["k", "ts", "y", "m_k", "tau_k", "phi", "w_dist", "w_time", "weight"]
keep = set(feature_list) | set(non_feature_cols)
extras = [c for c in df.columns if c not in keep]
if extras:
    print(f"Dropping {len(extras)} columns not in feature_list (raw cols carried alongside features): "
          f"{extras[:6]}{'...' if len(extras) > 6 else ''}")
    df = df.drop(columns=extras)

train_df, val_df, test_df = utils.chronological_split_with_embargo(
    df, train_frac=utils.TRAIN_FRAC, val_frac=utils.VAL_FRAC, embargo_k=utils.EMBARGO_K,
)
# Strict pre-training validation. Data quality (NaN handling, undef flags,
# imputation) is owned by 02_feature_building; this notebook only reacts —
# checkpoint raises with a precise diagnostic if anything leaked through.
utils.checkpoint_before_training(train_df, val_df, test_df, embargo_k=utils.EMBARGO_K)

split_summary = pd.DataFrame([
    {"split": "train", "n": len(train_df), "ts_start": train_df["ts"].min(), "ts_end": train_df["ts"].max(), "base_rate": float(train_df["y"].mean())},
    {"split": "val",   "n": len(val_df),   "ts_start": val_df["ts"].min(),   "ts_end": val_df["ts"].max(),   "base_rate": float(val_df["y"].mean())},
    {"split": "test",  "n": len(test_df),  "ts_start": test_df["ts"].min(),  "ts_end": test_df["ts"].max(),  "base_rate": float(test_df["y"].mean())},
])
print(f"Total rows: {len(df):,}    Features: {len(feature_list):,}")
display(split_summary)
"""))

CELLS.append(md("""## 3. Fit research model on recent train slice

Single seed, `posterior_sampling=True` (so virtual-ensemble uncertainty is available in Phase 5).
Iterations are capped high; early stopping does the work.

Learning curve is shown inline so it is observable mid-pipeline.
"""))

CELLS.append(code("""train_recent = select_recent_train_slice(train_df, TRAIN_SLICE)
print(f"Recent train slice : {len(train_recent):,} rows  ({train_recent['ts'].min()} -> {train_recent['ts'].max()})")
print(f"Slice base rate    : {train_recent['y'].mean():.4f}")

params = research_train_params(verbose=200)
print("\\nResearch model params:")
for k, v in params.items():
    print(f"  {k}: {v}")

t0 = time.perf_counter()
model = fit_research_model(train_recent, val_df, feature_list, params=params)
fit_seconds = time.perf_counter() - t0
print(f"\\nFit completed in {fit_seconds:.1f}s   best_iteration={model.get_best_iteration()}")
"""))

CELLS.append(code("""# Learning curves (train + val LogLoss; full and zoomed)
evals = model.get_evals_result()
train_ll = evals.get("learn", {}).get("Logloss", [])
val_ll = evals.get("validation", evals.get("validation_0", {})).get("Logloss", [])

if train_ll and val_ll:
    iters = list(range(1, len(train_ll) + 1))
    best = int(model.get_best_iteration())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].plot(iters, train_ll, label="Train", alpha=0.85)
    axes[0].plot(iters, val_ll, label="Validation", alpha=0.85)
    axes[0].axvline(best + 1, color="red", linestyle="--", alpha=0.7, label=f"Best ({best})")
    axes[0].set(xlabel="Iteration", ylabel="LogLoss", title="Learning Curve (full)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    zoom = max(0, len(train_ll) - 500)
    axes[1].plot(iters[zoom:], train_ll[zoom:], label="Train", alpha=0.85)
    axes[1].plot(iters[zoom:], val_ll[zoom:], label="Validation", alpha=0.85)
    axes[1].axvline(best + 1, color="red", linestyle="--", alpha=0.7)
    axes[1].set(xlabel="Iteration", ylabel="LogLoss", title="Learning Curve (last 500 iter)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    print(f"Final train LL : {train_ll[best]:.4f}")
    print(f"Final val   LL : {val_ll[best]:.4f}")
    print(f"Gap (val - tr) : {val_ll[best] - train_ll[best]:+.4f}")
else:
    print("Could not extract learning curves; eval keys =", list(evals.keys()))
"""))

CELLS.append(md("""## 4. Compute and cache predictions

Predictions on val + test are saved to a parquet cache. **Every downstream analytics phase reads this cache** instead of retraining the model, so iteration on charts is essentially free.

Cache schema: `k, ts, y, m_k, tau_k, phi, regime, p, split`.
"""))

CELLS.append(code("""cache = compute_predictions(
    model,
    {"val": val_df, "test": test_df},
    feature_list,
    regime_signal_col=REGIME_SIGNAL,
)
save_predictions_cache(cache, RESEARCH_CACHE_PATH)
print(f"Saved cache : {RESEARCH_CACHE_PATH}  ({len(cache):,} rows)")

cache_summary = (
    cache.groupby("split")
    .agg(
        n=("p", "size"),
        base_rate=("y", "mean"),
        p_min=("p", "min"),
        p_med=("p", "median"),
        p_max=("p", "max"),
        regime_med=("regime", "median"),
    )
    .round(5)
)
display(cache_summary)
"""))

CELLS.append(md("""## 5. Certified core metrics with bootstrap CIs

The headline metrics (ROC-AUC, PR-AUC, log-loss, Brier, ECE) are bootstrapped class-stratified iid (preserves base rate). Reading order: **point** is the metric on the full split; **CI** is the bootstrap percentile interval at `BOOTSTRAP_CI`.

Pay attention to the **width of the CI on test PR-AUC** - that is exactly the uncertainty that the legacy notebook does not report.
"""))

CELLS.append(code("""val_cache = cache[cache["split"] == "val"].reset_index(drop=True)
test_cache = cache[cache["split"] == "test"].reset_index(drop=True)

print(f"Bootstrapping core metrics: B={BOOTSTRAP_B}, CI={BOOTSTRAP_CI}, stratify={BOOTSTRAP_STRATIFY} ...")
t0 = time.perf_counter()
val_metrics = bootstrap_all_metrics(
    val_cache["y"].to_numpy(), val_cache["p"].to_numpy(),
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED,
)
test_metrics = bootstrap_all_metrics(
    test_cache["y"].to_numpy(), test_cache["p"].to_numpy(),
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED,
)
print(f"Bootstrap completed in {time.perf_counter() - t0:.1f}s")

def _ci(r: BootstrapResult) -> str:
    return f"[{r.ci_low:.4f}, {r.ci_high:.4f}]"
def _width(r: BootstrapResult) -> float:
    return float(r.ci_high - r.ci_low)

rows = []
for name in val_metrics:
    v, t = val_metrics[name], test_metrics[name]
    rows.append({
        "metric": name,
        "val (point)": v.point,
        "val 95% CI": _ci(v),
        "val width": _width(v),
        "test (point)": t.point,
        "test 95% CI": _ci(t),
        "test width": _width(t),
    })
metrics_table = pd.DataFrame(rows)
display(metrics_table.style.format({"val (point)": "{:.4f}", "val width": "{:.4f}", "test (point)": "{:.4f}", "test width": "{:.4f}"}))

bundle = {
    "config": {"B": BOOTSTRAP_B, "ci": BOOTSTRAP_CI, "stratify": BOOTSTRAP_STRATIFY, "seed": BOOTSTRAP_SEED, "regime_signal": REGIME_SIGNAL},
    "val": {"n_samples": int(len(val_cache)), "base_rate": float(val_cache["y"].mean()), **to_summary_dict(val_metrics)},
    "test": {"n_samples": int(len(test_cache)), "base_rate": float(test_cache["y"].mean()), **to_summary_dict(test_metrics)},
    "best_iteration": int(model.get_best_iteration()),
    "fit_seconds": float(fit_seconds),
    "train_slice": {
        "months_back": TRAIN_SLICE.months_back,
        "frac_back": TRAIN_SLICE.frac_back,
        "n_train": int(len(train_recent)),
        "ts_start": str(train_recent["ts"].min()),
        "ts_end": str(train_recent["ts"].max()),
    },
}
utils.save_json(RESEARCH_METRICS_PATH, bundle)
print(f"\\nSaved certified metrics : {RESEARCH_METRICS_PATH}")
"""))

CELLS.append(md("""## 6. Per-regime certified metrics

Tercile breakdown by `vol__rs__f__w240`. The legacy analytics flagged that high-vol calibration is materially worse than low-vol; the bootstrap CIs let us judge whether that gap is real or sampling noise.

Two views:

1. Tabular: per (split, regime), point + CI for every metric.
2. Chart: ECE-by-regime barplot with CI error bars (val vs test side-by-side).
"""))

CELLS.append(code("""print("Bootstrapping metrics per volatility regime ...")
t0 = time.perf_counter()
val_by_regime = bootstrap_metrics_by_regime(
    val_cache["y"].to_numpy(), val_cache["p"].to_numpy(), val_cache["regime"].to_numpy(),
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED,
)
test_by_regime = bootstrap_metrics_by_regime(
    test_cache["y"].to_numpy(), test_cache["p"].to_numpy(), test_cache["regime"].to_numpy(),
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED,
)
print(f"Per-regime bootstrap completed in {time.perf_counter() - t0:.1f}s")

def _regime_table(by_regime, split_name, cache_df):
    rows = []
    terciles = pd.qcut(cache_df["regime"], 3, labels=["low", "med", "high"])
    for label in ["low", "med", "high"]:
        if label not in by_regime:
            continue
        m = by_regime[label]
        mask = np.asarray(terciles == label)
        row = {
            "split": split_name,
            "regime": label,
            "n": int(mask.sum()),
            "base_rate": float(cache_df.loc[mask, "y"].mean()),
        }
        for name in ["roc_auc", "pr_auc", "brier_score", "ece_10bin"]:
            r = m.get(name)
            if r is None:
                continue
            row[f"{name}"] = r.point
            row[f"{name}_ci"] = f"[{r.ci_low:.4f}, {r.ci_high:.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)

regime_table = pd.concat([
    _regime_table(val_by_regime, "val", val_cache),
    _regime_table(test_by_regime, "test", test_cache),
], ignore_index=True)
display(regime_table.style.format({"base_rate": "{:.4f}", "roc_auc": "{:.4f}", "pr_auc": "{:.4f}", "brier_score": "{:.4f}", "ece_10bin": "{:.4f}"}))
"""))

CELLS.append(code("""# ECE-by-regime barplot with bootstrap CI error bars (val vs test)
regimes = ["low", "med", "high"]
x = np.arange(len(regimes))
width = 0.36
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, (split, by_regime) in enumerate([("val", val_by_regime), ("test", test_by_regime)]):
    points, lo_err, hi_err = [], [], []
    for r in regimes:
        if r not in by_regime or "ece_10bin" not in by_regime[r]:
            points.append(np.nan); lo_err.append(0.0); hi_err.append(0.0); continue
        m = by_regime[r]["ece_10bin"]
        points.append(m.point)
        lo_err.append(max(0.0, m.point - m.ci_low))
        hi_err.append(max(0.0, m.ci_high - m.point))
    ax.bar(x + (i - 0.5) * width, points, width=width, yerr=[lo_err, hi_err], capsize=4, label=split, alpha=0.85)
ax.set(xticks=x, xticklabels=regimes, ylabel="ECE (10-bin)", xlabel="volatility regime",
       title=f"ECE by volatility regime  (95% bootstrap CI, B={BOOTSTRAP_B})")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_ece_by_regime.png", dpi=150)
plt.show()

regime_bundle = {
    "val": by_regime_to_summary_dict(val_by_regime),
    "test": by_regime_to_summary_dict(test_by_regime),
}
utils.save_json(RESEARCH_REGIME_PATH, regime_bundle)
print(f"Saved per-regime metrics : {RESEARCH_REGIME_PATH}")
print(f"Saved chart              : {PLOTS_DIR / 'research_ece_by_regime.png'}")
"""))

CELLS.append(md("""## 7. Curves with bootstrap quantile bands

Three curves, each with the 95% per-grid-point bootstrap percentile band:

- **ROC** — TPR vs FPR (linear interpolation onto a 201-point FPR grid; upper-envelope at duplicate FPRs)
- **PR** — Precision vs recall (Pascal-VOC max-envelope onto a 201-point recall grid)
- **Calibration** — empirical y vs mean predicted p (10 equal-width bins, per-bin CI on empirical y)

Tight CIs in the operating range mean the model's behavior there is well-determined; wide CIs (especially in PR's high-precision low-recall corner) flag where the test-set point estimate is misleading.
"""))

CELLS.append(code("""from src.analytics.curves import (
    bootstrap_calibration_curve,
    bootstrap_pr_curve,
    bootstrap_roc_curve,
    plot_calibration_with_ci,
    plot_pr_with_band,
    plot_roc_with_band,
)

print(f"Bootstrapping curves: B={BOOTSTRAP_B}, CI={BOOTSTRAP_CI}, stratify={BOOTSTRAP_STRATIFY} ...")
t0 = time.perf_counter()
val_roc = bootstrap_roc_curve(val_cache["y"].to_numpy(), val_cache["p"].to_numpy(),
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED)
test_roc = bootstrap_roc_curve(test_cache["y"].to_numpy(), test_cache["p"].to_numpy(),
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED)
val_pr = bootstrap_pr_curve(val_cache["y"].to_numpy(), val_cache["p"].to_numpy(),
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED)
test_pr = bootstrap_pr_curve(test_cache["y"].to_numpy(), test_cache["p"].to_numpy(),
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED)
val_cal = bootstrap_calibration_curve(val_cache["y"].to_numpy(), val_cache["p"].to_numpy(),
    n_bins=10, B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED)
test_cal = bootstrap_calibration_curve(test_cache["y"].to_numpy(), test_cache["p"].to_numpy(),
    n_bins=10, B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED)
print(f"Curve bootstrap completed in {time.perf_counter() - t0:.1f}s")
"""))

CELLS.append(code("""# Side-by-side ROC: val vs test, both with 95% bootstrap bands
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
plot_roc_with_band(val_roc,  ax=axes[0], color="#e67e22", label="Validation")
plot_roc_with_band(test_roc, ax=axes[1], color="#3498db", label="Test", plot_diagonal=True)
axes[0].set_title(f"ROC val  (n={len(val_cache):,}, base rate={val_cache['y'].mean():.3f})")
axes[1].set_title(f"ROC test (n={len(test_cache):,}, base rate={test_cache['y'].mean():.3f})")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_roc_with_band.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_roc_with_band.png'}")
"""))

CELLS.append(code("""# Side-by-side PR: val vs test, with base-rate reference and 95% bands
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
plot_pr_with_band(val_pr,  ax=axes[0], base_rate=float(val_cache["y"].mean()),
                  color="#e67e22", label="Validation")
plot_pr_with_band(test_pr, ax=axes[1], base_rate=float(test_cache["y"].mean()),
                  color="#3498db", label="Test")
axes[0].set_title(f"PR val  (n={len(val_cache):,}, base rate={val_cache['y'].mean():.3f})")
axes[1].set_title(f"PR test (n={len(test_cache):,}, base rate={test_cache['y'].mean():.3f})")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_pr_with_band.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_pr_with_band.png'}")
"""))

CELLS.append(code("""# Calibration: val + test on a single axis with bootstrap CIs
fig, ax = plt.subplots(figsize=(7, 5.2))
plot_calibration_with_ci(val_cal,  ax=ax, color="#e67e22", label="Validation", plot_perfect=True)
plot_calibration_with_ci(test_cal, ax=ax, color="#3498db", label="Test",       plot_perfect=False)
ax.set_title(f"Calibration (95% bootstrap CI on empirical y, B={BOOTSTRAP_B})")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_calibration_with_ci.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_calibration_with_ci.png'}")
"""))

CELLS.append(code("""# Save curve summaries (JSON, excludes raw bootstrap samples)
curves_bundle = {
    "val":  {
        "roc": val_roc.to_summary_dict(),
        "pr":  val_pr.to_summary_dict(),
        "calibration": val_cal.to_summary_dict(),
    },
    "test": {
        "roc": test_roc.to_summary_dict(),
        "pr":  test_pr.to_summary_dict(),
        "calibration": test_cal.to_summary_dict(),
    },
}
utils.save_json(ANALYTICS_DIR / "research_curves_summary.json", curves_bundle)
display(pd.DataFrame([
    {"split": "val",  "ROC-AUC": val_roc.auc_point,  "ROC-AUC CI": f"[{val_roc.auc_ci_low:.3f}, {val_roc.auc_ci_high:.3f}]",
     "PR-AP":  val_pr.auc_point,  "PR-AP CI":  f"[{val_pr.auc_ci_low:.3f}, {val_pr.auc_ci_high:.3f}]",
     "ECE":    val_cal.auc_point, "ECE CI":    f"[{val_cal.auc_ci_low:.3f}, {val_cal.auc_ci_high:.3f}]"},
    {"split": "test", "ROC-AUC": test_roc.auc_point, "ROC-AUC CI": f"[{test_roc.auc_ci_low:.3f}, {test_roc.auc_ci_high:.3f}]",
     "PR-AP":  test_pr.auc_point, "PR-AP CI":  f"[{test_pr.auc_ci_low:.3f}, {test_pr.auc_ci_high:.3f}]",
     "ECE":    test_cal.auc_point,"ECE CI":    f"[{test_cal.auc_ci_low:.3f}, {test_cal.auc_ci_high:.3f}]"},
]).style.format({"ROC-AUC": "{:.4f}", "PR-AP": "{:.4f}", "ECE": "{:.4f}"}))
print(f"Saved curve summaries: {ANALYTICS_DIR / 'research_curves_summary.json'}")
"""))

CELLS.append(md("""## 8. Time-degradation diagnostics

Operating on the cached predictions (no retraining), four diagnostics:

1. **Rolling metrics with CI bands** — per-window ROC-AUC, PR-AUC, Brier, ECE. Reveals stability over time and where the model is reliably better/worse than its split-aggregate.
2. **Brier-Murphy decomposition over time** — splits Brier into Reliability + Resolution + Uncertainty + within-bin variance. Identity holds on the *binned* Brier: `BS_binned = REL - RES + UNC`. When Reliability rises but Resolution holds, the right action is recalibration; when Resolution falls, you need a refit.
3. **PSI / KS rolling** — measures probability-distribution drift of test windows vs the full validation distribution. PSI > 0.1 = moderate drift; > 0.2 = significant.
4. **Conditional precision heatmap** — precision at a fixed operating threshold, conditioned on (vol regime × hour-of-day). Wilson 95% CIs handle small per-cell counts robustly.
"""))

CELLS.append(code("""from src.analytics.degradation import (
    conditional_precision,
    plot_brier_decomposition_over_time,
    plot_conditional_precision_heatmap,
    plot_psi_ks_over_time,
    plot_rolling_metric_with_band,
    psi_ks_rolling,
    rolling_brier_decomposition,
    rolling_metrics_with_ci,
)

ROLL_WINDOW = "3D"
ROLL_STEP = "1D"
ROLL_B = 300
ROLL_MIN_N = 150
ROLL_MIN_POS = 15

print(f"Rolling window={ROLL_WINDOW}, step={ROLL_STEP}, B={ROLL_B}")

t0 = time.perf_counter()
rolling_test = rolling_metrics_with_ci(
    cache, split="test", window=ROLL_WINDOW, step=ROLL_STEP,
    B=ROLL_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED,
    min_n=ROLL_MIN_N, min_pos=ROLL_MIN_POS,
)
print(f"Rolling metrics: {len(rolling_test)} windows, {time.perf_counter() - t0:.1f}s")
display(rolling_test[["window_end", "n_samples", "n_pos", "base_rate",
                      "roc_auc_point", "pr_auc_point", "brier_score_point", "ece_10bin_point"]].head())
"""))

CELLS.append(code("""# Rolling metrics 4-panel: PR-AUC / ROC-AUC / Brier / ECE with bootstrap bands
fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
plot_rolling_metric_with_band(rolling_test, "pr_auc",      ax=axes[0, 0], color="C0", label="PR-AUC")
plot_rolling_metric_with_band(rolling_test, "roc_auc",     ax=axes[0, 1], color="C1", label="ROC-AUC")
plot_rolling_metric_with_band(rolling_test, "brier_score", ax=axes[1, 0], color="C2", label="Brier")
plot_rolling_metric_with_band(rolling_test, "ece_10bin",   ax=axes[1, 1], color="C3", label="ECE (10-bin)")
# Overlay base rate on PR-AUC panel for reference
axes[0, 0].plot(rolling_test["window_end"], rolling_test["base_rate"],
                linestyle=":", color="gray", alpha=0.7, label="base rate")
axes[0, 0].legend(loc="upper right", fontsize=9)
fig.suptitle(f"Rolling metrics on test ({ROLL_WINDOW} window, daily step, 95% CI)")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_rolling_metrics.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_rolling_metrics.png'}")
"""))

CELLS.append(code("""# Rolling Brier-Murphy decomposition (REL, RES, UNC, WBV components)
t0 = time.perf_counter()
rolling_brier = rolling_brier_decomposition(
    cache, split="test", window=ROLL_WINDOW, step=ROLL_STEP, n_bins=10,
    B=ROLL_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED,
    min_n=ROLL_MIN_N, min_pos=ROLL_MIN_POS,
)
print(f"Brier decomposition: {len(rolling_brier)} windows, {time.perf_counter() - t0:.1f}s")

fig, ax = plt.subplots(figsize=(13, 5))
plot_brier_decomposition_over_time(rolling_brier, ax=ax)
ax.set_title(f"Brier decomposition on test ({ROLL_WINDOW} window, daily step)")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_brier_decomposition.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_brier_decomposition.png'}")
"""))

CELLS.append(code("""# Rolling PSI / KS — test windows vs full validation distribution
drift = psi_ks_rolling(
    cache, reference_split="val", target_split="test",
    window=ROLL_WINDOW, step=ROLL_STEP, n_bins=10, min_n=ROLL_MIN_N,
)
print(f"Drift windows: {len(drift)}")

fig, ax = plt.subplots(figsize=(13, 4.5))
plot_psi_ks_over_time(drift, ax=ax)
ax.set_title("Probability-distribution drift: test windows vs full validation")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_psi_ks_over_time.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_psi_ks_over_time.png'}")
print()
print(f"PSI median {drift['psi'].median():.3f}, max {drift['psi'].max():.3f}  "
      f"(>0.1 = moderate drift, >0.2 = significant)")
print(f"KS  median {drift['ks'].median():.3f},  max {drift['ks'].max():.3f}")
"""))

CELLS.append(code("""# Conditional precision heatmap (vol regime x hour-of-day) at the operating threshold
OPERATING_THRESHOLD = 0.20  # near the knee of the threshold curve from Section 6

cond = conditional_precision(
    cache, threshold=OPERATING_THRESHOLD, split="test",
    by=("regime_bucket", "hour"), n_regime_buckets=3,
)

fig, ax = plt.subplots(figsize=(15, 4.2))
plot_conditional_precision_heatmap(
    cond, index_col="regime_bucket", column_col="hour",
    value_col="precision", annotate="value", ax=ax,
    vmin=0.0, vmax=cond["precision"].max() * 1.05 if len(cond) else 1.0,
)
ax.set_title(
    f"Conditional precision @ p>={OPERATING_THRESHOLD:.2f}  "
    f"(test, base rate={test_cache['y'].mean():.3f}, n_predictions={int(cond['n_predictions'].sum())})"
)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_conditional_precision.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_conditional_precision.png'}")

# Also show the supporting table of cell counts and CIs
display(cond.pivot(index="regime_bucket", columns="hour", values="n_predictions").fillna(0).astype(int))
"""))

CELLS.append(code("""# Save degradation summaries
import json
degradation_bundle = {
    "config": {"window": ROLL_WINDOW, "step": ROLL_STEP, "B": ROLL_B,
               "min_n": ROLL_MIN_N, "min_pos": ROLL_MIN_POS,
               "operating_threshold": OPERATING_THRESHOLD},
    "n_rolling_windows": int(len(rolling_test)),
    "psi_summary": {
        "median": float(drift["psi"].median()) if len(drift) else None,
        "max": float(drift["psi"].max()) if len(drift) else None,
        "n_above_0.1": int((drift["psi"] > 0.1).sum()) if len(drift) else 0,
        "n_above_0.2": int((drift["psi"] > 0.2).sum()) if len(drift) else 0,
    },
    "ks_summary": {
        "median": float(drift["ks"].median()) if len(drift) else None,
        "max": float(drift["ks"].max()) if len(drift) else None,
    },
    "conditional_precision": {
        "n_cells_populated": int(len(cond)),
        "median_precision": float(cond["precision"].median()) if len(cond) else None,
        "min_precision": float(cond["precision"].min()) if len(cond) else None,
        "max_precision": float(cond["precision"].max()) if len(cond) else None,
    },
}
utils.save_json(ANALYTICS_DIR / "research_degradation_summary.json", degradation_bundle)
print(f"Saved degradation summary: {ANALYTICS_DIR / 'research_degradation_summary.json'}")
print(json.dumps(degradation_bundle, indent=2, default=str))
"""))

CELLS.append(md("""## 9. Tail / edge analysis

The operating-point view: which thresholds are worth using, what's the realized edge per trade, and where does Kelly turn negative? Five charts:

1. **Threshold sweep with bootstrap CIs** — precision, recall, trade-rate, lift over base rate at every threshold.
2. **Net EV per trade vs trades-per-day** (the canonical operating chart) — log-x trades-per-day; net EV in log-return units; bootstrap band per threshold. Overlaid break-even line at 0.
3. **Kelly fraction by probability decile** — empirical hit rate per decile (Wilson 95% CI) plus the implied Kelly fraction under the configured outcome model.
4. **Lift / gain curve** — cumulative precision when taking the top-k by p, vs random baseline.
5. **Partial AUC** in the operating bands (FPR <= 5%, recall <= 10%) with bootstrap CIs.

**Outcome model** (configurable in the cell below): default assumes `gain_per_hit = phi`, `loss_per_miss = phi` (symmetric — conservative for entry-only), `cost_per_trade = 5 bps`. For accurate EV with a real exit policy, add an `r_realized` column to the dataset in 02_feature_building and pass `use_realized_return=True`.
"""))

CELLS.append(code("""from src.analytics.edge import (
    OutcomeModel,
    bootstrap_partial_pr_auc,
    bootstrap_partial_roc_auc,
    bootstrap_threshold_sweep,
    kelly_by_bin,
    lift_curve,
    plot_kelly_by_bin,
    plot_lift_curve,
    plot_net_ev_vs_trades_per_day,
    plot_threshold_sweep_with_bands,
)

OUTCOME_MODEL = OutcomeModel(
    gain_per_hit=float(test_cache["phi"].iloc[0]),
    loss_per_miss=float(test_cache["phi"].iloc[0]),  # symmetric default; override per strategy
    cost_per_trade=0.0005,
    use_realized_return=False,
)
print(f"Outcome model: gain={OUTCOME_MODEL.gain_per_hit:.5f}, "
      f"loss={OUTCOME_MODEL.loss_per_miss:.5f}, "
      f"cost={OUTCOME_MODEL.cost_per_trade:.5f}, "
      f"use_realized={OUTCOME_MODEL.use_realized_return}")

EDGE_THRESHOLDS = np.linspace(0.05, float(test_cache["p"].max()), 80)

t0 = time.perf_counter()
sweep = bootstrap_threshold_sweep(
    cache, split="test", thresholds=EDGE_THRESHOLDS,
    outcome_model=OUTCOME_MODEL,
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED,
)
print(f"Threshold sweep: {len(sweep)} thresholds, {time.perf_counter() - t0:.1f}s")
display(sweep[["threshold", "n_trades", "trades_per_day", "precision", "recall", "lift",
               "ev_per_trade", "sharpe_per_trade"]].head(10).style.format({
    "threshold": "{:.3f}", "trades_per_day": "{:.1f}", "precision": "{:.3f}",
    "recall": "{:.3f}", "lift": "{:.2f}", "ev_per_trade": "{:.5f}",
    "sharpe_per_trade": "{:.3f}",
}))
"""))

CELLS.append(code("""# Threshold sweep with bootstrap bands (precision / recall / trade rate / lift)
fig, ax = plt.subplots(figsize=(11, 5))
plot_threshold_sweep_with_bands(sweep, ax=ax, metrics=("precision", "recall", "trade_rate", "lift"))
ax.set_title(f"Threshold sweep on test (95% bootstrap CI, B={BOOTSTRAP_B})")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_threshold_sweep.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_threshold_sweep.png'}")
"""))

CELLS.append(code("""# THE canonical chart: Net EV per trade vs trades-per-day (log-x, with band)
fig, ax = plt.subplots(figsize=(11, 5.4))
plot_net_ev_vs_trades_per_day(
    sweep, ax=ax, color="C0", label="Test",
    base_rate=float(test_cache["y"].mean()),
)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_net_ev_curve.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_net_ev_curve.png'}")

# Annotate the EV-maximizing threshold (within the bootstrap band of the optimum)
sweep_clean = sweep.dropna(subset=["ev_per_trade"]).reset_index(drop=True)
if len(sweep_clean) > 0:
    best = sweep_clean.iloc[sweep_clean["ev_per_trade"].idxmax()]
    print()
    print(f"EV-maximizing threshold: {best['threshold']:.3f}  "
          f"=>  EV/trade {best['ev_per_trade']:+.5f} "
          f"[{best['ev_per_trade_ci_low']:+.5f}, {best['ev_per_trade_ci_high']:+.5f}]  "
          f"@ {best['trades_per_day']:.1f} trades/day,  "
          f"precision {best['precision']:.3f},  Sharpe/trade {best['sharpe_per_trade']:.3f}")
"""))

CELLS.append(code("""# Kelly fraction per probability decile (Wilson CI on hit rate, propagated to Kelly)
kbin = kelly_by_bin(cache, split="test", n_bins=10, outcome_model=OUTCOME_MODEL)
display(kbin.style.format({
    "p_lo": "{:.3f}", "p_hi": "{:.3f}", "mean_p": "{:.3f}",
    "hit_rate": "{:.3f}", "hit_rate_ci_low": "{:.3f}", "hit_rate_ci_high": "{:.3f}",
    "kelly": "{:.3f}", "kelly_ci_low": "{:.3f}", "kelly_ci_high": "{:.3f}",
    "half_kelly": "{:.3f}",
}))

fig, ax = plt.subplots(figsize=(11, 5))
plot_kelly_by_bin(kbin, base_rate=float(test_cache["y"].mean()), ax=ax)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_kelly_by_bin.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_kelly_by_bin.png'}")
"""))

CELLS.append(code("""# Lift / gain curve: cumulative precision as the top-k is taken
gain = lift_curve(cache, split="test")
fig, ax = plt.subplots(figsize=(10, 5))
plot_lift_curve(gain, ax=ax, color="C0", label="Test")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_lift_curve.png", dpi=150)
plt.show()

# Headline lift numbers at common operating points
for k_pct in [0.01, 0.025, 0.05, 0.10, 0.20]:
    k_target = int(round(k_pct * len(gain)))
    if k_target < 1:
        continue
    row = gain.iloc[k_target - 1]
    print(f"top-{k_pct:>5.1%}: precision {row['precision_at_k']:.3f}  "
          f"lift {row['lift_at_k']:.2f}x  recall {row['recall_at_k']:.3f}  (k={int(row['k'])})")
"""))

CELLS.append(code("""# Partial AUC over operating bands (low FPR, low recall)
y_test = test_cache["y"].to_numpy()
p_test = test_cache["p"].to_numpy()

partial_roc = bootstrap_partial_roc_auc(y_test, p_test, fpr_max=0.05,
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED)
partial_pr  = bootstrap_partial_pr_auc(y_test, p_test, recall_max=0.10,
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, stratify=BOOTSTRAP_STRATIFY, seed=BOOTSTRAP_SEED)

print(f"Partial ROC-AUC (FPR <= 0.05) : {partial_roc.point:.3f} "
      f"[{partial_roc.ci_low:.3f}, {partial_roc.ci_high:.3f}]")
print(f"Partial PR-AUC  (recall <=0.10): {partial_pr.point:.3f} "
      f"[{partial_pr.ci_low:.3f}, {partial_pr.ci_high:.3f}]")
print()
print("These are the metrics to track on a production entry gate that operates at low FPR / low recall.")
"""))

CELLS.append(code("""# Save edge bundle
import json
edge_bundle = {
    "outcome_model": {
        "gain_per_hit": OUTCOME_MODEL.gain_per_hit,
        "loss_per_miss": OUTCOME_MODEL.loss_per_miss,
        "cost_per_trade": OUTCOME_MODEL.cost_per_trade,
        "use_realized_return": OUTCOME_MODEL.use_realized_return,
    },
    "ev_max": {
        "threshold": float(best["threshold"]) if len(sweep_clean) else None,
        "ev_per_trade": float(best["ev_per_trade"]) if len(sweep_clean) else None,
        "ev_ci_low": float(best["ev_per_trade_ci_low"]) if len(sweep_clean) else None,
        "ev_ci_high": float(best["ev_per_trade_ci_high"]) if len(sweep_clean) else None,
        "trades_per_day": float(best["trades_per_day"]) if len(sweep_clean) else None,
        "precision": float(best["precision"]) if len(sweep_clean) else None,
        "sharpe_per_trade": float(best["sharpe_per_trade"]) if len(sweep_clean) else None,
    },
    "kelly_first_negative_bin": int(kbin[kbin["kelly"] < 0]["bin"].min()) if (kbin["kelly"] < 0).any() else None,
    "partial_roc_auc_fpr05": partial_roc.to_dict(include_samples=False),
    "partial_pr_auc_recall10": partial_pr.to_dict(include_samples=False),
    "top_decile_lift": float(gain.iloc[int(0.10 * len(gain)) - 1]["lift_at_k"]) if len(gain) > 10 else None,
}
utils.save_json(ANALYTICS_DIR / "research_edge_summary.json", edge_bundle)
print(f"Saved edge summary: {ANALYTICS_DIR / 'research_edge_summary.json'}")
print(json.dumps(edge_bundle, indent=2, default=str))
"""))

CELLS.append(md("""## 10. SHAP cohort decomposition (TP / FP / TN / FN)

Where do the model's *errors* come from at the feature level? TreeSHAP gives per-row, per-feature contribution to the log-odds; aggregating within cohorts then reveals systematic biases.

The headline view: **signed effect-size disagreement** = `(mean SHAP on FPs - mean SHAP on FNs) / pooled std`. Features at the top of this ranking are pushing the model toward FPs and away from FNs (or vice versa) — exactly the "inconsistent direction" features that are candidates for carve-out or stabilization.

Three views:
1. **Top-N by signed effect-size disagreement** (the standardized ranking).
2. **Top-N with bootstrap CI on the raw mean(FP) - mean(FN) difference** — filter the shortlist by survival under sampling noise; bars whose CI excludes 0 are statistically significant disagreements.
3. **Discriminative SHAP** — L2 logistic regression of `{FP=1, FN=0}` on per-row SHAP, which handles cross-feature correlations the mean-only views ignore. Ranked by `|coef|`.

Bonus: the grouped-cohort barplot for the top features shows the actual SHAP magnitudes per (TP / FP / TN / FN), not just the differences.
"""))

CELLS.append(code("""from src.analytics.cohorts import (
    bootstrap_shap_diff,
    cohort_assignments,
    cohort_counts,
    cohort_mean_shap,
    compute_shap_values,
    discriminative_shap,
    plot_bootstrap_shap_diff_with_ci,
    plot_signed_effect_size,
    plot_top_features_grouped_by_cohort,
    signed_effect_size_disagreement,
)

COHORT_THRESHOLD = 0.30  # operating threshold from the threshold sweep
TOP_N_FEATURES = 15

print(f"Computing SHAP on test ({len(test_df):,} rows x {len(feature_list):,} features) ...")
t0 = time.perf_counter()
shap_test = compute_shap_values(model, test_df, feature_list=feature_list)
print(f"SHAP computed in {time.perf_counter() - t0:.1f}s, shape={shap_test.shape}")

# Cohort assignment at the operating threshold
test_y = test_cache["y"].to_numpy()
test_p = test_cache["p"].to_numpy()
cohorts = cohort_assignments(test_y, test_p, threshold=COHORT_THRESHOLD)
counts = cohort_counts(cohorts)
print(f"Cohort counts at threshold {COHORT_THRESHOLD}: {counts}")
"""))

CELLS.append(code("""# Per-cohort mean SHAP (long-form)
mean_shap_long = cohort_mean_shap(shap_test, cohorts, feature_list)

# Signed effect-size disagreement
eff = signed_effect_size_disagreement(shap_test, cohorts, feature_list)
print(f"Top {TOP_N_FEATURES} features by signed effect-size disagreement:")
display(eff.head(TOP_N_FEATURES).style.format({
    "mean_shap_fp": "{:.4f}", "mean_shap_fn": "{:.4f}",
    "shap_diff_fp_minus_fn": "{:.4f}", "pooled_std": "{:.4f}",
    "effect_size": "{:.3f}", "abs_effect_size": "{:.3f}",
}))

# Plot
fig, ax = plt.subplots(figsize=(10, 6.5))
plot_signed_effect_size(eff, top_n=TOP_N_FEATURES, ax=ax)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_signed_effect_size.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_signed_effect_size.png'}")
"""))

CELLS.append(code("""# Bootstrap CI on (mean SHAP FP - mean SHAP FN) per feature
t0 = time.perf_counter()
diff_ci = bootstrap_shap_diff(
    shap_test, cohorts, feature_list,
    B=BOOTSTRAP_B, ci=BOOTSTRAP_CI, seed=BOOTSTRAP_SEED,
)
print(f"Bootstrap shap-diff: {time.perf_counter() - t0:.1f}s")
print(f"Top {TOP_N_FEATURES} features by |mean SHAP FP - FN| (with CI excluding 0 marker):")
display(diff_ci.head(TOP_N_FEATURES).style.format({
    "shap_diff": "{:.4f}", "shap_diff_ci_low": "{:.4f}", "shap_diff_ci_high": "{:.4f}",
    "abs_diff": "{:.4f}",
}))

fig, ax = plt.subplots(figsize=(10, 6.5))
plot_bootstrap_shap_diff_with_ci(diff_ci, top_n=TOP_N_FEATURES, ax=ax)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_shap_diff_with_ci.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_shap_diff_with_ci.png'}")
"""))

CELLS.append(code("""# Discriminative SHAP: logistic regression of {FP=1, FN=0} on per-row SHAP
t0 = time.perf_counter()
disc = discriminative_shap(shap_test, cohorts, feature_list, C=1.0)
print(f"Discriminative SHAP fit in {time.perf_counter() - t0:.1f}s")
print(f"Top {TOP_N_FEATURES} features by |discriminative coef|:")
display(disc.head(TOP_N_FEATURES).style.format({"discriminative_coef": "{:.4f}", "abs_coef": "{:.4f}"}))

# Plot horizontal-bar of top discriminative coefs
fig, ax = plt.subplots(figsize=(10, 6.5))
sub = disc.head(TOP_N_FEATURES).iloc[::-1]
colors = ["#e74c3c" if v > 0 else "#3498db" for v in sub["discriminative_coef"]]
ax.barh(sub["feature"], sub["discriminative_coef"], color=colors, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("L2 logistic-regression coefficient (FP=1 vs FN=0)")
ax.set_title(
    f"Top {TOP_N_FEATURES} features by discriminative SHAP\\n"
    "red = positive coef -> SHAP value pushes prediction toward FP cohort"
)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_discriminative_shap.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_discriminative_shap.png'}")
"""))

CELLS.append(code("""# Grouped barplot: top features by abs effect size, mean SHAP per cohort
top_features_by_effect = eff.head(TOP_N_FEATURES)["feature"].tolist()
fig, ax = plt.subplots(figsize=(13, 5.5))
plot_top_features_grouped_by_cohort(mean_shap_long, top_features_by_effect, ax=ax)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_cohort_mean_shap_grouped.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_cohort_mean_shap_grouped.png'}")

# Compare the three rankings (signed-effect, bootstrap-diff, discriminative)
top_overlap = pd.DataFrame({
    "by_effect_size": eff.head(TOP_N_FEATURES)["feature"].tolist(),
    "by_bootstrap_diff": diff_ci.head(TOP_N_FEATURES)["feature"].tolist(),
    "by_discriminative_coef": disc.head(TOP_N_FEATURES)["feature"].tolist(),
})
print("Top-N feature lists across the three rankings:")
display(top_overlap)

# Set overlap stats
set_eff = set(eff.head(TOP_N_FEATURES)["feature"])
set_diff = set(diff_ci.head(TOP_N_FEATURES)["feature"])
set_disc = set(disc.head(TOP_N_FEATURES)["feature"])
print()
print(f"Common across all three rankings: {len(set_eff & set_diff & set_disc)} features")
print(f"  - in eff and diff but not disc: {len((set_eff & set_diff) - set_disc)}")
print(f"  - in eff and disc but not diff: {len((set_eff & set_disc) - set_diff)}")
print(f"  - in diff and disc but not eff: {len((set_diff & set_disc) - set_eff)}")
"""))

CELLS.append(code("""# Save cohort summary
import json
cohort_bundle = {
    "config": {
        "operating_threshold": COHORT_THRESHOLD,
        "top_n": TOP_N_FEATURES,
        "B": BOOTSTRAP_B,
        "ci": BOOTSTRAP_CI,
    },
    "cohort_counts": counts,
    "top_by_signed_effect_size": eff.head(TOP_N_FEATURES)[
        ["feature", "mean_shap_fp", "mean_shap_fn", "shap_diff_fp_minus_fn",
         "pooled_std", "effect_size"]
    ].to_dict(orient="records"),
    "top_by_bootstrap_diff": diff_ci.head(TOP_N_FEATURES)[
        ["feature", "shap_diff", "shap_diff_ci_low", "shap_diff_ci_high",
         "ci_excludes_zero"]
    ].to_dict(orient="records"),
    "top_by_discriminative_coef": disc.head(TOP_N_FEATURES)[
        ["feature", "discriminative_coef"]
    ].to_dict(orient="records"),
    "intersection_count_all_three": int(len(set_eff & set_diff & set_disc)),
}
utils.save_json(ANALYTICS_DIR / "research_cohort_summary.json", cohort_bundle)
print(f"Saved cohort summary: {ANALYTICS_DIR / 'research_cohort_summary.json'}")
"""))

CELLS.append(md("""## 11. Virtual-ensemble uncertainty

The research model was trained with `posterior_sampling=True` (Stochastic Gradient Langevin Boosting) which lets us slice the trained tree sequence into K overlapping virtual ensembles and decompose predictive uncertainty:

- **Total** uncertainty = H[E_b[p_b]] — what the model is unsure about overall
- **Data** uncertainty = E_b[H(p_b)] — irreducible aleatoric noise the data carries
- **Knowledge** uncertainty = MI = Total - Data — epistemic; high values flag inputs far from training distribution

Operating idea: instead of a scalar gate `p >= τ_p`, use a **joint gate** `p >= τ_p AND knowledge_unc <= τ_unc`. If knowledge uncertainty is informative, this should improve precision at fixed trade rate (the points in the joint-gate scatter sit *above* the scalar-p baseline).

Validation that uncertainty is informative: the **(p decile × knowledge_unc decile) heatmap of empirical hit rate** should show monotone decrease left-to-right at fixed row. If not, the uncertainty signal is dead.
"""))

CELLS.append(code("""from src.analytics.uncertainty import (
    hit_rate_heatmap,
    joint_gate_sweep,
    plot_hit_rate_heatmap_2d,
    plot_joint_gate_vs_scalar_p,
    plot_uncertainty_distributions_by_cohort,
    plot_variance_reliability,
    predictive_uncertainty,
    variance_reliability,
    virtual_ensemble_predictions,
)

VE_K = 10  # number of virtual ensembles

print(f"Computing virtual-ensemble predictions on test ({len(test_df):,} rows, K={VE_K}) ...")
t0 = time.perf_counter()
p_ve_test = virtual_ensemble_predictions(
    model, test_df, virtual_ensembles_count=VE_K, feature_list=feature_list
)
print(f"VE predictions in {time.perf_counter() - t0:.1f}s, shape={p_ve_test.shape}")

unc_test = predictive_uncertainty(p_ve_test)
print()
print(f"mean_p              : mean={unc_test['mean_p'].mean():.4f}, max={unc_test['mean_p'].max():.4f}")
print(f"total_uncertainty   : mean={unc_test['total_uncertainty'].mean():.4f}, max={unc_test['total_uncertainty'].max():.4f}")
print(f"data_uncertainty    : mean={unc_test['data_uncertainty'].mean():.4f}, max={unc_test['data_uncertainty'].max():.4f}")
print(f"knowledge_uncertainty: mean={unc_test['knowledge_uncertainty'].mean():.4f}, max={unc_test['knowledge_uncertainty'].max():.4f}")
print(f"knowledge / total ratio (mean): {(unc_test['knowledge_uncertainty'] / np.maximum(unc_test['total_uncertainty'], 1e-12)).mean():.3f}")
"""))

CELLS.append(code("""# Per-cohort distribution of knowledge uncertainty
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, comp in zip(axes, ["total_uncertainty", "data_uncertainty", "knowledge_uncertainty"]):
    plot_uncertainty_distributions_by_cohort(cohorts, unc_test, component=comp, ax=ax)
fig.suptitle("Uncertainty per cohort (TP / FP / TN / FN)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOTS_DIR / "research_uncertainty_by_cohort.png", dpi=150)
plt.show()
print(f"Saved: {PLOTS_DIR / 'research_uncertainty_by_cohort.png'}")

# Quick summary table
unc_by_cohort = pd.DataFrame({
    "cohort": ["TP", "FP", "TN", "FN"],
    "n": [int((cohorts == c).sum()) for c in ["TP", "FP", "TN", "FN"]],
    "mean_p_mean": [float(unc_test["mean_p"][cohorts == c].mean()) if (cohorts == c).any() else np.nan for c in ["TP", "FP", "TN", "FN"]],
    "knowledge_unc_mean": [float(unc_test["knowledge_uncertainty"][cohorts == c].mean()) if (cohorts == c).any() else np.nan for c in ["TP", "FP", "TN", "FN"]],
    "knowledge_unc_median": [float(np.median(unc_test["knowledge_uncertainty"][cohorts == c])) if (cohorts == c).any() else np.nan for c in ["TP", "FP", "TN", "FN"]],
})
display(unc_by_cohort.style.format({"mean_p_mean": "{:.4f}", "knowledge_unc_mean": "{:.5f}", "knowledge_unc_median": "{:.5f}"}))
"""))

CELLS.append(code("""# 2D heatmap: empirical hit rate by (mean_p decile x knowledge_unc decile)
hr_grid = hit_rate_heatmap(
    test_y, unc_test["mean_p"], unc_test["knowledge_uncertainty"],
    n_bins_p=8, n_bins_unc=8,
)
fig, ax = plt.subplots(figsize=(11, 5.2))
plot_hit_rate_heatmap_2d(hr_grid, ax=ax, annotate=True)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_hit_rate_by_p_unc.png", dpi=150)
plt.show()

# Validation: at the highest p decile, hit rate should drop as unc rises
top_p_bin = hr_grid["p_bin"].max()
top_p_row = hr_grid[hr_grid["p_bin"] == top_p_bin].sort_values("unc_bin")
print(f"At top mean_p decile (p_bin={top_p_bin}, mean_p in [{top_p_row['p_lo'].iloc[0]:.3f}, {top_p_row['p_hi'].iloc[0]:.3f}]):")
for _, row in top_p_row.iterrows():
    print(f"  unc_bin {int(row['unc_bin'])}  unc in [{row['unc_lo']:.5f}, {row['unc_hi']:.5f}]  "
          f"n={int(row['n'])}  hit_rate={row['hit_rate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
"""))

CELLS.append(code("""# Variance reliability: predicted data_uncertainty vs observed (y - mean_p)^2
predicted_var_test = unc_test["data_uncertainty"]
observed_sq_err = (test_y - unc_test["mean_p"]) ** 2
var_rel = variance_reliability(predicted_var_test, observed_sq_err, n_bins=10)

fig, ax = plt.subplots(figsize=(7, 6))
plot_variance_reliability(var_rel, ax=ax)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_variance_reliability.png", dpi=150)
plt.show()

print()
print("Variance reliability per bin (perfect calibration -> ratio = 1):")
display(var_rel.style.format({
    "var_lo": "{:.5f}", "var_hi": "{:.5f}",
    "mean_predicted_var": "{:.5f}", "mean_observed_sq_error": "{:.5f}",
    "ratio_obs_over_pred": "{:.3f}",
}))
print()
mean_ratio = float(var_rel["ratio_obs_over_pred"].mean())
print(f"Mean ratio (observed / predicted) = {mean_ratio:.3f}")
print(f"  > 1 -> model is under-confident (predicted variance underestimates true)")
print(f"  < 1 -> model is over-confident (predicted variance overstates true)")
"""))

CELLS.append(code("""# Joint gate sweep: precision at every (tau_p, tau_unc) combination
p_thresholds = np.linspace(0.10, float(unc_test["mean_p"].max()) - 0.05, 12)
unc_thresholds = np.quantile(unc_test["knowledge_uncertainty"], np.linspace(0.1, 1.0, 8))

sweep_joint = joint_gate_sweep(
    test_y, unc_test["mean_p"], unc_test["knowledge_uncertainty"],
    p_thresholds=p_thresholds, unc_thresholds=unc_thresholds,
)
print(f"Joint-gate sweep: {len(sweep_joint)} (tau_p, tau_unc) cells")

fig, ax = plt.subplots(figsize=(11, 5.5))
plot_joint_gate_vs_scalar_p(sweep_joint, ax=ax)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "research_joint_gate.png", dpi=150)
plt.show()

# Comparison: at a fixed trade rate, what's the precision lift from the joint gate?
scalar_only = sweep_joint[sweep_joint["unc_threshold"] == sweep_joint["unc_threshold"].max()]
joint_strict = sweep_joint[sweep_joint["unc_threshold"] == sweep_joint["unc_threshold"].min()]
print()
print("Scalar-p baseline (no uncertainty filtering):")
display(scalar_only[["p_threshold", "trade_rate", "n_selected", "precision",
                     "precision_ci_low", "precision_ci_high"]].head(8).style.format({
    "p_threshold": "{:.3f}", "trade_rate": "{:.4f}", "precision": "{:.3f}",
    "precision_ci_low": "{:.3f}", "precision_ci_high": "{:.3f}",
}))
print()
print("Joint gate (strictest unc threshold = lowest decile):")
display(joint_strict[["p_threshold", "unc_threshold", "trade_rate", "n_selected",
                      "precision", "precision_ci_low", "precision_ci_high"]].head(8).style.format({
    "p_threshold": "{:.3f}", "unc_threshold": "{:.5f}", "trade_rate": "{:.4f}",
    "precision": "{:.3f}", "precision_ci_low": "{:.3f}", "precision_ci_high": "{:.3f}",
}))
"""))

CELLS.append(code("""# Save uncertainty bundle
import json
unc_bundle = {
    "config": {"K": VE_K, "n_bins_p": 8, "n_bins_unc": 8},
    "summary_stats": {
        "mean_p_mean": float(unc_test["mean_p"].mean()),
        "total_uncertainty_mean": float(unc_test["total_uncertainty"].mean()),
        "data_uncertainty_mean": float(unc_test["data_uncertainty"].mean()),
        "knowledge_uncertainty_mean": float(unc_test["knowledge_uncertainty"].mean()),
        "knowledge_over_total_mean_ratio": float(
            (unc_test["knowledge_uncertainty"] / np.maximum(unc_test["total_uncertainty"], 1e-12)).mean()
        ),
    },
    "by_cohort": unc_by_cohort.to_dict(orient="records"),
    "variance_reliability_mean_ratio": float(var_rel["ratio_obs_over_pred"].mean()),
    "joint_gate_strictest_unc": float(unc_thresholds[0]),
}
utils.save_json(ANALYTICS_DIR / "research_uncertainty_summary.json", unc_bundle)
print(f"Saved uncertainty summary: {ANALYTICS_DIR / 'research_uncertainty_summary.json'}")
print(json.dumps(unc_bundle, indent=2, default=str))
"""))

CELLS.append(md("""## 12. Production-trading audits

Sanity checks the model has to pass before it goes near a live execution path:

1. **Causal feature audit** — every feature in `feature_list` carries the `__f__` (frozen-up-to-now) or `__h__` (instantaneous) suffix; nothing leaks future information by name.
2. **Label-shuffle baseline** — under permutation of `y`, ROC-AUC concentrates on 0.5 and PR-AUC on the base rate. The real model's metrics must sit above this null distribution.
3. **Time-block permutation importance** — distinguishes features whose value carries the signal (drop under both shuffles) from features that ride on time-of-row only (drop only under across-block shuffle, suggesting time leak).
4. **Decision turnover** — at the operating threshold, what's the flip rate of the binary entry decision? Frequent flips destroy net-of-cost performance.
5. **Deflated Sharpe** (Bailey & Lopez de Prado) — adjusts the per-trade Sharpe for selection bias under `n_trials` HPO attempts and for non-normality of returns.
6. **Half-vs-half drift audit** — fits a model on the first half of train and another on the second; same predictions on test ⇒ covariate shift only; disagreement ⇒ concept drift in the gap between halves.
"""))

CELLS.append(code("""from src.analytics.audits import (
    causal_feature_audit,
    label_shuffle_baseline,
    time_block_permutation_importance,
    decision_turnover,
    deflated_sharpe,
    expected_max_sharpe,
    half_vs_half_drift_audit,
    plot_label_shuffle_baseline,
    plot_time_block_permutation,
    plot_half_vs_half_scatter,
    plot_decision_turnover_runs,
)
from sklearn.metrics import average_precision_score, roc_auc_score
"""))

# 12.1 — Causal feature audit
CELLS.append(md("""### 12.1 Causal feature audit

Static naming-convention check on `feature_list`. Pass criterion: no suspect tokens, no unmatched names. CI-runnable.
"""))

CELLS.append(code("""causal_res = causal_feature_audit(feature_list)
print(f"features audited: {causal_res.n_features}")
print(f"  causal (__f__ or __h__): {causal_res.n_causal}")
print(f"  suspect tokens: {causal_res.n_suspect}")
print(f"  unmatched (no known suffix): {len(causal_res.unmatched)}")
print(f"  PASS: {causal_res.passed}")
if causal_res.suspect:
    print("\\nSUSPECT TOKENS (first 20):")
    for f in causal_res.suspect[:20]:
        print(f"  {f}")
if causal_res.unmatched:
    print("\\nUNMATCHED (first 20):")
    for f in causal_res.unmatched[:20]:
        print(f"  {f}")
"""))

# 12.2 — Label-shuffle baseline
CELLS.append(md("""### 12.2 Label-shuffle baseline

Permute `y_test` 1000 times, recompute ROC-AUC and PR-AUC each time. The real model's metrics should sit comfortably above the 97.5% quantile of the shuffle distribution; if not, the model is statistically indistinguishable from random at this `n`.
"""))

CELLS.append(code("""y_t = test_cache["y"].to_numpy()
p_t = test_cache["p"].to_numpy()

shuffle_out = label_shuffle_baseline(y_t, p_t, n_shuffles=1000, random_seed=0)

real_roc = float(roc_auc_score(y_t, p_t))
real_pr = float(average_precision_score(y_t, p_t))
print(f"REAL test ROC-AUC: {real_roc:.4f}    shuffle median: {shuffle_out['roc_auc'].point:.4f}    "
      f"shuffle 95% CI: [{shuffle_out['roc_auc'].ci_low:.4f}, {shuffle_out['roc_auc'].ci_high:.4f}]")
print(f"REAL test PR-AUC : {real_pr:.4f}    shuffle median: {shuffle_out['pr_auc'].point:.4f}    "
      f"shuffle 95% CI: [{shuffle_out['pr_auc'].ci_low:.4f}, {shuffle_out['pr_auc'].ci_high:.4f}]")
print(f"base rate: {shuffle_out['base_rate'].point:.4f}")

passes_roc = real_roc > shuffle_out["roc_auc"].ci_high
passes_pr = real_pr > shuffle_out["pr_auc"].ci_high
print(f"\\n  ROC-AUC > shuffle 97.5%? {passes_roc}")
print(f"  PR-AUC  > shuffle 97.5%? {passes_pr}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
plot_label_shuffle_baseline(real_roc, shuffle_out["roc_auc"], metric_name="ROC-AUC", ax=axes[0])
plot_label_shuffle_baseline(real_pr, shuffle_out["pr_auc"], metric_name="PR-AUC",
                            base_rate=shuffle_out["base_rate"].point, ax=axes[1])
plt.tight_layout()
plt.savefig(PLOTS_DIR / "research_label_shuffle.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# 12.3 — Time-block permutation importance
CELLS.append(md("""### 12.3 Time-block permutation importance

For each candidate feature, shuffle within blocks vs across blocks. A feature with `drop_across >> drop_within` derives most of its value from time-of-row (a smell), not from local feature value.

To keep this fast we run on the top-25 features by SHAP importance from Phase 4, with `n_blocks=8` and `n_repeats=2`.
"""))

CELLS.append(code("""# Top-25 features by Phase 4 signed effect size (the strongest FP-vs-FN drivers)
top_feats_for_perm = eff.head(25)["feature"].tolist()
test_perm_df = test_df.copy()
perm_imp = time_block_permutation_importance(
    model,
    test_perm_df,
    feature_list,
    metric="pr_auc",
    n_blocks=8,
    features_to_test=top_feats_for_perm,
    n_repeats=2,
    random_seed=0,
)
print(f"base PR-AUC (test): {perm_imp.attrs['base_score']:.4f}")
print()
print(perm_imp.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 8))
plot_time_block_permutation(perm_imp, top_n=20, ax=ax)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "research_time_block_permutation.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# 12.4 — Decision turnover
CELLS.append(md("""### 12.4 Decision turnover at the operating threshold

Binary entry decision `act_t = (p_t >= τ)` at the operating threshold. High flip rate ⇒ the model toggles frequently across the threshold ⇒ trade churn under cost. A hysteresis band (`τ_in > τ_out`) is the standard mitigation.
"""))

CELLS.append(code("""ts_t = test_cache["ts"].to_numpy()
turnover_threshold = float(COHORT_THRESHOLD)

turnover = decision_turnover(p_t, threshold=turnover_threshold, ts=ts_t)
print(json.dumps(turnover, indent=2, default=str))

# Compare across a small grid of thresholds
turnover_grid = pd.DataFrame([
    decision_turnover(p_t, threshold=t, ts=ts_t)
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
])
turnover_grid["threshold"] = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
print()
print(turnover_grid[["threshold", "trade_rate", "flip_rate", "lag1_autocorr",
                     "mean_run_length_active", "mean_run_length_idle"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
plot_decision_turnover_runs(p_t, threshold=turnover_threshold, ax=ax)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "research_decision_turnover.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# 12.5 — Deflated Sharpe
CELLS.append(md("""### 12.5 Deflated Sharpe (Bailey & Lopez de Prado)

Per-trade returns at the operating threshold, with the **same outcome model** used in Phase 3 (gain on hit = `phi`, loss on miss = `phi`, cost = 5 bps). DSR > 0.95 is the conventional threshold for "significantly better than the best of `n_trials` random backtests, after correcting for non-normality."

We report DSR at three trial counts (`n_trials = 1, 100, 1000`) so you can see the selection-bias correction directly.
"""))

CELLS.append(code("""# Per-trade realized log-returns at the operating threshold (no realized-return col yet, so use phi)
sel = (p_t >= turnover_threshold)
y_sel = y_t[sel]
phi_t = test_cache["phi"].to_numpy()[sel]
gain = phi_t
loss = phi_t  # symmetric outcome — conservative
cost = 0.0005  # 5 bps fees + half-spread
per_trade = np.where(y_sel == 1, gain, -loss) - cost
print(f"trades selected: {len(per_trade)}    hit rate: {y_sel.mean():.3f}    "
      f"mean per-trade return: {per_trade.mean():+.4f}")

dsr_results = {}
for n_trials in [1, 100, 1000]:
    out = deflated_sharpe(per_trade, n_trials=n_trials)
    dsr_results[n_trials] = out
    print(f"\\nn_trials = {n_trials}")
    print(f"  Sharpe (per trade)        : {out['sharpe']:+.4f}")
    print(f"  E[max Sharpe] (selection) : {out['expected_max_sharpe']:+.4f}")
    print(f"  z statistic               : {out['deflated_sharpe_z']:+.4f}")
    print(f"  DSR (Pr Sharpe > E[max])  : {out['deflated_sharpe_prob']:.4f}")
    print(f"  skew = {out['skew']:+.3f}    kurt = {out['kurtosis']:.3f}    n = {out['n']}")
"""))

# 12.6 — Half-vs-half drift audit
CELLS.append(md("""### 12.6 Half-vs-half drift audit

Splits `train_recent` chronologically into two halves; trains a model on each (with smaller iterations to keep this fast); predicts on test. Same predictions ⇒ covariate shift only; disagreement ⇒ concept drift between the halves.
"""))

CELLS.append(code("""# Use halved iterations to keep this audit fast (we only need the disagreement structure)
half_params = research_train_params(iterations=600, depth=6, verbose=0,
                                    early_stopping_rounds=100, random_seed=42)
half_params2 = research_train_params(iterations=600, depth=6, verbose=0,
                                     early_stopping_rounds=100, random_seed=43)

t0 = time.time()
half_res = half_vs_half_drift_audit(
    train_recent, val_df, test_df, feature_list,
    threshold=COHORT_THRESHOLD,
    train_params_first=half_params,
    train_params_second=half_params2,
)
elapsed = time.time() - t0
print(f"trained two half-models in {elapsed:.1f}s")
print(f"  n_first = {half_res.n_first}    n_second = {half_res.n_second}")
print(f"  Spearman rank corr (test predictions): {half_res.spearman_corr:.4f}")
print(f"  Pearson         corr (test predictions): {half_res.pearson_corr:.4f}")
print(f"  mean |Δp|: {half_res.mean_abs_diff:.4f}    median |Δp|: {half_res.median_abs_diff:.4f}")
print(f"  decisions disagree at τ={half_res.threshold}: "
      f"{half_res.n_disagree_at_threshold} / {len(half_res.p_first)} "
      f"({half_res.n_disagree_at_threshold / len(half_res.p_first) * 100:.1f}%)")
print()
print(f"first-half model on test:  ROC-AUC = {half_res.metrics_first['roc_auc']:.4f}    "
      f"PR-AUC = {half_res.metrics_first['pr_auc']:.4f}")
print(f"second-half model on test: ROC-AUC = {half_res.metrics_second['roc_auc']:.4f}    "
      f"PR-AUC = {half_res.metrics_second['pr_auc']:.4f}")

fig, ax = plt.subplots(figsize=(7, 6.5))
plot_half_vs_half_scatter(half_res, ax=ax)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "research_half_vs_half.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# 12.7 — Persist the audit bundle
CELLS.append(md("""### 12.7 Persist the audit bundle

Save all audit outputs to a single JSON file for posterity.
"""))

CELLS.append(code("""audit_bundle = {
    "causal_audit": {
        "n_features": causal_res.n_features,
        "n_causal": causal_res.n_causal,
        "n_suspect": causal_res.n_suspect,
        "n_unmatched": len(causal_res.unmatched),
        "passed": causal_res.passed,
        "suspect": causal_res.suspect,
        "unmatched_first_20": causal_res.unmatched[:20],
    },
    "label_shuffle": {
        "real_roc_auc": real_roc,
        "real_pr_auc": real_pr,
        "shuffle_roc_auc_median": shuffle_out["roc_auc"].point,
        "shuffle_roc_auc_ci": [shuffle_out["roc_auc"].ci_low, shuffle_out["roc_auc"].ci_high],
        "shuffle_pr_auc_median": shuffle_out["pr_auc"].point,
        "shuffle_pr_auc_ci": [shuffle_out["pr_auc"].ci_low, shuffle_out["pr_auc"].ci_high],
        "base_rate": shuffle_out["base_rate"].point,
        "passes_roc": bool(passes_roc),
        "passes_pr": bool(passes_pr),
    },
    "time_block_permutation": {
        "metric": perm_imp.attrs["metric"],
        "base_score": perm_imp.attrs["base_score"],
        "n_blocks": perm_imp.attrs["n_blocks"],
        "top_5_by_drop_across": perm_imp.head(5).to_dict(orient="records"),
    },
    "decision_turnover": turnover,
    "deflated_sharpe": {
        f"n_trials_{k}": v for k, v in dsr_results.items()
    },
    "half_vs_half": {
        "n_first": half_res.n_first,
        "n_second": half_res.n_second,
        "spearman_corr": half_res.spearman_corr,
        "pearson_corr": half_res.pearson_corr,
        "mean_abs_diff": half_res.mean_abs_diff,
        "median_abs_diff": half_res.median_abs_diff,
        "n_disagree_at_threshold": half_res.n_disagree_at_threshold,
        "threshold": half_res.threshold,
        "metrics_first": half_res.metrics_first,
        "metrics_second": half_res.metrics_second,
    },
}
utils.save_json(ANALYTICS_DIR / "research_audits_bundle.json", audit_bundle)
print(f"Saved audit bundle: {ANALYTICS_DIR / 'research_audits_bundle.json'}")
print(json.dumps(audit_bundle, indent=2, default=str))
"""))

CELLS.append(md("""---
End of Phase 0+1+2+2b+3+4+5+6 deliverables.

The notebook is now a complete bootstrap-certified offline-study harness covering: certified core metrics + per-regime breakdown, banded ROC / PR / calibration curves, rolling-window degradation + Murphy decomposition + PSI/KS + Page-Hinkley, conditional precision heatmap, threshold sweep + EV vs trades-per-day + Kelly + partial AUC + lift, SHAP cohort decomposition (TP/FP/TN/FN) with discriminative-SHAP, virtual-ensemble uncertainty (decomposition + hit-rate heatmap + joint gate + variance reliability), and the production-trading audit suite (causal, shuffle, time-block permutation, turnover, deflated Sharpe, half-vs-half drift).
"""))


def main() -> None:
    cells_with_ids = [{**cell, "id": f"cell-{i:02d}"} for i, cell in enumerate(CELLS)]
    nb = {
        "cells": cells_with_ids,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path("notebooks/04_offline_study.ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
