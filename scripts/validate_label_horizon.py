"""Phase 0 validation: relabel the existing 1-min feature dataset at longer M,
co-scale phi by sqrt(M/M_baseline), retrain CatBoost, and report block-bootstrap
metrics with block_size=M_test.

The features stay frozen, including the ~5-10% barrier-aware family that
encodes the OLD (M=20, PHI=0.0025) regime. That is a CONSERVATIVE bias:
a passing M_test under this handicap is a stronger signal than a passing
M_test with regenerated barrier-aware features. If a horizon fails here,
re-run Phase 4b-style feature regeneration before concluding NO-GO.

Outputs:
  data/model_dataset/horizon_sweep/metrics_M{M}.json
  data/model_dataset/horizon_sweep/predictions_M{M}.parquet
  data/model_dataset/horizon_sweep/summary.json   (cross-M comparison)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import utils  # noqa: E402
from src.analytics.bootstrap import bootstrap_metric  # noqa: E402
from src.analytics.sampling import compute_uniqueness_weights  # noqa: E402
from src.features.boundary import construct_labels_pl  # noqa: E402

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DATASET_DIR = ROOT / "data" / "model_dataset"
DATASET_PATH = DATASET_DIR / "dataset_1min.parquet"
FEATURE_LIST_PATH = DATASET_DIR / "feature_list_1min.json"
RAW_PATH = ROOT / "data" / "raw_data" / "klines_1m.parquet"
BASELINE_METRICS_PATH = DATASET_DIR / "analytics" / "research_metrics_1min.json"
OUT_DIR = DATASET_DIR / "horizon_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Constants (must match notebook 03_train_model.ipynb exactly)
# -----------------------------------------------------------------------------
M_BASELINE = 20
PHI_BASELINE = 0.0025
C_FIXED = 0.0023  # cost component — does NOT scale with M (per-trade cost)
# eta(M) = phi(M) - C; phi(M) = PHI_BASELINE * sqrt(M / M_BASELINE)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
B_BOOT = 500
SEED = 42

# Match notebook 03 hyperparameters bit-for-bit so the only changing input is
# (M, phi) and its downstream effects on labels/weights/embargo/block_size.
LEGACY_BEST_PARAMS = {
    "learning_rate": 0.01,
    "l2_leaf_reg": 0.1,
    "depth": 6,
    "rsm": 1.0,
    "subsample": 1.0,
    "mvs_reg": 3.0,
    "diffusion_temperature": 10000,
    "random_strength": 0.0,
}

# The horizons to validate. M=60 first because it's the user's stated target;
# 45 and 90 bracket it so we can interpolate the sweet spot.
HORIZONS = [60, 45, 90]


def _ci_block(fn, y, p, M_test: int) -> dict:
    """Block-bootstrap CI for a metric, block_size=M_test."""
    res = bootstrap_metric(
        fn, y, p, B=B_BOOT, stratify=False, seed=SEED, block_size=int(M_test)
    )
    return {
        "point": float(res.point),
        "ci_low": float(res.ci_low),
        "ci_high": float(res.ci_high),
        "ci_width": float(res.ci_high - res.ci_low),
    }


def _metrics(name: str, sub: pd.DataFrame, p: np.ndarray, M_test: int) -> dict:
    y = sub["y"].astype(int).to_numpy()
    return {
        "split": name,
        "n_samples": int(len(sub)),
        "base_rate": float(y.mean()),
        "roc_auc": _ci_block(lambda y_, p_: float(roc_auc_score(y_, p_)), y, p, M_test),
        "pr_auc": _ci_block(
            lambda y_, p_: float(average_precision_score(y_, p_)), y, p, M_test
        ),
        "log_loss": _ci_block(
            lambda y_, p_: float(log_loss(y_, p_, labels=[0, 1])), y, p, M_test
        ),
        "brier": _ci_block(
            lambda y_, p_: float(brier_score_loss(y_, p_)), y, p, M_test
        ),
    }


def relabel_dataset(
    raw_pl: pl.DataFrame,
    df_features: pd.DataFrame,
    M_test: int,
    phi_test: float,
) -> pd.DataFrame:
    """Build a relabeled dataset: features frozen, labels regenerated at M_test."""
    eta_test = phi_test - C_FIXED
    if eta_test < 0:
        raise ValueError(
            f"phi_test={phi_test:.6f} is below C={C_FIXED}; eta would be negative"
        )

    # Build full-raw-range df_boundaries with k = bar position (matches what
    # the pipeline does at 1-min cadence: pl.int_range(pl.len()) as 'k').
    # construct_labels_pl uses iteration index, not the k column, so the
    # df_boundaries row at position i MUST correspond to raw bar i.
    df_boundaries = raw_pl.select(pl.int_range(pl.len()).alias("k"))
    df_labeled = construct_labels_pl(
        df_boundaries=df_boundaries,
        df_raw=raw_pl,
        M=int(M_test),
        eta=float(eta_test),
        c=float(C_FIXED),
        bar_stride=1,
        barrier_source="high",
    )

    # Convert to pandas, keep only mature labels (last M_test-1 rows have NaN y).
    df_labels_pd = df_labeled.to_pandas()
    mature = df_labels_pd["y"].notna()
    df_labels_pd = df_labels_pd.loc[mature, ["k", "y", "m_k", "tau_k", "phi"]].copy()
    df_labels_pd["y"] = df_labels_pd["y"].astype(int)

    # Inner-join with existing feature dataset on k. This drops the last
    # (M_test - M_baseline) rows from the dataset's tail since they no longer
    # have mature labels at the longer horizon.
    df_features_keys = df_features[["k"]].copy()
    df_features_keys["__pos__"] = np.arange(len(df_features), dtype=np.int64)
    merged = df_features_keys.merge(df_labels_pd, on="k", how="inner")
    if not merged["__pos__"].is_monotonic_increasing:
        raise RuntimeError(
            "Merge produced non-monotonic positions; alignment assumption broke."
        )

    # Pull the matching feature rows by position and stitch on the new labels.
    pos = merged["__pos__"].to_numpy()
    df_out = df_features.iloc[pos].copy()
    # Replace the old label columns with the new ones (column order preserved).
    df_out["y"] = merged["y"].to_numpy()
    df_out["m_k"] = merged["m_k"].to_numpy()
    df_out["tau_k"] = merged["tau_k"].to_numpy()
    df_out["phi"] = float(phi_test)

    # Recompute the asymmetric barrier-distance weight from new m_k + new phi.
    # Same scheme as notebook 02 (use_dist=True, use_time=False, normalize=False).
    w_combined, w_dist, w_time, w_info = utils.compute_training_weights(
        m_k=df_out["m_k"].to_numpy(dtype=float),
        phi=float(phi_test),
    )
    df_out["weight"] = w_combined
    df_out = df_out.reset_index(drop=True)
    return df_out, w_info


def run_horizon(
    raw_pl: pl.DataFrame,
    df_features: pd.DataFrame,
    feature_list: list[str],
    M_test: int,
) -> dict:
    phi_test = PHI_BASELINE * float(np.sqrt(M_test / M_BASELINE))
    eta_test = phi_test - C_FIXED
    embargo_k = utils.recommended_embargo_for_cadence(
        "1min", base_embargo=60, M=M_test
    )

    print()
    print("=" * 78)
    print(
        f"M={M_test}  PHI={phi_test:.6f}  C={C_FIXED}  ETA={eta_test:.6f}  "
        f"embargo_k={embargo_k}"
    )
    print("=" * 78)

    # ---- Relabel + reweight -------------------------------------------------
    t0 = time.perf_counter()
    df_work, w_info = relabel_dataset(
        raw_pl=raw_pl, df_features=df_features, M_test=M_test, phi_test=phi_test
    )
    dt_relabel = time.perf_counter() - t0
    base_rate = float(df_work["y"].mean())
    print(
        f"[relabel] {dt_relabel:.1f}s  n={len(df_work):,}  base_rate={base_rate:.4f}  "
        f"phi={phi_test:.6f}  weight range "
        f"[{df_work['weight'].min():.4f}, {df_work['weight'].max():.4f}]  "
        f"effective_n={w_info['combined']['effective_n']:.1f}"
    )

    # ---- Split (with cadence-aware embargo at M_test) -----------------------
    df_work = df_work.sort_values("k").reset_index(drop=True)
    train_df, val_df, test_df = utils.chronological_split_with_embargo(
        df_work, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, embargo_k=embargo_k
    )
    gap_tv = int(val_df["k"].min() - train_df["k"].max())
    gap_vt = int(test_df["k"].min() - val_df["k"].max())
    if gap_tv < M_test or gap_vt < M_test:
        raise RuntimeError(
            f"Embargo insufficient at M={M_test}: train|val={gap_tv}, val|test={gap_vt}"
        )
    print(
        f"[split] train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}  "
        f"gaps train|val={gap_tv} val|test={gap_vt} (>= M={M_test})"
    )

    # ---- Combined sample weights: barrier_distance * uniqueness -------------
    u_train = compute_uniqueness_weights(
        n_rows=len(train_df), M=int(M_test), bar_stride=1, normalize=False
    )
    u_val = compute_uniqueness_weights(
        n_rows=len(val_df), M=int(M_test), bar_stride=1, normalize=False
    )
    train_weights = train_df["weight"].to_numpy(dtype=float) * u_train
    val_weights = val_df["weight"].to_numpy(dtype=float) * u_val
    for name, w in [
        ("u_train", u_train), ("u_val", u_val),
        ("train_weights", train_weights), ("val_weights", val_weights),
    ]:
        if not np.isfinite(w).all() or (w <= 0).any():
            raise RuntimeError(f"{name} has bad values: min={w.min()}, max={w.max()}")
    print(
        f"[weights] u_train mean={u_train.mean():.6f}  (1/M={1.0/M_test:.6f})  "
        f"combined train sum={train_weights.sum():.1f}  ESS={u_train.sum():.1f}"
    )

    # ---- Train CatBoost (identical hyperparameters to notebook 03) ----------
    cb_params = {
        **utils.CB_FIXED_PARAMS,
        **LEGACY_BEST_PARAMS,
        "iterations": 2000,
        "verbose": 100,
    }
    if not cb_params.get("has_time", False):
        raise RuntimeError("CB_FIXED_PARAMS lost has_time=True; refuse to train.")

    X_train = train_df[feature_list].to_numpy()
    X_val = val_df[feature_list].to_numpy()
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        raise RuntimeError("NaN/inf in X_train; dataset deformed.")
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        raise RuntimeError("NaN/inf in X_val; dataset deformed.")

    train_pool = Pool(
        data=X_train,
        label=train_df["y"].astype(int).to_numpy(),
        timestamp=train_df["k"].to_numpy(dtype=np.uint32),
        weight=train_weights,
        feature_names=list(feature_list),
    )
    val_pool = Pool(
        data=X_val,
        label=val_df["y"].astype(int).to_numpy(),
        timestamp=val_df["k"].to_numpy(dtype=np.uint32),
        weight=val_weights,
        feature_names=list(feature_list),
    )

    model = CatBoostClassifier(**cb_params)
    t0 = time.perf_counter()
    print(f"[fit] start at {time.strftime('%H:%M:%S')} ...")
    model.fit(train_pool, eval_set=val_pool)
    fit_dt = time.perf_counter() - t0
    best_iter = int(model.get_best_iteration())
    tree_count = int(model.tree_count_)
    print(
        f"[fit] done in {fit_dt:.1f}s ({fit_dt / 60:.1f} min); "
        f"best_iter={best_iter}; tree_count={tree_count}"
    )

    # Save the trained model for later SHAP / feature-importance / Phase 3 work.
    model_path = OUT_DIR / f"catboost_model_M{M_test}.cbm"
    model.save_model(str(model_path))
    print(f"[save] {model_path}")

    # Free the training pools before scoring to keep memory tight.
    del train_pool, val_pool, X_train
    gc.collect()

    # ---- Score + metrics ----------------------------------------------------
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(test_df[feature_list].to_numpy())[:, 1]
    del X_val
    gc.collect()
    val_m = _metrics("val", val_df, p_val, M_test)
    test_m = _metrics("test", test_df, p_test, M_test)
    print(f"[val]  ROC-AUC {val_m['roc_auc']['point']:.4f} "
          f"[{val_m['roc_auc']['ci_low']:.4f}, {val_m['roc_auc']['ci_high']:.4f}]  "
          f"PR-AUC {val_m['pr_auc']['point']:.4f} "
          f"[{val_m['pr_auc']['ci_low']:.4f}, {val_m['pr_auc']['ci_high']:.4f}]")
    print(f"[test] ROC-AUC {test_m['roc_auc']['point']:.4f} "
          f"[{test_m['roc_auc']['ci_low']:.4f}, {test_m['roc_auc']['ci_high']:.4f}]  "
          f"PR-AUC {test_m['pr_auc']['point']:.4f} "
          f"[{test_m['pr_auc']['ci_low']:.4f}, {test_m['pr_auc']['ci_high']:.4f}]")

    # ---- Persist ------------------------------------------------------------
    out = {
        "M": int(M_test),
        "PHI": float(phi_test),
        "C": float(C_FIXED),
        "ETA": float(eta_test),
        "embargo_k": int(embargo_k),
        "base_rate_full": base_rate,
        "best_iter": best_iter,
        "tree_count": tree_count,
        "fit_seconds": float(fit_dt),
        "relabel_seconds": float(dt_relabel),
        "n_total": int(len(df_work)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "block_bootstrap_size": int(M_test),
        "u_train_ess": float(u_train.sum()),
        "u_val_ess": float(u_val.sum()),
        "val": val_m,
        "test": test_m,
        "cb_params_overrides": LEGACY_BEST_PARAMS,
        "feature_caveat": (
            "barrier_aware family encodes M=20, PHI=0.0025 (stale at M_test!=20); "
            "this conservatively handicaps M_test>20 in this experiment."
        ),
        "git_sha": utils.get_git_sha(),
    }
    out_path = OUT_DIR / f"metrics_M{M_test}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[save] {out_path}")

    pred_path = OUT_DIR / f"predictions_M{M_test}.parquet"
    pred_df = pd.DataFrame(
        {
            "k": np.concatenate(
                [val_df["k"].to_numpy(), test_df["k"].to_numpy()]
            ),
            "y": np.concatenate(
                [
                    val_df["y"].astype(int).to_numpy(),
                    test_df["y"].astype(int).to_numpy(),
                ]
            ),
            "m_k": np.concatenate(
                [val_df["m_k"].to_numpy(), test_df["m_k"].to_numpy()]
            ),
            "p": np.concatenate([p_val, p_test]),
            "split": ["val"] * len(val_df) + ["test"] * len(test_df),
        }
    )
    pred_df.to_parquet(pred_path, index=False)
    print(f"[save] {pred_path}")

    return out


def _load_baseline() -> dict:
    """Load M=20 baseline metrics already on disk (from notebook 03)."""
    with open(BASELINE_METRICS_PATH) as f:
        baseline = json.load(f)
    return baseline


def _print_summary(baseline: dict, per_M: list[dict]) -> None:
    """Print a tidy cross-M comparison and write summary.json."""
    print()
    print("=" * 78)
    print("SUMMARY: M=20 baseline vs validation sweep")
    print("=" * 78)

    def _row(label: str, m: dict, split: str) -> str:
        roc = m[split]["roc_auc"]
        pr = m[split]["pr_auc"]
        return (
            f"  {label:<12} n={m[split]['n_samples']:>7,}  "
            f"base={m[split]['base_rate']:.4f}  "
            f"ROC-AUC {roc['point']:.4f} [{roc['ci_low']:.4f},{roc['ci_high']:.4f}] "
            f"(w={roc['ci_width']:.4f})  "
            f"PR-AUC {pr['point']:.4f} [{pr['ci_low']:.4f},{pr['ci_high']:.4f}] "
            f"(w={pr['ci_width']:.4f})"
        )

    base_val_roc = baseline["val"]["roc_auc"]
    base_val_pr = baseline["val"]["pr_auc"]
    base_test_roc = baseline["test"]["roc_auc"]
    base_test_pr = baseline["test"]["pr_auc"]
    baseline_compact = {
        "val": {
            "n_samples": baseline["val"]["n_samples"],
            "base_rate": baseline["val"]["base_rate"],
            "roc_auc": {
                "point": base_val_roc["point"],
                "ci_low": base_val_roc["block_ci"][0],
                "ci_high": base_val_roc["block_ci"][1],
                "ci_width": base_val_roc["block_width"],
            },
            "pr_auc": {
                "point": base_val_pr["point"],
                "ci_low": base_val_pr["block_ci"][0],
                "ci_high": base_val_pr["block_ci"][1],
                "ci_width": base_val_pr["block_width"],
            },
        },
        "test": {
            "n_samples": baseline["test"]["n_samples"],
            "base_rate": baseline["test"]["base_rate"],
            "roc_auc": {
                "point": base_test_roc["point"],
                "ci_low": base_test_roc["block_ci"][0],
                "ci_high": base_test_roc["block_ci"][1],
                "ci_width": base_test_roc["block_width"],
            },
            "pr_auc": {
                "point": base_test_pr["point"],
                "ci_low": base_test_pr["block_ci"][0],
                "ci_high": base_test_pr["block_ci"][1],
                "ci_width": base_test_pr["block_width"],
            },
        },
    }

    print("VAL split:")
    print(_row("M=20 (base)", baseline_compact, "val"))
    for m in per_M:
        print(_row(f"M={m['M']}", m, "val"))
    print()
    print("TEST split:")
    print(_row("M=20 (base)", baseline_compact, "test"))
    for m in per_M:
        print(_row(f"M={m['M']}", m, "test"))

    # GO/NO-GO decision rule from the plan: lower CI bound at M_test beats the
    # POINT estimate at M=20 (apples-to-apples for the same metric/split).
    print()
    print("GO/NO-GO test (M_test lower CI vs M=20 point):")
    for split in ("val", "test"):
        for m in per_M:
            for metric in ("roc_auc", "pr_auc"):
                base_pt = baseline_compact[split][metric]["point"]
                test_lo = m[split][metric]["ci_low"]
                verdict = "GO " if test_lo >= base_pt else "no "
                print(
                    f"  [{verdict}] {split:>4}/{metric:<7}  "
                    f"M={m['M']:>2} CI_low={test_lo:.4f} vs M=20 point={base_pt:.4f}  "
                    f"(delta {test_lo - base_pt:+.4f})"
                )

    summary = {
        "baseline_M": M_BASELINE,
        "baseline_PHI": PHI_BASELINE,
        "horizons": [int(m["M"]) for m in per_M],
        "baseline_metrics": baseline_compact,
        "per_M_metrics": per_M,
    }
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[save] {summary_path}")


def main() -> None:
    t_total = time.perf_counter()
    print(f"ROOT: {ROOT}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Raw: {RAW_PATH}")

    # Load raw (only need close + high for label construction, but read all
    # for simplicity).
    raw_pl = pl.read_parquet(RAW_PATH)
    print(f"[load] raw bars: {raw_pl.height:,}")

    # Load feature dataset (we keep everything; relabeling replaces y/m_k/tau_k/phi/weight).
    df_features = pd.read_parquet(DATASET_PATH)
    feature_list = utils.load_json(FEATURE_LIST_PATH)
    print(
        f"[load] dataset: {len(df_features):,} rows x {len(df_features.columns)} cols, "
        f"{len(feature_list):,} features"
    )

    # Sanity: dataset k must be sorted and start at N_WARMUP, end before raw.height.
    if not df_features["k"].is_monotonic_increasing:
        df_features = df_features.sort_values("k").reset_index(drop=True)
    if int(df_features["k"].iloc[0]) < utils.N_WARMUP:
        raise RuntimeError(
            f"Dataset starts at k={df_features['k'].iloc[0]} < N_WARMUP="
            f"{utils.N_WARMUP}; alignment assumption broke."
        )

    per_M = []
    for M_test in HORIZONS:
        m = run_horizon(
            raw_pl=raw_pl, df_features=df_features, feature_list=feature_list,
            M_test=M_test,
        )
        per_M.append(m)
        gc.collect()

    baseline = _load_baseline()
    _print_summary(baseline, per_M)
    print(f"\nTotal wall time: {time.perf_counter() - t_total:.1f}s "
          f"({(time.perf_counter() - t_total) / 60:.1f} min)")


if __name__ == "__main__":
    main()
