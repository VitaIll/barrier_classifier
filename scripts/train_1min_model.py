"""Train a CatBoost model on the 1-min cadence dataset and produce a
predictions cache for the strategy/analytics layer.

Uses cadence-aware helpers:
- ``recommended_embargo_for_cadence('1min')`` for the chronological split
- ``recommended_time_discount_delta_for_cadence('1min')`` for training weights

Output:
    data/model_dataset/catboost_model_1min.cbm
    data/model_dataset/research_predictions_1min.parquet
    data/model_dataset/analytics/research_metrics_1min.json

HPO is intentionally skipped here — at 500k rows × 1243 features each Optuna
trial would take many minutes. We use sensible defaults from the
boundary-cadence research model; HPO can come later if the headline
numbers warrant the compute.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

sys.path.insert(0, ".")

from src import utils
from src.analytics.bootstrap import bootstrap_metric
from src.analytics.sampling import compute_uniqueness_weights
from src.features.config import M, PHI


# ----- Output paths ---------------------------------------------------------
DATASET_DIR = "data/model_dataset"
DATASET_PATH = f"{DATASET_DIR}/dataset_1min.parquet"
FEATURE_LIST_PATH = f"{DATASET_DIR}/feature_list_1min.json"
MODEL_PATH = f"{DATASET_DIR}/catboost_model_1min.cbm"
PREDICTIONS_PATH = f"{DATASET_DIR}/research_predictions_1min.parquet"
METRICS_PATH = f"{DATASET_DIR}/analytics/research_metrics_1min.json"
META_PATH = f"{DATASET_DIR}/dataset_metadata_1min.json"


# ----- Training config ------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Cadence-aware embargo: at 1-min cadence, 60 boundary-equivalent rows = 60*M = 1200
EMBARGO_K = utils.recommended_embargo_for_cadence("1min", base_embargo=60, M=M)

# CatBoost params: fast-iter defaults. Skip Optuna here.
CB_PARAMS = {
    "iterations": 1500,
    "learning_rate": 0.02,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "random_seed": 42,
    "early_stopping_rounds": 150,
    "border_count": 128,
    "thread_count": -1,
    "use_best_model": True,
    "allow_writing_files": False,
    "verbose": 200,
    "posterior_sampling": True,
}


def _bps(x: float) -> str:
    return f"{x * 100:+.3f}%"


def main() -> None:
    print(f"Embargo at 1-min cadence: {EMBARGO_K} rows  ({EMBARGO_K / M:.1f} boundaries = {EMBARGO_K / 60:.1f} hours)")

    # ----- Load dataset + feature list ---------------------------------------
    print(f"Loading {DATASET_PATH} ...")
    t0 = time.perf_counter()
    df = pd.read_parquet(DATASET_PATH)
    feature_list = utils.load_json(FEATURE_LIST_PATH)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s: {len(df):,} rows x {len(df.columns)} cols")
    print(f"  features in feature_list: {len(feature_list):,}")
    print(f"  base rate y = {float(df['y'].mean()):.4f}")

    # Sanity: every feature exists
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise ValueError(f"feature_list missing from dataset: {missing[:5]}...")

    # ----- Chronological split with cadence-aware embargo --------------------
    df = df.sort_values("k").reset_index(drop=True)
    train_df, val_df, test_df = utils.chronological_split_with_embargo(
        df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, embargo_k=EMBARGO_K
    )
    print()
    print(f"Splits:")
    print(f"  train: n={len(train_df):,}  k=[{train_df['k'].min()}, {train_df['k'].max()}]  base_rate={float(train_df['y'].mean()):.4f}")
    print(f"  val  : n={len(val_df):,}  k=[{val_df['k'].min()}, {val_df['k'].max()}]  base_rate={float(val_df['y'].mean()):.4f}")
    print(f"  test : n={len(test_df):,}  k=[{test_df['k'].min()}, {test_df['k'].max()}]  base_rate={float(test_df['y'].mean()):.4f}")
    # Verify embargo
    gap_train_val = int(val_df["k"].min() - train_df["k"].max())
    gap_val_test = int(test_df["k"].min() - val_df["k"].max())
    assert gap_train_val >= M and gap_val_test >= M, (
        f"insufficient gap: train_val={gap_train_val} val_test={gap_val_test} (need >= M={M})"
    )
    print(f"  embargo gaps: train|val={gap_train_val} val|test={gap_val_test} (>= M={M} required)")

    # ----- Sample weights with cadence-aware delta ---------------------------
    # The TIME_DECAY base used at boundary cadence corresponds to per-boundary
    # decay. Convert to per-1-min-row decay via the helper.
    base_delta = float(utils.WEIGHT_TIME_DELTA)  # boundary-cadence value
    delta_1min = utils.recommended_time_discount_delta_for_cadence(
        "1min", base_delta=base_delta, M=M
    )
    print()
    print(f"Time-discount delta: base={base_delta:.4f} -> 1-min={delta_1min:.6f}")
    print("Computing training weights on train slice ...")
    t0 = time.perf_counter()
    train_weights, _w_dist, _w_time, weight_info = utils.compute_training_weights(
        m_k=train_df["m_k"].to_numpy(),
        phi=PHI,
        delta=delta_1min,
        k_index=train_df["k"].to_numpy(),
    )
    print(f"  weights computed in {time.perf_counter()-t0:.1f}s; "
          f"range=[{weight_info['combined']['weight_range'][0]:.4f}, {weight_info['combined']['weight_range'][1]:.4f}]")
    print(f"  effective_n: {weight_info['combined']['effective_n']:.0f} of {len(train_weights)}")

    # Label-uniqueness weights for overlapping 1-min barrier labels.
    # Adjacent labels share M-1 future bars; without uniqueness, a 20-bar
    # monotonic trend contributes 20 highly redundant rows. Multiplying by
    # u_i (mean-normalized so total weight is preserved) downweights
    # redundant rows and gives an isolated event ~M times the weight of a
    # fully-overlapping one. See ``analytics.sampling.sample_uniqueness``.
    u = compute_uniqueness_weights(
        n_rows=len(train_df), M=int(M), bar_stride=1, normalize=True
    )
    train_weights = train_weights * u
    print(
        f"  applied label-uniqueness weights: u range=[{u.min():.4f}, {u.max():.4f}], "
        f"effective_n after = {float(train_weights.sum()):.0f}"
    )
    # val weights stay at 1.0 for early stopping comparability (legacy convention)
    val_weights = np.ones(len(val_df), dtype=float)

    # ----- CatBoost training -------------------------------------------------
    print()
    print(f"Training CatBoost with params:")
    for k, v in CB_PARAMS.items():
        print(f"  {k}: {v}")

    train_pool = Pool(
        data=train_df[feature_list].to_numpy(),
        label=train_df["y"].astype(int).to_numpy(),
        timestamp=train_df["k"].to_numpy(dtype=np.uint32),
        weight=train_weights,
        feature_names=list(feature_list),
    )
    val_pool = Pool(
        data=val_df[feature_list].to_numpy(),
        label=val_df["y"].astype(int).to_numpy(),
        timestamp=val_df["k"].to_numpy(dtype=np.uint32),
        weight=val_weights,
        feature_names=list(feature_list),
    )

    model = CatBoostClassifier(**CB_PARAMS)
    t0 = time.perf_counter()
    print()
    print(f"Fit start: {time.strftime('%H:%M:%S')}")
    model.fit(train_pool, eval_set=val_pool)
    fit_dt = time.perf_counter() - t0
    best_iter = int(model.get_best_iteration())
    print(f"Fit done in {fit_dt:.1f}s ({fit_dt/60:.1f} min); best_iteration={best_iter}")

    model.save_model(MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")

    # ----- Score val + test, save predictions cache --------------------------
    print()
    print("Computing predictions on val + test ...")
    p_val = model.predict_proba(val_df[feature_list].to_numpy())[:, 1]
    p_test = model.predict_proba(test_df[feature_list].to_numpy())[:, 1]
    print(f"  val p: min={p_val.min():.3f} med={np.median(p_val):.3f} max={p_val.max():.3f}")
    print(f"  test p: min={p_test.min():.3f} med={np.median(p_test):.3f} max={p_test.max():.3f}")

    REGIME_COL = "vol__rs__f__w240"
    cache_frames = []
    for name, sub, p in [("val", val_df, p_val), ("test", test_df, p_test)]:
        cache = pd.DataFrame({
            "k": sub["k"].to_numpy(),
            "ts": sub["ts"].to_numpy(),
            "y": sub["y"].astype(int).to_numpy(),
            "m_k": sub["m_k"].to_numpy(),
            "tau_k": sub["tau_k"].to_numpy() if "tau_k" in sub.columns else np.full(len(sub), np.nan),
            "phi": sub["phi"].to_numpy() if "phi" in sub.columns else np.full(len(sub), PHI),
            "regime": sub[REGIME_COL].to_numpy(dtype=float) if REGIME_COL in sub.columns else np.full(len(sub), np.nan),
            "p": p.astype(float),
            "split": name,
        })
        cache_frames.append(cache)
    cache_all = pd.concat(cache_frames, ignore_index=True)
    cache_all.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"Saved predictions: {PREDICTIONS_PATH} ({len(cache_all):,} rows)")

    # ----- Headline metrics with BLOCK BOOTSTRAP -----------------------------
    # Block bootstrap is mandatory at 1-min cadence — IID bootstrap would
    # produce CIs that are too tight by a factor of ~sqrt(M).
    print()
    print(f"Computing headline metrics with block bootstrap (block_size=M={M}) ...")
    os.makedirs(f"{DATASET_DIR}/analytics", exist_ok=True)

    def _ci(metric_name, fn, y, p):
        res_iid = bootstrap_metric(fn, y, p, B=500, stratify=True, seed=0)
        res_block = bootstrap_metric(fn, y, p, B=500, stratify=False, seed=0, block_size=M)
        return {
            "point": res_iid.point,
            "iid_ci": [res_iid.ci_low, res_iid.ci_high],
            "block_ci": [res_block.ci_low, res_block.ci_high],
            "iid_width": res_iid.ci_high - res_iid.ci_low,
            "block_width": res_block.ci_high - res_block.ci_low,
            "block_vs_iid_width_x": (res_block.ci_high - res_block.ci_low)
                                     / max(res_iid.ci_high - res_iid.ci_low, 1e-18),
        }

    def _metrics(name, sub, p):
        y = sub["y"].astype(int).to_numpy()
        return {
            "split": name,
            "n_samples": int(len(sub)),
            "base_rate": float(y.mean()),
            "roc_auc": _ci("roc_auc", lambda y_, p_: float(roc_auc_score(y_, p_)), y, p),
            "pr_auc":  _ci("pr_auc",  lambda y_, p_: float(average_precision_score(y_, p_)), y, p),
            "log_loss": _ci("log_loss", lambda y_, p_: float(log_loss(y_, p_, labels=[0, 1])), y, p),
            "brier":    _ci("brier",   lambda y_, p_: float(brier_score_loss(y_, p_)), y, p),
        }

    val_metrics = _metrics("val", val_df, p_val)
    test_metrics = _metrics("test", test_df, p_test)

    out = {
        "config": {
            "embargo_k": int(EMBARGO_K),
            "delta_1min": delta_1min,
            "best_iteration": best_iter,
            "fit_seconds": fit_dt,
            "block_bootstrap_size": int(M),
        },
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print()
    print("Headline metrics (1-min cadence):")
    for split_name, m in [("val", val_metrics), ("test", test_metrics)]:
        print(f"  [{split_name}] n={m['n_samples']:,}  base_rate={m['base_rate']:.4f}")
        for metric_name in ["roc_auc", "pr_auc", "log_loss", "brier"]:
            mm = m[metric_name]
            print(f"      {metric_name:>9}: {mm['point']:.4f}  "
                  f"iid CI=[{mm['iid_ci'][0]:.4f}, {mm['iid_ci'][1]:.4f}]  "
                  f"block CI=[{mm['block_ci'][0]:.4f}, {mm['block_ci'][1]:.4f}]  "
                  f"(block/iid width = {mm['block_vs_iid_width_x']:.2f}x)")
    print(f"\nSaved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
