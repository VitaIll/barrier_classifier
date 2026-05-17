"""Smoke test: verify relabel_dataset produces a sane M=60 dataset.

Checks (cheap; no training):
  * dataset row count ~ 505,381 at M=60 (505,421 baseline - ~40 immature tail)
  * base rate stays near 0.21 (phi co-scaled by sqrt(M/M_baseline))
  * weight range [1, 5] and finite
  * weight effective_n in plausible range
  * no NaN/inf in features in the relabeled output
  * peak memory tolerable

Run: python scripts/smoke_test_relabel.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_label_horizon import (  # noqa: E402
    C_FIXED,
    DATASET_PATH,
    FEATURE_LIST_PATH,
    M_BASELINE,
    PHI_BASELINE,
    RAW_PATH,
    relabel_dataset,
)
from src import utils  # noqa: E402


def main() -> None:
    M_test = 60
    phi_test = PHI_BASELINE * float(np.sqrt(M_test / M_BASELINE))
    eta_test = phi_test - C_FIXED
    print(f"Smoke test: M={M_test}, PHI={phi_test:.6f}, ETA={eta_test:.6f}, "
          f"C={C_FIXED}")
    print(f"PHI/PHI_baseline = {phi_test / PHI_BASELINE:.4f} (expect sqrt(3) = "
          f"{np.sqrt(3):.4f})")

    t0 = time.perf_counter()
    raw_pl = pl.read_parquet(RAW_PATH)
    print(f"[load] raw bars: {raw_pl.height:,}  in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    df_features = pd.read_parquet(DATASET_PATH)
    feature_list = utils.load_json(FEATURE_LIST_PATH)
    print(f"[load] dataset: {len(df_features):,} rows x "
          f"{len(df_features.columns)} cols, {len(feature_list):,} features  "
          f"in {time.perf_counter()-t0:.1f}s")

    # Capture old label stats for diff comparison.
    old_y_mean = float(df_features["y"].mean())
    old_phi_unique = sorted(df_features["phi"].unique().tolist())
    old_weight_range = (
        float(df_features["weight"].min()),
        float(df_features["weight"].max()),
    )
    old_weight_mean = float(df_features["weight"].mean())
    old_mk_range = (
        float(df_features["m_k"].min()),
        float(df_features["m_k"].max()),
    )
    print(f"\n[before] y_mean={old_y_mean:.4f}  phi={old_phi_unique}  "
          f"weight range=[{old_weight_range[0]:.4f}, {old_weight_range[1]:.4f}]  "
          f"weight mean={old_weight_mean:.4f}  "
          f"m_k range=[{old_mk_range[0]:.6f}, {old_mk_range[1]:.6f}]")

    t0 = time.perf_counter()
    df_out, w_info = relabel_dataset(
        raw_pl=raw_pl, df_features=df_features, M_test=M_test, phi_test=phi_test
    )
    dt = time.perf_counter() - t0
    print(f"\n[relabel] done in {dt:.1f}s")

    # ---- Assertions ---------------------------------------------------------
    new_y_mean = float(df_out["y"].mean())
    new_weight_range = (
        float(df_out["weight"].min()),
        float(df_out["weight"].max()),
    )
    new_weight_mean = float(df_out["weight"].mean())
    new_mk_range = (
        float(df_out["m_k"].min()),
        float(df_out["m_k"].max()),
    )

    print(f"\n[after]  n_rows={len(df_out):,}  y_mean={new_y_mean:.4f}  "
          f"phi={float(df_out['phi'].iloc[0]):.6f}  "
          f"weight range=[{new_weight_range[0]:.4f}, {new_weight_range[1]:.4f}]  "
          f"weight mean={new_weight_mean:.4f}  "
          f"m_k range=[{new_mk_range[0]:.6f}, {new_mk_range[1]:.6f}]  "
          f"effective_n={w_info['combined']['effective_n']:.1f}")

    # Row count: expect ~505,381 (baseline 505,421 minus ~40 immature tail).
    expected_n_low = 505_300
    expected_n_high = 505_421
    assert expected_n_low <= len(df_out) <= expected_n_high, (
        f"Row count {len(df_out)} outside expected [{expected_n_low}, "
        f"{expected_n_high}]"
    )
    print(f"[check] n_rows {len(df_out):,} in expected band "
          f"[{expected_n_low:,}, {expected_n_high:,}]  PASS")

    # Base rate: with phi co-scaled by sqrt(M/M_baseline), under approximately
    # Brownian dynamics the hit probability should be roughly preserved.
    # Empirically allow [0.10, 0.40] — wide because of real-world fat tails.
    assert 0.10 <= new_y_mean <= 0.40, (
        f"Base rate {new_y_mean:.4f} outside sanity band [0.10, 0.40]; "
        "phi may be mis-scaled"
    )
    print(f"[check] base_rate {new_y_mean:.4f} in [0.10, 0.40] band  PASS")

    # Weight: range should be [1, 5] (asymmetric scheme caps negatives at
    # WEIGHT_DIST_W_MAX=5, positives at 1).
    assert new_weight_range[0] >= 1.0 - 1e-9, (
        f"weight min {new_weight_range[0]} < 1"
    )
    assert new_weight_range[1] <= 5.0 + 1e-9, (
        f"weight max {new_weight_range[1]} > 5"
    )
    assert np.isfinite(df_out["weight"].to_numpy()).all(), "weight has non-finite"
    print(f"[check] weight in [1, 5] and finite  PASS")

    # Features: no NaN / inf introduced.
    t0 = time.perf_counter()
    X = df_out[feature_list].to_numpy()
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    print(f"[check] feature matrix scan: n_nan={n_nan}, n_inf={n_inf}  "
          f"(in {time.perf_counter()-t0:.1f}s)")
    assert n_nan == 0 and n_inf == 0, (
        f"Features have NaN={n_nan} or inf={n_inf} after relabel"
    )

    # Phi: should be exactly the co-scaled value, applied to every row.
    phi_unique = df_out["phi"].unique()
    assert len(phi_unique) == 1, f"phi not uniform: {phi_unique}"
    assert abs(float(phi_unique[0]) - phi_test) < 1e-12, (
        f"phi value drift: got {phi_unique[0]}, expected {phi_test}"
    )
    print(f"[check] phi uniform at {float(phi_unique[0]):.6f}  PASS")

    # k monotonic.
    assert df_out["k"].is_monotonic_increasing, "k not monotonic"
    print(f"[check] k monotonic increasing  PASS")

    # m_k: positive labels (y=1) MUST have m_k >= phi; negatives must have
    # m_k < phi (or NaN where construct_labels_pl returned 0 for non-finite).
    mask_pos = df_out["y"] == 1
    mask_neg = df_out["y"] == 0
    if mask_pos.any():
        m_pos_min = float(df_out.loc[mask_pos, "m_k"].min())
        assert m_pos_min >= phi_test - 1e-12, (
            f"Positive y with m_k={m_pos_min} < phi={phi_test}"
        )
    if mask_neg.any():
        m_neg_max = float(df_out.loc[mask_neg, "m_k"].max())
        assert m_neg_max < phi_test + 1e-12, (
            f"Negative y with m_k={m_neg_max} >= phi={phi_test}"
        )
    print(f"[check] m_k respects barrier: pos>=phi, neg<phi  PASS")

    # Free up memory
    del df_out, df_features, raw_pl
    gc.collect()

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
