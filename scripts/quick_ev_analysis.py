"""Quick cost-adjusted EV analysis to complement the classification metrics.

Compares M=20 baseline vs M_test predictions at a sweep of probability
thresholds, reporting:
  - Precision (P(y=1|p>=thr))
  - Trades/period (P(p>=thr))
  - Naive EV per trade (assuming win pays +phi, loss pays -phi, cost is fixed bps)

This is NOT a full simulator — it ignores cluster caps, intrabar fills,
and overlap. But it captures the headline economics: does the larger
phi at longer M offset the precision loss?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / "data" / "model_dataset"
OUT_DIR = DATASET_DIR / "horizon_sweep"

# Round-trip transaction cost assumptions to sweep.
# BTC USDT taker fees on Binance perps ~ 4 bps each side = 8 bps RT; with smart
# execution (maker entries) closer to 2-5 bps RT.
COST_BPS_GRID = [2.0, 5.0, 8.0, 10.0]


def _load_baseline_preds() -> tuple[pd.DataFrame, float]:
    """Load the M=20 baseline predictions and PHI."""
    p = pd.read_parquet(DATASET_DIR / "research_predictions_1min.parquet")
    # M=20 PHI = 0.0025 in log return = 25 bps approx
    return p, 0.0025


def _load_M_preds(M_test: int) -> tuple[pd.DataFrame, float]:
    p = pd.read_parquet(OUT_DIR / f"predictions_M{M_test}.parquet")
    with open(OUT_DIR / f"metrics_M{M_test}.json") as f:
        m = json.load(f)
    return p, float(m["PHI"])


def ev_curve(df: pd.DataFrame, phi: float, split: str) -> pd.DataFrame:
    """For each threshold, compute precision, fire rate, lift, and EV at each cost."""
    sub = df[df["split"] == split].copy()
    y = sub["y"].astype(int).to_numpy()
    p = sub["p"].to_numpy()
    base = float(y.mean())
    n = len(sub)
    phi_bps = float(phi) * 10000.0

    quantiles = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    rows = []
    for q in quantiles:
        thr = float(np.quantile(p, q))
        mask = p >= thr
        n_take = int(mask.sum())
        if n_take == 0:
            row = {"q": q, "thr": thr, "n_take": 0, "precision": np.nan,
                   "lift": np.nan, "fire_rate": 0.0}
            for c in COST_BPS_GRID:
                row[f"ev@{int(c)}bps"] = np.nan
            rows.append(row)
            continue
        precision = float(y[mask].mean())
        lift = precision / base if base > 0 else np.nan
        fire_rate = float(n_take / n)
        row = {
            "q": q, "thr": thr, "n_take": n_take, "precision": precision,
            "lift": lift, "fire_rate": fire_rate,
        }
        for c in COST_BPS_GRID:
            row[f"ev@{int(c)}bps"] = phi_bps * (2 * precision - 1) - c
        rows.append(row)
    return pd.DataFrame(rows), base


def _format(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["q"] = out["q"].map(lambda x: f"{int(x*100)}")
    out["thr"] = out["thr"].map(lambda x: f"{x:.4f}")
    out["precision"] = out["precision"].map(lambda x: f"{x:.4f}")
    out["lift"] = out["lift"].map(lambda x: f"{x:.2f}x")
    out["fire_rate"] = out["fire_rate"].map(lambda x: f"{x:.4f}")
    for c in COST_BPS_GRID:
        col = f"ev@{int(c)}bps"
        out[col] = out[col].map(lambda x: f"{x:+.2f}")
    return out


def main() -> None:
    print(f"Cost assumption grid: {COST_BPS_GRID} bps round trip")

    baseline_df, phi_20 = _load_baseline_preds()
    print(f"\n=== M=20 baseline (PHI={phi_20:.5f} = {phi_20*10000:.2f} bps) ===")
    for split in ("val", "test"):
        ev, base = ev_curve(baseline_df, phi_20, split)
        print(f"\n--- {split} (base_rate={base:.4f}) ---")
        print(_format(ev).to_string(index=False))

    for M_test in [60, 45, 90]:
        try:
            df, phi = _load_M_preds(M_test)
        except FileNotFoundError:
            print(f"\n=== M={M_test} — predictions not yet on disk; skipping ===")
            continue
        print(f"\n=== M={M_test} (PHI={phi:.5f} = {phi*10000:.2f} bps) ===")
        for split in ("val", "test"):
            ev, base = ev_curve(df, phi, split)
            print(f"\n--- {split} (base_rate={base:.4f}) ---")
            print(_format(ev).to_string(index=False))


if __name__ == "__main__":
    main()
