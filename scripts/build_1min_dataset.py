"""Build the 1-min-cadence dataset on the full year of BTC data and save.

Produces:
    data/model_dataset/dataset_1min.parquet
    data/model_dataset/dataset_metadata_1min.json
    data/model_dataset/feature_list_1min.json

The legacy boundary-cadence artifacts (dataset.parquet, etc.) are left
untouched — a 1-min model lives alongside the boundary model.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

import polars as pl
import pandas as pd

sys.path.insert(0, ".")

from src.analytics.audits import causal_feature_audit
from src.features.config import C, ETA, K_WARMUP, M, N_WARMUP, PHI
from src.features.pipeline import (
    _BASE_COLS,
    _DERIV_BASE_COLS,
    _LABEL_AUX_COLS,
    _RAW_COLS,
    run_pipeline,
)


# Target/execution alignment: at 1-min cadence the production strategy
# fills a long TP on intrabar HIGH crossing, so the label tests future
# highs (default in run_pipeline). The build records the source it used
# so a model train + strategy backtest cannot silently drift apart.
BARRIER_SOURCE = "high"


def main() -> None:
    raw = pd.read_parquet("data/raw_data/klines_1m.parquet")
    print(f"Raw bars: {len(raw):,} rows  range {raw.index.min()} -> {raw.index.max()}")

    print(
        f"Building dataset at label_cadence='1min', barrier_source='{BARRIER_SOURCE}'..."
    )
    t0 = time.perf_counter()
    ds_1min = run_pipeline(
        raw,
        with_derivatives=False,
        label_cadence="1min",
        barrier_source=BARRIER_SOURCE,
    )
    dt = time.perf_counter() - t0
    print(f"  ran in {dt:.1f}s")
    print(f"  rows={len(ds_1min):,}  cols={len(ds_1min.columns)}")
    print(f"  base_rate y = {float(ds_1min['y'].mean()):.4f}")
    print(f"  ts range: {ds_1min['ts'].min()} -> {ds_1min['ts'].max()}")
    n_autocorr = sum(1 for c in ds_1min.columns if c.startswith("target__autocorr_"))
    print(f"  autocorr cols: {n_autocorr}")
    n_roll_excursion = sum(
        1 for c in ds_1min.columns
        if c.startswith("excursion__roll_max_drawup__f__")
        or c.startswith("excursion__roll_max_drawdown__f__")
    )
    print(f"  rolling-excursion cols (every-row trailing): {n_roll_excursion}")
    n_sparse_excursion = sum(
        1 for c in ds_1min.columns
        if c.startswith("excursion__max_drawup__f__")
        or c.startswith("excursion__max_drawdown__f__")
    )
    assert n_sparse_excursion == 0, (
        f"boundary-sparse excursion cols leaked into 1-min dataset: {n_sparse_excursion}"
    )

    out_dir = "data/model_dataset"
    os.makedirs(out_dir, exist_ok=True)
    ds_1min.write_parquet(f"{out_dir}/dataset_1min.parquet")
    print(f"  wrote {out_dir}/dataset_1min.parquet")

    # Feature list: every column that is not a label, raw OHLCV, base
    # series, or derivatives base series. Reuses the pipeline's
    # authoritative constants so the exclusion set cannot drift from the
    # imputation-step contract. Triple-barrier aux columns (m_dn, tau_dn)
    # are included in _LABEL_AUX_COLS — sidecar diagnostics, not features.
    NON_FEATURE = set(_LABEL_AUX_COLS) | set(_RAW_COLS) | set(_BASE_COLS) | set(_DERIV_BASE_COLS)
    feature_cols = [
        c for c in ds_1min.columns
        if c not in NON_FEATURE and not c.startswith("undef__")
    ]
    # Static causal-naming audit: every feature must contain ``__f__`` or
    # ``__h__`` and no suspect tokens (``fwd``, ``future``, ``ahead``, ...).
    audit = causal_feature_audit(feature_cols)
    if not audit.passed:
        raise RuntimeError(
            "causal_feature_audit failed on 1-min build:\n"
            f"  suspect = {audit.suspect[:10]}\n"
            f"  unmatched = {audit.unmatched[:10]}"
        )
    print(f"  causal audit passed: {audit.n_causal}/{audit.n_features} causal")
    with open(f"{out_dir}/feature_list_1min.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  feature_list_1min.json: {len(feature_cols)} features")

    # Metadata
    metadata = {
        "label_cadence": "1min",
        "barrier_source": BARRIER_SOURCE,
        "M": int(M),
        "N_WARMUP": int(N_WARMUP),
        "K_WARMUP": int(K_WARMUP),
        "PHI": float(PHI),
        "C": float(C),
        "ETA": float(ETA),
        "n_rows": int(len(ds_1min)),
        "n_features": len(feature_cols),
        "ts_start": str(ds_1min["ts"].min()),
        "ts_end": str(ds_1min["ts"].max()),
        "base_rate": float(ds_1min["y"].mean()),
        "build_time_seconds": dt,
    }
    with open(f"{out_dir}/dataset_metadata_1min.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata: {metadata}")


if __name__ == "__main__":
    main()
