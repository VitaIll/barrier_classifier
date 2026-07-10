"""Inference pipeline ≡ training pipeline (anti-skew guards).

Three invariants keep the serving path honest:

1. **Row equality** — for every row the training pipeline emits, the
   inference pipeline emits identical feature values (training mode only
   *filters* rows, never changes values).
2. **Tail coverage** — inference keeps the last M rows (null labels),
   ending exactly at the input's last bar; training mode drops them.
3. **Slice invariance** — running the boundary stages on a bounded tail
   slice yields the same final row as running them over the full window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.features.config import M, N_WARMUP
from src.features.pipeline import run_inference_pipeline, run_pipeline

pytestmark = [pytest.mark.engine_slow]


def synthetic_raw(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01 00:01:00", periods=n, freq="1min", tz="UTC")
    ret = rng.normal(0, 6e-4, n)
    close = 40_000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]])
    # OHLC-sane by construction (mirrors nb01-validated data): the wick
    # extends beyond the body on both sides.
    spread = np.abs(rng.normal(0, 4e-4, n)) + 1e-4
    high = np.maximum(open_, close) * np.exp(spread)
    low = np.minimum(open_, close) * np.exp(-spread)
    volume = np.abs(rng.normal(2.0, 0.5, n)) + 0.1
    frame = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": volume * close,
        "num_trades": np.round(np.abs(rng.normal(50, 10, n))) + 1,
        "taker_buy_base": volume * rng.uniform(0.3, 0.7, n),
        "taker_buy_quote": volume * close * 0.5,
    }, index=idx)
    frame.index.name = "ts"
    return frame


@pytest.fixture(scope="module")
def small_frame() -> pd.DataFrame:
    # Small: below warmup — exercises slice invariance cheaply.
    return synthetic_raw(4_000)


@pytest.fixture(scope="module")
def warm_frame() -> pd.DataFrame:
    # Past warmup so training mode emits rows.
    return synthetic_raw(N_WARMUP + 400)


def test_slice_invariance_of_boundary_stages(small_frame):
    full = run_inference_pipeline(small_frame, boundary_tail_rows=None,
                                  label_cadence="1min")
    sliced = run_inference_pipeline(small_frame, boundary_tail_rows=3_200,
                                    label_cadence="1min")
    assert full["ts"][-1] == sliced["ts"][-1]
    # Compare the last row across every shared feature column. ``k`` is a
    # frame-local row index by construction, not a feature — exclude it.
    # Floats compare at ULP scale: polars' sliding-sum kernels accumulate
    # from the start of the array they see, so a tail slice reorders the
    # floating-point additions (≈1 ULP wobble). Semantic divergence would
    # show up orders of magnitude above this tolerance.
    shared = [c for c in sliced.columns if c in full.columns and c != "k"]
    last_full = full.select(shared).tail(1)
    last_sliced = sliced.select(shared).tail(1)
    for col in shared:
        a = last_full[col][0]
        b = last_sliced[col][0]
        if a is None or b is None:
            assert a == b, f"null mismatch in {col}: {a} vs {b}"
        elif isinstance(a, float):
            both_nan = np.isnan(a) and np.isnan(b)
            close = a == b or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
            assert both_nan or close, f"value mismatch in {col}: {a} vs {b}"
        else:
            assert a == b, f"value mismatch in {col}: {a} vs {b}"


def test_inference_keeps_tail_and_matches_training_rows(warm_frame):
    train_out = run_pipeline(warm_frame, label_cadence="1min")
    infer_out = run_inference_pipeline(warm_frame, boundary_tail_rows=None,
                                       label_cadence="1min")

    # (2) Tail coverage: inference ends at the last input bar; training
    # drops the unlabeled tail (M rows) and the warmup head.
    assert pd.Timestamp(infer_out["ts"][-1]) == warm_frame.index[-1].tz_localize(None)
    assert pd.Timestamp(train_out["ts"][-1]) < pd.Timestamp(infer_out["ts"][-1])
    n_labeled_tail = infer_out.filter(pl.col("y").is_not_null()).height
    assert train_out.height <= n_labeled_tail

    # (1) Row equality on the training rows (join by ts).
    feature_cols = [
        c for c in train_out.columns
        if c in infer_out.columns and c not in ("ts", "k")
    ]
    merged = train_out.select(["ts"] + feature_cols).join(
        infer_out.select(["ts"] + feature_cols), on="ts", how="inner",
        suffix="_inf",
    )
    assert merged.height == train_out.height, "inference lost training rows"
    for col in feature_cols:
        a = merged[col].to_numpy()
        b = merged[f"{col}_inf"].to_numpy()
        if a.dtype.kind == "f":
            same = (a == b) | (np.isnan(a.astype(float)) & np.isnan(b.astype(float)))
        else:
            same = a == b
        assert bool(np.all(same)), (
            f"training/inference divergence in {col!r}: "
            f"{int((~same).sum())} rows differ (first at {np.argmax(~same)})"
        )


def test_last_inference_row_is_fully_finite_on_features(warm_frame):
    out = run_inference_pipeline(warm_frame, boundary_tail_rows=6_000,
                                 label_cadence="1min")
    from src.features.pipeline import (
        _BASE_COLS,
        _DERIV_BASE_COLS,
        _LABEL_AUX_COLS,
        _RAW_COLS,
    )
    non_feature = set(_LABEL_AUX_COLS + _RAW_COLS + _BASE_COLS + _DERIV_BASE_COLS)
    feature_cols = [
        c for c in out.columns
        if c not in non_feature and not c.startswith("undef__")
    ]
    last = out.select(feature_cols).tail(1).to_numpy().astype(float)
    assert np.isfinite(last).all(), "non-finite feature values on the serving row"
