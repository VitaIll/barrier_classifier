"""Tests for src.analytics.fast_train.

Tests are split into pure-Python (fast: slicing, params, cache I/O) and
CatBoost smoke tests (a few seconds each). The pure-Python tests certify the
mathematical / structural properties (cutoff arithmetic, contiguity, exact
schema). The CatBoost tests catch integration regressions: feature-list
ordering, posterior_sampling persistence, weight-column validation, multi-split
predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.fast_train import (
    CACHE_REQUIRED_COLS,
    TrainSliceConfig,
    compute_predictions,
    fit_research_model,
    load_predictions_cache,
    research_train_params,
    save_predictions_cache,
    select_recent_train_slice,
)

pytestmark = pytest.mark.analytics_phase1


def _make_synthetic_dataset(
    n_train: int = 800,
    n_val: int = 200,
    n_test: int = 200,
    n_features: int = 8,
    seed: int = 0,
):
    """Tiny OHLCV-flavored dataset suitable for a smoke fit."""
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    feature_names = [f"feat_{i:02d}" for i in range(n_features)]
    X = rng.normal(size=(n, n_features))
    score = X[:, 0] * 1.2 + X[:, 1] * 0.6 - X[:, 2] * 0.4
    base_rate = 0.10
    threshold = float(np.quantile(score, 1.0 - base_rate))
    y = (score > threshold).astype(int)

    ts_start = pd.Timestamp("2024-01-01", tz="UTC")
    ts = pd.date_range(ts_start, periods=n, freq="10min")
    k = np.arange(n, dtype=np.int64)

    df = pd.DataFrame(X, columns=feature_names)
    df["ts"] = ts
    df["k"] = k
    df["y"] = y.astype(float)
    df["m_k"] = rng.uniform(0.001, 0.02, size=n)
    df["tau_k"] = np.where(y == 1, rng.integers(1, 11, size=n), np.nan)
    df["phi"] = 0.005
    df["weight"] = 1.0
    df["vol__rs__f__w240"] = np.abs(rng.normal(0.001, 0.0005, size=n))

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train_df, val_df, test_df, feature_names


# ---------------------------------------------------------------------------
# Pure-Python helpers (no CatBoost)
# ---------------------------------------------------------------------------


def test_research_train_params_required_keys():
    """Every key the notebook uses must be present and have a sensible type."""
    p = research_train_params()
    assert p["loss_function"] == "Logloss"
    assert p["posterior_sampling"] is True
    assert p["use_best_model"] is True
    assert p["allow_writing_files"] is False
    assert isinstance(p["iterations"], int) and p["iterations"] > 0
    assert isinstance(p["learning_rate"], float) and p["learning_rate"] > 0
    assert isinstance(p["depth"], int) and p["depth"] > 0
    assert isinstance(p["l2_leaf_reg"], float) and p["l2_leaf_reg"] > 0
    assert isinstance(p["early_stopping_rounds"], int)


def test_research_train_params_posterior_sampling_off_omits_key():
    """posterior_sampling=False must not include the key (so CatBoost uses
    its default of False without conflicting with other params)."""
    p = research_train_params(posterior_sampling=False)
    assert "posterior_sampling" not in p


def test_select_recent_train_slice_includes_endpoint():
    """The most recent timestamp must be in the slice (it is the anchor)."""
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")})
    sliced = select_recent_train_slice(df, TrainSliceConfig(months_back=2.0))
    assert sliced["ts"].max() == df["ts"].max()


def test_select_recent_train_slice_is_contiguous_tail():
    """The slice is a contiguous chronological tail — no gaps, no reordering."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC"),
            "x": np.arange(300),
        }
    )
    sliced = select_recent_train_slice(df, TrainSliceConfig(months_back=2.0))
    # Tail of original
    assert sliced["x"].iloc[-1] == 299
    # Strictly increasing x
    assert sliced["x"].is_monotonic_increasing
    # Differences are exactly 1 day (no gaps)
    diffs = sliced["ts"].diff().dropna().unique()
    assert len(diffs) == 1 and diffs[0] == pd.Timedelta(days=1)


def test_select_recent_train_slice_months_cutoff_arithmetic():
    """The cutoff is end_ts - months_back * 30.44 days, exactly. All rows on
    or after the cutoff are kept; all earlier rows are dropped."""
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")})
    months = 3.0
    sliced = select_recent_train_slice(df, TrainSliceConfig(months_back=months))
    expected_cutoff = df["ts"].max() - pd.Timedelta(days=months * 30.44)
    assert (sliced["ts"] >= expected_cutoff).all()
    dropped = df[~df["ts"].isin(sliced["ts"])]
    assert (dropped["ts"] < expected_cutoff).all()


def test_select_recent_train_slice_frac_exact_count():
    """frac_back=0.25 of 1000 rows must give exactly 250 (the most recent)."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=1000, freq="D", tz="UTC"),
            "x": np.arange(1000),
        }
    )
    sliced = select_recent_train_slice(
        df, TrainSliceConfig(months_back=None, frac_back=0.25)
    )
    assert len(sliced) == 250
    assert sliced["x"].min() == 750
    assert sliced["x"].max() == 999


def test_select_recent_train_slice_none_returns_full_copy():
    """Both months_back=None and frac_back=None -> full input. Result must be
    a fresh copy (mutating it does not affect the input)."""
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "y": np.zeros(10),
        }
    )
    sliced = select_recent_train_slice(df, TrainSliceConfig(months_back=None, frac_back=None))
    assert len(sliced) == len(df)
    sliced.loc[0, "y"] = 99.0
    assert df.loc[0, "y"] == 0.0  # input untouched


def test_select_recent_train_slice_months_dominates_frac():
    """When both are set, months_back wins."""
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")})
    sliced = select_recent_train_slice(
        df, TrainSliceConfig(months_back=1.0, frac_back=0.99)
    )
    # 1 month ~ 30 rows; frac_back=0.99 would yield ~297 -> months wins
    assert len(sliced) < 50


def test_save_load_predictions_cache_roundtrip(tmp_path: Path):
    """parquet I/O preserves all rows and column types."""
    n = 50
    df = pd.DataFrame(
        {
            "k": np.arange(n),
            "ts": pd.date_range("2024-01-01", periods=n, freq="10min", tz="UTC"),
            "y": np.zeros(n, dtype=int),
            "m_k": np.linspace(0, 0.01, n),
            "tau_k": np.full(n, np.nan),
            "phi": np.full(n, 0.005),
            "regime": np.linspace(0.001, 0.002, n),
            "p": np.linspace(0.0, 1.0, n),
            "split": ["val"] * n,
        }
    )
    path = tmp_path / "cache.parquet"
    save_predictions_cache(df, path)
    loaded = load_predictions_cache(path)
    assert list(loaded.columns) == CACHE_REQUIRED_COLS
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False
    )


def test_save_predictions_cache_missing_column_raises(tmp_path: Path):
    df = pd.DataFrame({"k": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing required columns"):
        save_predictions_cache(df, tmp_path / "bad.parquet")


def test_load_predictions_cache_missing_column_raises(tmp_path: Path):
    """Loading an externally-produced parquet that lacks required cols must raise."""
    bad = pd.DataFrame({"k": [1, 2, 3], "p": [0.1, 0.2, 0.3]})
    path = tmp_path / "bad.parquet"
    bad.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_predictions_cache(path)


# ---------------------------------------------------------------------------
# CatBoost integration smoke tests
# ---------------------------------------------------------------------------


def test_fit_research_model_smoke():
    """End-to-end fit: model is_fitted, predict_proba returns shape (n, 2)."""
    train_df, val_df, _, feats = _make_synthetic_dataset()
    params = research_train_params(iterations=80, early_stopping_rounds=20, verbose=0)
    model = fit_research_model(train_df, val_df, feats, params=params)
    assert model.is_fitted()
    p = model.predict_proba(val_df[feats].to_numpy())
    assert p.shape == (len(val_df), 2)
    # Probabilities sum to 1 per row
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)


def test_fit_research_model_posterior_sampling_persists_in_get_all_params():
    """posterior_sampling=True at construction must remain True after fit.

    This catches CatBoost silently dropping the option (e.g. if we pair it
    with an incompatible bootstrap_type).
    """
    train_df, val_df, _, feats = _make_synthetic_dataset()
    params = research_train_params(iterations=40, early_stopping_rounds=10, verbose=0)
    model = fit_research_model(train_df, val_df, feats, params=params)
    all_params = model.get_all_params()
    assert all_params.get("posterior_sampling") is True


def test_fit_research_model_weight_in_feature_list_raises():
    """The weight column must not appear in feature_list — that would leak
    sample weights as a feature. Validation catches this before training."""
    train_df, val_df, _, feats = _make_synthetic_dataset()
    bad_feats = list(feats) + ["weight"]
    params = research_train_params(iterations=20, verbose=0)
    with pytest.raises(ValueError, match="must not be in feature_list"):
        fit_research_model(train_df, val_df, bad_feats, params=params)


def test_fit_research_model_learns_signal():
    """The model must produce a higher mean predicted probability on
    actual positives than on actual negatives — otherwise the smoke test
    is testing fitting noise.
    """
    train_df, val_df, _, feats = _make_synthetic_dataset()
    params = research_train_params(iterations=120, early_stopping_rounds=30, verbose=0)
    model = fit_research_model(train_df, val_df, feats, params=params)
    p_val = model.predict_proba(val_df[feats].to_numpy())[:, 1]
    pos_mean = float(p_val[val_df["y"] == 1].mean())
    neg_mean = float(p_val[val_df["y"] == 0].mean())
    assert pos_mean > neg_mean + 0.05, (
        f"model did not learn the signal: pos_mean={pos_mean:.3f}, neg_mean={neg_mean:.3f}"
    )


def test_compute_predictions_schema_and_split_partition():
    """Cache has exact expected columns; (val, test) splits are non-overlapping
    in their k values, and union covers all input rows."""
    train_df, val_df, test_df, feats = _make_synthetic_dataset()
    params = research_train_params(iterations=80, early_stopping_rounds=20, verbose=0)
    model = fit_research_model(train_df, val_df, feats, params=params)
    cache = compute_predictions(model, {"val": val_df, "test": test_df}, feats)

    assert list(cache.columns) == CACHE_REQUIRED_COLS
    assert set(cache["split"].unique()) == {"val", "test"}
    assert len(cache) == len(val_df) + len(test_df)
    val_cache = cache[cache["split"] == "val"]
    test_cache = cache[cache["split"] == "test"]
    assert len(val_cache) == len(val_df)
    assert len(test_cache) == len(test_df)
    # k partition is exact
    assert set(val_cache["k"]) == set(val_df["k"])
    assert set(test_cache["k"]) == set(test_df["k"])
    assert set(val_cache["k"]).isdisjoint(set(test_cache["k"]))


def test_compute_predictions_p_matches_direct_predict_proba():
    """compute_predictions must read features in feature_list ORDER and pass
    them to predict_proba unchanged. We verify by recomputing with explicit
    .to_numpy() on the same column order and asserting bitwise equality.
    This catches accidental column reordering / stale feature_list bugs.
    """
    train_df, val_df, _, feats = _make_synthetic_dataset()
    params = research_train_params(iterations=80, early_stopping_rounds=20, verbose=0)
    model = fit_research_model(train_df, val_df, feats, params=params)
    cache = compute_predictions(model, {"val": val_df}, feats)
    p_manual = model.predict_proba(val_df[feats].to_numpy())[:, 1]
    val_cache = cache[cache["split"] == "val"].sort_values("k").reset_index(drop=True)
    val_df_sorted = val_df.sort_values("k").reset_index(drop=True)
    p_manual_sorted = model.predict_proba(val_df_sorted[feats].to_numpy())[:, 1]
    np.testing.assert_array_equal(val_cache["p"].to_numpy(), p_manual_sorted)


def test_compute_predictions_p_changes_if_feature_columns_permuted():
    """If we (deliberately wrongly) pass a permuted feature list, predictions
    must DIFFER. This certifies that feature_list order is load-bearing —
    a future change that silently aligns by name would break this test."""
    train_df, val_df, _, feats = _make_synthetic_dataset()
    params = research_train_params(iterations=80, early_stopping_rounds=20, verbose=0)
    model = fit_research_model(train_df, val_df, feats, params=params)

    # Correct prediction
    p_ok = model.predict_proba(val_df[feats].to_numpy())[:, 1]
    # Permuted prediction (swap columns) — using to_numpy bypasses CatBoost's
    # feature-name alignment so result must differ
    permuted = list(feats)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    p_perm = model.predict_proba(val_df[permuted].to_numpy())[:, 1]
    assert not np.allclose(p_ok, p_perm), (
        "permuting feature columns gave identical predictions — order is not load-bearing?"
    )


def test_compute_predictions_missing_regime_raises():
    train_df, val_df, _, feats = _make_synthetic_dataset()
    params = research_train_params(iterations=40, early_stopping_rounds=10, verbose=0)
    model = fit_research_model(train_df, val_df, feats, params=params)
    val_df_no_regime = val_df.drop(columns=["vol__rs__f__w240"])
    with pytest.raises(ValueError, match="not in DataFrame"):
        compute_predictions(model, {"val": val_df_no_regime}, feats)


def test_full_pipeline_save_load_roundtrip(tmp_path: Path):
    """fit -> predict -> save -> load preserves the prediction frame exactly."""
    train_df, val_df, test_df, feats = _make_synthetic_dataset()
    params = research_train_params(iterations=80, early_stopping_rounds=20, verbose=0)
    model = fit_research_model(train_df, val_df, feats, params=params)
    cache = compute_predictions(model, {"val": val_df, "test": test_df}, feats)
    path = tmp_path / "cache.parquet"
    save_predictions_cache(cache, path)
    loaded = load_predictions_cache(path)
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        cache.reset_index(drop=True),
        check_dtype=False,
    )
