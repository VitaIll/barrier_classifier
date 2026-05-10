"""Fast-iter offline research model and prediction cache.

Trains a single CatBoost classifier with ``posterior_sampling=True`` (so the
downstream virtual-ensemble uncertainty pipeline can decompose total/data/
knowledge entropy) on a recent slice of the training window, then caches
predictions so every other analytics module re-loads instantly.

Cache schema (parquet):

    k        int    boundary index
    ts       ts     boundary timestamp (UTC)
    y        int    label (0/1)
    m_k      float  max excursion (label diagnostic)
    tau_k    float  time-to-barrier (NaN for negatives)
    phi      float  barrier (per-row, typically constant)
    regime   float  regime signal (default: vol__rs__f__w240)
    p        float  predicted probability of class 1
    split    str    {train, val, test}

The cache is the canonical input to all downstream analytics phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

CACHE_REQUIRED_COLS = ["k", "ts", "y", "m_k", "tau_k", "phi", "regime", "p", "split"]

DEFAULT_REGIME_SIGNAL = "vol__rs__f__w240"


@dataclass
class TrainSliceConfig:
    """Slice the training window for fast iteration.

    ``months_back`` -> keep only the last N months of ``train_df`` by ``ts``.
    ``frac_back`` -> alternative: keep the last ``frac_back`` fraction of rows
    (chronological). If both are provided, ``months_back`` wins.
    Set both to ``None`` to keep the full window.
    """

    months_back: Optional[float] = 6.0
    frac_back: Optional[float] = None


def select_recent_train_slice(
    train_df: pd.DataFrame,
    config: TrainSliceConfig,
    *,
    ts_col: str = "ts",
) -> pd.DataFrame:
    """Return the most-recent chunk of ``train_df`` per ``config``."""
    if config.months_back is None and config.frac_back is None:
        return train_df.copy()
    if config.months_back is not None:
        if ts_col not in train_df.columns:
            raise ValueError(f"select_recent_train_slice requires '{ts_col}' column")
        end_ts = train_df[ts_col].max()
        delta = pd.Timedelta(days=float(config.months_back) * 30.44)
        cutoff = end_ts - delta
        sliced = train_df[train_df[ts_col] >= cutoff].copy()
        if len(sliced) == 0:
            raise ValueError(
                f"select_recent_train_slice returned 0 rows (cutoff={cutoff}, end_ts={end_ts})"
            )
        return sliced
    n = len(train_df)
    take = max(1, int(np.ceil(n * float(config.frac_back))))
    return train_df.iloc[-take:].copy()


def research_train_params(
    *,
    iterations: int = 3000,
    learning_rate: float = 0.01,
    depth: int = 6,
    l2_leaf_reg: float = 5.0,
    random_seed: int = 42,
    posterior_sampling: bool = True,
    early_stopping_rounds: int = 200,
    border_count: int = 128,
    thread_count: int = -1,
    verbose: int = 200,
) -> dict:
    """CatBoost params for the research model.

    Defaults tuned for fast iteration (single seed, lower iterations than
    production, early stopping). ``posterior_sampling=True`` enables Stochastic
    Gradient Langevin Boosting; CatBoost will set ``langevin=True`` and a
    Bayesian-compatible bootstrap automatically. We do not set ``subsample``
    or ``rsm`` here — they are incompatible with the Bayesian bootstrap that
    posterior sampling requires.
    """
    params = {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": random_seed,
        "early_stopping_rounds": early_stopping_rounds,
        "border_count": border_count,
        "thread_count": thread_count,
        "use_best_model": True,
        "allow_writing_files": False,
        "verbose": verbose,
    }
    if posterior_sampling:
        params["posterior_sampling"] = True
    return params


def _make_pool(
    df: pd.DataFrame,
    feature_list: List[str],
    label_col: str,
    timestamp_col: str,
    weight_col: Optional[str],
) -> Pool:
    X = df[feature_list].to_numpy()
    y = df[label_col].astype(int).to_numpy()
    timestamps = df[timestamp_col].to_numpy(dtype=np.uint32)
    pool_kwargs: dict = {
        "data": X,
        "label": y,
        "timestamp": timestamps,
        "feature_names": feature_list,
    }
    if weight_col is not None and weight_col in df.columns:
        pool_kwargs["weight"] = df[weight_col].to_numpy(dtype=float)
    return Pool(**pool_kwargs)


def fit_research_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_list: List[str],
    *,
    label_col: str = "y",
    timestamp_col: str = "k",
    weight_col: Optional[str] = "weight",
    params: Optional[dict] = None,
) -> CatBoostClassifier:
    """Fit a single CatBoostClassifier on (train_df, val_df) with time-aware Pools."""
    if params is None:
        params = research_train_params()
    if weight_col is not None and weight_col in feature_list:
        raise ValueError(f"weight_col '{weight_col}' must not be in feature_list")

    train_pool = _make_pool(train_df, feature_list, label_col, timestamp_col, weight_col)
    val_pool = _make_pool(val_df, feature_list, label_col, timestamp_col, weight_col)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)
    return model


def compute_predictions(
    model: CatBoostClassifier,
    splits: Dict[str, pd.DataFrame],
    feature_list: List[str],
    *,
    regime_signal_col: str = DEFAULT_REGIME_SIGNAL,
) -> pd.DataFrame:
    """Run ``predict_proba`` on each split DataFrame and stack into one cache frame.

    ``splits`` is e.g. ``{"val": val_df, "test": test_df}``. The frame has
    columns ``CACHE_REQUIRED_COLS`` and is sorted by ``(split, k)``.
    """
    frames = []
    for split_name, df in splits.items():
        if regime_signal_col not in df.columns:
            raise ValueError(
                f"regime_signal_col '{regime_signal_col}' not in DataFrame for split '{split_name}'"
            )
        p = model.predict_proba(df[feature_list].to_numpy())[:, 1]
        n = len(df)
        cache = pd.DataFrame(
            {
                "k": df["k"].to_numpy(),
                "ts": df["ts"].to_numpy(),
                "y": df["y"].astype(int).to_numpy(),
                "m_k": df["m_k"].to_numpy() if "m_k" in df.columns else np.full(n, np.nan),
                "tau_k": df["tau_k"].to_numpy() if "tau_k" in df.columns else np.full(n, np.nan),
                "phi": df["phi"].to_numpy() if "phi" in df.columns else np.full(n, np.nan),
                "regime": df[regime_signal_col].to_numpy(dtype=float),
                "p": p.astype(float),
                "split": split_name,
            }
        )
        frames.append(cache)
    out = pd.concat(frames, ignore_index=True)
    out = out[CACHE_REQUIRED_COLS]
    out = out.sort_values(["split", "k"]).reset_index(drop=True)
    return out


def save_predictions_cache(cache: pd.DataFrame, path: str | Path) -> None:
    """Write the prediction cache to parquet, validating schema first."""
    missing = [c for c in CACHE_REQUIRED_COLS if c not in cache.columns]
    if missing:
        raise ValueError(f"cache missing required columns: {missing}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(path, index=False)


def load_predictions_cache(path: str | Path) -> pd.DataFrame:
    """Load and validate the prediction cache from parquet."""
    df = pd.read_parquet(path)
    missing = [c for c in CACHE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"cache at {path} missing required columns: {missing}")
    return df
