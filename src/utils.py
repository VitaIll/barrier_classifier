from __future__ import annotations

"""Project utilities (spec-driven).

Single source of truth: `docs/MINIMAL_PROJECT_SPEC_v2.md` (Minimal Barrier-Crossing
Classifier Specification v4.0).
"""

import hashlib
import json
import math
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# Appendix B: Configuration Constants (defaults must match spec)
# =============================================================================

# Numerical stability
EPS: float = 1e-10

# Symbol and interval
SYMBOL: str = "BTCUSDT"
INTERVAL: str = "1m"

# Date range
START_YEAR: int = 2025
START_MONTH: int = 1
END_YEAR: int = 2025
END_MONTH: int = 12

# Time indexing
M: int = 20  # Decision multiplier

# Label construction
ETA: float = 0.0002
C: float = 0.0023
PHI: float = C + ETA

# Feature windows
WINDOWS_F: List[int] = [
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    15,
    20,
    25,
    30,
    35,
    45,
    60,
    75,
    90,
    120,
    150,
    180,
    240,
    300,
    360,
    480,
    600,
    720,
    960,
    1440,
    1920,
    2880,
    4320,
    10080,
    20160,
]
WINDOWS_H: List[int] = [2, 3, 6, 12, 24, 36, 72, 144]
LAGS_F: List[int] = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    30,
    32,
    35,
    40,
    42,
    45,
    50,
    55,
    60,
    66,
    75,
    84,
    90,
    105,
    120,
    150,
    180,
    240,
    300,
    360,
    480,
    600,
    720,
    960,
    1440,
    2880,
    4320,
]
VOL_PAIRS: List[Tuple[int, int]] = [
    (10, 60),
    (10, 240),
    (20, 120),
    (20, 480),
    (30, 180),
    (60, 360),
    (60, 1440),
    (120, 720),
    (120, 2880),
    (240, 1440),
    (240, 4320),
    (720, 4320),
]

# Per-group window subsets (must match Appendix B)
WINDOWS_B: List[int] = WINDOWS_F
WINDOWS_BPLUS: List[int] = [30, 45, 60, 90, 120, 180, 240, 360, 720, 1440, 2880, 4320]
WINDOWS_VOL_OHLC: List[int] = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 1920, 2880, 4320]
WINDOWS_VOL_DECOMP: List[int] = [60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320]
WINDOWS_BARRIER: List[int] = [10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320]

WINDOWS_CANDLE_ROLL: List[int] = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480]
WINDOWS_BREAKOUT: List[int] = [20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 1920, 2880, 4320]

WINDOWS_EXCURSION: List[int] = [10, 20, 30, 60, 120, 240, 480, 960, 1440]
WINDOWS_MAXRET: List[int] = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]

WINDOWS_LOGP_Z: List[int] = [30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320]
WINDOWS_RSI: List[int] = [7, 10, 14, 20, 30, 45, 60, 90, 120, 180, 240, 360]

WINDOWS_LIQ_AMIHUD: List[int] = [15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
WINDOWS_LIQ_RPV: List[int] = [15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
WINDOWS_OFI_IMPULSE: List[int] = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360]

WINDOWS_CORR: List[int] = [30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
WINDOWS_PENTROPY: List[int] = [60, 120, 240, 720]

HITRATE_WINDOWS_H: List[int] = [3, 6, 12, 24, 36, 72, 144]

# Warmup & trimming (derived)
N_WARMUP: int = max(max(WINDOWS_F) - 1, max(LAGS_F), M * max(WINDOWS_H))
K_WARMUP: int = (N_WARMUP + M - 1) // M

# Train/val/test split
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
EMBARGO_K: int = 60
N_CV_FOLDS: int = 1

# CatBoost defaults
CB_ITERATIONS: int = 2000
CB_LEARNING_RATE: float = 0.03
CB_DEPTH: int = 6
CB_EARLY_STOPPING: int = 100
CB_SEED: int = 42
N_ENSEMBLE_MODELS: int = 5  # Number of models in ensemble
CB_L2_LEAF_REG: float = 3.0

# Optuna hyperparameter search
ENABLE_HPO: bool = False
OPTUNA_N_TRIALS: int = 500
OPTUNA_SEED: int = 42
# HPO train truncation (drop oldest fraction of observations; HPO-only)
HPO_DROP_OLDEST_FRAC: float = 0.6

# Observation weighting (optional)
WEIGHT_USE_BARRIER_DISTANCE: bool = True
WEIGHT_USE_TIME_DISCOUNT: bool = False
WEIGHT_DIST_W_MAX: float = 5.0     # negative-class cap (m_k < phi)
WEIGHT_DIST_Q_TAIL: float = 0.001  # negative-class tail quantile
WEIGHT_DIST_USE_POSITIVE: bool = False
WEIGHT_DIST_W_MAX_POS: float = 2.0
WEIGHT_DIST_Q_TAIL_POS: float = 0.01
WEIGHT_TIME_R: float = 0.0
WEIGHT_TIME_DELTA: float = 0.99999
WEIGHT_NORMALIZE: bool = False

# Catboost fixed params
CB_FIXED_PARAMS: Dict[str, Any] = {
    # Objective / metrics
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "custom_metric": ["Logloss", "AUC", "PRAUC"],

    # Training control
    "iterations": 1000,
    "early_stopping_rounds": 100,
    "use_best_model": True,

    # Imbalance
    "auto_class_weights": None,

    # Repro / perf
    "random_seed": CB_SEED,
    "verbose": False,
    "thread_count": -1,
    "allow_writing_files": False,

    # Ordered/time
    "boosting_type": "Ordered",
    "has_time": True,

    # Quantization
    "feature_border_type": "GreedyLogSum",
    "border_count": 128,

    # Tree shape
    "grow_policy": "SymmetricTree",

    # Sampling (MVS)
    "bootstrap_type": "MVS",

    # Leaf value optimization: Newton with 10 steps
    "leaf_estimation_method": "Newton",
    "leaf_estimation_iterations": 10,

    # Langevin / SGLB mode (CPU-only)
    "langevin": True,
}

# Hyperparameter search ranges (Optuna)
CB_HP_RANGES: Dict[str, Any] = {
    "learning_rate": (0.001, 0.05),
    "l2_leaf_reg": (0.1, 5.0),
    "diffusion_temperature": (5000,15000),
    "depth": (5, 8),
    "rsm": (0.65, 0.9),
    "subsample": (0.8, 1.0),
    "mvs_reg": (1.0, 10.0),
}

# =============================================================================
# Appendix E: Derivatives Configuration Constants
# =============================================================================

# Master switch (if False: skip all derivatives downloads + feature computation)
ENABLE_DERIVATIVES_FEATURES: bool = True

# Granular switches (all gated by ENABLE_DERIVATIVES_FEATURES)
ENABLE_FUTURES_KLINES: bool = True
ENABLE_FUNDING_RATE: bool = True
ENABLE_FUTURES_METRICS: bool = True
ENABLE_EOH_SUMMARY: bool = True
ENABLE_BVOL_INDEX: bool = True

# Optional extension (Section E.3.3)
ENABLE_DELIVERY_TERM_STRUCTURE: bool = False

# Derivatives feature windows
WINDOWS_BASIS: List[int] = [0, 5, 60]
WINDOWS_FLOW_CSUM: List[int] = [5, 10, 20]
WINDOWS_LIQ: List[int] = [0, 15]
WINDOWS_ACTIVITY_Z: List[int] = [30]
WINDOWS_OI_CHG: List[int] = [60, 120]
WINDOWS_FUNDING: List[int] = [0, 1440, 4320]
WINDOWS_OPTIONS: List[int] = [0, 1440]
WINDOWS_VOL_IDX: List[int] = [0, 1440, 43200]

# Default imputation helper (E.9)
MEDIAN_OI_USD: float = 15e9

# Derivatives data paths
DERIVATIVES_RAW_DIR: str = "data/raw_data/derivatives"
FUTURES_KLINES_PARQUET: str = f"{DERIVATIVES_RAW_DIR}/futures_klines_1m.parquet"
FUNDING_RATE_PARQUET: str = f"{DERIVATIVES_RAW_DIR}/funding_rate_1m.parquet"
FUTURES_METRICS_PARQUET: str = f"{DERIVATIVES_RAW_DIR}/futures_metrics_1m.parquet"
EOH_SUMMARY_PARQUET: str = f"{DERIVATIVES_RAW_DIR}/eoh_summary_1m.parquet"
BVOL_INDEX_PARQUET: str = f"{DERIVATIVES_RAW_DIR}/bvol_index_1m.parquet"
DERIVATIVES_VALIDATION_JSON: str = f"{DERIVATIVES_RAW_DIR}/derivatives_validation.json"

# Derivatives data availability
FUTURES_DATA_START: str = "2019-09-08"
FUTURES_METRICS_DATA_START: str = "2021-12-01"
EOH_DATA_START: str = "2023-05-18"
EOH_DATA_END: str = "2023-10-23"
BVOL_DATA_START: str = "2023-06-20"


# =============================================================================
# Utilities
# =============================================================================


class CatBoostEnsemble:
    """Wrapper for ensemble of CatBoost models with averaged predictions."""

    def __init__(self, models: List[Any]):
        if not models:
            raise ValueError("CatBoostEnsemble requires at least one model.")
        self.models = list(models)

    def predict_proba(self, X):
        probas = np.stack([m.predict_proba(X) for m in self.models], axis=0)
        return probas.mean(axis=0)

    def get_feature_importance(self, data=None, type: str = "PredictionValuesChange"):
        imps = [np.asarray(m.get_feature_importance(data=data, type=type), dtype=float) for m in self.models]
        return np.mean(imps, axis=0)

    def get_best_iteration(self) -> int:
        return int(np.mean([m.get_best_iteration() for m in self.models]))

    def get_evals_result(self) -> Dict[str, Dict[str, List[float]]]:
        results = [m.get_evals_result() for m in self.models]
        averaged: Dict[str, Dict[str, List[float]]] = {}
        for split_name, metrics in results[0].items():
            averaged[split_name] = {}
            for metric_name in metrics:
                curves = [r[split_name][metric_name] for r in results]
                min_len = min(len(c) for c in curves)
                if min_len == 0:
                    averaged[split_name][metric_name] = []
                    continue
                truncated = [np.asarray(c[:min_len], dtype=float) for c in curves]
                averaged[split_name][metric_name] = np.mean(truncated, axis=0).tolist()
        return averaged

    def save_model(self, path: str | Path) -> None:
        base = Path(path)
        for i, m in enumerate(self.models):
            m.save_model(str(base.with_suffix(f".{i}.cbm")))
        self.models[0].save_model(str(base))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, default=_json_default)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def assert_columns(df: pd.DataFrame, required: Iterable[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def assert_index_is_utc_datetime_index(df: pd.DataFrame, *, context: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{context}: expected DatetimeIndex, got {type(df.index)}")
    if df.index.tz is None:
        raise ValueError(f"{context}: expected timezone-aware UTC index, got tz=None")
    if str(df.index.tz) != "UTC":
        raise ValueError(f"{context}: expected UTC timezone, got {df.index.tz}")


# =============================================================================
# Section 4: Binance Data Acquisition
# =============================================================================


def generate_download_urls(
    symbol: str,
    interval: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> List[Dict[str, Any]]:
    urls: List[Dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == start_year and month < start_month:
                continue
            if year == end_year and month > end_month:
                continue

            filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
            base = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}"
            urls.append(
                {
                    "year": year,
                    "month": month,
                    "data_url": f"{base}/{filename}",
                    "checksum_url": f"{base}/{filename}.CHECKSUM",
                    "filename": filename,
                }
            )
    return urls


def generate_futures_klines_urls(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    start_year: int = START_YEAR,
    start_month: int = START_MONTH,
    end_year: int = END_YEAR,
    end_month: int = END_MONTH,
) -> List[Dict[str, Any]]:
    urls: List[Dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == start_year and month < start_month:
                continue
            if year == end_year and month > end_month:
                continue

            filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
            base = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}"
            urls.append(
                {
                    "year": year,
                    "month": month,
                    "data_url": f"{base}/{filename}",
                    "checksum_url": f"{base}/{filename}.CHECKSUM",
                    "filename": filename,
                    "source": "futures_klines",
                }
            )
    return urls


def generate_funding_rate_urls(
    symbol: str = SYMBOL,
    start_year: int = START_YEAR,
    start_month: int = START_MONTH,
    end_year: int = END_YEAR,
    end_month: int = END_MONTH,
) -> List[Dict[str, Any]]:
    urls: List[Dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == start_year and month < start_month:
                continue
            if year == end_year and month > end_month:
                continue

            filename = f"{symbol}-fundingRate-{year}-{month:02d}.zip"
            base = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}"
            urls.append(
                {
                    "year": year,
                    "month": month,
                    "data_url": f"{base}/{filename}",
                    "checksum_url": f"{base}/{filename}.CHECKSUM",
                    "filename": filename,
                    "source": "funding_rate",
                }
            )
    return urls


def generate_futures_metrics_urls(
    symbol: str = SYMBOL,
    start_year: int = START_YEAR,
    start_month: int = START_MONTH,
    end_year: int = END_YEAR,
    end_month: int = END_MONTH,
) -> List[Dict[str, Any]]:
    start_date = pd.Timestamp(f"{start_year}-{start_month:02d}-01", tz="UTC")
    end_date = pd.Timestamp(f"{end_year}-{end_month:02d}-01", tz="UTC") + pd.offsets.MonthEnd(1)
    metrics_start = max(start_date, pd.Timestamp(FUTURES_METRICS_DATA_START, tz="UTC"))

    urls: List[Dict[str, Any]] = []
    for date in pd.date_range(metrics_start, end_date, freq="D"):
        ymd = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-metrics-{ymd}.zip"
        base = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}"
        urls.append(
            {
                "date": ymd,
                "data_url": f"{base}/{filename}",
                "checksum_url": f"{base}/{filename}.CHECKSUM",
                "filename": filename,
                "source": "futures_metrics",
            }
        )
    return urls


def generate_eoh_summary_urls(
    symbol: str = SYMBOL,
    start_year: int = START_YEAR,
    start_month: int = START_MONTH,
    end_year: int = END_YEAR,
    end_month: int = END_MONTH,
) -> List[Dict[str, Any]]:
    start_date = pd.Timestamp(f"{start_year}-{start_month:02d}-01", tz="UTC")
    end_date = pd.Timestamp(f"{end_year}-{end_month:02d}-01", tz="UTC") + pd.offsets.MonthEnd(1)

    eoh_start = max(start_date, pd.Timestamp(EOH_DATA_START, tz="UTC"))
    eoh_end = min(end_date, pd.Timestamp(EOH_DATA_END, tz="UTC"))

    urls: List[Dict[str, Any]] = []
    if eoh_start <= eoh_end:
        for date in pd.date_range(eoh_start, eoh_end, freq="D"):
            ymd = date.strftime("%Y-%m-%d")
            filename = f"{symbol}-EOHSummary-{ymd}.zip"
            base = f"https://data.binance.vision/data/option/daily/EOHSummary/{symbol}"
            urls.append(
                {
                    "date": ymd,
                    "data_url": f"{base}/{filename}",
                    "checksum_url": f"{base}/{filename}.CHECKSUM",
                    "filename": filename,
                    "source": "eoh_summary",
                }
            )
    return urls


def generate_bvol_index_urls(
    symbol: str = "BTCBVOLUSDT",
    start_year: int = START_YEAR,
    start_month: int = START_MONTH,
    end_year: int = END_YEAR,
    end_month: int = END_MONTH,
) -> List[Dict[str, Any]]:
    start_date = pd.Timestamp(f"{start_year}-{start_month:02d}-01", tz="UTC")
    end_date = pd.Timestamp(f"{end_year}-{end_month:02d}-01", tz="UTC") + pd.offsets.MonthEnd(1)
    bvol_start = max(start_date, pd.Timestamp(BVOL_DATA_START, tz="UTC"))

    urls: List[Dict[str, Any]] = []
    for date in pd.date_range(bvol_start, end_date, freq="D"):
        ymd = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-BVOLIndex-{ymd}.zip"
        base = f"https://data.binance.vision/data/option/daily/BVOLIndex/{symbol}"
        urls.append(
            {
                "date": ymd,
                "data_url": f"{base}/{filename}",
                "checksum_url": f"{base}/{filename}.CHECKSUM",
                "filename": filename,
                "source": "bvol_index",
            }
        )
    return urls


def generate_all_derivatives_urls(
    start_year: int = START_YEAR,
    start_month: int = START_MONTH,
    end_year: int = END_YEAR,
    end_month: int = END_MONTH,
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "futures_klines": generate_futures_klines_urls(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
        ),
        "funding_rate": generate_funding_rate_urls(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
        ),
        "futures_metrics": generate_futures_metrics_urls(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
        ),
        "eoh_summary": generate_eoh_summary_urls(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
        ),
        "bvol_index": generate_bvol_index_urls(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
        ),
    }


def generate_derivatives_download_urls(
    start_year: int = START_YEAR,
    start_month: int = START_MONTH,
    end_year: int = END_YEAR,
    end_month: int = END_MONTH,
) -> Dict[str, List[Dict[str, Any]]]:
    return generate_all_derivatives_urls(
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
    )


def download_file(
    url: str,
    output_path: Path,
    *,
    max_retries: int = 3,
    timeout: int = 60,
    chunk_size: int = 8192,
) -> bool:
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            with temp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            temp_path.replace(output_path)
            return True
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            continue
    return False


def verify_checksum(zip_path: Path, checksum_path: Path) -> bool:
    with checksum_path.open("r", encoding="utf-8") as f:
        expected_hash = f.read().strip().split()[0].lower()

    sha256 = hashlib.sha256()
    with zip_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest().lower()
    return expected_hash == actual_hash


def extract_and_load_csv(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = zip_path.stem + ".csv"
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f,
                header=None,
                names=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "num_trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
                dtype={
                    "open_time": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                    "close_time": "int64",
                    "quote_volume": "float64",
                    "num_trades": "int64",
                    "taker_buy_base": "float64",
                    "taker_buy_quote": "float64",
                    "ignore": "string",
                },
            )

    return df


def _repair_out_of_bounds_open_time(open_time_ms: pd.Series, step_ms: int) -> Tuple[pd.Series, int]:
    ts = pd.to_datetime(open_time_ms + step_ms, unit="ms", utc=True, errors="coerce")
    invalid = ts.isna()
    n_invalid = int(invalid.sum())
    if n_invalid == 0:
        return open_time_ms, 0

    values = open_time_ms.to_numpy(copy=True, dtype="int64")
    invalid_mask = invalid.to_numpy()
    if invalid_mask.any():
        min_ms = pd.Timestamp.min.value // 1_000_000
        max_ms = pd.Timestamp.max.value // 1_000_000
        invalid_values = values[invalid_mask]

        scaled_1k = invalid_values // 1_000
        in_range_1k = (scaled_1k + step_ms >= min_ms) & (scaled_1k + step_ms <= max_ms)
        if in_range_1k.any():
            invalid_idx = np.flatnonzero(invalid_mask)
            values[invalid_idx[in_range_1k]] = scaled_1k[in_range_1k]
            invalid_mask[invalid_idx[in_range_1k]] = False

        if invalid_mask.any():
            invalid_values = values[invalid_mask]
            scaled_1m = invalid_values // 1_000_000
            in_range_1m = (scaled_1m + step_ms >= min_ms) & (scaled_1m + step_ms <= max_ms)
            if in_range_1m.any():
                invalid_idx = np.flatnonzero(invalid_mask)
                values[invalid_idx[in_range_1m]] = scaled_1m[in_range_1m]
                invalid_mask[invalid_idx[in_range_1m]] = False
    valid_idx = np.flatnonzero(~invalid_mask)
    if valid_idx.size == 0:
        raise ValueError("convert_timestamps: all open_time values out of bounds")

    last_valid = None
    for i in range(len(values)):
        if not invalid_mask[i]:
            last_valid = i
        elif last_valid is not None:
            values[i] = values[last_valid] + step_ms * (i - last_valid)

    first_valid = int(valid_idx[0])
    for i in range(first_valid - 1, -1, -1):
        values[i] = values[i + 1] - step_ms

    ts_check = pd.to_datetime(values + step_ms, unit="ms", utc=True, errors="coerce")
    if ts_check.isna().any():
        bad = int(ts_check.isna().sum())
        raise ValueError(f"convert_timestamps: {bad} timestamps remain out of bounds after repair")

    return pd.Series(values, index=open_time_ms.index), n_invalid


def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Canonical timestamp: moment when the bar is complete.
    # Spec convention: ts_n = close_time_n + 1ms = open_time_n + 60s (for 1m bars).
    #
    # Using open_time + 60s is numerically equivalent under the intended Binance
    # schema and is more robust to rare close_time anomalies.
    open_time, _ = _repair_out_of_bounds_open_time(df["open_time"], 60_000)
    df["ts"] = pd.to_datetime(open_time + 60_000, unit="ms", utc=True)
    df = df.set_index("ts")
    df = df.drop(columns=["open_time", "close_time"])
    return df


def load_futures_klines(zip_path: Path) -> pd.DataFrame:
    """
    Load futures klines CSV from ZIP archive (header row present).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = zip_path.stem + ".csv"
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f,
                dtype={
                    "open_time": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                    "close_time": "int64",
                    "quote_volume": "float64",
                    "count": "int64",
                    "taker_buy_volume": "float64",
                    "taker_buy_quote_volume": "float64",
                    "ignore": "string",
                },
            )

    df = df.rename(
        columns={
            "count": "num_trades",
            "taker_buy_volume": "taker_buy_base",
            "taker_buy_quote_volume": "taker_buy_quote",
        }
    )
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    required = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
        "taker_buy_base",
    ]
    assert_columns(df, required, context="load_futures_klines")

    df = convert_timestamps(df)
    df = df.sort_index()
    return df


def load_funding_rate(zip_path: Path) -> pd.DataFrame:
    """
    Load funding rate CSV from ZIP archive.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = zip_path.stem + ".csv"
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    assert_columns(df, ["calc_time", "last_funding_rate"], context="load_funding_rate")
    df["ts"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[["last_funding_rate"]].rename(columns={"last_funding_rate": "funding_rate"})
    return df


def load_futures_metrics(zip_path: Path) -> pd.DataFrame:
    """
    Load futures metrics CSV from ZIP archive (5-minute snapshots).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = zip_path.stem + ".csv"
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    assert_columns(df, ["create_time", "sum_open_interest_value"], context="load_futures_metrics")
    df["ts"] = pd.to_datetime(df["create_time"], utc=True)
    df = df.set_index("ts").sort_index()

    df["oi_usd"] = pd.to_numeric(df["sum_open_interest_value"], errors="coerce")
    ratio_cols = [
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    keep_cols = ["oi_usd"]
    for col in ratio_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            keep_cols.append(col)

    return df[keep_cols]


def load_eoh_summary(zip_path: Path) -> pd.DataFrame:
    """
    Load and aggregate EOHSummary CSV from ZIP archive.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = zip_path.stem + ".csv"
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    required = ["date", "hour", "type", "volume_usdt", "openinterest_usdt"]
    assert_columns(df, required, context="load_eoh_summary")

    ts = pd.to_datetime(df["date"] + " " + df["hour"].astype(str).str.zfill(2) + ":00:00", utc=True)
    df = df.assign(ts=ts)

    df["volume_usdt"] = pd.to_numeric(df["volume_usdt"], errors="coerce").fillna(0.0)
    df["openinterest_usdt"] = pd.to_numeric(df["openinterest_usdt"], errors="coerce").fillna(0.0)

    is_put = df["type"].astype(str).str.upper().eq("P")
    is_call = df["type"].astype(str).str.upper().eq("C")

    g = df.groupby("ts", sort=True)
    out = pd.DataFrame(index=g.size().index)
    out["opt_oi"] = g["openinterest_usdt"].sum()
    out["opt_volume"] = g["volume_usdt"].sum()
    out["put_open_interest"] = df[is_put].groupby("ts", sort=True)["openinterest_usdt"].sum()
    out["call_open_interest"] = df[is_call].groupby("ts", sort=True)["openinterest_usdt"].sum()
    out["put_volume"] = df[is_put].groupby("ts", sort=True)["volume_usdt"].sum()
    out["call_volume"] = df[is_call].groupby("ts", sort=True)["volume_usdt"].sum()

    return out.sort_index().fillna(0.0)


def load_bvol_index(zip_path: Path) -> pd.DataFrame:
    """
    Load BVOL index CSV from ZIP archive.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = zip_path.stem + ".csv"
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    assert_columns(df, ["calc_time", "index_value"], context="load_bvol_index")
    df["ts"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df["bvol"] = pd.to_numeric(df["index_value"], errors="coerce")
    return df[["bvol"]]


def align_to_1m_grid(
    df_source: pd.DataFrame,
    index_1m: pd.DatetimeIndex,
    method: Optional[str],
) -> pd.DataFrame:
    """
    Align a source DataFrame to a 1-minute grid.

    For piecewise-constant sources, use method="ffill".
    Do not extrapolate beyond source coverage.
    """
    df = df_source.copy()
    assert_index_is_utc_datetime_index(df, context="align_to_1m_grid: df_source")
    if not isinstance(index_1m, pd.DatetimeIndex):
        raise ValueError("align_to_1m_grid: index_1m must be a DatetimeIndex")
    if index_1m.tz is None or str(index_1m.tz) != "UTC":
        raise ValueError("align_to_1m_grid: index_1m must be UTC")

    df = df.sort_index()
    if method is None:
        aligned = df.reindex(index_1m)
    else:
        if method not in ("ffill", "pad"):
            raise ValueError(f"align_to_1m_grid: unsupported method '{method}'")
        aligned = df.reindex(index_1m, method="ffill")

    if len(df) > 0:
        aligned.loc[aligned.index < df.index.min(), :] = np.nan
        aligned.loc[aligned.index > df.index.max(), :] = np.nan
    return aligned


def expected_bar_count(start_open: datetime, end_open_exclusive: datetime) -> int:
    delta = end_open_exclusive - start_open
    return int(delta.total_seconds() / 60)


def validate_klines(df: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "date_range": (df.index.min(), df.index.max()),
        "issues": [],
    }

    assert_index_is_utc_datetime_index(df, context="validate_klines")

    invalid_high = df["high"] < df[["open", "close"]].max(axis=1)
    invalid_low = df["low"] > df[["open", "close"]].min(axis=1)
    invalid_range = df["high"] < df["low"]
    non_positive = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)

    if invalid_high.any():
        results["issues"].append(f"High < max(O,C): {int(invalid_high.sum())} bars")
    if invalid_low.any():
        results["issues"].append(f"Low > min(O,C): {int(invalid_low.sum())} bars")
    if invalid_range.any():
        results["issues"].append(f"High < Low: {int(invalid_range.sum())} bars")
    if non_positive.any():
        results["issues"].append(f"Non-positive OHLC: {int(non_positive.sum())} bars")

    negative_vol = df["volume"] < 0
    invalid_taker = df["taker_buy_base"] > df["volume"]

    if negative_vol.any():
        results["issues"].append(f"Negative volume: {int(negative_vol.sum())} bars")
    if invalid_taker.any():
        results["issues"].append(f"Taker > Volume: {int(invalid_taker.sum())} bars")

    dups = df.index.duplicated()
    if dups.any():
        results["issues"].append(f"Duplicate timestamps: {int(dups.sum())}")

    time_diffs = df.index.to_series().diff().dropna()
    expected_diff = pd.Timedelta(minutes=1)
    gaps = time_diffs[time_diffs != expected_diff]
    if len(gaps) > 0:
        results["issues"].append(f"Gaps detected: {int(len(gaps))}")
        results["gap_locations"] = {k.isoformat(): str(v) for k, v in gaps.head(10).items()}

    if not df.index.is_monotonic_increasing:
        results["issues"].append("Timestamps not monotonic increasing")

    results["is_valid"] = len(results["issues"]) == 0
    return results


# =============================================================================
# Section 8A: Pipeline Validation Checkpoints
# =============================================================================


def checkpoint_raw_data(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    assert_index_is_utc_datetime_index(df, context="checkpoint_raw_data")

    actual_start = df.index.min()
    actual_end = df.index.max()

    expected_start_open = pd.Timestamp(
        f"{config['START_YEAR']}-{config['START_MONTH']:02d}-01",
        tz="UTC",
    )
    expected_end_open_exclusive = pd.Timestamp(
        f"{config['END_YEAR']}-{config['END_MONTH']:02d}-01",
        tz="UTC",
    ) + pd.offsets.MonthBegin(1)

    expected_start_ts = expected_start_open + pd.Timedelta(minutes=1)
    expected_end_ts = expected_end_open_exclusive
    tol = pd.Timedelta(minutes=1)

    results["date_range"] = {
        "actual": (actual_start, actual_end),
        "expected": (expected_start_ts, expected_end_ts),
        "start_ok": (expected_start_ts - tol) <= actual_start <= (expected_start_ts + tol),
        "end_ok": (expected_end_ts - tol) <= actual_end <= (expected_end_ts + tol),
    }

    expected_minutes = int((expected_end_open_exclusive - expected_start_open).total_seconds() / 60)
    results["row_count"] = {
        "actual": int(len(df)),
        "expected_approx": int(expected_minutes),
        "ratio": float(len(df) / expected_minutes),
        "ok": 0.99 < len(df) / expected_minutes < 1.01,
    }

    dup_count = int(df.index.duplicated().sum())
    results["duplicates"] = {"count": dup_count, "ok": dup_count == 0}
    results["monotonic"] = {"ok": bool(df.index.is_monotonic_increasing)}

    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ).sum()
    results["ohlc_valid"] = {"invalid_count": int(invalid_ohlc), "ok": int(invalid_ohlc) == 0}

    negative_vol = int((df["volume"] < 0).sum())
    results["volume_valid"] = {"negative_count": negative_vol, "ok": negative_vol == 0}

    critical_ok = all(
        [
            results["date_range"]["start_ok"],
            results["date_range"]["end_ok"],
            results["duplicates"]["ok"],
            results["monotonic"]["ok"],
        ]
    )
    if not critical_ok:
        raise ValueError(f"Critical data validation failed: {results}")

    print("OK: Raw data validation passed")
    return results


def checkpoint_base_series(df: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    required = ["p", "r", "rho", "ofi", "clv", "logvol", "b"]
    results["columns_exist"] = {
        "missing": [c for c in required if c not in df.columns],
        "ok": all(c in df.columns for c in required),
    }
    if not results["columns_exist"]["ok"]:
        raise ValueError(f"Base series missing required columns: {results['columns_exist']['missing']}")

    results["return_nan_at_0"] = {
        "r_0_is_nan": bool(pd.isna(df["r"].iloc[0])),
        "ok": bool(pd.isna(df["r"].iloc[0])),
    }
    if not results["return_nan_at_0"]["ok"]:
        raise ValueError("Base series r[0] must be NaN (return undefined at first bar)")

    sample_idx = len(df) // 2
    results["log_price_correct"] = {
        "ok": bool(np.isclose(df["p"].iloc[sample_idx], np.log(df["close"].iloc[sample_idx]))),
    }
    if not results["log_price_correct"]["ok"]:
        raise ValueError("Base series p must equal log(close)")

    ofi_valid = df["ofi"].dropna()
    if len(ofi_valid) > 0:
        results["ofi_bounded"] = {
            "min": float(ofi_valid.min()),
            "max": float(ofi_valid.max()),
            "ok": float(ofi_valid.min()) >= -1.0 and float(ofi_valid.max()) <= 1.0,
        }
        if not results["ofi_bounded"]["ok"]:
            raise ValueError("OFI must be bounded in [-1, 1] where defined")

    print("OK: Base series validation passed")
    return results


def checkpoint_derivatives_data(
    df_futures: pd.DataFrame,
    df_funding: pd.DataFrame,
    df_metrics: pd.DataFrame,
    df_eoh: pd.DataFrame,
    df_bvol: pd.DataFrame,
    df_spot: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Validate derivatives data before feature computation.
    """
    results: Dict[str, Any] = {}

    futures_aligned = df_futures.index.equals(df_spot.index)
    results["futures_index_aligned"] = {"ok": bool(futures_aligned)}

    corr = df_futures["close"].corr(df_spot["close"])
    results["futures_spot_corr"] = {"value": float(corr) if corr is not None else float("nan"), "ok": corr > 0.99}

    funding_range_ok = df_funding["funding_rate"].dropna().between(-0.1, 0.1).all()
    results["funding_range"] = {"ok": bool(funding_range_ok)}

    oi_ok = df_metrics["oi_usd"].dropna().ge(0).all()
    results["oi_non_negative"] = {"ok": bool(oi_ok)}

    eoh_coverage = df_eoh.notna().mean().mean() if len(df_eoh) else 0.0
    results["eoh_coverage"] = {"value": float(eoh_coverage)}

    bvol_valid = df_bvol["bvol"].between(10, 300).mean() if len(df_bvol) else 0.0
    results["bvol_valid_frac"] = {"value": float(bvol_valid), "ok": float(bvol_valid) > 0.95}

    all_ok = all(r.get("ok", True) for r in results.values() if isinstance(r, dict))
    results["all_ok"] = bool(all_ok)

    if all_ok:
        print("OK: Derivatives data validation passed")
    else:
        print("WARN: Derivatives data validation FAILED")
        for key, val in results.items():
            if isinstance(val, dict) and not val.get("ok", True):
                print(f"  - {key}: {val}")
    return results


def checkpoint_derivatives_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate derivatives features after computation.
    """
    results: Dict[str, Any] = {}

    expected = [
        "basis__pct__f__w0",
        "flow__taker_buy_ratio__f__w0",
        "funding__rate__f__w0",
        "opt_pcr__oi__f__w0",
        "vol_idx__bvol30d__f__w0",
    ]
    missing = [f for f in expected if f not in df.columns]
    results["features_exist"] = {"missing": missing, "ok": len(missing) == 0}

    if "basis__pct__f__w0" in df.columns:
        basis_pct = df["basis__pct__f__w0"].dropna()
        results["basis_range"] = {
            "min": float(basis_pct.min()) if len(basis_pct) else float("nan"),
            "max": float(basis_pct.max()) if len(basis_pct) else float("nan"),
            "ok": float(basis_pct.between(-5, 5).mean()) > 0.99 if len(basis_pct) else True,
        }

    if "opt_pcr__oi__f__w0" in df.columns:
        pcr = df["opt_pcr__oi__f__w0"].dropna()
        results["pcr_range"] = {
            "min": float(pcr.min()) if len(pcr) else float("nan"),
            "max": float(pcr.max()) if len(pcr) else float("nan"),
            "ok": float(pcr.between(0.1, 10).mean()) > 0.95 if len(pcr) else True,
        }

    all_ok = all(r.get("ok", True) for r in results.values() if isinstance(r, dict))
    results["all_ok"] = bool(all_ok)

    if all_ok:
        print("OK: Derivatives features validation passed")
    else:
        print("WARN: Derivatives features validation FAILED")
    return results


def checkpoint_boundaries(df_boundaries: pd.DataFrame, df_raw: pd.DataFrame, M_: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    expected_k = ((len(df_raw) - 1) // M_) + 1
    results["boundary_count"] = {
        "actual": int(len(df_boundaries)),
        "expected": int(expected_k),
        "ok": int(len(df_boundaries)) == int(expected_k),
    }
    if not results["boundary_count"]["ok"]:
        raise ValueError(f"Boundary count mismatch: {results['boundary_count']}")

    results["k_sequential"] = {"ok": bool((df_boundaries["k"].diff().dropna() == 1).all())}
    if not results["k_sequential"]["ok"]:
        raise ValueError("Boundary k column must be sequential increments of 1")

    if "ts" not in df_boundaries.columns:
        raise ValueError("Boundary DataFrame must contain 'ts' column (reset_index() required)")
    ts_diffs = df_boundaries["ts"].diff().dropna()
    expected_diff = pd.Timedelta(minutes=M_)
    results["timestamp_spacing"] = {
        "all_equal": bool((ts_diffs == expected_diff).all()),
        "ok": float((ts_diffs == expected_diff).mean()) > 0.99,
    }
    if not results["timestamp_spacing"]["ok"]:
        raise ValueError(f"Boundary timestamp spacing invalid: {results['timestamp_spacing']}")

    print("OK: Decision boundary validation passed")
    return results


def checkpoint_labels(df: pd.DataFrame, phi: float) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    unique_labels = df["y"].dropna().unique()
    results["labels_binary"] = {"unique_values": sorted(unique_labels.tolist()), "ok": set(unique_labels) <= {0, 1}}
    if not results["labels_binary"]["ok"]:
        raise ValueError(f"Labels not binary: {results['labels_binary']}")

    base_rate = float(df["y"].mean())
    results["base_rate"] = {"value": base_rate, "ok": 0.05 < base_rate < 0.70}
    if not results["base_rate"]["ok"]:
        print(f"Warning: base rate outside typical range: {base_rate:.4f}")

    results["m_k_reasonable"] = {
        "min": float(df["m_k"].min()),
        "max": float(df["m_k"].max()),
        "ok": float(df["m_k"].min()) > -0.5 and float(df["m_k"].max()) < 0.5,
    }
    if not results["m_k_reasonable"]["ok"]:
        raise ValueError(f"m_k values out of sanity bounds: {results['m_k_reasonable']}")

    label_check = ((df["m_k"] >= phi) == (df["y"] == 1)).dropna().all()
    results["label_consistent"] = {"ok": bool(label_check)}
    if not results["label_consistent"]["ok"]:
        raise ValueError("Label y is inconsistent with m_k and phi")

    print("OK: Label validation passed")
    return results


def checkpoint_warmup_trimmed(df: pd.DataFrame, k_warmup: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    results["warmup_removed"] = {
        "min_k": int(df["k"].min()),
        "k_warmup": int(k_warmup),
        "ok": int(df["k"].min()) >= int(k_warmup),
    }
    if not results["warmup_removed"]["ok"]:
        raise ValueError(f"Warmup trimming failed: {results['warmup_removed']}")

    nan_labels = int(df["y"].isna().sum())
    results["no_nan_labels"] = {"nan_count": nan_labels, "ok": nan_labels == 0}
    if not results["no_nan_labels"]["ok"]:
        raise ValueError("Warmup-trimmed dataset must not contain NaN labels")

    results["rows_remaining"] = {"count": int(len(df)), "ok": int(len(df)) > 10000}
    if not results["rows_remaining"]["ok"]:
        raise ValueError(f"Too few rows after trimming: {results['rows_remaining']}")

    print("OK: Warmup trimming validation passed")
    return results


def checkpoint_final_dataset(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    nan_counts = df[feature_cols].isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    results["no_nans"] = {"features_with_nan": nan_features.to_dict(), "ok": len(nan_features) == 0}
    if not results["no_nans"]["ok"]:
        raise ValueError(f"Features with NaN after imputation: {nan_features.to_dict()}")

    results["feature_count"] = {
        "actual": int(len(feature_cols)),
        "expected_min": 1200,
        "expected_max": 1700,
        "ok": 1200 <= len(feature_cols) <= 1700,
    }
    if not results["feature_count"]["ok"]:
        raise ValueError(f"Feature count outside expected range: {results['feature_count']}")

    label_aux_cols = {"m_k", "tau_k", "phi"}
    present = sorted(label_aux_cols.intersection(set(feature_cols)))
    results["no_label_aux_in_features"] = {"present": present, "ok": len(present) == 0}
    if not results["no_label_aux_in_features"]["ok"]:
        raise ValueError(f"Label-derived columns in feature set: {present}")

    weight_cols = {"w_dist", "w_time", "weight"}
    present_weights = sorted(weight_cols.intersection(set(feature_cols)))
    results["no_weights_in_features"] = {"present": present_weights, "ok": len(present_weights) == 0}
    if not results["no_weights_in_features"]["ok"]:
        raise ValueError(f"Weight columns included in feature set: {present_weights}")

    numeric = df[feature_cols].select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric).sum()
    inf_features = inf_counts[inf_counts > 0]
    results["no_infs"] = {"features_with_inf": inf_features.to_dict(), "ok": len(inf_features) == 0}
    if not results["no_infs"]["ok"]:
        raise ValueError(f"Features with inf after imputation: {inf_features.to_dict()}")

    results["labels_valid"] = {"ok": bool(df["y"].isin([0, 1]).all())}
    if not results["labels_valid"]["ok"]:
        raise ValueError("Labels must be 0/1 with no NaNs")

    results["ts_monotonic"] = {"ok": bool(df["ts"].is_monotonic_increasing)}
    if not results["ts_monotonic"]["ok"]:
        raise ValueError("Final dataset timestamps must be monotonic increasing")

    results["k_sequential"] = {"ok": bool((df["k"].diff().dropna() == 1).all())}
    if not results["k_sequential"]["ok"]:
        raise ValueError("Final dataset k must be sequential after trimming")

    print("OK: Final dataset validation passed")
    print(f"  Rows: {len(df):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Base rate: {df['y'].mean():.3f}")
    print(f"  Date range: {df['ts'].min()} to {df['ts'].max()}")
    return results


def checkpoint_before_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embargo_k: int,
    *,
    train_frac: Optional[float] = None,
    val_frac: Optional[float] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    if train_frac is None:
        train_frac = TRAIN_FRAC
    if val_frac is None:
        val_frac = VAL_FRAC
    test_frac = 1.0 - float(train_frac) - float(val_frac)
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"train_frac + val_frac must be < 1, got {train_frac + val_frac}")

    results["chronological"] = {
        "train_max_k": int(train_df["k"].max()),
        "val_min_k": int(val_df["k"].min()),
        "val_max_k": int(val_df["k"].max()),
        "test_min_k": int(test_df["k"].min()),
        "train_before_val": bool(train_df["k"].max() < val_df["k"].min()),
        "val_before_test": bool(val_df["k"].max() < test_df["k"].min()),
        "ok": bool((train_df["k"].max() < val_df["k"].min()) and (val_df["k"].max() < test_df["k"].min())),
    }
    if not results["chronological"]["ok"]:
        raise ValueError(f"Split chronological order invalid: {results['chronological']}")

    results["embargo"] = {
        "train_val_gap": int(val_df["k"].min() - train_df["k"].max()),
        "val_test_gap": int(test_df["k"].min() - val_df["k"].max()),
        "embargo_k": int(embargo_k),
        "ok": bool(
            (val_df["k"].min() - train_df["k"].max() >= embargo_k)
            and (test_df["k"].min() - val_df["k"].max() >= embargo_k)
        ),
    }
    if not results["embargo"]["ok"]:
        raise ValueError(f"Embargo violated: {results['embargo']}")

    total = len(train_df) + len(val_df) + len(test_df)
    n_total_est = int(total + 2 * int(embargo_k))
    train_end_idx = int(float(train_frac) * n_total_est)
    val_end_idx = int((float(train_frac) + float(val_frac)) * n_total_est)
    expected_train_n = int(train_end_idx)
    expected_val_n = int(val_end_idx - (train_end_idx + int(embargo_k)))
    expected_test_n = int(n_total_est - (val_end_idx + int(embargo_k)))
    if expected_val_n <= 0 or expected_test_n <= 0:
        raise ValueError(
            "Configured split fractions are incompatible with embargo "
            f"(n_total_est={n_total_est}, train_frac={train_frac}, val_frac={val_frac}, embargo_k={embargo_k}; "
            f"expected sizes train={expected_train_n}, val={expected_val_n}, test={expected_test_n})"
        )

    results["split_sizes"] = {
        "train_frac": float(len(train_df) / total),
        "val_frac": float(len(val_df) / total),
        "test_frac": float(len(test_df) / total),
        "configured": {
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
            "test_frac": float(test_frac),
            "embargo_k": int(embargo_k),
        },
        "expected_sizes": {
            "n_total_est": int(n_total_est),
            "train_n": int(expected_train_n),
            "val_n": int(expected_val_n),
            "test_n": int(expected_test_n),
        },
        "actual_sizes": {
            "train_n": int(len(train_df)),
            "val_n": int(len(val_df)),
            "test_n": int(len(test_df)),
        },
        "ok": bool(
            (len(train_df) == expected_train_n)
            and (len(val_df) == expected_val_n)
            and (len(test_df) == expected_test_n)
        ),
    }
    if not results["split_sizes"]["ok"]:
        raise ValueError(f"Split sizes do not match configured fractions: {results['split_sizes']}")

    # Non-feature columns — must mirror the pipeline's NON_FEATURE_COLS
    # contract: labels, weights, raw OHLCV, base series, and derivatives
    # base series. The dataset preserves these as sidecar columns (they're
    # carried through for diagnostics but excluded from imputation), so
    # they can carry NaN where coverage is missing (e.g. EOH options
    # outside their coverage window). Treating them as features would
    # trip on legitimate sidecar NaN.
    non_feature_cols = [
        "k", "ts", "y", "m_k", "tau_k", "phi", "m_dn", "tau_dn",
        "w_dist", "w_time", "weight",
        # Raw OHLCV
        "open", "high", "low", "close", "volume", "quote_volume",
        "num_trades", "taker_buy_base", "taker_buy_quote",
        # Base series (from compute_base_series)
        "p", "r", "rho", "r_oc", "g", "logvol", "logtrades", "logquotevol",
        "b", "ofi", "clv", "bodyfrac", "wickup", "wickdn", "vwap", "vwapdev",
        "qpertrade",
        # Derivatives base (from compute_derivatives_base_series)
        "close_fut", "volume_fut", "quote_volume_fut", "taker_buy_base_fut",
        "num_trades_fut", "funding_rate", "oi_usd", "opt_oi",
        "put_open_interest", "call_open_interest", "opt_volume", "put_volume",
        "call_volume", "bvol", "basis_abs", "basis_pct", "tb_ratio_fut",
        "net_vol_fut", "pcr_oi", "pcr_vol",
    ]
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        drop_cols = [c for c in non_feature_cols if c in split_df.columns]
        check_df = split_df.drop(columns=drop_cols, errors="ignore")
        nan_per_col = check_df.isna().sum()
        nan_features = nan_per_col[nan_per_col > 0]
        nan_count = int(nan_features.sum())
        results[f"{name}_no_nans"] = {"nan_count": nan_count, "ok": nan_count == 0}
        if nan_count != 0:
            raise ValueError(
                f"NaNs present in {name} split feature columns: {nan_features.to_dict()} "
                f"(excluded non-feature cols: {drop_cols})"
            )

    print("OK: Pre-training validation passed")
    print(f"  Train: {len(train_df):,} ({len(train_df)/total:.1%})")
    print(f"  Val:   {len(val_df):,} ({len(val_df)/total:.1%})")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/total:.1%})")
    return results


# =============================================================================
# Section 7: Feature Computation
# =============================================================================


def compute_base_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all per-bar base series from raw OHLCV.

    Adds columns:
      p, r, rho, r_oc, g,
      logvol, logtrades, logquotevol,
      b, ofi,
      clv, bodyfrac, wickup, wickdn,
      vwap, vwapdev, qpertrade
    """
    assert_columns(
        df,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "num_trades",
            "taker_buy_base",
        ],
        context="compute_base_series",
    )

    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]
    quote_volume = df["quote_volume"]
    num_trades = df["num_trades"]
    taker_buy_base = df["taker_buy_base"]

    p = np.log(close)
    r = p.diff()

    high_gt_low = high > low
    rho = np.where(high_gt_low, np.log(high / low), np.nan)

    r_oc = np.where(open_ > 0, np.log(close / open_), np.nan)

    prev_close = close.shift(1)
    g = np.where((open_ > 0) & (prev_close > 0), np.log(open_ / prev_close), np.nan)

    logvol = np.log1p(volume)
    logtrades = np.log1p(num_trades)
    logquotevol = np.log1p(quote_volume)

    vol_pos = volume > 0
    b = np.where(vol_pos, taker_buy_base / volume, np.nan)
    ofi = np.where(vol_pos, 2.0 * b - 1.0, np.nan)

    candle_range = high - low
    range_pos = candle_range > 0

    clv = np.where(
        range_pos,
        (2.0 * close - high - low) / candle_range,
        np.nan,
    )
    bodyfrac = np.where(range_pos, (close - open_).abs() / candle_range, np.nan)
    wickup = np.where(range_pos, (high - np.maximum(open_, close)) / candle_range, np.nan)
    wickdn = np.where(range_pos, (np.minimum(open_, close) - low) / candle_range, np.nan)

    vwap = np.where(vol_pos, quote_volume / volume, np.nan)
    vwapdev = np.where(vol_pos, np.log(close / vwap), np.nan)

    trades_pos = num_trades > 0
    qpertrade = np.where(trades_pos, quote_volume / num_trades, np.nan)

    new_cols: Dict[str, Any] = {
        "p": p,
        "r": r,
        "rho": rho,
        "r_oc": r_oc,
        "g": g,
        "logvol": logvol,
        "logtrades": logtrades,
        "logquotevol": logquotevol,
        "b": b,
        "ofi": ofi,
        "clv": clv,
        "bodyfrac": bodyfrac,
        "wickup": wickup,
        "wickdn": wickdn,
        "vwap": vwap,
        "vwapdev": vwapdev,
        "qpertrade": qpertrade,
    }
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_lag_features(df: pd.DataFrame, lags_f: List[int]) -> pd.DataFrame:
    """Compute Group A lag features (Section 7.4)."""
    assert_columns(df, ["r", "rho", "clv", "logvol", "logtrades", "ofi"], context="compute_lag_features")

    r = df["r"]
    rho = df["rho"]
    clv = df["clv"]
    logvol = df["logvol"]
    logtrades = df["logtrades"]
    ofi = df["ofi"]

    # Build new columns in one shot to avoid DataFrame fragmentation.
    new_cols: Dict[str, pd.Series] = {}
    for L in lags_f:
        r_shift = r.shift(L)
        new_cols[f"ret__lag{L}__f__w0"] = r_shift
        new_cols[f"absret__lag{L}__f__w0"] = r_shift.abs()
        new_cols[f"range__lag{L}__f__w0"] = rho.shift(L)
        new_cols[f"clv__lag{L}__f__w0"] = clv.shift(L)
        new_cols[f"logvol__lag{L}__f__w0"] = logvol.shift(L)
        new_cols[f"logtrades__lag{L}__f__w0"] = logtrades.shift(L)
        new_cols[f"ofi__lag{L}__f__w0"] = ofi.shift(L)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_rolling_stats(df: pd.DataFrame, windows_f: List[int]) -> pd.DataFrame:
    """Compute Group B rolling distribution statistics (Section 7.5)."""
    assert_columns(df, ["r", "rho", "logvol", "ofi"], context="compute_rolling_stats")

    r = df["r"]
    rho = df["rho"]
    logvol = df["logvol"]
    ofi = df["ofi"]

    abs_r = r.abs()
    r2 = r ** 2
    pos_r = (r > 0).astype(float)

    # Build new columns in one shot to avoid DataFrame fragmentation.
    new_cols: Dict[str, pd.Series] = {}
    for W in windows_f:
        roll_r = r.rolling(W, min_periods=W)
        new_cols[f"ret__mean__f__w{W}"] = roll_r.mean()
        new_cols[f"ret__std__f__w{W}"] = roll_r.std(ddof=0)
        new_cols[f"ret__rms__f__w{W}"] = np.sqrt(r2.rolling(W, min_periods=W).mean())
        new_cols[f"absret__mean__f__w{W}"] = abs_r.rolling(W, min_periods=W).mean()
        new_cols[f"ret__posfrac__f__w{W}"] = pos_r.rolling(W, min_periods=W).mean()

        new_cols[f"range__mean__f__w{W}"] = rho.rolling(W, min_periods=W).mean()
        new_cols[f"logvol__mean__f__w{W}"] = logvol.rolling(W, min_periods=W).mean()
        new_cols[f"logvol__std__f__w{W}"] = logvol.rolling(W, min_periods=W).std(ddof=0)
        new_cols[f"ofi__std__f__w{W}"] = ofi.rolling(W, min_periods=W).std(ddof=0)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def _boundary_indices(n_rows: int) -> np.ndarray:
    return np.arange(0, n_rows, M, dtype=np.int64)


def compute_quantile_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Compute Group B+ quantile and MAD features (Section 7.6)."""
    assert_columns(df, ["r"], context="compute_quantile_features")

    r = df["r"].to_numpy(dtype=float)
    n = len(r)
    bidx = _boundary_indices(n)

    new_cols: Dict[str, Any] = {}
    for W in windows:
        q10 = np.full(n, np.nan, dtype=float)
        q50 = np.full(n, np.nan, dtype=float)
        q90 = np.full(n, np.nan, dtype=float)
        mad = np.full(n, np.nan, dtype=float)

        eligible = bidx[bidx >= (W - 1)]
        if len(eligible) == 0:
            new_cols[f"ret__q10__f__w{W}"] = q10
            new_cols[f"ret__q50__f__w{W}"] = q50
            new_cols[f"ret__q90__f__w{W}"] = q90
            new_cols[f"ret__mad__f__w{W}"] = mad
            continue

        offsets = np.arange(W - 1, -1, -1, dtype=np.int64)
        chunk_size = 2000 if W >= 720 else 5000
        for start in range(0, len(eligible), chunk_size):
            idx = eligible[start : start + chunk_size]
            rows = idx[:, None] - offsets[None, :]
            window_vals = r[rows]

            invalid = np.isnan(window_vals).any(axis=1)

            q10_chunk = np.quantile(window_vals, 0.10, axis=1, method="linear")
            q50_chunk = np.quantile(window_vals, 0.50, axis=1, method="linear")
            q90_chunk = np.quantile(window_vals, 0.90, axis=1, method="linear")

            med = q50_chunk
            abs_dev = np.abs(window_vals - med[:, None])
            mad_chunk = np.quantile(abs_dev, 0.50, axis=1, method="linear")

            q10_chunk[invalid] = np.nan
            q50_chunk[invalid] = np.nan
            q90_chunk[invalid] = np.nan
            mad_chunk[invalid] = np.nan

            q10[idx] = q10_chunk
            q50[idx] = q50_chunk
            q90[idx] = q90_chunk
            mad[idx] = mad_chunk

        new_cols[f"ret__q10__f__w{W}"] = q10
        new_cols[f"ret__q50__f__w{W}"] = q50
        new_cols[f"ret__q90__f__w{W}"] = q90
        new_cols[f"ret__mad__f__w{W}"] = mad

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_volatility_ohlc(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Compute Group C OHLC volatility estimators (Section 7.7)."""
    assert_columns(df, ["open", "high", "low", "close"], context="compute_volatility_ohlc")

    log_hl = np.where(df["high"] > df["low"], np.log(df["high"] / df["low"]), np.nan)
    log_co = np.where(df["open"] > 0, np.log(df["close"] / df["open"]), np.nan)

    var_p = (log_hl**2) / (4.0 * np.log(2.0))
    var_gk = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
    var_rs = (
        np.log(df["high"] / df["open"]) * np.log(df["high"] / df["close"])
        + np.log(df["low"] / df["open"]) * np.log(df["low"] / df["close"])
    )

    var_p_s = pd.Series(var_p, index=df.index)
    var_gk_s = pd.Series(var_gk, index=df.index)
    var_rs_s = pd.Series(var_rs, index=df.index)

    new_cols: Dict[str, Any] = {}
    for W in windows:
        new_cols[f"vol__parkinson__f__w{W}"] = np.sqrt(var_p_s.rolling(W, min_periods=W).mean())
        new_cols[f"vol__gk__f__w{W}"] = np.sqrt(np.maximum(0.0, var_gk_s.rolling(W, min_periods=W).mean()))
        new_cols[f"vol__rs__f__w{W}"] = np.sqrt(np.maximum(0.0, var_rs_s.rolling(W, min_periods=W).mean()))

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_volatility_rs_only(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Compute Rogers-Satchell volatility for additional windows.

    Used to support vol-ratio pairs that include long windows not listed in
    `WINDOWS_VOL_OHLC` (e.g., W=1440), without adding extra Group C features.
    """
    assert_columns(df, ["open", "high", "low", "close"], context="compute_volatility_rs_only")

    var_rs = (
        np.log(df["high"] / df["open"]) * np.log(df["high"] / df["close"])
        + np.log(df["low"] / df["open"]) * np.log(df["low"] / df["close"])
    )
    var_rs_s = pd.Series(var_rs, index=df.index)
    new_cols: Dict[str, Any] = {}
    for W in windows:
        new_cols[f"vol__rs__f__w{W}"] = np.sqrt(np.maximum(0.0, var_rs_s.rolling(W, min_periods=W).mean()))
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_volatility_decomposition(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Compute Group C+ volatility decomposition features (Section 7.8)."""
    assert_columns(df, ["r"], context="compute_volatility_decomposition")

    abs_r = df["r"].abs()
    r2 = df["r"] ** 2
    prod = abs_r * abs_r.shift(1)

    if "ret__rms__f__w20" not in df.columns:
        raise ValueError("compute_volatility_decomposition requires ret__rms__f__w20 to exist")

    new_cols: Dict[str, Any] = {}
    for W in windows:
        rv = r2.rolling(W, min_periods=W).mean()
        bpv = (math.pi / 2.0) * prod.rolling(W - 1, min_periods=W - 1).mean()
        new_cols[f"vol__bpv_ratio__f__w{W}"] = rv / (bpv + EPS)

        down = np.minimum(df["r"], 0.0) ** 2
        up = np.maximum(df["r"], 0.0) ** 2
        sv_down = down.rolling(W, min_periods=W).mean()
        sv_up = up.rolling(W, min_periods=W).mean()

        semidown = np.sqrt(np.maximum(0.0, sv_down))
        semiup = np.sqrt(np.maximum(0.0, sv_up))
        new_cols[f"vol__semivar_down__f__w{W}"] = semidown
        new_cols[f"vol__semivar_up__f__w{W}"] = semiup
        new_cols[f"vol__semivar_ratio__f__w{W}"] = semidown / (semiup + EPS)

        sigma = df["ret__rms__f__w20"]
        new_cols[f"vol__vov__f__w{W}"] = sigma.rolling(W, min_periods=W).std(ddof=0)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_candle_geometry(
    df: pd.DataFrame,
    windows_candle_roll: List[int],
    windows_breakout: List[int],
) -> pd.DataFrame:
    """Compute Group D candle geometry and breakout state (Section 7.10)."""
    assert_columns(df, ["p", "clv", "bodyfrac", "wickup", "wickdn", "g"], context="compute_candle_geometry")

    p = df["p"]
    clv = df["clv"]
    bodyfrac = df["bodyfrac"]
    wickup = df["wickup"]
    wickdn = df["wickdn"]
    g = df["g"]

    new_cols: Dict[str, Any] = {
        "clv__inst__f__w0": clv,
        "bodyfrac__inst__f__w0": bodyfrac,
        "wickup__inst__f__w0": wickup,
        "wickdn__inst__f__w0": wickdn,
        "gap__inst__f__w0": g,
    }

    for W in windows_candle_roll:
        new_cols[f"clv__mean__f__w{W}"] = clv.rolling(W, min_periods=W).mean()

    for W in windows_breakout:
        p_max = p.rolling(W, min_periods=W).max()
        p_min = p.rolling(W, min_periods=W).min()
        denom = p_max - p_min

        pos = (p - p_min) / denom
        pos = pos.where(denom != 0.0, np.nan)
        new_cols[f"logp__pos__f__w{W}"] = pos
        new_cols[f"logp__dd__f__w{W}"] = p_max - p
        new_cols[f"logp__du__f__w{W}"] = p - p_min

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def _wilder_rsi(r: np.ndarray, W: int) -> np.ndarray:
    n = len(r)
    rsi = np.full(n, np.nan, dtype=float)
    if n <= W:
        return rsi

    gain = np.where(r > 0, r, 0.0)
    loss = np.where(r < 0, -r, 0.0)

    avg_gain = np.full(n, np.nan, dtype=float)
    avg_loss = np.full(n, np.nan, dtype=float)

    avg_gain[W] = float(np.mean(gain[1 : W + 1]))
    avg_loss[W] = float(np.mean(loss[1 : W + 1]))

    for i in range(W + 1, n):
        avg_gain[i] = ((W - 1) * avg_gain[i - 1] + gain[i]) / W
        avg_loss[i] = ((W - 1) * avg_loss[i - 1] + loss[i]) / W

    rs = avg_gain / (avg_loss + EPS)
    rsi[W:] = 100.0 - 100.0 / (1.0 + rs[W:])
    return rsi


def compute_trend_momentum(df: pd.DataFrame, windows_logp_z: List[int], windows_rsi: List[int]) -> pd.DataFrame:
    """Compute Group E trend/momentum features (Section 7.12)."""
    assert_columns(df, ["p", "r"], context="compute_trend_momentum")

    p = df["p"]
    r = df["r"]
    new_cols: Dict[str, Any] = {}
    for W in windows_logp_z:
        roll = p.rolling(W, min_periods=W)
        mu = roll.mean()
        sigma = roll.std(ddof=0)
        z = (p - mu) / sigma
        z = z.where(sigma != 0.0, np.nan)
        new_cols[f"logp__z__f__w{W}"] = z

    def ema(series: pd.Series, W: int) -> pd.Series:
        alpha = 2.0 / (W + 1.0)
        return series.ewm(alpha=alpha, adjust=False).mean()

    ema10 = ema(p, 10)
    ema20 = ema(p, 20)
    ema60 = ema(p, 60)
    ema120 = ema(p, 120)
    ema240 = ema(p, 240)

    new_cols["logp__ema_spread__f__w0__fast10__slow60"] = ema10 - ema60
    new_cols["logp__ema_spread__f__w0__fast20__slow120"] = ema20 - ema120
    new_cols["logp__ema_spread__f__w0__fast60__slow240"] = ema60 - ema240

    r_np = r.to_numpy(dtype=float)
    for W in windows_rsi:
        new_cols[f"ret__rsi__f__w{W}"] = _wilder_rsi(r_np, W)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_activity_flow(df: pd.DataFrame, windows_activity: List[int]) -> pd.DataFrame:
    """Compute Group F activity/flow/liquidity features (Section 7.13)."""
    assert_columns(
        df,
        ["b", "ofi", "qpertrade", "vwapdev", "logvol", "quote_volume", "r"],
        context="compute_activity_flow",
    )

    b = df["b"]
    ofi = df["ofi"]
    qpertrade = df["qpertrade"]
    vwapdev = df["vwapdev"]
    logvol = df["logvol"]
    quote_volume = df["quote_volume"]
    r = df["r"]

    new_cols: Dict[str, Any] = {
        "tb_ratio__inst__f__w0": b,
        "ofi__inst__f__w0": ofi,
        "qpertrade__inst__f__w0": qpertrade,
        "vwapdev__inst__f__w0": vwapdev,
    }

    for W in windows_activity:
        roll = logvol.rolling(W, min_periods=W)
        mu = roll.mean()
        sigma = roll.std(ddof=0)
        z = (logvol - mu) / sigma
        z = z.where(sigma != 0.0, np.nan)
        new_cols[f"logvol__z__f__w{W}"] = z

        num = quote_volume.rolling(W, min_periods=W).sum()
        den = r.abs().rolling(W, min_periods=W).sum()
        new_cols[f"liq__quote_per_absret__f__w{W}"] = num / (den + EPS)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def _rolling_corr_population(x: pd.Series, y: pd.Series, W: int) -> pd.Series:
    mean_x = x.rolling(W, min_periods=W).mean()
    mean_y = y.rolling(W, min_periods=W).mean()
    mean_x2 = (x**2).rolling(W, min_periods=W).mean()
    mean_y2 = (y**2).rolling(W, min_periods=W).mean()
    mean_xy = (x * y).rolling(W, min_periods=W).mean()

    var_x = mean_x2 - mean_x**2
    var_y = mean_y2 - mean_y**2
    cov_xy = mean_xy - mean_x * mean_y

    std_x = np.sqrt(np.maximum(0.0, var_x))
    std_y = np.sqrt(np.maximum(0.0, var_y))

    denom = std_x * std_y
    corr = cov_xy / denom
    corr = corr.where((var_x != 0.0) & (var_y != 0.0), np.nan)
    return corr


def compute_correlations(df: pd.DataFrame, windows_corr: List[int]) -> pd.DataFrame:
    """Compute Group G serial dependence and correlation features (Section 7.15)."""
    assert_columns(df, ["r", "logvol", "ofi"], context="compute_correlations")

    r = df["r"]
    logvol = df["logvol"]
    ofi = df["ofi"]

    new_cols: Dict[str, Any] = {}
    for W in windows_corr:
        new_cols[f"ret__acf1__f__w{W}"] = _rolling_corr_population(r, r.shift(1), W)
        new_cols[f"ret__corr_logvol__f__w{W}"] = _rolling_corr_population(r, logvol, W)
        new_cols[f"absret__corr_logvol__f__w{W}"] = _rolling_corr_population(r.abs(), logvol, W)
        new_cols[f"ret__corr_ofi__f__w{W}"] = _rolling_corr_population(r, ofi, W)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Group J event-based features (Section 7.19)."""
    assert_columns(df, ["r"], context="compute_event_features")

    r = df["r"].to_numpy(dtype=float)
    run_dir = np.zeros(len(r), dtype=np.int8)
    run_len = np.zeros(len(r), dtype=np.int32)
    run_cum = np.zeros(len(r), dtype=float)

    direction = 0
    length = 0
    cumret = 0.0

    for i, ri in enumerate(r):
        if not np.isfinite(ri) or ri == 0.0:
            direction = 0
            length = 0
            cumret = 0.0
        else:
            sign = 1 if ri > 0 else -1
            if sign == direction:
                length += 1
                cumret += float(ri)
            else:
                direction = sign
                length = 1
                cumret = float(ri)

        run_dir[i] = direction
        run_len[i] = length
        run_cum[i] = cumret

    new_cols: Dict[str, Any] = {
        "event__run_dir__f__w0": run_dir,
        "event__run_len__f__w0": run_len,
        "event__run_cumret__f__w0": run_cum,
    }
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_data_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Group M data quality guard features (Section 7.22)."""
    assert_columns(df, ["open", "high", "low", "close"], context="compute_data_quality_flags")
    assert_index_is_utc_datetime_index(df, context="compute_data_quality_flags")

    bad_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    diffs = df.index.to_series().diff()
    gap = (diffs != pd.Timedelta(minutes=1)).astype(np.int8)
    if len(gap) > 0:
        gap.iloc[0] = 0

    new_cols: Dict[str, Any] = {
        "data__bad_ohlc__f__w0": bad_ohlc.astype(np.int8),
        "data__gap__f__w0": gap,
    }
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Group Q seasonality/time context features (Section 7.17)."""
    assert_index_is_utc_datetime_index(df, context="compute_seasonality")

    idx = df.index
    minute_of_day = (idx.hour * 60 + idx.minute).astype(int)
    day_of_week = idx.dayofweek.astype(int)

    new_cols: Dict[str, Any] = {
        "time__sin_minute__f__w0": np.sin(2.0 * math.pi * minute_of_day / 1440.0),
        "time__cos_minute__f__w0": np.cos(2.0 * math.pi * minute_of_day / 1440.0),
        "time__sin_dow__f__w0": np.sin(2.0 * math.pi * day_of_week / 7.0),
        "time__cos_dow__f__w0": np.cos(2.0 * math.pi * day_of_week / 7.0),
    }
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_excursion_features(df: pd.DataFrame, windows_excursion: List[int], windows_maxret: List[int]) -> pd.DataFrame:
    """Compute Group O excursion and burstiness features (Section 7.11)."""
    assert_columns(df, ["p", "r"], context="compute_excursion_features")

    p = df["p"].to_numpy(dtype=float)
    n = len(p)
    bidx = _boundary_indices(n)

    new_cols: Dict[str, Any] = {}
    for W in windows_excursion:
        out_up = np.full(n, np.nan, dtype=float)
        out_dn = np.full(n, np.nan, dtype=float)

        eligible = bidx[bidx >= (W - 1)]
        offsets = np.arange(W - 1, -1, -1, dtype=np.int64)
        max_elements = 5_000_000  # cap chunk memory ~O(chunk_size * W)
        chunk_size = int(min(20000, max(1, max_elements // int(W))))
        for start in range(0, len(eligible), chunk_size):
            idx = eligible[start : start + chunk_size]
            rows = idx[:, None] - offsets[None, :]
            window_vals = p[rows]

            invalid = np.isnan(window_vals).any(axis=1)

            running_min = np.minimum.accumulate(window_vals, axis=1)
            drawups = window_vals - running_min
            max_drawup = np.max(drawups, axis=1)

            running_max = np.maximum.accumulate(window_vals, axis=1)
            drawdowns = running_max - window_vals
            max_drawdown = np.max(drawdowns, axis=1)

            max_drawup[invalid] = np.nan
            max_drawdown[invalid] = np.nan

            out_up[idx] = max_drawup
            out_dn[idx] = max_drawdown

        new_cols[f"excursion__max_drawup__f__w{W}"] = out_up
        new_cols[f"excursion__max_drawdown__f__w{W}"] = out_dn

    r2 = df["p"] - df["p"].shift(2)
    for W in windows_maxret:
        new_cols[f"ret__max1m__f__w{W}"] = df["r"].rolling(W, min_periods=W).max()
        new_cols[f"ret__max2m__f__w{W}"] = r2.rolling(W, min_periods=W).max()
        new_cols[f"ret__min1m__f__w{W}"] = df["r"].rolling(W, min_periods=W).min()

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_enhanced_liquidity(
    df: pd.DataFrame,
    windows_liq_amihud: List[int],
    windows_liq_rpv: List[int],
    windows_ofi_impulse: List[int],
) -> pd.DataFrame:
    """Compute Group P enhanced liquidity proxy features (Section 7.14)."""
    assert_columns(df, ["r", "volume", "high", "low", "ofi"], context="compute_enhanced_liquidity")

    abs_r = df["r"].abs()
    hl_range = df["high"] - df["low"]

    new_cols: Dict[str, Any] = {}
    for W in windows_liq_amihud:
        num = abs_r.rolling(W, min_periods=W).sum()
        den = df["volume"].rolling(W, min_periods=W).sum()
        new_cols[f"liq__amihud__f__w{W}"] = num / (den + EPS)

    for W in windows_liq_rpv:
        num = hl_range.rolling(W, min_periods=W).sum()
        den = df["volume"].rolling(W, min_periods=W).sum()
        new_cols[f"liq__range_per_vol__f__w{W}"] = num / (den + EPS)

    for W in windows_ofi_impulse:
        new_cols[f"ofi__delta__f__w{W}"] = df["ofi"] - df["ofi"].shift(W)
        new_cols[f"ofi__max__f__w{W}"] = df["ofi"].rolling(W, min_periods=W).max()
        new_cols[f"ofi__min__f__w{W}"] = df["ofi"].rolling(W, min_periods=W).min()
        new_cols[f"ofi__ret_interaction__f__w{W}"] = (df["ofi"] * abs_r).rolling(W, min_periods=W).sum()

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_barrier_aware_features(
    df_boundaries: pd.DataFrame,
    windows_barrier: List[int],
    phi: float,
    M_: int,
    vol_pairs: List[Tuple[int, int]],
) -> pd.DataFrame:
    """Compute Group N barrier-aware features at decision boundaries (Section 7.9)."""
    df = df_boundaries

    new_cols: Dict[str, Any] = {}
    for W in windows_barrier:
        col = f"vol__rs__f__w{W}"
        if col not in df.columns:
            raise ValueError(f"compute_barrier_aware_features requires {col} at boundaries")

        vol = df[col]
        new_cols[f"barrier__z_tight__f__w{W}"] = phi / (vol * math.sqrt(M_) + EPS)
        new_cols[f"barrier__emax_ratio__f__w{W}"] = (vol * math.sqrt(2.0 * math.log(M_))) / (phi + EPS)

    for ws, wl in vol_pairs:
        col_s = f"vol__rs__f__w{ws}"
        col_l = f"vol__rs__f__w{wl}"
        if col_s not in df.columns or col_l not in df.columns:
            raise ValueError(
                f"compute_barrier_aware_features requires {col_s} and {col_l} (precompute long vol via compute_volatility_rs_only)"
            )
        new_cols[f"vol__ratio__f__ws{ws}__wl{wl}"] = df[col_s] / (df[col_l] + EPS)

    new_cols["cost__c__h__w0"] = float(C)
    new_cols["barrier__phi__h__w0"] = float(phi)
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


# =============================================================================
# Section 7.16: Permutation Entropy (Group H)
# =============================================================================


def permutation_entropy_normalized(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    """
    Compute normalized permutation entropy in [0, 1].

    Tie handling: stable argsort (equivalent to NumPy argsort(..., kind='mergesort')).
    """
    n = len(x)
    n_patterns = n - (m - 1) * tau

    if n_patterns < 5 * math.factorial(m):
        return float("nan")

    patterns: List[Tuple[int, ...]] = []
    for i in range(n_patterns):
        window = [x[i + j * tau] for j in range(m)]
        if any(not np.isfinite(v) for v in window):
            return float("nan")
        patterns.append(tuple(np.argsort(window, kind="mergesort")))

    from collections import Counter

    counts = Counter(patterns)
    probs = np.array(list(counts.values()), dtype=float) / float(n_patterns)
    H = -np.sum(probs * np.log(probs))
    H_max = np.log(math.factorial(m))
    return float(H / H_max)


def _perm_codes_m3_tau1(x: np.ndarray) -> np.ndarray:
    """
    Vectorized ordinal-pattern encoding for m=3, tau=1 with stable tie-breaking.

    Returns codes in {0,1,2,3,4,5} for valid patterns, else -1 when any value is NaN/inf.
    """
    a = x[:-2]
    b = x[1:-1]
    c = x[2:]

    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)

    v0 = a.copy()
    v1 = b.copy()
    v2 = c.copy()
    i0 = np.zeros_like(v0, dtype=np.int8)
    i1 = np.ones_like(v0, dtype=np.int8)
    i2 = np.full_like(v0, 2, dtype=np.int8)

    # Stable sorting network for 3 elements: swap only when strictly greater.
    mask = v0 > v1
    v0, v1 = np.where(mask, v1, v0), np.where(mask, v0, v1)
    i0, i1 = np.where(mask, i1, i0), np.where(mask, i0, i1)

    mask = v1 > v2
    v1, v2 = np.where(mask, v2, v1), np.where(mask, v1, v2)
    i1, i2 = np.where(mask, i2, i1), np.where(mask, i1, i2)

    mask = v0 > v1
    v0, v1 = np.where(mask, v1, v0), np.where(mask, v0, v1)
    i0, i1 = np.where(mask, i1, i0), np.where(mask, i0, i1)

    key = (i0.astype(np.int16) * 9) + (i1.astype(np.int16) * 3) + i2.astype(np.int16)
    mapping = np.full(27, -1, dtype=np.int8)
    mapping[(0 * 9) + (1 * 3) + 2] = 0
    mapping[(0 * 9) + (2 * 3) + 1] = 1
    mapping[(1 * 9) + (0 * 3) + 2] = 2
    mapping[(1 * 9) + (2 * 3) + 0] = 3
    mapping[(2 * 9) + (0 * 3) + 1] = 4
    mapping[(2 * 9) + (1 * 3) + 0] = 5

    codes = mapping[key]
    codes[~valid] = -1
    return codes


def compute_permutation_entropy(df: pd.DataFrame, windows: List[int], m: int = 3, tau: int = 1) -> pd.DataFrame:
    """Compute Group H permutation entropy features (Section 7.16)."""
    if m != 3 or tau != 1:
        raise ValueError("compute_permutation_entropy implemented for m=3, tau=1 (spec default)")

    assert_columns(df, ["r"], context="compute_permutation_entropy")

    r = df["r"].to_numpy(dtype=float)
    new_cols: Dict[str, Any] = {}
    if len(r) < 3:
        for W in windows:
            new_cols[f"pentropy_norm__inst__f__w{W}__m3__tau1"] = np.nan
        new_df = pd.DataFrame(new_cols, index=df.index)
        return pd.concat([df, new_df], axis=1)

    codes = _perm_codes_m3_tau1(r)  # length N-2
    n_patterns_total = len(codes)
    n_bars = len(r)
    max_entropy = math.log(math.factorial(3))

    for W in windows:
        n_patterns = W - 2
        if n_patterns < 5 * math.factorial(3):
            new_cols[f"pentropy_norm__inst__f__w{W}__m3__tau1"] = np.nan
            continue

        out = np.full(n_bars, np.nan, dtype=float)
        if n_patterns_total < n_patterns:
            new_cols[f"pentropy_norm__inst__f__w{W}__m3__tau1"] = out
            continue

        end_idx = np.arange(n_patterns - 1, n_patterns_total, dtype=np.int64)
        start_idx = end_idx - (n_patterns - 1)

        invalid = (codes == -1).astype(np.int32)
        invalid_cum = np.cumsum(invalid, dtype=np.int64)
        invalid_in_window = invalid_cum[end_idx] - np.where(start_idx > 0, invalid_cum[start_idx - 1], 0)
        valid_window = invalid_in_window == 0

        probs = []
        for code in range(6):
            ind = (codes == code).astype(np.int32)
            cum = np.cumsum(ind, dtype=np.int64)
            cnt = cum[end_idx] - np.where(start_idx > 0, cum[start_idx - 1], 0)
            probs.append(cnt.astype(float) / float(n_patterns))
        P = np.vstack(probs).T  # (n_windows, 6)

        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
        H_norm = H / max_entropy
        H_norm[~valid_window] = np.nan

        bar_positions = end_idx + 2
        out[bar_positions] = H_norm
        new_cols[f"pentropy_norm__inst__f__w{W}__m3__tau1"] = out

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


# =============================================================================
# Section 7.18 + 7.11.3: Block Features (Group I + block-level O)
# =============================================================================


def compute_block_features(
    df_boundaries: pd.DataFrame,
    df_raw: pd.DataFrame,
    M_: int,
    windows_h: List[int],
) -> pd.DataFrame:
    """
    Compute Group I decision-block features (Section 7.18) and the decision-block
    excursion features from Section 7.11.3.
    """
    dfb = df_boundaries
    assert_columns(dfb, ["k"], context="compute_block_features(df_boundaries)")
    assert_columns(
        df_raw,
        ["high", "low", "close", "volume", "quote_volume", "num_trades", "taker_buy_base"],
        context="compute_block_features(df_raw)",
    )

    K = int(len(dfb))
    n_max = (K - 1) * M_
    if n_max >= len(df_raw):
        raise ValueError("compute_block_features: boundary count implies n_max beyond raw data length")

    close = df_raw["close"].to_numpy(dtype=float)
    p = np.log(close)

    high = df_raw["high"].to_numpy(dtype=float)
    low = df_raw["low"].to_numpy(dtype=float)
    volume = df_raw["volume"].to_numpy(dtype=float)
    quote_volume = df_raw["quote_volume"].to_numpy(dtype=float)
    num_trades = df_raw["num_trades"].to_numpy(dtype=float)
    taker_buy_base = df_raw["taker_buy_base"].to_numpy(dtype=float)

    # Block aggregates H^h_k, L^h_k, etc. Block 0 = {0}; for k>=1: bars (k-1)M+1 .. kM.
    H = np.full(K, np.nan, dtype=float)
    L = np.full(K, np.nan, dtype=float)
    V = np.full(K, np.nan, dtype=float)
    Q = np.full(K, np.nan, dtype=float)
    Ntr = np.full(K, np.nan, dtype=float)
    VTB = np.full(K, np.nan, dtype=float)

    H[0] = high[0]
    L[0] = low[0]
    V[0] = volume[0]
    Q[0] = quote_volume[0]
    Ntr[0] = num_trades[0]
    VTB[0] = taker_buy_base[0]

    if K > 1:
        sl = slice(1, n_max + 1)
        high_blocks = high[sl].reshape(K - 1, M_)
        low_blocks = low[sl].reshape(K - 1, M_)
        vol_blocks = volume[sl].reshape(K - 1, M_)
        q_blocks = quote_volume[sl].reshape(K - 1, M_)
        ntr_blocks = num_trades[sl].reshape(K - 1, M_)
        vtb_blocks = taker_buy_base[sl].reshape(K - 1, M_)

        H[1:] = np.max(high_blocks, axis=1)
        L[1:] = np.min(low_blocks, axis=1)
        V[1:] = np.sum(vol_blocks, axis=1)
        Q[1:] = np.sum(q_blocks, axis=1)
        Ntr[1:] = np.sum(ntr_blocks, axis=1)
        VTB[1:] = np.sum(vtb_blocks, axis=1)

    close_boundary = close[::M_][:K]
    p_boundary = p[::M_][:K]

    ret_inst = np.full(K, np.nan, dtype=float)
    ret_inst[1:] = np.log(close_boundary[1:] / close_boundary[:-1])
    new_cols: Dict[str, Any] = {
        "ret__inst__h__w0": ret_inst,
        "range__inst__h__w0": np.where(H > L, np.log(H / L), np.nan),
        "logvol__inst__h__w0": np.log1p(V),
        "ofi__inst__h__w0": np.where(V > 0, 2.0 * (VTB / V) - 1.0, np.nan),
    }

    ret_inst_s = pd.Series(ret_inst)
    for W in windows_h:
        new_cols[f"ret__std__h__w{W}"] = ret_inst_s.rolling(W, min_periods=W).std(ddof=0).to_numpy()

    # Decision-block excursion features (Section 7.11.3)
    block_maxret = np.full(K, np.nan, dtype=float)
    block_minret = np.full(K, np.nan, dtype=float)
    if K > 1:
        p_blocks = p[1 : n_max + 1].reshape(K - 1, M_)
        p_prev = p_boundary[:-1].reshape(K - 1, 1)
        diffs = p_blocks - p_prev
        block_maxret[1:] = np.max(diffs, axis=1)
        block_minret[1:] = np.min(diffs, axis=1)
    new_cols["block__maxret__h__w0"] = block_maxret
    new_cols["block__minret__h__w0"] = block_minret

    denom_hl = np.log(H) - np.log(L)
    close_to_high = (p_boundary - np.log(L)) / (denom_hl + EPS)
    close_to_high = np.where(denom_hl != 0.0, close_to_high, np.nan)
    new_cols["block__close_to_high__h__w0"] = close_to_high

    new_df = pd.DataFrame(new_cols, index=dfb.index)
    return pd.concat([dfb, new_df], axis=1)


# =============================================================================
# Section 6: Label Construction + Section 7.20: Past Targets
# =============================================================================


def construct_labels(df_boundaries: pd.DataFrame, df_raw: pd.DataFrame, M_: int, eta: float, c: float) -> pd.DataFrame:
    """
    Construct binary labels at decision boundaries.

    Returns DataFrame with columns added: y, m_k, tau_k, phi
    """
    dfb = df_boundaries
    assert_columns(dfb, ["k"], context="construct_labels(df_boundaries)")
    assert_columns(df_raw, ["close"], context="construct_labels(df_raw)")

    close = df_raw["close"].to_numpy(dtype=float)
    n_total = len(close)
    phi = float(c + eta)

    y = np.full(len(dfb), np.nan, dtype=float)
    m_k = np.full(len(dfb), np.nan, dtype=float)
    tau_k = np.full(len(dfb), np.nan, dtype=float)

    for k in range(len(dfb)):
        n_k = k * M_
        if n_k + M_ >= n_total:
            continue
        base = close[n_k]
        future = close[n_k + 1 : n_k + M_ + 1]
        future_ret = np.log(future / base)

        m_val = float(np.max(future_ret))
        m_k[k] = m_val

        hit = m_val >= phi
        y[k] = 1.0 if hit else 0.0
        if hit:
            tau_k[k] = float(np.argmax(future_ret >= phi) + 1)

    new_cols: Dict[str, Any] = {
        "y": y,
        "m_k": m_k,
        "tau_k": tau_k,
        "phi": phi,
    }
    new_df = pd.DataFrame(new_cols, index=dfb.index)
    return pd.concat([dfb, new_df], axis=1)


def compute_past_target_features(
    df_boundaries: pd.DataFrame,
    windows_h: List[int],
    hitrate_windows_h: List[int],
) -> pd.DataFrame:
    """Compute Group K past-target features using matured labels only (Section 7.20)."""
    df = df_boundaries
    assert_columns(df, ["k", "y"], context="compute_past_target_features")

    y = df["y"]
    new_cols: Dict[str, Any] = {
        "hit__prev__h__w0": y.shift(1),
    }

    y_shift = y.shift(1)
    for W in hitrate_windows_h:
        new_cols[f"hit__rate__h__w{W}"] = y_shift.rolling(W, min_periods=W).mean()

    hit_k = df["k"].where(df["y"] == 1)
    last_hit_before = hit_k.shift(1).ffill()
    new_cols["hit__since__h__w0"] = df["k"] - last_hit_before
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


# =============================================================================
# Appendix E: Derivatives Feature Computation
# =============================================================================


def compute_derivatives_base_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute base series from derivatives data.
    """
    required = [
        "close",
        "close_fut",
        "volume_fut",
        "taker_buy_base_fut",
        "num_trades_fut",
        "funding_rate",
        "opt_oi",
        "put_open_interest",
        "call_open_interest",
        "opt_volume",
        "put_volume",
        "call_volume",
        "bvol",
    ]
    assert_columns(df, required, context="compute_derivatives_base_series")

    close = df["close"]
    close_fut = df["close_fut"]
    volume_fut = df["volume_fut"]
    taker_buy_base_fut = df["taker_buy_base_fut"]
    call_open_interest = df["call_open_interest"]
    put_open_interest = df["put_open_interest"]
    call_volume = df["call_volume"]
    put_volume = df["put_volume"]

    basis_abs = close_fut - close
    basis_pct = (close_fut - close) / (close + EPS) * 100.0

    vol_pos = volume_fut > 0
    tb_ratio_fut = np.where(
        vol_pos,
        taker_buy_base_fut / (volume_fut + EPS),
        np.nan,
    )
    net_vol_fut = 2.0 * taker_buy_base_fut - volume_fut

    call_oi_pos = call_open_interest > 0
    call_vol_pos = call_volume > 0
    pcr_oi = np.where(
        call_oi_pos,
        put_open_interest / (call_open_interest + EPS),
        np.nan,
    )
    pcr_vol = np.where(
        call_vol_pos,
        put_volume / (call_volume + EPS),
        np.nan,
    )

    new_cols: Dict[str, Any] = {
        "basis_abs": basis_abs,
        "basis_pct": basis_pct,
        "tb_ratio_fut": tb_ratio_fut,
        "net_vol_fut": net_vol_fut,
        "pcr_oi": pcr_oi,
        "pcr_vol": pcr_vol,
    }
    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_basis_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Compute basis features (Group R; perpetual futures vs spot).
    """
    assert_columns(df, ["basis_abs", "basis_pct", "close", "close_fut"], context="compute_basis_features")

    basis_abs = df["basis_abs"]
    basis_pct = df["basis_pct"]
    close = df["close"]
    close_fut = df["close_fut"]

    new_cols: Dict[str, Any] = {
        "basis__abs__f__w0": basis_abs,
        "basis__pct__f__w0": basis_pct,
    }

    minute_of_day = df.index.hour * 60 + df.index.minute
    next_funding_min = np.where(
        minute_of_day < 8 * 60,
        8 * 60,
        np.where(minute_of_day < 16 * 60, 16 * 60, 24 * 60),
    )
    tau = pd.Series((next_funding_min - minute_of_day).astype("float64"), index=df.index).clip(
        lower=60,
        upper=480,
    )
    basis_ratio = (close_fut - close) / (close + EPS)
    new_cols["basis__ann_yield__f__w0"] = basis_ratio * (365.0 * 24.0 * 60.0 / tau) * 100.0

    if 5 in windows:
        new_cols["basis__chg__f__w5"] = basis_pct - basis_pct.shift(5)
        new_cols["basis__mean__f__w5"] = basis_pct.rolling(5, min_periods=5).mean()
        new_cols["basis__std__f__w5"] = basis_pct.rolling(5, min_periods=5).std(ddof=0)

    if 60 in windows:
        new_cols["basis__mean__f__w60"] = basis_pct.rolling(60, min_periods=60).mean()
        new_cols["basis__std__f__w60"] = basis_pct.rolling(60, min_periods=60).std(ddof=0)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_flow_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Compute derivatives volume and order flow features (Group S).
    """
    required = [
        "tb_ratio_fut",
        "net_vol_fut",
        "quote_volume",
        "quote_volume_fut",
        "volume_fut",
        "num_trades_fut",
    ]
    assert_columns(df, required, context="compute_flow_features")

    tb_ratio_fut = df["tb_ratio_fut"]
    net_vol_fut = df["net_vol_fut"]
    quote_volume = df["quote_volume"]
    quote_volume_fut = df["quote_volume_fut"]
    volume_fut = df["volume_fut"]
    num_trades_fut = df["num_trades_fut"]

    new_cols: Dict[str, Any] = {
        "flow__taker_buy_ratio__f__w0": tb_ratio_fut,
        "flow__net_vol_btcs__f__w0": net_vol_fut,
    }

    for W in [w for w in windows if w in (5, 10, 20)]:
        new_cols[f"flow__net_vol_csum__f__w{W}"] = net_vol_fut.rolling(W, min_periods=W).sum()

    spot_qv_pos = quote_volume > 0
    new_cols["liq__fut_vs_spot_vol__f__w0"] = np.where(
        spot_qv_pos,
        quote_volume_fut / (quote_volume + EPS),
        np.nan,
    )

    trades_pos = num_trades_fut > 0
    new_cols["liq__avg_trade_size__f__w0"] = np.where(
        trades_pos,
        volume_fut / (num_trades_fut + EPS),
        np.nan,
    )

    vol_15 = volume_fut.rolling(15, min_periods=15).sum()
    trades_15 = num_trades_fut.rolling(15, min_periods=15).sum()
    new_cols["liq__avg_trade_size__f__w15"] = np.where(
        trades_15 > 0,
        vol_15 / (trades_15 + EPS),
        np.nan,
    )

    mean_30 = num_trades_fut.rolling(30, min_periods=30).mean()
    std_30 = num_trades_fut.rolling(30, min_periods=30).std(ddof=0)
    new_cols["activity__trades_zscore__f__w30"] = np.where(
        std_30 > 0,
        (num_trades_fut - mean_30) / (std_30 + EPS),
        np.nan,
    )

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_oi_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Compute open interest features (Group T - OI part).
    """
    assert_columns(df, ["oi_usd", "quote_volume_fut", "r"], context="compute_oi_features")

    oi_usd = df["oi_usd"]
    quote_volume_fut = df["quote_volume_fut"]
    r = df["r"]

    new_cols: Dict[str, Any] = {
        "oi__total_usd__f__w0": oi_usd,
    }

    if 60 in windows:
        new_cols["oi__chg__f__w60"] = oi_usd - oi_usd.shift(60)
        new_cols["oi__chg_pct__f__w60"] = (oi_usd - oi_usd.shift(60)) / (oi_usd.shift(60) + EPS) * 100.0

        quote_60 = quote_volume_fut.rolling(60, min_periods=60).sum()
        new_cols["oi__vol_ratio__f__w60"] = np.where(quote_60 > 0, oi_usd / (quote_60 + EPS), np.nan)

    if 120 in windows:
        oi_diff = oi_usd.diff()
        new_cols["oi__price_corr__f__w120"] = oi_diff.rolling(120, min_periods=120).corr(r)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_funding_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Compute funding rate features (Group T - Funding part).
    """
    assert_columns(df, ["funding_rate"], context="compute_funding_features")

    funding_rate = df["funding_rate"]
    funding_rate_pct = funding_rate * 100.0
    new_cols: Dict[str, Any] = {
        "funding__rate__f__w0": funding_rate_pct,
    }

    if 1440 in windows:
        new_cols["funding__ewma__f__w1440"] = funding_rate_pct.ewm(span=1440, adjust=False).mean()

    if 4320 in windows:
        mean_3d = funding_rate_pct.rolling(4320, min_periods=4320).mean()
        new_cols["funding__trend__f__w4320"] = funding_rate_pct - mean_3d

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_options_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Compute options market sentiment features (Group U).
    """
    assert_columns(df, ["pcr_oi", "pcr_vol", "opt_oi", "opt_volume"], context="compute_options_features")

    pcr_oi = df["pcr_oi"]
    pcr_vol = df["pcr_vol"]
    opt_oi = df["opt_oi"]
    opt_volume = df["opt_volume"]

    new_cols: Dict[str, Any] = {
        "opt_pcr__oi__f__w0": pcr_oi,
        "opt_pcr__vol__f__w0": pcr_vol,
    }

    if 1440 in windows:
        new_cols["opt_pcr__oi_chg__f__w1440"] = pcr_oi - pcr_oi.shift(1440)

    new_cols["opt_oi__total_usd__f__w0"] = opt_oi
    new_cols["opt_vol__24h_usd__f__w0"] = opt_volume

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


def compute_vol_index_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Compute implied volatility and risk premium features (Group V).
    """
    assert_columns(df, ["bvol", "r"], context="compute_vol_index_features")

    bvol = df["bvol"]
    r = df["r"]
    new_cols: Dict[str, Any] = {
        "vol_idx__bvol30d__f__w0": bvol,
    }

    if 1440 in windows:
        new_cols["vol_idx__bvol_chg__f__w1440"] = bvol - bvol.shift(1440)

    if 43200 in windows:
        rv_30d = r.rolling(43200, min_periods=43200).std(ddof=0) * np.sqrt(525600.0) * 100.0
    else:
        rv_30d = np.full(len(df), np.nan, dtype=float)

    new_cols["vol_realized__30d__f__w43200"] = rv_30d
    new_cols["vol_risk_premium__diff__f__w0"] = bvol - rv_30d
    new_cols["vol_risk_premium__ratio__f__w0"] = bvol / (rv_30d + EPS)

    new_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, new_df], axis=1)


# =============================================================================
# Section 8: Missing Value Handling
# =============================================================================


def get_imputation_value(feature_name: str, p_hit_prior: float = 0.5, cap_h_blocks: int = 144) -> float:
    """
    Get imputation value for a feature based on its name pattern (Section 8.3).

    Order matters: more specific patterns must come first.
    """
    patterns: List[Tuple[str, Optional[float]]] = [
        (r"^cost__c", None),
        (r"^barrier__phi", None),
        (r"^barrier__z_tight", 10.0),
        (r"^barrier__emax_ratio", 0.0),
        (r"^vol__bpv_ratio", 1.0),
        (r"^vol__semivar_ratio", 1.0),
        (r"^vol__ratio", 1.0),
        (r"^vol__", 0.0),
        (r"^ret__rsi", 50.0),
        (r"^ret__posfrac", 0.5),
        (r"^logp__pos", 0.5),
        (r"^tb_ratio", 0.5),
        (r"^block__close_to_high", 0.5),
        (r"^pentropy_norm", 0.5),
        (r"^hit__rate", float(p_hit_prior)),
        (r"^hit__since", float(cap_h_blocks)),
        (r"^hit__prev", 0.0),
        # Target autocorrelation: null when the rolling window has zero
        # label variance (all-zero or all-one). 0.0 = "no detectable
        # autocorrelation" is the right default here (also matches the
        # catch-all fallback below, but we list it explicitly so the
        # intent is auditable and not a happy accident of pattern order).
        (r"^target__autocorr", 0.0),
        (r"^basis__abs", 0.0),
        (r"^basis__pct", 0.0),
        (r"^basis__ann_yield", 0.0),
        (r"^basis__chg", 0.0),
        (r"^basis__mean", 0.0),
        (r"^basis__std", 0.0),
        (r"^term_struct__", 0.0),
        (r"^flow__taker_buy_ratio", 0.5),
        (r"^flow__net_vol", 0.0),
        (r"^liq__fut_vs_spot_vol", 1.0),
        (r"^liq__avg_trade_size", 0.0),
        (r"^activity__trades_zscore", 0.0),
        (r"^oi__total_usd", MEDIAN_OI_USD),
        (r"^oi__chg", 0.0),
        (r"^oi__vol_ratio", 0.0),
        (r"^oi__price_corr", 0.0),
        (r"^funding__rate", 0.0),
        (r"^funding__trend", 0.0),
        (r"^funding__ewma", 0.0),
        (r"^opt_pcr__oi", 1.0),
        (r"^opt_pcr__vol", 1.0),
        (r"^opt_pcr__oi_chg", 0.0),
        (r"^opt_oi__total_usd", 0.0),
        (r"^opt_vol__24h_usd", 0.0),
        (r"^vol_idx__bvol30d", 60.0),
        (r"^vol_idx__bvol_chg", 0.0),
        (r"^vol_realized__30d", 60.0),
        (r"^vol_risk_premium__diff", 0.0),
        (r"^vol_risk_premium__ratio", 1.0),
        (r".*", 0.0),
    ]

    for pattern, value in patterns:
        if re.match(pattern, feature_name):
            if value is None:
                raise ValueError(f"Feature {feature_name} should never be NaN")
            return float(value)
    return 0.0


def create_undef_flags_and_impute(
    df: pd.DataFrame,
    feature_cols: List[str],
    p_hit_prior: float,
    cap_h_blocks: int = 144,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create undefined flags and impute NaNs.

    Returns (imputed_df, list_of_new_undef_columns).
    """
    out = df.copy()
    undef_cols: List[str] = []
    new_flags: Dict[str, Any] = {}

    for col in feature_cols:
        if col not in out.columns:
            raise ValueError(f"create_undef_flags_and_impute: feature column missing: {col}")

        if out[col].isna().any():
            undef_col = f"undef__{col}"
            new_flags[undef_col] = out[col].isna().astype(np.int8)
            undef_cols.append(undef_col)

            impute_value = get_imputation_value(col, p_hit_prior=p_hit_prior, cap_h_blocks=cap_h_blocks)
            out[col] = out[col].fillna(impute_value)

    if new_flags:
        flags_df = pd.DataFrame(new_flags, index=out.index)
        out = pd.concat([out, flags_df], axis=1)

    remaining = int(out[feature_cols].isna().sum().sum())
    if remaining != 0:
        raise ValueError(f"NaNs remain after imputation: {remaining}")

    numeric = out[feature_cols].select_dtypes(include=[np.number])
    if np.isinf(numeric).any().any():
        bad = np.isinf(numeric).sum()
        bad = bad[bad > 0]
        raise ValueError(f"Infs remain after imputation: {bad.to_dict()}")

    return out, undef_cols


# =============================================================================
# Section W: Observation Weighting
# =============================================================================


def compute_barrier_distance_weight(
    m_k: np.ndarray,
    phi: float,
    *,
    w_max: float = WEIGHT_DIST_W_MAX,
    q_tail: float = WEIGHT_DIST_Q_TAIL,
    w_max_pos: float = WEIGHT_DIST_W_MAX_POS,
    q_tail_pos: float = WEIGHT_DIST_Q_TAIL_POS,
    use_pos: bool = WEIGHT_DIST_USE_POSITIVE,
    enabled: bool = WEIGHT_USE_BARRIER_DISTANCE,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute barrier-distance weights with a continuous cap.

    Downside distances use w_max/q_tail, and (optionally) upside distances use
    w_max_pos/q_tail_pos. Lambdas are derived as log(w_max) / d_star and
    log(w_max_pos) / g_star for continuity at the cap.
    """
    m_k = np.asarray(m_k, dtype=float)
    n = len(m_k)
    w_dist = np.ones(n, dtype=float)

    pos_mask = m_k >= phi
    neg_mask = ~pos_mask
    n_positive = int(pos_mask.sum())
    n_negative = int(neg_mask.sum())

    d_k = np.maximum(0.0, phi - m_k)
    g_k = np.maximum(0.0, m_k - phi)

    d_star = 0.0
    d_max = 0.0
    lam_neg = 0.0
    n_capped_neg = 0

    g_star = 0.0
    g_max = 0.0
    lam_pos = 0.0
    n_capped_pos = 0

    if enabled:
        if not (0.0 < q_tail < 1.0):
            raise ValueError(f"q_tail must be in (0, 1), got {q_tail}")
        if w_max < 1.0:
            raise ValueError(f"w_max must be >= 1.0, got {w_max}")

        if n_negative > 0:
            d_neg = d_k[neg_mask]
            d_star = float(np.quantile(d_neg, 1.0 - q_tail))
            d_max = float(d_neg.max())
            if d_star > 0.0 and w_max > 1.0:
                lam_neg = math.log(w_max) / d_star
                log_cap = math.log(w_max)
                exp_arg = np.minimum(lam_neg * d_neg, log_cap)
                w_neg = np.exp(exp_arg)
                n_capped_neg = int((d_neg >= d_star).sum())
            else:
                w_neg = np.ones_like(d_neg)
            w_dist[neg_mask] = w_neg

        if use_pos:
            if not (0.0 < q_tail_pos < 1.0):
                raise ValueError(f"q_tail_pos must be in (0, 1), got {q_tail_pos}")
            if w_max_pos < 1.0:
                raise ValueError(f"w_max_pos must be >= 1.0, got {w_max_pos}")

            if n_positive > 0:
                g_pos = g_k[pos_mask]
                g_star = float(np.quantile(g_pos, 1.0 - q_tail_pos))
                g_max = float(g_pos.max())
                if g_star > 0.0 and w_max_pos > 1.0:
                    lam_pos = math.log(w_max_pos) / g_star
                    log_cap = math.log(w_max_pos)
                    exp_arg = np.minimum(lam_pos * g_pos, log_cap)
                    w_pos = np.exp(exp_arg)
                    n_capped_pos = int((g_pos >= g_star).sum())
                else:
                    w_pos = np.ones_like(g_pos)
                w_dist[pos_mask] = w_pos

    max_weight_neg = 1.0 if n_negative == 0 else float(w_dist[neg_mask].max())
    max_weight_pos = 1.0 if n_positive == 0 else float(w_dist[pos_mask].max())
    weight_range = (1.0, 1.0) if n == 0 else (float(w_dist.min()), float(w_dist.max()))
    info = {
        "enabled": bool(enabled),
        "enabled_pos": bool(enabled and use_pos),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "d_star": d_star,
        "g_star": g_star,
        "lambda": lam_neg,
        "lambda_pos": lam_pos,
        "d_max": d_max,
        "g_max": g_max,
        "n_capped": n_capped_neg,
        "n_capped_pos": n_capped_pos,
        "max_weight_neg": max_weight_neg,
        "max_weight_pos": max_weight_pos,
        "weight_range": weight_range,
        "params": {
            "phi": float(phi),
            "w_max": float(w_max),
            "q_tail": float(q_tail),
            "w_max_neg": float(w_max),
            "q_tail_neg": float(q_tail),
            "w_max_pos": float(w_max_pos),
            "q_tail_pos": float(q_tail_pos),
            "use_pos": bool(use_pos),
        },
    }
    return w_dist, info


def compute_time_discount_weight(
    N: int,
    *,
    r: float = WEIGHT_TIME_R,
    delta: float = WEIGHT_TIME_DELTA,
    k_index: Optional[np.ndarray] = None,
    enabled: bool = WEIGHT_USE_TIME_DISCOUNT,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute time-discount weights with geometric decay into the past."""
    if N < 0:
        raise ValueError(f"N must be >= 0, got {N}")

    w_time = np.ones(N, dtype=float)
    if N == 0:
        info = {
            "enabled": bool(enabled),
            "k0_rank": -1,
            "n_discounted": 0,
            "n_undiscounted": 0,
            "oldest_weight": 1.0,
            "weight_range": (1.0, 1.0),
            "params": {"N": int(N), "r": float(r), "delta": float(delta)},
        }
        return w_time, info

    if enabled:
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"r must be in [0, 1], got {r}")
        if not (0.0 < delta <= 1.0):
            raise ValueError(f"delta must be in (0, 1], got {delta}")

    if k_index is None:
        rank = np.arange(N, dtype=int)
    else:
        k = np.asarray(k_index)
        if len(k) != N:
            raise ValueError(f"k_index length {len(k)} != N={N}")
        order = np.argsort(k)
        rank = np.empty(N, dtype=int)
        rank[order] = np.arange(N, dtype=int)

    k0_rank = int(math.ceil((1.0 - r) * N)) - 1 if enabled else -1
    older_mask = np.zeros(N, dtype=bool)
    n_clipped = 0
    min_weight_floor = float(np.finfo(float).tiny)

    if enabled and k0_rank >= 0:
        older_mask = rank <= k0_rank
        exponents = (k0_rank - rank[older_mask]).astype(float)
        log_delta = float(math.log(delta))
        log_w = exponents * log_delta
        log_floor = float(math.log(min_weight_floor))
        n_clipped = int((log_w < log_floor).sum())
        log_w = np.maximum(log_w, log_floor)
        w_time[older_mask] = np.exp(log_w)

    n_discounted = int(older_mask.sum())
    n_undiscounted = int(N - n_discounted)
    oldest_idx = int(np.argmin(rank))
    oldest_weight = float(w_time[oldest_idx])
    weight_range = (float(w_time.min()), float(w_time.max()))

    info = {
        "enabled": bool(enabled),
        "k0_rank": k0_rank,
        "n_discounted": n_discounted,
        "n_undiscounted": n_undiscounted,
        "oldest_weight": oldest_weight,
        "weight_range": weight_range,
        "n_clipped": n_clipped,
        "min_weight_floor": min_weight_floor,
        "params": {"N": int(N), "r": float(r), "delta": float(delta)},
    }
    return w_time, info


def compute_training_weights(
    m_k: np.ndarray,
    phi: float,
    *,
    use_dist: bool = WEIGHT_USE_BARRIER_DISTANCE,
    use_time: bool = WEIGHT_USE_TIME_DISCOUNT,
    w_max: float = WEIGHT_DIST_W_MAX,
    q_tail: float = WEIGHT_DIST_Q_TAIL,
    w_max_pos: float = WEIGHT_DIST_W_MAX_POS,
    q_tail_pos: float = WEIGHT_DIST_Q_TAIL_POS,
    use_pos: bool = WEIGHT_DIST_USE_POSITIVE,
    r: float = WEIGHT_TIME_R,
    delta: float = WEIGHT_TIME_DELTA,
    k_index: Optional[np.ndarray] = None,
    normalize: bool = WEIGHT_NORMALIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute combined training weights: w = w_dist * w_time."""
    m_k = np.asarray(m_k, dtype=float)
    n = len(m_k)

    w_dist, dist_info = compute_barrier_distance_weight(
        m_k=m_k,
        phi=phi,
        w_max=w_max,
        q_tail=q_tail,
        w_max_pos=w_max_pos,
        q_tail_pos=q_tail_pos,
        use_pos=use_pos,
        enabled=use_dist,
    )
    w_time, time_info = compute_time_discount_weight(
        N=n,
        r=r,
        delta=delta,
        k_index=k_index,
        enabled=use_time,
    )

    w_combined = w_dist * w_time
    if normalize and n > 0:
        w_sum = float(w_combined.sum())
        if w_sum > 0.0:
            w_combined = w_combined * (float(n) / w_sum)

    if n > 0:
        w_mean = float(w_combined.mean())
        w_std = float(w_combined.std())
        denom = float((w_combined ** 2).sum())
        effective_n = float((w_combined.sum() ** 2) / denom) if denom > 0.0 else 0.0
        weight_range = (float(w_combined.min()), float(w_combined.max()))
    else:
        w_mean = 0.0
        w_std = 0.0
        effective_n = 0.0
        weight_range = (1.0, 1.0)

    info = {
        "barrier_distance": dist_info,
        "time_discount": time_info,
        "combined": {
            "weight_range": weight_range,
            "weight_mean": w_mean,
            "weight_std": w_std,
            "effective_n": effective_n,
            "normalized": bool(normalize),
        },
        "config": {
            "use_dist": bool(use_dist),
            "use_time": bool(use_time),
            "use_pos": bool(use_pos),
            "normalize": bool(normalize),
        },
    }
    return w_combined, w_dist, w_time, info


def checkpoint_weights(
    w_combined: np.ndarray,
    w_dist: np.ndarray,
    w_time: np.ndarray,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate computed weights for consistency."""
    results: Dict[str, Any] = {}

    w_combined = np.asarray(w_combined, dtype=float)
    w_dist = np.asarray(w_dist, dtype=float)
    w_time = np.asarray(w_time, dtype=float)

    if not (len(w_combined) == len(w_dist) == len(w_time)):
        raise ValueError("Weight arrays must have the same length")

    results["all_positive"] = {
        "w_dist": bool((w_dist > 0).all()),
        "w_time": bool((w_time > 0).all()),
        "w_combined": bool((w_combined > 0).all()),
        "ok": bool((w_dist > 0).all() and (w_time > 0).all() and (w_combined > 0).all()),
    }
    if not results["all_positive"]["ok"]:
        def _count_nonpos(x: np.ndarray) -> int:
            return int(np.sum(~(x > 0)))

        raise ValueError(
            "Weights must be strictly positive "
            f"(min w_dist={float(np.nanmin(w_dist))}, nonpos={_count_nonpos(w_dist)}; "
            f"min w_time={float(np.nanmin(w_time))}, nonpos={_count_nonpos(w_time)}; "
            f"min w_combined={float(np.nanmin(w_combined))}, nonpos={_count_nonpos(w_combined)})"
        )

    recomputed = w_dist * w_time
    results["multiplicative_consistent"] = {
        "max_diff": float(np.abs(w_combined - recomputed).max()) if len(w_combined) > 0 else 0.0,
        "ok": bool(np.allclose(w_combined, recomputed)),
    }
    if not results["multiplicative_consistent"]["ok"]:
        raise ValueError("Combined weights not equal to w_dist * w_time")

    dist_info = info.get("barrier_distance", {})
    if dist_info.get("enabled", False):
        params = dist_info.get("params", {})
        w_max_neg = params.get("w_max_neg", params.get("w_max", None))
        w_max_pos = params.get("w_max_pos", None)
        max_weight_neg = dist_info.get("max_weight_neg", None)
        max_weight_pos = dist_info.get("max_weight_pos", None)
        if w_max_neg is not None:
            max_weight_neg = float(max_weight_neg) if max_weight_neg is not None else 1.0
            results["dist_bounded"] = {
                "max_weight": max_weight_neg,
                "w_max": float(w_max_neg),
                "ok": bool(max_weight_neg <= float(w_max_neg) + 1e-9),
            }
            if not results["dist_bounded"]["ok"]:
                raise ValueError(
                    f"Barrier-distance weight exceeds w_max (negative side): {max_weight_neg} > {w_max_neg}"
                )
        if dist_info.get("enabled_pos", False) and w_max_pos is not None and max_weight_pos is not None:
            max_weight_pos = float(max_weight_pos)
            results["dist_bounded_pos"] = {
                "max_weight": max_weight_pos,
                "w_max": float(w_max_pos),
                "ok": bool(max_weight_pos <= float(w_max_pos) + 1e-9),
            }
            if not results["dist_bounded_pos"]["ok"]:
                raise ValueError(
                    f"Barrier-distance weight exceeds w_max_pos (positive side): {max_weight_pos} > {w_max_pos}"
                )

    time_info = info.get("time_discount", {})
    if time_info.get("enabled", False):
        results["time_bounded"] = {
            "min_weight": float(w_time.min()) if len(w_time) > 0 else 1.0,
            "max_weight": float(w_time.max()) if len(w_time) > 0 else 1.0,
            "ok": bool(len(w_time) == 0 or (w_time.min() > 0 and w_time.max() <= 1.0 + 1e-9)),
        }
        if not results["time_bounded"]["ok"]:
            raise ValueError(f"Time-discount weight out of (0, 1]: [{w_time.min()}, {w_time.max()}]")

    results["summary"] = {
        "effective_n": info.get("combined", {}).get("effective_n", 0.0),
        "n_observations": int(len(w_combined)),
        "efficiency_ratio": float(info.get("combined", {}).get("effective_n", 0.0) / len(w_combined))
        if len(w_combined) > 0
        else 0.0,
    }

    print("OK: Weight validation passed")
    return results


# =============================================================================
# Section 12.4: Output Helpers
# =============================================================================


def save_feature_list(path: str, feature_names: List[str]) -> None:
    save_json(path, feature_names, indent=2)


def save_metadata(path: str, metadata: Dict[str, Any]) -> None:
    save_json(path, metadata, indent=2)


# =============================================================================
# Section 9: Train/Val/Test Split
# =============================================================================


def recommended_embargo_for_cadence(
    label_cadence: str, *, base_embargo: int = 60, M: int | None = None
) -> int:
    """Calendar-time-preserving embargo for the train/val and val/test splits.

    At **boundary** cadence (one row per M bars, labels non-overlapping by
    construction), an embargo of ``base_embargo`` boundary rows is the
    standard López-de-Prado recommendation (~ EMBARGO_K = 60 boundaries
    = 20 hours at M=20).

    At **1-min** cadence, adjacent labels share M-1 of their M future bars,
    so the **minimum** embargo to avoid any label overlap between train and
    val is ``M`` rows. For calendar-time parity with boundary cadence, we
    scale by M: ``base_embargo * M`` rows. The returned value is the maximum
    of these two, so callers always get something at least M.

    Critical: if you split a 1-min cadence dataset with embargo_k=60 (the
    boundary-cadence default), val's first row has its label-prediction
    window overlapping train's last 19 rows' windows. The model "validates"
    on the same future bars it trained on. Use this helper.
    """
    from src.features.config import M as M_DEFAULT

    M_int = int(M) if M is not None else int(M_DEFAULT)
    if label_cadence == "boundary":
        return int(base_embargo)
    if label_cadence == "1min":
        return max(M_int, int(base_embargo) * M_int)
    raise ValueError(
        f"label_cadence must be 'boundary' or '1min', got {label_cadence!r}"
    )


def recommended_time_discount_delta_for_cadence(
    label_cadence: str, *, base_delta: float, M: int | None = None
) -> float:
    """Per-step decay factor scaled to preserve calendar-time decay.

    ``compute_time_discount_weight`` applies ``delta`` once per row. At
    boundary cadence each row is M bars apart; at 1-min cadence each row
    is 1 bar apart. To make a one-day-old sample have the same weight in
    both cadences, the 1-min delta must equal ``base_delta ** (1/M)``.

    For example, with base_delta=0.5 and M=20 the 1-min delta is ~0.9659:
    the weight halves over each 20 minutes, identical to the boundary
    cadence's "halve per boundary".
    """
    from src.features.config import M as M_DEFAULT

    M_int = int(M) if M is not None else int(M_DEFAULT)
    if not (0.0 < float(base_delta) <= 1.0):
        raise ValueError(f"base_delta must be in (0, 1], got {base_delta}")
    if label_cadence == "boundary":
        return float(base_delta)
    if label_cadence == "1min":
        return float(base_delta) ** (1.0 / float(M_int))
    raise ValueError(
        f"label_cadence must be 'boundary' or '1min', got {label_cadence!r}"
    )


def chronological_split_with_embargo(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo_k: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split with an embargo gap.

    ``embargo_k`` is in **rows of the input dataframe** — its meaning
    depends on the cadence used to build ``df``. Use
    ``recommended_embargo_for_cadence`` to pick a safe value:

        embargo = recommended_embargo_for_cadence(label_cadence)
        train, val, test = chronological_split_with_embargo(df, embargo_k=embargo)

    Calling with the boundary-cadence default ``embargo_k=60`` on a 1-min
    cadence dataset leaks label-window overlap into the validation set.
    """
    n = len(df)
    train_end_idx = int(train_frac * n)
    val_end_idx = int((train_frac + val_frac) * n)

    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx + embargo_k : val_end_idx].copy()
    test_df = df.iloc[val_end_idx + embargo_k :].copy()

    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("chronological_split_with_embargo produced empty validation or test split")

    if not (train_df["k"].max() < val_df["k"].min()):
        raise ValueError("Train/val overlap or ordering violation")
    if not (val_df["k"].max() < test_df["k"].min()):
        raise ValueError("Val/test overlap or ordering violation")

    return train_df, val_df, test_df


def walk_forward_cv(df: pd.DataFrame, n_folds: int = 3, embargo_k: int = 1) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Walk-forward CV folds with an embargo gap between train and val of each fold.

    Same cadence caveat as ``chronological_split_with_embargo``: pass an
    ``embargo_k`` sized to the input dataframe's cadence. At 1-min cadence
    use ``recommended_embargo_for_cadence('1min')`` (>= M rows).
    """
    n = len(df)
    fold_size = n // (n_folds + 1)

    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        val_start = train_end + embargo_k
        val_end = min(val_start + fold_size, n)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        if len(val_df) > 0:
            folds.append((train_df, val_df))
    return folds


# =============================================================================
# Section 11: Evaluation Metrics
# =============================================================================


def compute_all_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "log_loss": float(log_loss(y_true, y_pred_proba)),
        "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
        "ece": float(expected_calibration_error(y_true, y_pred_proba, n_bins=10)),
    }


def expected_calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_pred_proba >= lo) & (y_pred_proba <= hi)
        else:
            mask = (y_pred_proba >= lo) & (y_pred_proba < hi)
        if mask.sum() == 0:
            continue

        bin_accuracy = float(y_true[mask].mean())
        bin_confidence = float(y_pred_proba[mask].mean())
        bin_weight = float(mask.sum() / len(y_true))
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def calibration_by_regime(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    vol_series: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    from sklearn.metrics import brier_score_loss

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    vol_series = np.asarray(vol_series)

    vol_terciles = pd.qcut(vol_series, 3, labels=["low", "med", "high"])
    results: Dict[str, Dict[str, float]] = {}
    for regime in ["low", "med", "high"]:
        mask = np.asarray(vol_terciles == regime)
        if mask.sum() < 50:
            continue
        results[regime] = {
            "n_samples": int(mask.sum()),
            "base_rate": float(y_true[mask].mean()),
            "ece": float(expected_calibration_error(y_true[mask], y_pred_proba[mask], n_bins=n_bins)),
            "brier": float(brier_score_loss(y_true[mask], y_pred_proba[mask])),
        }
    return results


def threshold_analysis(y_true: np.ndarray, y_pred_proba: np.ndarray, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    rows: List[Dict[str, Any]] = []
    n = len(y_true)
    n_pos = int((y_true == 1).sum())

    for t in thresholds:
        pred = y_pred_proba >= float(t)
        n_trades = int(pred.sum())
        tp = int(((y_true == 1) & pred).sum())
        precision = float(tp / n_trades) if n_trades > 0 else float("nan")
        recall = float(tp / n_pos) if n_pos > 0 else float("nan")
        rows.append(
            {
                "threshold": float(t),
                "n_trades": n_trades,
                "trade_rate": float(n_trades / n),
                "hit_rate": precision,
                "precision": precision,
                "recall": recall,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Section 12.1: Plotting
# =============================================================================


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    ax=None,
    *,
    label: str = "Model",
    color: str | None = None,
    plot_perfect: bool = True,
    show_ece: bool = True,
):
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs: List[float] = []
    ys: List[float] = []
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_pred_proba >= lo) & (y_pred_proba <= hi)
        else:
            mask = (y_pred_proba >= lo) & (y_pred_proba < hi)
        if mask.sum() == 0:
            continue
        xs.append(float(y_pred_proba[mask].mean()))
        ys.append(float(y_true[mask].mean()))

    if plot_perfect:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.plot(xs, ys, marker="o", label=label, color=color)
    ece = expected_calibration_error(y_true, y_pred_proba, n_bins=n_bins)
    if show_ece:
        ax.set_title(f"Calibration Curve (ECE={ece:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return ax


def plot_calibration_by_regime(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    vol_series: np.ndarray,
    n_bins: int = 10,
    fig=None,
):
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    vol_series = np.asarray(vol_series)

    regimes = pd.qcut(vol_series, 3, labels=["low", "med", "high"])
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    else:
        axes = fig.subplots(1, 3, sharex=True, sharey=True)

    for ax, regime in zip(axes, ["low", "med", "high"]):
        mask = np.asarray(regimes == regime)
        if mask.sum() == 0:
            ax.set_title(f"{regime} (n=0)")
            ax.axis("off")
            continue
        plot_calibration_curve(y_true[mask], y_pred_proba[mask], n_bins=n_bins, ax=ax)
        ax.set_title(f"{regime} vol (n={int(mask.sum())})")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: List[str], top_n: int = 30, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    importances = np.asarray(model.get_feature_importance(), dtype=float)
    order = np.argsort(importances)[::-1][:top_n]
    feats = [feature_names[i] for i in order]
    vals = importances[order]

    ax.barh(list(reversed(feats)), list(reversed(vals)))
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    return ax


def plot_threshold_curves(thresh_df: pd.DataFrame, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.plot(thresh_df["threshold"], thresh_df["trade_rate"], label="Trade rate")
    ax.plot(thresh_df["threshold"], thresh_df["precision"], label="Precision / Hit rate")
    ax.plot(thresh_df["threshold"], thresh_df["recall"], label="Recall")
    ax.set_title("Threshold Analysis")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_xlim(0, 1)
    ax.legend()
    return ax


def plot_weight_profiles(
    m_k: np.ndarray,
    phi: float,
    w_dist: np.ndarray,
    w_time: np.ndarray,
    *,
    k_index: Optional[np.ndarray] = None,
    info: Optional[Dict[str, Any]] = None,
    fig: Optional[Any] = None,
) -> Any:
    """Plot barrier-distance and time-discount weight profiles."""
    import matplotlib.pyplot as plt

    m_k = np.asarray(m_k, dtype=float)
    w_dist = np.asarray(w_dist, dtype=float)
    w_time = np.asarray(w_time, dtype=float)
    n = len(m_k)
    if len(w_dist) != n or len(w_time) != n:
        raise ValueError("plot_weight_profiles: m_k, w_dist, w_time must have the same length")

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        axes = fig.subplots(1, 2)

    ax1, ax2 = axes

    # Barrier-distance profile
    pos_mask = m_k >= phi
    if n > 50000:
        step = max(1, n // 50000)
        idx = np.arange(0, n, step)
        m_k_plot = m_k[idx]
        w_dist_plot = w_dist[idx]
        pos_plot = pos_mask[idx]
    else:
        m_k_plot = m_k
        w_dist_plot = w_dist
        pos_plot = pos_mask

    ax1.scatter(m_k_plot[pos_plot], w_dist_plot[pos_plot], s=6, alpha=0.4, color="#2ecc71", label="m_k >= phi")
    ax1.scatter(m_k_plot[~pos_plot], w_dist_plot[~pos_plot], s=6, alpha=0.4, color="#e74c3c", label="m_k < phi")
    ax1.axvline(phi, color="black", linestyle="--", linewidth=1)
    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    if info is not None:
        dist_info = info.get("barrier_distance", {})
        params = dist_info.get("params", {})
        w_max_neg = params.get("w_max_neg", params.get("w_max", None))
        w_max_pos = params.get("w_max_pos", None)
        enabled_pos = dist_info.get("enabled_pos", False)
        if w_max_neg is not None:
            ax1.axhline(float(w_max_neg), color="orange", linestyle="--", linewidth=1)
        if enabled_pos and w_max_pos is not None and w_max_pos != w_max_neg:
            ax1.axhline(float(w_max_pos), color="#2ecc71", linestyle="--", linewidth=1)
        lam_neg = dist_info.get("lambda", 0.0)
        d_star = dist_info.get("d_star", 0.0)
        if enabled_pos:
            lam_pos = dist_info.get("lambda_pos", 0.0)
            g_star = dist_info.get("g_star", 0.0)
            ax1.set_title(
                f"Barrier-distance weights (lambda-={lam_neg:.4g}, d*={d_star:.4g}; "
                f"lambda+={lam_pos:.4g}, g*={g_star:.4g})"
            )
        else:
            ax1.set_title(f"Barrier-distance weights (lambda={lam_neg:.4g}, d*={d_star:.4g})")
    else:
        ax1.set_title("Barrier-distance weights")
    ax1.set_xlabel("m_k")
    ax1.set_ylabel("w_dist")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Time-discount profile
    if k_index is None:
        rank = np.arange(n, dtype=int)
    else:
        k = np.asarray(k_index)
        if len(k) != n:
            raise ValueError("plot_weight_profiles: k_index length mismatch")
        order = np.argsort(k)
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n, dtype=int)

    if n > 1:
        frac = rank / (n - 1)
    else:
        frac = rank.astype(float)

    order = np.argsort(rank)
    ax2.plot(frac[order], w_time[order], color="#3498db", linewidth=1.5)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    if info is not None:
        r_val = info.get("time_discount", {}).get("params", {}).get("r", None)
        delta_val = info.get("time_discount", {}).get("params", {}).get("delta", None)
        if r_val is not None:
            ax2.axvline(1.0 - float(r_val), color="black", linestyle="--", linewidth=1)
        ax2.set_title(f"Time-discount weights (r={r_val}, delta={delta_val})")
    else:
        ax2.set_title("Time-discount weights")
    ax2.set_xlabel("Observation position (0=oldest, 1=newest)")
    ax2.set_ylabel("w_time")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_weight_distributions(
    w_dist: np.ndarray,
    w_time: np.ndarray,
    w_combined: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    info: Optional[Dict[str, Any]] = None,
    fig: Optional[Any] = None,
) -> Any:
    """Plot distributions for w_dist, w_time, and combined weights."""
    import matplotlib.pyplot as plt

    w_dist = np.asarray(w_dist, dtype=float)
    w_time = np.asarray(w_time, dtype=float)
    w_combined = np.asarray(w_combined, dtype=float)

    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    else:
        axes = fig.subplots(1, 3)

    ax1, ax2, ax3 = axes

    w_max_neg = None
    w_max_pos = None
    w_max_cap = None
    enabled_pos = False
    if info is not None:
        dist_info = info.get("barrier_distance", {})
        params = dist_info.get("params", {})
        w_max_neg = params.get("w_max_neg", params.get("w_max", None))
        w_max_pos = params.get("w_max_pos", None)
        enabled_pos = dist_info.get("enabled_pos", False)
        if w_max_neg is not None:
            w_max_cap = float(w_max_neg)
        if enabled_pos and w_max_pos is not None:
            w_max_cap = float(w_max_pos) if w_max_cap is None else max(w_max_cap, float(w_max_pos))

    if y is not None:
        y = np.asarray(y, dtype=int)
        pos_mask = y == 1
        neg_mask = y == 0
        if w_max_cap is None:
            bins_dist = 50
        else:
            bins_dist = np.linspace(0.9, float(w_max_cap) + 0.1, 50)
        ax1.hist(w_dist[neg_mask], bins=bins_dist, alpha=0.6, color="#e74c3c", label="y=0", density=True)
        ax1.hist(w_dist[pos_mask], bins=bins_dist, alpha=0.6, color="#2ecc71", label="y=1", density=True)
        ax1.legend(loc="upper right")
    else:
        ax1.hist(w_dist, bins=50, alpha=0.7, color="#2ecc71", density=True)
    ax1.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    if w_max_neg is not None:
        ax1.axvline(float(w_max_neg), color="orange", linestyle="--", linewidth=1)
    if enabled_pos and w_max_pos is not None and w_max_pos != w_max_neg:
        ax1.axvline(float(w_max_pos), color="#2ecc71", linestyle="--", linewidth=1)
    ax1.set_title("Barrier-distance weights")
    ax1.set_xlabel("w_dist")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)

    ax2.hist(w_time, bins=50, alpha=0.7, color="#3498db", density=True)
    ax2.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_title("Time-discount weights")
    ax2.set_xlabel("w_time")
    ax2.set_ylabel("Density")
    ax2.grid(True, alpha=0.3)

    w_log = np.log10(w_combined + EPS)
    if y is not None:
        ax3.hist(w_log[neg_mask], bins=50, alpha=0.6, color="#e74c3c", label="y=0", density=True)
        ax3.hist(w_log[pos_mask], bins=50, alpha=0.6, color="#2ecc71", label="y=1", density=True)
        ax3.legend(loc="upper right")
    else:
        ax3.hist(w_log, bins=50, alpha=0.7, color="#8e44ad", density=True)
    ax3.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax3.set_title("Combined weights (log10)")
    ax3.set_xlabel("log10(w_combined)")
    ax3.set_ylabel("Density")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# =============================================================================
# Section 13: Feature Selection Utilities
# =============================================================================


def importance_by_window(model, feature_names: List[str]) -> Dict[int, float]:
    importances = dict(zip(feature_names, np.asarray(model.get_feature_importance(), dtype=float).tolist()))
    by_window: Dict[int, float] = {}
    for name, imp in importances.items():
        match = re.search(r"__w(\\d+)", name)
        if match:
            w = int(match.group(1))
            by_window[w] = by_window.get(w, 0.0) + float(imp)
    return dict(sorted(by_window.items(), key=lambda x: -x[1]))


def stability_selection(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_folds: int = 4,
    importance_threshold: float = 0.001,
) -> List[str]:
    from catboost import CatBoostClassifier

    importance_counts: Dict[str, int] = {c: 0 for c in feature_cols}
    for train_df, val_df in walk_forward_cv(df, n_folds=n_folds, embargo_k=EMBARGO_K):
        X_train = train_df[feature_cols]
        y_train = train_df["y"].astype(int)
        X_val = val_df[feature_cols]
        y_val = val_df["y"].astype(int)

        model = CatBoostClassifier(
            iterations=CB_ITERATIONS,
            learning_rate=CB_LEARNING_RATE,
            depth=CB_DEPTH,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=CB_L2_LEAF_REG,
            early_stopping_rounds=CB_EARLY_STOPPING,
            use_best_model=True,
            auto_class_weights="Balanced",
            random_seed=CB_SEED,
            verbose=False,
            thread_count=-1,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        imp = np.asarray(model.get_feature_importance(), dtype=float)
        for feat, val in zip(feature_cols, imp):
            if float(val) > float(importance_threshold):
                importance_counts[feat] += 1

    stable = [f for f, cnt in importance_counts.items() if cnt >= n_folds // 2]
    return stable
