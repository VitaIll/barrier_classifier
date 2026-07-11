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
import warnings
import zipfile
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
    # Long-horizon lags added for hourly-target-friendly feature design
    # (longer-term dynamics carry more signal at M >= 45). Capped at 14400
    # so N_WARMUP stays at 20159 (dominated by max(WINDOWS_F)-1).
    5760,
    7200,
    10080,
    14400,
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

# Excursion windows extended to multi-day scales for hourly-target-friendly
# feature design (P2 from the strategy-improvement plan). The added windows
# capture multi-day drawdown / drawup context that 20-min trades barely use
# but hourly / 45-min trades care about. Calendar-stable at 1-min cadence.
WINDOWS_EXCURSION: List[int] = [10, 20, 30, 60, 120, 240, 480, 960, 1440, 2880, 4320, 7200, 10080]
WINDOWS_MAXRET: List[int] = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]

# Causal local-extreme structure: dist-to-trailing-low/high (volatility-normalized)
# and bounded price rank. Mirrors the WINDOWS_EXCURSION ladder so the two
# families compose on the same scales.
WINDOWS_EXTREME: List[int] = [30, 60, 120, 240, 480, 960, 1440]

# Quadratic trend fits: slope and curvature over a trailing window. Limited
# to mid-sized windows — quadratic fits over very short windows are noise,
# and over very long windows the linear term dominates.
WINDOWS_QUAD_TREND: List[int] = [30, 60, 120, 240, 480]

# Pivot (Hi/Lo) carry-forward window length. Q (confirmation lag) is fixed
# per emitted column and listed separately.
WINDOWS_PIVOT: List[int] = [240, 480, 960]
# Q values: confirmed swing pivots require Q bars before AND after the
# candidate extremum. Two Q values let the model see fast and slow pivots.
PIVOT_Q_VALUES: List[int] = [5, 15]

# Signed semivariance ratio and bipower-jump share. Match the WINDOWS_VOL_DECOMP
# ladder so the model can compare them against existing decomposition features.
WINDOWS_VOL_SIGNED: List[int] = [60, 120, 240, 480, 960, 1440]
WINDOWS_VOL_JUMP: List[int] = [60, 120, 240, 480, 960, 1440]

# Rolling signed flow pressure + sell-absorption / buy-exhaustion interactions.
WINDOWS_FLOW_PRESSURE: List[int] = [60, 120, 240, 480, 960, 1440]

# OI regime decomposition (long_build / short_build / short_cover / long_liq):
# two windows because most OI snapshot feeds publish at coarser cadence than
# 1m, and the short window has to be large enough for ΔOI to register.
WINDOWS_OI_REGIME: List[int] = [60, 240, 480]

# Equilibrium-residual ladder. All ``eq__*`` features use strictly past-only
# windows (``rolling_X(...).shift(1)`` semantics) so the equilibrium estimate
# at row n is built from rows {n-W, ..., n-1} only. Mid-range windows: too
# short ⇒ noisy equilibrium, too long ⇒ stale fair value.
WINDOWS_EQ: List[int] = [30, 60, 120, 240, 480, 960]
# EWMA half-lives (in bars) for the recursive equilibrium proxy. Span used
# internally is ``2H - 1`` to match the canonical half-life parametrization.
HALFLIVES_EQ: List[int] = [30, 120, 480]
# (short, long) window pairs for rising-equilibrium pullback / falling-eq
# overextension interaction features. Short < long by construction.
WINDOWS_EQ_PAIRS: List[Tuple[int, int]] = [(30, 240), (60, 480), (120, 960)]

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
CB_L2_LEAF_REG: float = 3.0

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
    "iterations": CB_ITERATIONS,
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
    "border_count": 700,

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


def get_git_sha(repo_dir: str | Path | None = None) -> str:
    """Return the current git HEAD sha (40 chars) or ``"unknown"`` if
    ``git`` is unavailable / the path is not a repo.

    Used by metadata writers (build/train notebooks) so every artifact
    can be traced back to a commit. Stdlib-only — no new dependencies.
    """
    import subprocess
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    if res.returncode != 0:
        return "unknown"
    return res.stdout.strip() or "unknown"


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
    """Best-effort repair of out-of-pandas-bounds ``open_time`` values.

    Emits a ``UserWarning`` whenever any rescaling (us/ns mistakenly interpreted
    as ms) or linear extrapolation fires, since either path invents timestamps
    that were not present in the source CSV. Callers that need strict semantics
    should validate upstream rather than rely on this repair.
    """
    ts = pd.to_datetime(open_time_ms + step_ms, unit="ms", utc=True, errors="coerce")
    invalid = ts.isna()
    n_invalid = int(invalid.sum())
    if n_invalid == 0:
        return open_time_ms, 0

    values = open_time_ms.to_numpy(copy=True, dtype="int64")
    invalid_mask = invalid.to_numpy()
    n_rescaled = 0
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
            n_rescaled += int(in_range_1k.sum())

        if invalid_mask.any():
            invalid_values = values[invalid_mask]
            scaled_1m = invalid_values // 1_000_000
            in_range_1m = (scaled_1m + step_ms >= min_ms) & (scaled_1m + step_ms <= max_ms)
            if in_range_1m.any():
                invalid_idx = np.flatnonzero(invalid_mask)
                values[invalid_idx[in_range_1m]] = scaled_1m[in_range_1m]
                invalid_mask[invalid_idx[in_range_1m]] = False
                n_rescaled += int(in_range_1m.sum())
    valid_idx = np.flatnonzero(~invalid_mask)
    if valid_idx.size == 0:
        raise ValueError("convert_timestamps: all open_time values out of bounds")

    n_extrapolated = int(invalid_mask.sum())
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

    if n_rescaled or n_extrapolated:
        warnings.warn(
            "_repair_out_of_bounds_open_time: invented timestamps "
            f"(rescaled={n_rescaled}, extrapolated={n_extrapolated}, total_invalid={n_invalid}). "
            "These rows did not have a valid open_time in the source CSV; downstream "
            "timestamps are best-effort and may not match exchange data.",
            UserWarning,
            stacklevel=2,
        )

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
    """Compute Group K past-target features using matured labels only (Section 7.20).

    Convention note (parity contract, do not change):
        ``hit__since__h__w0`` is computed as ``df['k'] - hit_k.shift(1).ffill()``.
        The ``ffill()`` is intentional: before the first matured hit (i.e. across
        the warmup region) ``last_hit_before`` is NaN, which propagates into
        ``hit__since__h__w0`` and is dropped by downstream warmup trimming. After
        the first hit, the value forward-fills past subsequent non-hit rows so
        that "time since last hit" grows monotonically. This is the legacy
        behaviour and is the parity oracle for the Polars engine; the engine
        replicates it bit-for-bit.
    """
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
    """Compute derivatives volume and order flow features (Group S).

    Convention note (parity contract, do not change):
        ``liq__avg_trade_size__f__w15`` and ``activity__trades_zscore__f__w30``
        are emitted UNCONDITIONALLY regardless of whether 15 / 30 appear in
        ``windows``. ``windows`` here only gates the ``flow__net_vol_csum__f__w*``
        series (5/10/20). The legacy Polars engine
        (``src/features/families/derivatives.py``: ``FlowAvgTradeSize15`` /
        ``FlowTradesZscore30``) mirrors this with empty ``windows = ()``, so
        both 15 and 30 are always produced. Do not gate these columns — it
        would break the parity oracle in tests/features/test_family_step11.py.
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
        # Drifted-Brownian first-passage probability. Float in [0, 1].
        # Default fill = 0.5 (uninformative) when the underlying mu/sigma
        # window has not yet warmed up. The undef__ flag distinguishes
        # missing-due-to-warmup from a genuine 0.5 estimate.
        (r"^barrier__p_hit_drifted", 0.5),
        (r"^vol__bpv_ratio", 1.0),
        # Bipower-jump share in [0, 1]; "no detectable jump" is the safe
        # default when the window has not yet warmed up.
        (r"^vol__jump_ratio", 0.0),
        # Signed semivariance ratio in [-1, 1]; "no asymmetry" is the
        # safe neutral default.
        (r"^vol__semivar_signed", 0.0),
        (r"^vol__semivar_ratio", 1.0),
        (r"^vol__ratio", 1.0),
        (r"^vol__", 0.0),
        (r"^ret__rsi", 50.0),
        (r"^ret__posfrac", 0.5),
        (r"^logp__pos", 0.5),
        (r"^tb_ratio", 0.5),
        (r"^block__close_to_high", 0.5),
        (r"^pentropy_norm", 0.5),
        # Causal local-extreme features (Round 1a). Distance features get a
        # large "not near the extreme" fallback; price rank gets 0.5
        # (median of the bounded support).
        (r"^extreme__dist_low_z", 5.0),
        (r"^extreme__dist_high_z", 5.0),
        (r"^extreme__price_rank", 0.5),
        # Quadratic trend features (Round 1b). Both are real-valued and
        # centred at 0 in expectation; "no detected slope / curvature" is
        # the right neutral fill.
        (r"^trend__quad_slope_z", 0.0),
        (r"^trend__quad_curv_z", 0.0),
        # Pivot features (Round 1c). Distance neutral = 0; age sentinel
        # already lives in the column (= W) for no-pivot rows, so the
        # imputation handles only true warmup nulls — pick "old pivot" as
        # the safer default (high age = stale).
        (r"^pivot__last_low_dist_z", 0.0),
        (r"^pivot__last_high_dist_z", 0.0),
        (r"^pivot__last_low_age", 0.0),
        (r"^pivot__last_high_age", 0.0),
        # Rolling flow pressure and the two interactions. All bounded and
        # zero-centred in expectation; neutral fill = 0.
        (r"^flow__pressure", 0.0),
        (r"^flow__sell_absorption", 0.0),
        (r"^flow__buy_exhaustion", 0.0),
        # Equilibrium-residual family (Round 2).
        #
        # Residuals / innovations (``eq__*_resid_*``, ``eq__*_innov_*``,
        # ``eq__upside_to_eq_over_phi``, ``eq__barrier_vs_eq_hz``): real-
        # valued, zero-centred in expectation, so 0.0 is the neutral
        # ``no-information`` fill. The ``undef`` flag distinguishes warmup
        # from a genuine zero residual.
        #
        # Non-negative interactions (``eq__proxy_dispersion``,
        # ``eq__pullback_rising_eq``, ``eq__above_falling_eq``): 0.0 = "no
        # detected dispersion / pullback / overextension"; correct neutral
        # for a max-zero-clamped feature.
        #
        # Raw proxy / scale columns (``eq__mu_*``, ``eq__sigma_*``,
        # ``eq__mad_p``, ``eq__trend_sresid``): used only as tier-2 inputs;
        # post-impute values land downstream of the residual columns. Mu's
        # of log price filled at 0 is far from any real BTC log close
        # (~ln(50_000) ≈ 10.8), so a 0 fill is detectable as ``warmup`` via
        # the ``undef__`` flag; sigma / scale columns filled at 0 reduce
        # the residual denominator to EPS, which the residual feature
        # itself handles via the EPS guard. Catch-all 0.0 is correct for
        # all of them.
        (r"^eq__", 0.0),
        # Target-derived matured-label memory (Round 2). m_mean is a
        # log-return; 0 is the right "no observed burst" default. tau
        # gets the horizon M as the "slow / absent" sentinel. Near-miss
        # rates are in [0, 1]; 0 means "no near-misses observed".
        (r"^target__mature_m_mean", 0.0),
        (r"^target__mature_m_pos_mean", 0.0),
        (r"^target__mature_tau_pos_mean", float(cap_h_blocks)),
        (r"^target__mature_near_miss_up", 0.0),
        (r"^target__mature_near_miss_dn", 0.0),
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
        # OI regime decomposition (Round 4a). Each quadrant feature is in
        # [0, 1], with the four columns mutually exclusive on any given
        # row. Neutral fill = 0 (no regime detected).
        (r"^oi__long_build", 0.0),
        (r"^oi__short_build", 0.0),
        (r"^oi__short_cover", 0.0),
        (r"^oi__long_liq", 0.0),
        (r"^funding__rate", 0.0),
        (r"^funding__trend", 0.0),
        (r"^funding__ewma", 0.0),
        # Funding phase (Round 4b). phase ∈ (0, 1]; default = 0.5 puts
        # the imputed row "halfway between settlements", which is the
        # least informative value. Sin/cos ∈ [-1, 1] with neutral 0.
        (r"^funding__phase_sin", 0.0),
        (r"^funding__phase_cos", 0.0),
        (r"^funding__phase", 0.5),
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



# =============================================================================
# Section 12.4: Output Helpers
# =============================================================================






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
