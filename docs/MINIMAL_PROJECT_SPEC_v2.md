> **LEGACY — DO NOT USE AS SOURCE OF TRUTH.** This document predates the ongoing refactor and may not reflect current code. Kept for historical reference only.

# Minimal Barrier-Crossing Classifier Specification (BTCUSDT 1m)

**Version:** 4.0  
**Date:** 2026-01-03  
**Status:** Implementation-aligned (matches `src/utils.py` and the notebooks)  
**Scope:** Offline binary classification on Binance 1-minute data: predict whether price crosses an upward log-return barrier within the next `M=10` minutes (one decision interval).

---

## Table of Contents

- 0. Technical Executive Summary
- 1. High-Level Workflow
- 2. Research Objective
- 3. Repository Structure
- 4. Binance Data Acquisition (Detailed)
- 5. Time Indexing
- 6. Label Definition
- 7. Complete Feature Definitions
- 8. Missing Value Handling
- 9. Train/Validation/Test Split
- 10. CatBoost Configuration
- 11. Evaluation Metrics
- 12. Output Requirements
- 13. Feature Parsimony Strategy
- 14. Implementation Index
- Appendix A: Requirements
- Appendix B: Configuration Constants
- Appendix C: Quick Start
- Appendix D: Changelog (from v3.1 to v4.0)
- Appendix E: BTC/USDT Derivatives-Derived Features
- Appendix F: References

---

## 0. Technical Executive Summary

### Project Purpose and Scope
This repository implements an offline machine-learning pipeline that converts Binance 1-minute BTCUSDT market data into a supervised dataset and a calibrated probability model. At discrete **decision boundaries** spaced `M=10` minutes apart, the task is to estimate the probability that the price will realize an **upward barrier crossing** within the next decision interval. Formally, at decision index `k` we observe a feature snapshot `x_k` built only from information available up to the boundary time, and we predict `p(y_k=1 | x_k)` where `y_k` indicates whether the maximum future log return over the next `M` 1-minute bars exceeds a fixed barrier `phi`.

The primary use case is short-horizon probability forecasting for trading decision support and market regime analysis (e.g., risk-on/risk-off). The deliverables are designed for reproducible research: validated raw parquet(s), a fully materialized model dataset with an explicit feature contract, a trained CatBoost model artifact, and a complete evaluation bundle (metrics + plots) produced under a strict no-peeking protocol.

Out of scope: production deployment, real-time streaming ingestion, online learning, execution logic, PnL attribution, portfolio construction, multi-asset support, and exchange connectivity beyond public historical data downloads. This project intentionally prioritizes strict causality invariants and verifiable data contracts over “production readiness.”

### Architectural Overview
The system is notebook-driven and strictly offline. `01_data_download.ipynb` downloads Binance public-data ZIP archives (spot klines; optionally derivatives datasets), verifies SHA256 checksums, parses CSVs into typed DataFrames, performs schema and integrity checks, and writes validated parquet files. `02_feature_building.ipynb` loads the validated spot parquet, computes a large engineered feature set on the 1-minute grid, optionally joins and featurizes derivatives sources, samples decision boundaries every `M` bars, constructs labels using only future bars, trims a warmup region where lookbacks are incomplete, adds undefined flags, imputes remaining NaNs deterministically, and saves the final dataset and metadata. `03_model_training.ipynb` loads the dataset and `feature_list.json`, applies a chronological split with an explicit embargo, trains a CatBoost classifier using time-aware Pools (ordered boosting with a timestamp), and evaluates calibration and discrimination on held-out validation and test splits.

**High-level data flow:**

```
Binance public ZIPs (spot + optional derivatives)
  -> checksum-verified downloads
  -> parse + timestamp normalization + validation
  -> data/raw_data/klines_1m.parquet (+ data/raw_data/derivatives/*.parquet)
  -> feature engineering on 1m grid (Groups A–Q)
  -> decision-boundary sampling (every M bars) + labels + boundary-only features
  -> warmup trim + undefined flags + imputation (+ optional sample weights)
  -> data/model_dataset/dataset.parquet + feature_list.json + dataset_metadata.json
  -> chronological split + embargo
  -> CatBoost model + metrics + plots + saved artifacts
```

### Key Design Decisions
- **Decision cadence matches horizon:** A decision boundary every `M=10` minutes yields one label per 10-minute interval and makes the horizon explicit.
- **Barrier label:** Define `m_k = max_{j=1..M} log(P_{n_k+j}/P_{n_k})` and `y_k = 1[m_k >= phi]` with a fixed global barrier `phi = C + ETA` (constants in Appendix B).
- **No-lookahead invariant:** Features at boundary `k` use bars `<= n_k` only; the label uses bars `n_k+1..n_k+M` only; past-target features use only matured labels `<= k-1`.
- **Warmup trimming:** Drop early boundaries where any lookback window, lag, or block statistic is structurally undefined (`k < K_WARMUP`).
- **Explicit feature contract:** Persist `feature_list.json` and build `X` exclusively from it; never infer features as “all columns except `y`.”
- **Undefined flags as signal:** For features that can be undefined due to structural reasons (e.g., zero volume, missing derivatives coverage), create `undef__*` indicators and keep them as model inputs.
- **Deterministic imputation:** After flagging, impute remaining NaNs using a rule table keyed by feature name patterns so CatBoost never sees NaNs.
- **Chronological split + embargo:** Split only at training time, and enforce an embargo gap between adjacent splits to prevent horizon overlap leakage.
- **Time-aware boosting:** Use CatBoost Ordered boosting with timestamps (`k`) to respect temporal ordering and reduce leakage.
- **Optional sample weighting:** Optionally weight observations using barrier-distance and time-discount schemes to emphasize informative negatives and recent data.
- **Optional derivatives integration:** Join derivatives sources by as-of alignment onto the 1-minute grid without extrapolating outside each source’s coverage window (Appendix E).

### Feature Engineering Philosophy
The feature set is causal, multi-scale, and tailored to heavy-tailed, heteroskedastic returns. It combines distributional statistics of returns and ranges, multiple volatility estimators and decompositions, candle geometry and breakout state, trend/momentum transforms, microstructure proxies (volume, taker flow, OFI, VWAP deviation), serial dependence and complexity (permutation entropy), and label-aligned excursion features. Barrier-aware features normalize the barrier size by horizon-adjusted volatility to encode signal-to-noise and regime priors. Optional derivatives features add basis, funding, open interest, options sentiment, and implied volatility risk-premium signals, aligned to spot timestamps with strict no-lookahead rules.

### Model Selection Rationale
CatBoost is chosen for robust performance on noisy tabular features and for training-time controls that accommodate temporal structure. The implementation uses ordered boosting, early stopping, and time-aware Pools with a monotone timestamp. This combination yields a strong baseline that is easy to reproduce and diagnose with calibration-focused metrics. Alternatives (logistic regression, XGBoost/LightGBM, neural sequence models) are viable, but must preserve the same feature/label contracts, split discipline, and evaluation protocol to avoid subtle leakage.

### Success Metrics
Model quality is assessed on held-out validation and test splits using proper scoring rules and calibration: Log Loss, Brier score, Expected Calibration Error (ECE), ROC-AUC, PR-AUC, and regime-stratified calibration using volatility terciles of `vol__rs__f__w240` (Sections 11–12). A successful run produces the required artifacts and passes all pipeline checkpoints.

---

## 1. High-Level Workflow

### 1.0 Overview
**Purpose:** Define the end-to-end offline pipeline (data → features/labels → training → evaluation) and the required artifact flow.  
**Scope:** Notebook execution order, inputs/outputs, and high-level stage responsibilities (not the detailed formulas).  
**Dependencies:** Appendix B (constants) and Sections 4–12 (detailed contracts).  
**Implementation Location:** `notebooks/01_data_download.ipynb`, `notebooks/02_feature_building.ipynb`, `notebooks/03_model_training.ipynb`, `src/utils.py`.

### 1.1 Deliverable Requirements
This specification defines a complete, self-contained project that must be:

1. **Delivered:** All code, notebooks, and documentation are present and functional.
2. **Documented:** Every formula, parameter, file contract, and validation rule is explicit.
3. **Correct:** Causality invariants are enforced; no lookahead bias.
4. **Internally Consistent:** This document matches implementation (and vice versa).

**Acceptance Criteria:**
- All three notebooks execute top-to-bottom without error.
- Labels match Section 6 exactly.
- Feature names, windows, and formulas match Section 7 exactly.
- Train/val/test splits respect chronological ordering and embargo (Section 9).
- Model evaluation uses only held-out validation and test data (Section 11).

### 1.2 Notebook Workflow
The project follows a strict four-phase offline workflow:

1. **Data acquisition (`01_data_download.ipynb`):** Generate Binance public-data URLs, download ZIPs, and verify SHA256 checksums.
2. **Data preparation (`01_data_download.ipynb`):** Parse CSVs, normalize timestamps to the canonical 1-minute bar-complete index, repair rare missing bars deterministically, validate schemas/integrity, and write parquet(s).
3. **Dataset build (`02_feature_building.ipynb`):** Compute features, sample decision boundaries every `M`, construct labels, warmup-trim, create `undef__*` flags, impute deterministically, and save the model dataset + metadata.
4. **Modeling (`03_model_training.ipynb`):** Chronological split with embargo, train CatBoost with time-aware Pools, and write evaluation artifacts.

### 1.3 Validation Criteria
- Running `notebooks/01_data_download.ipynb` ? `notebooks/02_feature_building.ipynb` ? `notebooks/03_model_training.ipynb` produces all artifacts in Section 12 and passes all checkpoints in Section 8A.
- `data/model_dataset/dataset_metadata.json` records the configured constants (Appendix B) and the expected feature totals (Section 7.23).

---

## 2. Research Objective

### 2.0 Overview
**Purpose:** State the prediction task, inputs/outputs, and non-goals in one place.  
**Scope:** Offline classification for a single symbol/interval with a fixed label horizon and barrier.  
**Dependencies:** Sections 5–7 (time indexing, labels, features) and Appendix B (constants).  
**Implementation Location:** `notebooks/02_feature_building.ipynb` (label creation) and `src/utils.py::construct_labels()`.

Train and evaluate a CatBoost classifier to predict whether price will cross an upward barrier within a 10-minute horizon. The workflow is strictly offline: download data → build features/labels → train → evaluate.

**Target application:** Short-horizon probability forecasting for trading decision support.

**Non-goals:** Online inference, production systems, real-time streaming, multi-asset portfolios.

### 2.1 Validation Criteria
- The saved dataset contains `y` (binary) constructed exactly as in Section 6 with `M=10` and `phi=PHI` (Appendix B).
- `src/utils.py::checkpoint_labels()` passes on the boundary DataFrame.

---

## 3. Repository Structure

### 3.0 Overview
**Purpose:** Define the on-disk contracts for inputs/outputs and where each pipeline stage writes artifacts.  
**Scope:** Repository-relative paths only (no machine-specific absolute paths).  
**Dependencies:** Section 1 (workflow) and Section 12 (output requirements).  
**Implementation Location:** Notebooks write to these locations; utilities in `src/utils.py` enforce schemas and checkpoints.

```
/project
    /data
        /raw_data/
            /zips/                  # Downloaded ZIP files
            klines_1m.parquet       # Validated, concatenated data
            download_manifest.json  # Download tracking
            validation_report.json  # Data quality report
        /model_dataset/
            dataset.parquet         # Features + labels (see Section 12.4)
            dataset_metadata.json   # Dataset build metadata (config + derived)
            feature_list.json       # Ordered model feature names (inputs only)
            catboost_model.cbm      # Trained model
            /analytics/
                metrics.json
                calibration_by_regime.json
                threshold_analysis.csv
                best_hyperparameters.json
            /plots/
                feature_analysis.png    # Feature importance (top 30)
                *.png                   # Other evaluation plots
    /notebooks
        01_data_download.ipynb
        02_feature_building.ipynb
        03_model_training.ipynb
    /src
        utils.py                    # Helper functions (pure; see Section 14)
    requirements.txt
    README.md
```

### 3.1 Validation Criteria
- After a full run, the repository contains the expected directory structure and required artifacts under `data/` (Section 12.2).
- No stage requires machine-specific absolute paths to reproduce outputs (paths are repository-relative).

---

## 4. Binance Data Acquisition (Detailed)

### 4.0 Overview
**Purpose:** Specify the exact public-data endpoints, download/verification procedure, kline schema, timestamp normalization, and raw-data validation rules.  
**Scope:** Binance public historical data from `https://data.binance.vision/` (spot klines; derivatives sources are defined in Appendix E).  
**Dependencies:** Appendix B (SYMBOL/INTERVAL/date range) and Section 5 (time indexing assumptions).  
**Implementation Location:** `notebooks/01_data_download.ipynb`, `src/utils.py` (URL generation, checksum verification, parsing, timestamp conversion, validation).

### 4.1 Data Source

Binance provides historical kline data as monthly ZIP archives at `https://data.binance.vision/`.

**Base URL Pattern:**
```
https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YEAR}-{MONTH:02d}.zip
```

**Checksum URL Pattern:**
```
https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YEAR}-{MONTH:02d}.zip.CHECKSUM
```

**Example URLs for BTCUSDT 1m, January 2024:**
```
Data: https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip
Checksum: https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip.CHECKSUM
```

### 4.2 Download Procedure

**Step 1: Generate URL List**
```python
def generate_download_urls(symbol: str, interval: str, 
                           start_year: int, start_month: int,
                           end_year: int, end_month: int) -> list[dict]:
    """
    Generate list of download URLs for date range.
    Returns list of {year, month, data_url, checksum_url, filename}.
    """
    urls = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month):
                continue
            if (year == end_year and month > end_month):
                continue
            
            filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
            base = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}"
            
            urls.append({
                'year': year,
                'month': month,
                'data_url': f"{base}/{filename}",
                'checksum_url': f"{base}/{filename}.CHECKSUM",
                'filename': filename,
            })
    return urls
```

**Step 2: Download with Retry Logic**
```python
import requests
import time
from pathlib import Path

def download_file(url: str, output_path: Path, 
                  max_retries: int = 3, 
                  timeout: int = 60) -> bool:
    """
    Download file with retry logic.
    
    Returns True on success, False on failure.
    
    Handles:
    - Connection timeouts
    - HTTP errors (404, 500, etc.)
    - Incomplete downloads
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Write to temp file first, then rename (atomic)
            temp_path = output_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            temp_path.rename(output_path)
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return False
```

**Step 3: Verify Checksum**
```python
import hashlib

def verify_checksum(zip_path: Path, checksum_path: Path) -> bool:
    """
    Verify SHA256 checksum matches.
    
    Checksum file format: "<hash>  <filename>"
    """
    # Read expected checksum
    with open(checksum_path, 'r') as f:
        expected_hash = f.read().strip().split()[0].lower()
    
    # Compute actual checksum
    sha256 = hashlib.sha256()
    with open(zip_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest().lower()
    
    return expected_hash == actual_hash
```

**Step 4: Extract and Parse CSV**
```python
import zipfile
import pandas as pd

def extract_and_load_csv(zip_path: Path) -> pd.DataFrame:
    """
    Extract CSV from ZIP and load with correct schema.
    
    Binance kline CSVs have NO HEADER ROW.
    Column order is fixed per Binance documentation.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # ZIP contains single CSV with same name (minus .zip)
        csv_name = zip_path.stem + '.csv'
        
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f,
                header=None,  # CRITICAL: No header row
                names=[
                    'open_time', 'open', 'high', 'low', 'close',
                    'volume', 'close_time', 'quote_volume',
                    'num_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
                ],
                dtype={
                    'open_time': 'int64',
                    'open': 'float64',
                    'high': 'float64',
                    'low': 'float64',
                    'close': 'float64',
                    'volume': 'float64',
                    'close_time': 'int64',
                    'quote_volume': 'float64',
                    'num_trades': 'int64',
                    'taker_buy_base': 'float64',
                    'taker_buy_quote': 'float64',
                    'ignore': 'str',
                }
            )
    
    # Drop 'ignore' column
    df = df.drop(columns=['ignore'])
    
    return df
```

### 4.3 Kline Schema (Binance CSV Column Order)

**CRITICAL:** Binance CSVs have **no header row**. Columns are positional:

| Index | Field | Type | Description |
|-------|-------|------|-------------|
| 0 | `open_time` | int64 (ms) | Kline open timestamp (UTC milliseconds) |
| 1 | `open` | float | Open price |
| 2 | `high` | float | High price |
| 3 | `low` | float | Low price |
| 4 | `close` | float | Close price |
| 5 | `volume` | float | Base asset volume |
| 6 | `close_time` | int64 (ms) | Kline close timestamp (UTC milliseconds) |
| 7 | `quote_volume` | float | Quote asset volume |
| 8 | `num_trades` | int | Number of trades |
| 9 | `taker_buy_base` | float | Taker buy base volume |
| 10 | `taker_buy_quote` | float | Taker buy quote volume |
| 11 | `ignore` | str | Unused field |

**Reference:** [Binance Public Data GitHub](https://github.com/binance/binance-public-data)

### 4.4 Timestamp Conversion

```python
def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert millisecond timestamps to datetime index.
    
    The canonical bar timestamp is close_time + 1ms, which equals
    open_time + 60000ms for 1-minute bars. This represents the
    moment when the close price becomes known.
    """
    # Canonical timestamp: moment when bar is complete
    df['ts'] = pd.to_datetime(df['close_time'] + 1, unit='ms', utc=True)
    
    # Set as index
    df = df.set_index('ts')
    
    # Drop original timestamp columns (keep data clean)
    df = df.drop(columns=['open_time', 'close_time'])
    
    return df
```

### 4.5 Data Validation

**Expected Row Count:**
```python
def expected_bar_count(start_open: datetime, end_open_exclusive: datetime) -> int:
    """
    Calculate expected number of 1-minute bars.
    
    Convention (critical):
    - start_open is the FIRST bar open timestamp (inclusive)
    - end_open_exclusive is the FIRST bar open timestamp AFTER the end of the range (exclusive)
    
    For continuous trading (24/7 crypto), this equals the number of minutes in
    the half-open interval [start_open, end_open_exclusive).
    """
    delta = end_open_exclusive - start_open
    return int(delta.total_seconds() / 60)
```

**Validation Checks:**
```python
def validate_klines(df: pd.DataFrame) -> dict:
    """
    Comprehensive validation of kline data.
    
    Returns dict with validation results and any issues found.
    """
    results = {
        'n_rows': len(df),
        'date_range': (df.index.min(), df.index.max()),
        'issues': []
    }
    
    # 1. OHLC validity
    invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
    invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
    invalid_range = df['high'] < df['low']
    non_positive = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
    
    if invalid_high.any():
        results['issues'].append(f"High < max(O,C): {invalid_high.sum()} bars")
    if invalid_low.any():
        results['issues'].append(f"Low > min(O,C): {invalid_low.sum()} bars")
    if invalid_range.any():
        results['issues'].append(f"High < Low: {invalid_range.sum()} bars")
    if non_positive.any():
        results['issues'].append(f"Non-positive OHLC: {non_positive.sum()} bars")
    
    # 2. Volume validity
    negative_vol = df['volume'] < 0
    invalid_taker = df['taker_buy_base'] > df['volume']
    
    if negative_vol.any():
        results['issues'].append(f"Negative volume: {negative_vol.sum()} bars")
    if invalid_taker.any():
        results['issues'].append(f"Taker > Volume: {invalid_taker.sum()} bars")
    
    # 3. Timestamp checks
    dups = df.index.duplicated()
    if dups.any():
        results['issues'].append(f"Duplicate timestamps: {dups.sum()}")
    
    time_diffs = df.index.to_series().diff().dropna()
    expected_diff = pd.Timedelta(minutes=1)
    gaps = time_diffs[time_diffs != expected_diff]
    if len(gaps) > 0:
        results['issues'].append(f"Gaps detected: {len(gaps)}")
        results['gap_locations'] = gaps.head(10).to_dict()
    
    # 4. Monotonicity
    if not df.index.is_monotonic_increasing:
        results['issues'].append("Timestamps not monotonic increasing")
    
    results['is_valid'] = len(results['issues']) == 0
    
    return results
```

**Deterministic gap repair (implementation detail):** The pipeline requires a complete 1-minute grid for feature windows and label horizons. If the raw Binance monthly klines contain rare missing timestamps, `01_data_download.ipynb` reindexes to the expected UTC minute index and fills missing bars deterministically:
- `open/high/low/close` set to the previous minute’s close (flat synthetic bar)
- `volume`, `quote_volume`, `taker_buy_*`, `num_trades` set to 0

After repair, `validate_klines()` must report zero gaps. The count of filled bars is printed and can be logged alongside `validation_report.json`.

### 4.6 Common Download Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| 404 Not Found | Data not available for that month | Skip month, log warning |
| Connection timeout | Network issues | Retry with exponential backoff |
| Checksum mismatch | Corrupted download | Delete and re-download |
| Incomplete file | Connection dropped | Use streaming download with temp file |
| Rate limiting | Too many requests | Add delay between downloads (1-2 seconds) |
| Missing recent month | Data not yet published | Monthly data available ~1 week after month end |

### 4.7 Validation Criteria
- All downloaded ZIPs pass SHA256 verification against their `.CHECKSUM` files.
- The validated spot parquet has a UTC 1-minute index with no duplicates and passes `src/utils.py::checkpoint_raw_data()`.
- Any repaired gaps are explainable by the deterministic gap-repair rule (Section 4.2).

---

## 5. Time Indexing

### 5.0 Overview
**Purpose:** Define the two-cadence indexing system (1-minute bars vs 10-minute decision boundaries) and the boundary observation rule that enforces causality.  
**Scope:** Index definitions (`n`, `k`, `n_k = k*M`) and the constraint that feature snapshots only use information available at the boundary.  
**Dependencies:** Section 4 (timestamp normalization) and Appendix B (`M`).  
**Implementation Location:** `notebooks/02_feature_building.ipynb` (boundary sampling) and `src/utils.py` helpers that assume the indexing convention.

### 5.1 Two-Cadence System

| Symbol | Definition | Default |
|--------|------------|---------|
| `Δ_f` | Monitoring cadence | 1 minute |
| `M` | Decision multiplier | 10 |
| `Δ_h` | Decision cadence: `Δ_h = M × Δ_f` | 10 minutes |
| `n` | Monitoring bar index ∈ {0, 1, 2, ...} | — |
| `k` | Decision boundary index ∈ {0, 1, 2, ...} | — |
| `n_k` | Monitoring index at boundary k: `n_k = k × M` | — |

### 5.2 Boundary Observation Rule

At decision boundary `k`:
- Bar `n_k` is **fully observed** (close price known)
- Feature snapshot `x_k` uses only bars with index `≤ n_k`
- Feature snapshot `x_k` **must not** use bars with index `> n_k`

### 5.3 Validation Criteria
- The canonical timestamp convention holds: `ts = open_time + 60s` and the index advances in 1-minute steps (Section 4.4).
- Decision boundaries satisfy `ts_k - ts_{k-1} = M minutes` for >99% of rows and `k` is sequential (`src/utils.py::checkpoint_boundaries()`).

---

## 6. Label Definition

### 6.0 Overview
**Purpose:** Define the supervised target `y_k` at decision boundary `k`, including the fixed horizon `M`, barrier `phi`, and optional diagnostic quantities (`m_k`, `tau_k`).  
**Scope:** Binary barrier-crossing label on spot close prices; optional diagnostics are persisted but are never model inputs.  
**Dependencies:** Section 5 (decision boundaries) and Appendix B (`M`, `ETA`, `C`, `PHI`).  
**Implementation Location:** `src/utils.py::construct_labels()`, `notebooks/02_feature_building.ipynb` (Stage 5 label construction + checkpoints).

### 6.1 Price and Returns

```
P_n := close_n                          # Reference price (close)
p_n := ln(P_n)                          # Log price
r_n := p_n - p_{n-1} = ln(P_n / P_{n-1})  # 1-minute log return
```

### 6.2 Barrier Definition

```
φ := c + η
```

Where:
- `η > 0`: Net profit target (log-return units). Config: `ETA = 0.0002`.
- `c ≥ 0`: Round-trip cost estimate (log-return units). Config: `C = 0.0023`.

**Configuration:** `c` and `η` are constants set before label construction and stored in `data/model_dataset/dataset_metadata.json` as `C`, `ETA`, and `PHI = C + ETA = 0.0025` (Appendix B).

### 6.3 Intra-Horizon Returns

At decision boundary `k`, define returns over the **next** `M` bars:

```
r_{k,j} := ln(P_{n_k+j} / P_{n_k})    for j ∈ {1, 2, ..., M}
```

### 6.4 Maximum Future Return

```
m_k := max_{j=1..M} r_{k,j}
```

### 6.5 Binary Target

```
y_k := 1{m_k ≥ φ}
```

That is, `y_k = 1` if the maximum return over the horizon meets or exceeds the barrier.

### 6.6 First-Hit Time (Optional, for analysis)

```
τ_k := min{j ∈ {1,...,M} : r_{k,j} ≥ φ}    if y_k = 1
τ_k := NaN                                   if y_k = 0
```

**Implementation note:** If fewer than `M` future bars exist (near the end of the series), `y_k`, `m_k`, and `τ_k` are left undefined (NaN) and are dropped before saving `dataset.parquet`.

### 6.7 Causality Invariant (CRITICAL)

| Data | Available at boundary k | Used for |
|------|-------------------------|----------|
| Bars `0, 1, ..., n_k` | Yes | Features `x_k` |
| Bars `n_k+1, ..., n_{k+1}` | **No** | Label `y_k` |
| Label `y_k` | **No** (matures at k+1) | — |
| Labels `y_0, ..., y_{k-1}` | Yes | Past-target features |

**This invariant must be enforced in implementation. Violation constitutes lookahead bias.**

### 6.8 Validation Criteria
- For any boundary with sufficient future data, `y_k = 1[m_k >= phi]` and `tau_k` is in `{1, ?, M}` when `y_k=1`, otherwise `NaN`.
- The final boundary (and any boundary without a full future horizon) has `y=NaN` and is dropped during Stage 9 warmup/label trimming.
- `src/utils.py::checkpoint_labels()` passes with `phi=PHI` (Appendix B).

---

## 7. Complete Feature Definitions

### 7.0 Overview
**Purpose:** Define every engineered feature (names, formulas, windows, and undefined conditions) and the computation order constraints required for correctness.  
**Scope:** Spot features (Groups A–Q) plus boundary-level and past-target features; derivatives features are defined in Appendix E but counted here for completeness.  
**Dependencies:** Sections 4–6 (validated raw data, time indexing, label definition) and Appendix B (window sets and constants).  
**Implementation Location:** `src/utils.py` feature functions (`compute_*`) and `notebooks/02_feature_building.ipynb` (staged execution + checkpoints).

### 7.1 Notation

- `n`: Monitoring bar index
- `k`: Decision boundary index
- `n_k = k × M`: Bar index at boundary k
- `I_{k,W} = {n_k - W + 1, ..., n_k}`: Rolling window of W bars ending at n_k
- `B_k = {n_{k-1} + 1, ..., n_k}`: Decision block (M bars)

### 7.2 Base Series (Computed Per Bar)

#### 7.2.1 Price and Return Primitives

```python
P_n = close_n                                    # Reference price
p_n = np.log(P_n)                               # Log price
r_n = p_n - p_{n-1}                             # Log return (defined for n ≥ 1)
ρ_n = np.log(high_n / low_n)                    # Log range (defined for high > low)
r_oc_n = np.log(close_n / open_n)               # Open-close return
g_n = np.log(open_n / close_{n-1})              # Gap return (defined for n ≥ 1)
```

#### 7.2.2 Volume and Activity Transforms

```python
Ṽ_n = np.log1p(volume_n)                        # Log-transformed volume
Ñ_n = np.log1p(num_trades_n)                    # Log-transformed trade count
Q̃_n = np.log1p(quote_volume_n)                  # Log-transformed quote volume
```

#### 7.2.3 Taker-Buy Ratio and Order Flow Imbalance

```python
# Taker-buy ratio ∈ [0, 1] (undefined if volume = 0)
b_n = taker_buy_base_n / volume_n    if volume_n > 0 else NaN

# Order flow imbalance proxy ∈ [-1, 1]
ofi_n = 2 * b_n - 1                  if volume_n > 0 else NaN
```

#### 7.2.4 Candle Shape Primitives

```python
range_n = high_n - low_n

# Close Location Value ∈ [-1, 1] (undefined if range = 0)
clv_n = (2*close_n - high_n - low_n) / range_n    if range_n > 0 else NaN

# Body fraction ∈ [0, 1]
bodyfrac_n = abs(close_n - open_n) / range_n      if range_n > 0 else NaN

# Upper wick fraction ∈ [0, 1]
wickup_n = (high_n - max(open_n, close_n)) / range_n    if range_n > 0 else NaN

# Lower wick fraction ∈ [0, 1]
wickdn_n = (min(open_n, close_n) - low_n) / range_n     if range_n > 0 else NaN
```

#### 7.2.5 VWAP and Related

```python
# Minute VWAP (undefined if volume = 0)
vwap_n = quote_volume_n / volume_n    if volume_n > 0 else NaN

# VWAP deviation (undefined if volume = 0)
vwapdev_n = np.log(close_n / vwap_n)  if volume_n > 0 else NaN

# Quote per trade (undefined if num_trades = 0)
qpertrade_n = quote_volume_n / num_trades_n    if num_trades_n > 0 else NaN
```

### 7.3 Feature Windows and Lags

**Minute-scale windows (bars):**
```python
W_f = [
    3, 4, 5, 6, 8, 10, 12, 15,
    20, 25, 30, 35, 45, 60, 75, 90,
    120, 150, 180, 240, 300, 360, 480, 600,
    720, 960, 1440, 1920, 2880, 4320, 10080, 20160,
]
```

**Minute-scale lags:**
```python
L_f = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 26, 27, 28, 30, 32, 35,
    40, 42, 45, 50, 55, 60, 66, 75, 84, 90,
    105, 120, 150, 180, 240, 300, 360, 480, 600, 720,
    960, 1440, 2880, 4320,
]
```

**Decision-scale windows (blocks = 10 minutes each):**
```python
W_h = [2, 3, 6, 12, 24, 36, 72, 144]
```

**Short/long volatility window pairs (for regime detection):**
```python
VOL_PAIRS = [
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
]  # (short, long) in bars
```

**Canonical per-group window sets (implementation must match these exact sets):**

These are the window sets used by the feature definitions below. Keep them as distinct constants in code (see Appendix B) to prevent silently computing extra windows.

```python
# Rolling distribution stats (Group B)
WINDOWS_B = WINDOWS_F

# Quantiles / MAD (Group B+)
WINDOWS_BPLUS = [30, 45, 60, 90, 120, 180, 240, 360, 720, 1440, 2880, 4320]

# OHLC volatility estimators (Group C)
WINDOWS_VOL_OHLC = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 1920, 2880, 4320]

# Volatility decomposition (Group C+)
WINDOWS_VOL_DECOMP = [60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320]

# Barrier-aware (Group N)
WINDOWS_BARRIER = [10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320]

# Candle geometry rolling means (Group D rolling)
WINDOWS_CANDLE_ROLL = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480]

# Breakout geometry (Group D breakout)
WINDOWS_BREAKOUT = [20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 1920, 2880, 4320]

# Excursion windows (Group O drawup/drawdown)
WINDOWS_EXCURSION = [10, 20, 30, 60, 120, 240, 480, 960, 1440]

# Burstiness windows (Group O max returns)
WINDOWS_MAXRET = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]

# Trend z-score windows (Group E)
WINDOWS_LOGP_Z = [30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320]

# RSI windows (Group E)
WINDOWS_RSI = [7, 10, 14, 20, 30, 45, 60, 90, 120, 180, 240, 360]

# Enhanced liquidity (Group P)
WINDOWS_LIQ_AMIHUD = [15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
WINDOWS_LIQ_RPV = [15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
WINDOWS_OFI_IMPULSE = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360]

# Correlations (Group G)
WINDOWS_CORR = [30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]

# Permutation entropy (Group H)
WINDOWS_PENTROPY = [60, 120, 240, 720]

# Past-target hit_rate windows (Group K, in blocks)
HITRATE_WINDOWS_H = [3, 6, 12, 24, 36, 72, 144]
```

### 7.3A Rolling / Numerical Conventions (Implementation Contract)

To keep the project “low abstraction / high clarity”, these conventions are part of the spec:

- **Rolling-window requirement:** Unless explicitly stated otherwise, rolling features with window `W` use `min_periods=W` (warmup NaNs are expected and handled in Section 8).
- **Population moments:** All rolling `std` are population standard deviations (ddof=0), matching formulas of the form `(1/W) Σ(...)`.
- **No silent NaN filling:** Do **not** forward-fill/zero-fill during feature computation. Leave undefined values as `NaN`, then handle them via `undef__*` flags + imputation (Section 8).
- **Quantiles:** Use linear interpolation for quantiles (Pandas: `interpolation='linear'`; NumPy: `method='linear'`).
- **EPS:** All “+ ε”/“+ e” terms mean `EPS = 1e-10` exactly (Appendix B).

### 7.4 Group A: Lag Features

For each lag `L ∈ L_f` (Section 7.3 / Appendix B):

| Feature Name | Formula | Output Range | Undefined When |
|--------------|---------|--------------|----------------|
| `ret__lag{L}__f__w0` | `r_{n_k - L}` | ℝ | n_k - L < 1 |
| `absret__lag{L}__f__w0` | `\|r_{n_k - L}\|` | ℝ≥0 | n_k - L < 1 |
| `range__lag{L}__f__w0` | `ρ_{n_k - L}` | ℝ≥0 | H = L |
| `clv__lag{L}__f__w0` | `clv_{n_k - L}` | [-1, 1] | H = L |
| `logvol__lag{L}__f__w0` | `Ṽ_{n_k - L}` | ℝ≥0 | Never |
| `logtrades__lag{L}__f__w0` | `Ñ_{n_k - L}` | ℝ≥0 | Never |
| `ofi__lag{L}__f__w0` | `ofi_{n_k - L}` | [-1, 1] | volume = 0 |

**Total: 7 × 54 = 378 features**

### 7.5 Group B: Rolling Distribution Statistics

For each window `W ∈ W_f`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `ret__mean__f__w{W}` | `(1/W) Σ_{i∈I} r_i` | ℝ |
| `ret__std__f__w{W}` | `sqrt((1/W) Σ (r_i - μ)²)` | ℝ≥0 |
| `ret__rms__f__w{W}` | `sqrt((1/W) Σ r_i²)` | ℝ≥0 |
| `absret__mean__f__w{W}` | `(1/W) Σ \|r_i\|` | ℝ≥0 |
| `ret__posfrac__f__w{W}` | `(1/W) Σ 1{r_i > 0}` | [0, 1] |
| `range__mean__f__w{W}` | `(1/W) Σ ρ_i` | ℝ≥0 |
| `logvol__mean__f__w{W}` | `(1/W) Σ Ṽ_i` | ℝ |
| `logvol__std__f__w{W}` | Std of Ṽ | ℝ≥0 |
| `ofi__std__f__w{W}` | Std of ofi | ℝ≥0 |

**Total: 9 × 32 = 288 features**

### 7.6 Group B (Extended): Quantile and MAD Features

For `W ∈ {30, 45, 60, 90, 120, 180, 240, 360, 720, 1440, 2880, 4320}` (larger windows for stability):

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `ret__q10__f__w{W}` | 10th percentile of {r_i} | ℝ |
| `ret__q50__f__w{W}` | 50th percentile (median) | ℝ |
| `ret__q90__f__w{W}` | 90th percentile | ℝ |
| `ret__mad__f__w{W}` | `median(\|r_i - median(r)\|)` | ℝ≥0 |

**Total: 4 × 12 = 48 features**

**Computational Note:** Quantiles require sorting; for efficiency you may compute them only at decision boundaries (k) instead of for every bar, but the values must equal the per-bar rolling quantile evaluated at `n_k`. Use linear interpolation as specified in Section 7.3A.

### 7.7 Group C: OHLC Range-Based Volatility Estimators

Per-bar variance estimates:

**Parkinson (1980):**
```python
u_i = np.log(high_i / low_i)
σ²_P,i = u_i² / (4 * np.log(2))
```

**Garman-Klass (1980):**
```python
u_i = np.log(high_i / low_i)
c_i = np.log(close_i / open_i)
σ²_GK,i = 0.5 * u_i² - (2*np.log(2) - 1) * c_i²
```

**Rogers-Satchell (1991):**
```python
σ²_RS,i = np.log(high_i/open_i) * np.log(high_i/close_i) + \
          np.log(low_i/open_i) * np.log(low_i/close_i)
```

Rolling aggregate over window W:
```python
σ²_avg = (1/W) * Σ_{i∈I} σ²_*,i
σ_*,k,W = np.sqrt(np.maximum(0, σ²_avg))
```

**CAVEAT:** Garman-Klass can yield negative window averages in strong trending regimes. Always clamp to zero before sqrt.

For `W ∈ {5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 1920, 2880, 4320}`:

| Feature Name | Formula |
|--------------|---------|
| `vol__parkinson__f__w{W}` | `σ_P,k,W` |
| `vol__gk__f__w{W}` | `σ_GK,k,W` |
| `vol__rs__f__w{W}` | `σ_RS,k,W` |

**Total: 3 × 19 = 57 features**

**References:**
- Parkinson (1980): [DOI:10.1086/296071](https://doi.org/10.1086/296071)
- Garman-Klass (1980): [DOI:10.1086/296072](https://doi.org/10.1086/296072)
- Rogers-Satchell (1991): [DOI:10.1214/aoap/1177005835](https://doi.org/10.1214/aoap/1177005835)

### 7.8 Group C (Extended): Volatility Decomposition and Jump Proxies

These features capture the "quality" of volatility—diffusive vs jumpy behavior.

#### 7.8.1 Bipower Variation Ratio (Jumpiness Proxy)

Bipower variation is less sensitive to jumps than realized variance:

```python
# Per-bar absolute returns
|r_i| = abs(r_i)

# Bipower variation over window W
BPV_W = (π/2) * (1/(W-1)) * Σ_{i=2}^{W} |r_i| × |r_{i-1}|

# Realized variance over window W  
RV_W = (1/W) * Σ_{i∈I} r_i²

# Ratio (jumpiness indicator: values >> 1 suggest jump activity)
jump_ratio = RV_W / (BPV_W + ε)    # ε = 1e-10
```

For `W ∈ {60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320}`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `vol__bpv_ratio__f__w{W}` | `RV_W / (BPV_W + ε)` | ℝ≥0 |

**Total: 12 features**

#### 7.8.2 Semivariance (Directional Volatility)

```python
# Downside semivariance
SV_down,W = (1/W) * Σ_{i∈I} min(r_i, 0)²

# Upside semivariance  
SV_up,W = (1/W) * Σ_{i∈I} max(r_i, 0)²

# Asymmetry ratio
semivar_ratio = sqrt(SV_down,W) / (sqrt(SV_up,W) + ε)
```

For `W ∈ {60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320}`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `vol__semivar_down__f__w{W}` | `sqrt(SV_down,W)` | ℝ≥0 |
| `vol__semivar_up__f__w{W}` | `sqrt(SV_up,W)` | ℝ≥0 |
| `vol__semivar_ratio__f__w{W}` | `sqrt(SV_down,W) / (sqrt(SV_up,W) + ε)` | ℝ≥0 |

**Total: 3 × 12 = 36 features**

#### 7.8.3 Volatility-of-Volatility (Vol-of-Vol)

Rolling standard deviation of rolling volatility estimates:

```python
# Using RMS volatility as base estimate
σ_i = ret__rms__f__w20  # 20-bar RMS at bar i

# Vol-of-vol over window W
vov_W = std({σ_i : i ∈ I_{k,W}})
```

For `W ∈ {60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320}`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `vol__vov__f__w{W}` | `std of 20-bar RMS over W` | ℝ≥0 |

**Total: 12 features**

### 7.9 Group N: Barrier-Aware Features (NEW - Critical for Fixed φ)

These features normalize market conditions relative to the fixed barrier, capturing "signal-to-noise" for barrier crossing.

#### 7.9.1 Normalized Barrier Tightness

```python
# z^{(φ)}_{k,W} = φ / (σ_{k,W} × sqrt(M))
# Interpretation: how many "volatility units" is the barrier?
# Low z → barrier is easy; High z → barrier is hard

σ_k,W = vol__rs__f__w{W}  # Use Rogers-Satchell as base estimator
z_barrier = φ / (σ_k,W * np.sqrt(M) + ε)    # ε = 1e-10
```

For `W ∈ {10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320}`:

| Feature Name | Formula | Output Range | Undefined When |
|--------------|---------|--------------|----------------|
| `barrier__z_tight__f__w{W}` | `φ / (σ_RS,k,W × √M + ε)` | ℝ≥0 | Never (ε guards) |

**Total: 16 features**

#### 7.9.2 Expected-Max Proxy

Based on extreme value theory: for IID normals, E[max over M draws] ≈ σ√(2 ln M).

```python
# Expected max return magnitude under diffusion assumption
e_max_W = σ_k,W * np.sqrt(2 * np.log(M))

# Ratio to barrier (values > 1 suggest barrier is "reachable")
barrier_emax_ratio = e_max_W / (φ + ε)
```

For `W ∈ {10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320}`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `barrier__emax_ratio__f__w{W}` | `σ_RS × √(2 ln M) / φ` | ℝ≥0 |

**Total: 16 features**

#### 7.9.3 Short/Long Volatility Ratio (Regime Shift Detector)

```python
# vol_ratio = σ_{short} / σ_{long}
# Interpretation: ratio > 1 means vol is elevated vs recent average

vol_ratio = σ_k,W_s / (σ_k,W_l + ε)
```

For pairs `(W_s, W_l) ∈ {(10, 60), (10, 240), (20, 120), (20, 480), (30, 180), (60, 360), (60, 1440), (120, 720), (120, 2880), (240, 1440), (240, 4320), (720, 4320)}`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `vol__ratio__f__ws{W_s}__wl{W_l}` | `σ_RS,k,W_s / σ_RS,k,W_l` | ℝ≥0 |

**Total: 12 features**

### 7.10 Group D: Candle Geometry and Breakout State

**Instantaneous (at n_k):**

| Feature Name | Formula | Output Range | Undefined When |
|--------------|---------|--------------|----------------|
| `clv__inst__f__w0` | `clv_{n_k}` | [-1, 1] | H = L |
| `bodyfrac__inst__f__w0` | `bodyfrac_{n_k}` | [0, 1] | H = L |
| `wickup__inst__f__w0` | `wickup_{n_k}` | [0, 1] | H = L |
| `wickdn__inst__f__w0` | `wickdn_{n_k}` | [0, 1] | H = L |
| `gap__inst__f__w0` | `g_{n_k} = ln(O_{n_k}/C_{n_k-1})` | ℝ | n_k = 0 |

**Rolling (for W ∈ {10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480}):**

| Feature Name | Formula |
|--------------|---------|
| `clv__mean__f__w{W}` | Rolling mean of clv |

**Breakout geometry (for W ∈ {20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 1920, 2880, 4320}):**

```python
p_max = max_{i∈I} p_i
p_min = min_{i∈I} p_i
```

| Feature Name | Formula | Output Range | Undefined When |
|--------------|---------|--------------|----------------|
| `logp__pos__f__w{W}` | `(p_{n_k} - p_min) / (p_max - p_min)` | [0, 1] | p_max = p_min |
| `logp__dd__f__w{W}` | `p_max - p_{n_k}` (drawdown) | ℝ≥0 | Never |
| `logp__du__f__w{W}` | `p_{n_k} - p_min` (drawup) | ℝ≥0 | Never |

**Total: 5 + 12 + 48 = 65 features**

### 7.11 Group O: Path/Excursion Features (NEW - Label-Aligned)

These features capture recent "burstiness" and excursion patterns that align with the max-over-horizon label geometry.

#### 7.11.1 Rolling Max Drawup / Max Drawdown

```python
def compute_max_drawup(prices: np.ndarray, W: int) -> np.ndarray:
    """
    Compute maximum drawup within rolling window W.
    
    Drawup at position i within window = price[i] - running_min_up_to_i
    Max drawup = max of all drawups in window
    
    This measures the largest upward excursion from a local trough.
    """
    result = np.full(len(prices), np.nan)
    
    for i in range(W - 1, len(prices)):
        window = prices[i - W + 1 : i + 1]
        # Running minimum from start of window to each position
        running_min = np.minimum.accumulate(window)
        # Drawup at each position
        drawups = window - running_min
        result[i] = np.max(drawups)
    
    return result

def compute_max_drawdown(prices: np.ndarray, W: int) -> np.ndarray:
    """
    Compute maximum drawdown within rolling window W.
    
    Drawdown at position i within window = running_max_up_to_i - price[i]
    Max drawdown = max of all drawdowns in window
    
    This measures the largest downward excursion from a local peak.
    """
    result = np.full(len(prices), np.nan)
    
    for i in range(W - 1, len(prices)):
        window = prices[i - W + 1 : i + 1]
        # Running maximum from start of window to each position
        running_max = np.maximum.accumulate(window)
        # Drawdown at each position
        drawdowns = running_max - window
        result[i] = np.max(drawdowns)
    
    return result

# Vectorized version using stride tricks (more efficient):
def compute_max_drawup_vectorized(prices: pd.Series, W: int) -> pd.Series:
    """Vectorized implementation using rolling apply."""
    def max_drawup_window(window):
        running_min = np.minimum.accumulate(window)
        return np.max(window - running_min)
    
    return prices.rolling(W).apply(max_drawup_window, raw=True)
```

For `W ∈ {10, 20, 30, 60, 120, 240, 480, 960, 1440}`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `excursion__max_drawup__f__w{W}` | Max drawup in log-price | ℝ≥0 |
| `excursion__max_drawdown__f__w{W}` | Max drawdown in log-price | ℝ≥0 |

**Total: 2 × 9 = 18 features**

#### 7.11.2 Max Short-Term Returns (Burstiness)

```python
# Maximum 1-bar and 2-bar returns in recent window
max_1m_ret_W = max_{i∈I} r_i
max_2m_ret_W = max_{i∈I} (p_i - p_{i-2})
min_1m_ret_W = min_{i∈I} r_i
```

For `W ∈ {10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440}`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `ret__max1m__f__w{W}` | `max(r_i)` over W | ℝ |
| `ret__max2m__f__w{W}` | `max(r_{2-bar})` over W | ℝ |
| `ret__min1m__f__w{W}` | `min(r_i)` over W | ℝ |

**Total: 3 × 15 = 45 features**

#### 7.11.3 Decision-Block Excursion Features

Within the most recent decision block `B_k`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `block__maxret__h__w0` | `max_{i∈B_k} (p_i - p_{n_{k-1}})` | ℝ |
| `block__minret__h__w0` | `min_{i∈B_k} (p_i - p_{n_{k-1}})` | ℝ |
| `block__close_to_high__h__w0` | `(p_{n_k} - ln(L^h_k)) / (ln(H^h_k) - ln(L^h_k) + EPS)` | [0,1] |

**Total: 3 features**

**Note (unit consistency):** `p_{n_k} = ln(close_{n_k})` is a log-price, while `H^h_k`/`L^h_k` are block extrema in price space (Section 7.18). The definition above converts the block extrema to log space to avoid mixing units. When `H^h_k = L^h_k` (flat block), treat this feature as undefined (NaN), create an `undef__*` flag, and impute to `0.5`.

### 7.12 Group E: Trend, Momentum, Mean Reversion

**Z-score (for W ∈ {30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320}):**

```python
μ_p = (1/W) * Σ p_i
σ_p = std(p_i)
z_p = (p_{n_k} - μ_p) / σ_p    # undefined if σ_p = 0
```

| Feature Name | Formula | Undefined When |
|--------------|---------|----------------|
| `logp__z__f__w{W}` | `z_{p,k,W}` | σ_p = 0 |

**EMA Spread:**

```python
EMA^{(W)}_n = α * p_n + (1-α) * EMA^{(W)}_{n-1}    where α = 2/(W+1)
```

**Implementation detail (unambiguous):** Use the recursive definition above with `adjust=False` and initialize `EMA^{(W)}_0 = p_0` (equivalently: `p.ewm(alpha=2/(W+1), adjust=False).mean()` in Pandas).

| Feature Name | Formula |
|--------------|---------|
| `logp__ema_spread__f__w0__fast10__slow60` | `EMA^{(10)} - EMA^{(60)}` |
| `logp__ema_spread__f__w0__fast20__slow120` | `EMA^{(20)} - EMA^{(120)}` |
| `logp__ema_spread__f__w0__fast60__slow240` | `EMA^{(60)} - EMA^{(240)}` |

**RSI (Wilder smoothing, for W ∈ {7, 10, 14, 20, 30, 45, 60, 90, 120, 180, 240, 360}):**

```python
gain_n = max(r_n, 0)
loss_n = max(-r_n, 0)

# Wilder initialization (first W returns):
# avg_gain_W = (1/W) * Σ_{i=1..W} gain_i
# avg_loss_W = (1/W) * Σ_{i=1..W} loss_i
#
# Wilder recursion (n > W):
# avg_gain_n = ((W-1) * avg_gain_{n-1} + gain_n) / W
# avg_loss_n = ((W-1) * avg_loss_{n-1} + loss_n) / W
#
# RSI is undefined for n < W (warmup NaNs are handled in Section 8)
RS_n = avg_gain_n / (avg_loss_n + EPS)
RSI_n = 100 - 100 / (1 + RS_n)
```

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `ret__rsi__f__w{W}` | `RSI^{(W)}_{n_k}` | [0, 100] |

**Total: 14 + 3 + 12 = 29 features**

### 7.13 Group F: Activity, Flow, and Liquidity

**Instantaneous:**

| Feature Name | Formula | Output Range | Undefined When |
|--------------|---------|--------------|----------------|
| `tb_ratio__inst__f__w0` | `b_{n_k}` | [0, 1] | volume = 0 |
| `ofi__inst__f__w0` | `ofi_{n_k}` | [-1, 1] | volume = 0 |
| `qpertrade__inst__f__w0` | `Q_{n_k} / N_{n_k}` | ℝ≥0 | num_trades = 0 |
| `vwapdev__inst__f__w0` | `ln(C_{n_k} / vwap_{n_k})` | ℝ | volume = 0 |

**Volume Z-score (for W ∈ {60, 120, 240}):**

| Feature Name | Formula | Undefined When |
|--------------|---------|----------------|
| `logvol__z__f__w{W}` | `(Ṽ_{n_k} - μ_Ṽ) / σ_Ṽ` | σ_Ṽ = 0 |

**Liquidity proxy (for W ∈ {60, 120, 240}):**

```python
liq_{k,W} = Σ Q_i / (ε + Σ |r_i|)    # ε = 1e-10
```

| Feature Name | Formula |
|--------------|---------|
| `liq__quote_per_absret__f__w{W}` | `liq_{k,W}` |

**Total: 4 + 3 + 3 = 10 features**

### 7.14 Group P: Enhanced Liquidity Proxies (NEW)

#### 7.14.1 Amihud-Style Illiquidity

```python
# Price impact per unit volume (inverse liquidity)
illiq_W = Σ_{i∈I} |r_i| / (Σ_{i∈I} volume_i + ε)
```

For `W ∈ {15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440}`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `liq__amihud__f__w{W}` | `Σ|r| / (Σvol + ε)` | ℝ≥0 |

**Total: 12 features**

#### 7.14.2 Range Per Volume

```python
# Another illiquidity proxy
rpv_W = Σ_{i∈I} (high_i - low_i) / (Σ_{i∈I} volume_i + ε)
```

For `W ∈ {15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440}`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `liq__range_per_vol__f__w{W}` | `Σ(H-L) / (Σvol + ε)` | ℝ≥0 |

**Total: 12 features**

#### 7.14.3 OFI Impulse Features

```python
# OFI momentum and extremes
delta_ofi_W = ofi_{n_k} - ofi_{n_k - W}
max_ofi_W = max_{i∈I} ofi_i
min_ofi_W = min_{i∈I} ofi_i

# OFI × |return| interaction (flow-price impact)
ofi_ret_interaction = Σ_{i∈I} ofi_i × |r_i|
```

For `W ∈ {5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360}`:

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `ofi__delta__f__w{W}` | `ofi_{n_k} - ofi_{n_k-W}` | [-2, 2] |
| `ofi__max__f__w{W}` | `max(ofi_i)` over W | [-1, 1] |
| `ofi__min__f__w{W}` | `min(ofi_i)` over W | [-1, 1] |
| `ofi__ret_interaction__f__w{W}` | `Σ ofi × |r|` | ℝ |

**Total: 4 × 12 = 48 features**

### 7.15 Group G: Serial Dependence and Correlations

For `W ∈ {30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440}`:

| Feature Name | Formula | Undefined When |
|--------------|---------|----------------|
| `ret__acf1__f__w{W}` | `Corr({r_i}, {r_{i-1}})` | var = 0 |
| `ret__corr_logvol__f__w{W}` | `Corr({r_i}, {Ṽ_i})` | var = 0 |
| `absret__corr_logvol__f__w{W}` | `Corr({\|r_i\|}, {Ṽ_i})` | var = 0 |
| `ret__corr_ofi__f__w{W}` | `Corr({r_i}, {ofi_i})` | var = 0 |

**Total: 4 × 12 = 48 features**

**Computational Note:** Use Pearson correlation with the rolling conventions in Section 7.3A (population moments, no silent NaN filling). If either input series contains `NaN` inside a window (or variance is zero), the correlation feature is undefined (`NaN`) and is handled later via `undef__*` + imputation.

### 7.16 Group H: Permutation Entropy (Complexity)

Permutation entropy measures the complexity/predictability of a time series.

**Definition (Bandt & Pompe, 2002):**

For embedding dimension `m` and delay `τ`:
1. Extract ordinal patterns of length m from the series
2. Count frequency of each permutation pattern
3. Compute normalized entropy

```python
def permutation_entropy_normalized(x, m=3, tau=1):
    """
    Compute normalized permutation entropy ∈ [0, 1].
    
    Parameters:
    - x: time series (must have ≥ (m-1)*tau + 1 elements)
    - m: embedding dimension (pattern length)
    - tau: time delay
    
    Returns:
    - H_norm: normalized entropy (0 = deterministic, 1 = random)
    """
    n = len(x)
    n_patterns = n - (m - 1) * tau
    
    if n_patterns < 5 * math.factorial(m):
        return np.nan  # Insufficient data
    
    # Extract patterns and convert to ordinal ranks
    patterns = []
    for i in range(n_patterns):
        pattern = tuple(np.argsort([x[i + j*tau] for j in range(m)]))
        patterns.append(pattern)
    
    # Count frequencies
    from collections import Counter
    counts = Counter(patterns)
    probs = np.array(list(counts.values())) / n_patterns
    
    # Shannon entropy
    H = -np.sum(probs * np.log(probs))
    
    # Normalize by max entropy (log(m!))
    H_max = np.log(math.factorial(m))
    
    return H / H_max
```

**Tie Handling:** Break ties deterministically by time index using a stable argsort (NumPy: `np.argsort(..., kind='mergesort')`).

For `W ∈ {60, 120, 240, 720}`, `m ∈ {3}`, `τ = 1`:

| Feature Name | Formula | Undefined When |
|--------------|---------|----------------|
| `pentropy_norm__inst__f__w{W}__m3__tau1` | Normalized PE of {r_i} | n_patterns < 5×m! |

**Total: 4 features**

**Reference:** Bandt & Pompe (2002): [DOI:10.1103/PhysRevLett.88.174102](https://doi.org/10.1103/PhysRevLett.88.174102)

### 7.17 Group Q: Seasonality/Time Context (NEW)

Crypto markets exhibit strong hour-of-day and day-of-week patterns. Cyclical encoding avoids discontinuities.

```python
# Extract time components from timestamp
minute_of_day = ts.hour * 60 + ts.minute  # 0-1439
day_of_week = ts.dayofweek                 # 0-6 (Mon=0)

# Cyclical encoding
sin_minute = np.sin(2 * π * minute_of_day / 1440)
cos_minute = np.cos(2 * π * minute_of_day / 1440)
sin_dow = np.sin(2 * π * day_of_week / 7)
cos_dow = np.cos(2 * π * day_of_week / 7)
```

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `time__sin_minute__f__w0` | `sin(2π × minute_of_day / 1440)` | [-1, 1] |
| `time__cos_minute__f__w0` | `cos(2π × minute_of_day / 1440)` | [-1, 1] |
| `time__sin_dow__f__w0` | `sin(2π × day_of_week / 7)` | [-1, 1] |
| `time__cos_dow__f__w0` | `cos(2π × day_of_week / 7)` | [-1, 1] |

**Total: 4 features**

### 7.18 Group I: Decision-Block Features (10-minute scale)

Block definition: `B_k = {n_{k-1} + 1, ..., n_k}` (M = 10 bars)

**Block aggregates:**
```python
H^{(h)}_k = max_{i∈B_k} high_i
L^{(h)}_k = min_{i∈B_k} low_i
V^{(h)}_k = Σ_{i∈B_k} volume_i
Q^{(h)}_k = Σ_{i∈B_k} quote_volume_i
N^{(h)}_k = Σ_{i∈B_k} num_trades_i
V^{TB,(h)}_k = Σ_{i∈B_k} taker_buy_base_i
```

**Instantaneous block features:**

| Feature Name | Formula | Undefined When |
|--------------|---------|----------------|
| `ret__inst__h__w0` | `ln(P_{n_k} / P_{n_{k-1}})` | k = 0 |
| `range__inst__h__w0` | `ln(H^{(h)}_k / L^{(h)}_k)` | H = L |
| `logvol__inst__h__w0` | `ln(1 + V^{(h)}_k)` | Never |
| `ofi__inst__h__w0` | `2 × V^{TB,(h)}_k / V^{(h)}_k - 1` | V^{(h)} = 0 |

**Rolling block stats (for W ∈ {2, 3, 6, 12, 24, 36, 72, 144} blocks):**

| Feature Name | Formula |
|--------------|---------|
| `ret__std__h__w{W}` | Std of {r^{(h)}_i} over W blocks |

**Total: 4 + 8 = 12 features**

### 7.19 Group J: Event-Based Features

**Return-Sign Run Features:**

Track consecutive bars with same sign of return.

```python
# Run tracking state
run_direction = 0      # -1, 0, or 1
run_length = 0
run_cumulative_return = 0.0

# Update per bar:
sign_n = np.sign(r_n) if r_n != 0 else 0

if sign_n == 0:
    # Zero return breaks the run
    run_direction = 0
    run_length = 0
    run_cumulative_return = 0.0
elif sign_n == run_direction:
    # Continue run
    run_length += 1
    run_cumulative_return += r_n
else:
    # New run starts
    run_direction = sign_n
    run_length = 1
    run_cumulative_return = r_n
```

| Feature Name | Formula | Output Range |
|--------------|---------|--------------|
| `event__run_dir__f__w0` | Current run direction | {-1, 0, 1} |
| `event__run_len__f__w0` | Current run length | ℕ |
| `event__run_cumret__f__w0` | Cumulative return in run | ℝ |

**Total: 3 features**

### 7.20 Group K: Past-Target Statistics (Matured Labels Only)

**CRITICAL CAUSALITY RULE:** At boundary k, only labels `{y_i : i ≤ k-1}` are available (matured).

| Feature Name | Formula | Undefined When |
|--------------|---------|----------------|
| `hit__prev__h__w0` | `y_{k-1}` | k = 0 |
| `hit__rate__h__w{W}` | `(1/W) Σ_{i=1}^{W} y_{k-i}` | k < W |
| `hit__since__h__w0` | `min{j≥1 : y_{k-j} = 1}` | No hit in history |

Windows for hit_rate: `W ∈ {3, 6, 12, 24, 36, 72, 144}`

**Total: 1 + 7 + 1 = 9 features**

### 7.21 Group L: Cost/Barrier Context

| Feature Name | Formula | Notes |
|--------------|---------|-------|
| `cost__c__h__w0` | `c` | Constant (same as label) |
| `barrier__phi__h__w0` | `φ = c + η` | Constant (same as label) |

**Total: 2 features**

### 7.22 Group M: Data Quality Guards

| Feature Name | Formula | Output |
|--------------|---------|--------|
| `data__bad_ohlc__f__w0` | `1 if invalid OHLC else 0` | {0, 1} |
| `data__gap__f__w0` | `1 if timestamp gap else 0` | {0, 1} |

**Total: 2 features**

### 7.23 Feature Count Summary

| Group | Description | Count |
|-------|-------------|-------|
| A | Lag features | 378 |
| B | Rolling statistics | 288 |
| B+ | Quantiles and MAD | 48 |
| C | Volatility estimators (OHLC) | 57 |
| C+ | Volatility decomposition (bipower, semivar, vov) | 60 |
| N | **Barrier-aware features (NEW)** | 44 |
| D | Candle geometry + breakout | 65 |
| O | **Path/excursion features (NEW)** | 66 |
| E | Trend/momentum | 29 |
| F | Activity/flow/liquidity | 10 |
| P | **Enhanced liquidity proxies (NEW)** | 72 |
| G | Correlations | 48 |
| H | Permutation entropy | 4 |
| Q | **Seasonality/time context (NEW)** | 4 |
| I | Decision-block | 12 |
| J | Event-based | 3 |
| K | Past-target | 9 |
| L | Cost/barrier | 2 |
| M | Data quality | 2 |
| **TOTAL (spot features)** | | **1201** |
| R | Derivatives basis (Appendix E) | 8 |
| S | Derivatives volume/flow/liquidity (Appendix E) | 9 |
| T | Derivatives open interest + funding (Appendix E) | 8 |
| U | Options sentiment (Appendix E) | 5 |
| V | Implied vol + risk premium (Appendix E) | 5 |
| **TOTAL (engineered features)** | | **1236** |

Undefined flags (`undef__*`): data-dependent (primarily derivatives coverage + rare denom=0). Total model inputs (`feature_list.json`): expected in **[1200, 1700]**.

### 7.24 Validation Criteria
- `data/model_dataset/feature_list.json` matches the engineered feature columns produced by Notebook 02 (excluding labels/weights), and its length is within the expected range (Section 7.23).
- Rolling conventions match Section 7.3A (`min_periods=W`, `ddof=0`), otherwise feature distributions and counts will diverge.

---

## 7A. Feature Computation Order and Dependencies

### 7A.1 Why Order Matters
Feature computation is not commutative: rolling windows/lags require history (warmup), some feature groups depend on other derived series (e.g., barrier-aware features depend on volatility estimates), and past-target features must use only matured labels (shifted) to avoid leakage.

### 7A.2 Notebook 02 Execution Order (Authoritative)
The stage order in `notebooks/02_feature_building.ipynb` is:

| Stage | Grid | What happens | Key implementation |
|-------|------|--------------|-------------------|
| 0 | 1m | Load `df_raw` from `data/raw_data/klines_1m.parquet` | Notebook 02 Stage 0 |
| 1 | 1m | Base series (`p`, `r`, `rho`, ?) | `src/utils.py::compute_base_series()` |
| 2 | 1m | Tier?1 spot features (Groups A,B,B+,C,D,E,F,G,H,J,M,Q) | `src/utils.py::compute_*()` |
| 2.5 | 1m | Optional derivatives join + features (Appendix E) | `src/utils.py::compute_derivatives_base_series()` + `src/utils.py::compute_*_features()` |
| 3 | 1m | Tier?2 spot features (Groups C+,O,P) | `src/utils.py::compute_volatility_decomposition()`, `src/utils.py::compute_excursion_features()`, `src/utils.py::compute_enhanced_liquidity()` |
| 4 | 10m | Boundary sampling (`df.iloc[::M]`), add `ts` and sequential `k` | Notebook 02 Stage 4 + `src/utils.py::checkpoint_boundaries()` |
| 5 | 10m | Labels from future spot bars (Section 6) | `src/utils.py::construct_labels()` + `src/utils.py::checkpoint_labels()` |
| 6 | 10m | Past?target features from shifted labels (Group K) | `src/utils.py::compute_past_target_features()` |
| 7 | 10m | Barrier-aware features (Group N) | `src/utils.py::compute_barrier_aware_features()` |
| 8 | 10m | Block features (Group I) | `src/utils.py::compute_block_features()` |
| 9 | 10m | Warmup trim + drop `y` NaNs | Notebook 02 Stage 9 + `src/utils.py::checkpoint_warmup_trimmed()` |
| 10 | 10m | `undef__*` + imputation | `src/utils.py::create_undef_flags_and_impute()` + `src/utils.py::checkpoint_final_dataset()` |
| W | 10m | Optional observation weights | `src/utils.py::compute_training_weights()` (calls `compute_barrier_distance_weight()` + `compute_time_discount_weight()`) |
| 11 | n/a | Save dataset + metadata + feature list | Notebook 02 Stage 11 |

### 7A.3 Warmup Period Explained (Authoritative)
Warmup is defined on the 1-minute grid as `N_WARMUP = max(max(WINDOWS_F)-1, max(LAGS_F), M*max(WINDOWS_H))` and enforced on decision boundaries as `K_WARMUP = ceil(N_WARMUP / M) = (N_WARMUP + M - 1) // M`.

With the default constants (Appendix B): `N_WARMUP = 20159` minutes and `K_WARMUP = 2016` boundaries.

### 7A.4 Critical Implementation Notes
- Past-target features must not use `y_k` at boundary `k`; only `y_{<k}` is available by construction.
- Rolling statistics must use `min_periods=W` and `ddof=0` (Section 7.3A).
- After boundary sampling, raw/base/derivatives-join columns are dropped; only engineered features remain in `feature_list.json` (Section 12.4).
- Warmup trimming occurs before `undef__*` generation so flags represent structural undefinedness beyond warmup, not early-window NaNs.
- `vol__rs__f__w1440` is computed as an intermediate for `vol__ratio__f__w20_1440` and is dropped before saving.

## 8. Missing Value Handling

### 8.0 Overview
**Purpose:** Define missingness categories, the `undef__*` flag contract, deterministic imputation rules, and the required post-imputation invariants.  
**Scope:** Model-input columns only; label diagnostics (`m_k`, `tau_k`, `phi`) and weight columns are excluded from NaN constraints.  
**Dependencies:** Section 7 (feature definitions), Appendix B (constants), Appendix E (derivatives coverage rules).  
**Implementation Location:** `src/utils.py::get_imputation_value()`, `src/utils.py::create_undef_flags_and_impute()`, and `notebooks/02_feature_building.ipynb` (Stage 10 imputation).

### 8.1 Categories of Missingness

1. **Warmup NaNs:** Insufficient history for rolling window W
2. **Structural undefined:** Division by zero (volume=0, range=0), log of non-positive, zero variance
3. **Event-absent:** No event has occurred yet (e.g., no previous hit)
4. **Lag boundary:** Requesting data before series start

### 8.2 Undefined Flags

For each feature that can be structurally undefined, create a companion binary flag:

```python
undef__{feature_name} = 1 if feature_value is NaN else 0
```

**The flag is a model input.** After creating the flag, impute the NaN value.

### 8.3 Imputation Rules (Complete Table)

| Feature Pattern | Impute Value | Rationale |
|-----------------|--------------|-----------|
| **Returns & Movement** | | |
| `ret__*`, `absret__*` | 0.0 | Neutral drift/no movement |
| `gap__*` | 0.0 | No gap |
| `range__*` | 0.0 | Zero range (flat candle) |
| `ret__max1m*`, `ret__max2m*` | 0.0 | No extreme returns |
| `ret__min1m*` | 0.0 | No extreme returns |
| **Volatility** | | |
| `vol__parkinson*`, `vol__gk*`, `vol__rs*` | 0.0 | No volatility |
| `vol__bpv_ratio*` | 1.0 | No jumps (BPV ≈ RV) |
| `vol__semivar_down*`, `vol__semivar_up*` | 0.0 | No volatility |
| `vol__semivar_ratio*` | 1.0 | Symmetric volatility |
| `vol__vov*` | 0.0 | No vol-of-vol |
| `vol__ratio*` | 1.0 | No regime shift |
| `ret__std*`, `ret__rms*`, `ret__mad*` | 0.0 | No dispersion |
| **Barrier-Aware** | | |
| `barrier__z_tight*` | 10.0 | Very tight barrier (conservative) |
| `barrier__emax_ratio*` | 0.0 | Expected max << barrier |
| **Distributional** | | |
| `ret__posfrac*` | 0.5 | Symmetric baseline |
| `ret__q10*`, `ret__q50*`, `ret__q90*` | 0.0 | Neutral quantiles |
| **Position & Breakout** | | |
| `logp__pos*` | 0.5 | Mid-channel |
| `logp__dd*`, `logp__du*` | 0.0 | No drawdown/drawup |
| `logp__z*` | 0.0 | At-mean |
| `logp__ema_spread*` | 0.0 | No trend spread |
| **Excursion** | | |
| `excursion__max_drawup*` | 0.0 | No excursion |
| `excursion__max_drawdown*` | 0.0 | No excursion |
| `block__maxret*`, `block__minret*` | 0.0 | Neutral block |
| `block__close_to_high*` | 0.5 | Mid-position |
| **Momentum** | | |
| `ret__rsi*` | 50.0 | Neutral RSI |
| **Candle Shape** | | |
| `clv__*` | 0.0 | Neutral close location |
| `bodyfrac__*`, `wickup__*`, `wickdn__*` | 0.0 | Zero-range candle |
| **Volume & Activity** | | |
| `logvol__*`, `logtrades__*` | 0.0 | No activity (log(1+0)=0) |
| `logvol__z*` | 0.0 | Volume at-mean |
| `tb_ratio__*` | 0.5 | Balanced flow |
| **Order Flow** | | |
| `ofi__inst*`, `ofi__std*` | 0.0 | No imbalance |
| `ofi__delta*` | 0.0 | No OFI change |
| `ofi__max*`, `ofi__min*` | 0.0 | No extreme OFI |
| `ofi__ret_interaction*` | 0.0 | No flow-price interaction |
| **Liquidity** | | |
| `qpertrade__*` | 0.0 | No trades |
| `vwapdev__*` | 0.0 | No deviation |
| `liq__quote_per_absret*` | 0.0 | No liquidity info |
| `liq__amihud*` | 0.0 | No illiquidity signal |
| `liq__range_per_vol*` | 0.0 | No illiquidity signal |
| **Correlations** | | |
| `ret__acf1*`, `*__corr*` | 0.0 | No correlation |
| **Complexity** | | |
| `pentropy_norm__*` | 0.5 | Unknown complexity (mid-entropy) |
| **Seasonality** | | |
| `time__sin*`, `time__cos*` | 0.0 | Neutral time encoding |
| **Past Targets** | | |
| `hit__prev*` | 0 | No previous hit |
| `hit__rate*` | `p_hit_prior` | Training set hit rate |
| `hit__since*` | `cap_h_blocks` (144) | "At least cap since last hit" |
| **Events** | | |
| `event__*_dir*` | 0 | No direction |
| `event__*_len*` | 0 | No run |
| `event__*_cumret*` | 0.0 | No cumulative return |
| **Cost/Barrier Constants** | | |
| `cost__c*`, `barrier__phi*` | **DO NOT IMPUTE** | Must match label; missing = config error |

**Important (avoids a common implementation pitfall):** In the default pipeline, `compute_past_target_features()` (Stage 6) is executed **before** warmup trimming (Stage 9), and `K_WARMUP = max(W_h) = 144` boundaries. Therefore `hit__rate*` has enough history and should not be `NaN` in `df_final`. The `p_hit_prior` fallback exists only to keep the imputation logic complete if you change ordering/windows or trim earlier.

**Imputation Lookup Function:**

```python
def get_imputation_value(
    feature_name: str,
    p_hit_prior: float = 0.5,
    cap_h_blocks: int = 144,
) -> float:
    """
    Get imputation value for a feature based on its name pattern.
    
    Order matters: more specific patterns should come first.
    """
    patterns = [
        # Constants - should never be imputed
        (r'^cost__c', None),  # Raise error
        (r'^barrier__phi', None),  # Raise error
        
        # Barrier-aware
        (r'^barrier__z_tight', 10.0),
        (r'^barrier__emax_ratio', 0.0),
        
        # Volatility ratios
        (r'^vol__bpv_ratio', 1.0),
        (r'^vol__semivar_ratio', 1.0),
        (r'^vol__ratio', 1.0),
        
        # Volatility levels
        (r'^vol__', 0.0),
        
        # RSI
        (r'^ret__rsi', 50.0),
        
        # Fractions and ratios with 0.5 neutral
        (r'^ret__posfrac', 0.5),
        (r'^logp__pos', 0.5),
        (r'^tb_ratio', 0.5),
        (r'^block__close_to_high', 0.5),
        (r'^pentropy_norm', 0.5),
        
        # Hit rate uses prior
        (r'^hit__rate', p_hit_prior),
        (r'^hit__since', cap_h_blocks),
        (r'^hit__prev', 0),
        
        # Everything else defaults to 0.0
        (r'.*', 0.0),
    ]
    
    import re
    for pattern, value in patterns:
        if re.match(pattern, feature_name):
            if value is None:
                raise ValueError(f"Feature {feature_name} should never be NaN")
            return value
    
    return 0.0  # Fallback
```

### 8.4 Warmup Period Calculation

```python
W_f_max = max(W_f)  # 1440
L_f_max = max(L_f)  # 20
W_h_max = max(W_h)  # 144 blocks
M = 10

n_warmup = max(
    W_f_max - 1,           # 1439 (for rolling windows)
    L_f_max,               # 20 (for lags)
    M * W_h_max,           # 1440 (for block-level rolling)
)
# n_warmup = 1440

k_warmup = ceil(n_warmup / M)  # ceil(1440 / 10) = 144
```

**Action:** Drop all rows with `k < k_warmup = 144`.

### 8.5 Implementation Order

1. Compute all features (NaNs will appear)
2. Sample at decision boundaries
3. Drop warmup rows (k < k_warmup)
4. For each NaN-capable feature:
   - Create `undef__{name}` flag
   - Apply imputation value
5. **Assert: zero NaNs remain in feature columns**

### 8.6 Global Constants

Define once at module level:

```python
EPS = 1e-10  # Guard for division by zero, used in all ratio computations
```

### 8.7 Validation Criteria
- After Stage 10, no feature column contains `NaN` or `inf` and `src/utils.py::checkpoint_final_dataset()` passes.
- Any feature that was `NaN` prior to imputation has a corresponding `undef__{feature}` flag and the flags are included in `feature_list.json`.

---

## 8A. Pipeline Validation Checkpoints

### 8A.0 Overview
**Purpose:** Define the runtime checkpoint functions used in the notebooks to catch schema/time/leakage violations early.  
**Scope:** Notebook-executed sanity checks; not unit tests.  
**Dependencies:** Sections 4?10 and Appendix B (and Appendix E when derivatives are enabled).  
**Implementation Location:** `src/utils.py` (`checkpoint_*`) and calls in the notebooks.

### 8A.1 Raw Spot Data (`checkpoint_raw_data`)
Executed in `notebooks/01_data_download.ipynb` after timestamp normalization and gap repair.
- Index is a UTC `DatetimeIndex`, monotonic increasing, with no duplicates.
- Start/end timestamps match the configured month range within ?1 minute.
- Row count matches expected minutes within ?1%.
- OHLC is internally consistent (`high >= max(open, close)`, `low <= min(open, close)`, `high >= low`).
- No negative `volume`.

### 8A.2 Base Series (`checkpoint_base_series`)
Executed in `notebooks/02_feature_building.ipynb` Stage 1.
- Required columns exist: `{p, r, rho, ofi, clv, logvol, b}`.
- `r[0]` is `NaN`.
- `p == log(close)` at a sampled index.
- `ofi` is bounded in `[-1, 1]` where defined.

### 8A.3 Derivatives (Optional)
- `checkpoint_derivatives_data(...)` (Notebook 01): futures index aligned to spot; futures/spot close correlation `> 0.99`; funding rates within `[-0.1, 0.1]`; `oi_usd` non-negative; BVOL in `[10, 300]` for >95% of aligned minutes (NaNs count as failures, so this check may WARN during pre-coverage months); this checkpoint is non-fatal (prints `OK`/`WARN`).
- `checkpoint_derivatives_features(df)` (Notebook 02 Stage 2.5): expected derivatives feature columns exist; `basis__pct__f__w0` mostly within `[-5, 5]`; `opt_pcr__oi__f__w0` mostly within `[0.1, 10]`.

### 8A.4 Boundaries, Labels, and Final Dataset
- `checkpoint_boundaries(df_boundaries, df_raw, M)` (Notebook 02 Stage 4): expected boundary count `((len(df_raw)-1)//M)+1`; sequential `k`; `ts` exists; boundary spacing equals `M` minutes for >99% of rows.
- `checkpoint_labels(df, phi)` (Notebook 02 Stage 5): `y` is binary; base-rate warning outside `[0.05, 0.70]`; `m_k` sanity bounds `(-0.5, 0.5)`; label consistency `(m_k >= phi) == (y == 1)`.
- `checkpoint_warmup_trimmed(df_final, K_WARMUP)` (Notebook 02 Stage 9): `min(k) >= K_WARMUP`; no `NaN` labels; at least 10,000 rows remain.
- `checkpoint_final_dataset(df_final, feature_cols)` (Notebook 02 Stage 10): no NaNs/Infs in `feature_cols`; `len(feature_cols)` within `[1200, 1700]`; `{m_k, tau_k, phi}` and `{w_dist, w_time, weight}` are not present in `feature_cols`; `ts` monotonic; `k` sequential.

### 8A.5 Training-Time Checks
Executed in `notebooks/03_model_training.ipynb` after splitting.
- `checkpoint_before_training(...)`: splits are chronological; embargo gaps `>= EMBARGO_K`; split sizes match configured fractions when accounting for embargo; no NaNs in feature columns after excluding `{k, ts, y, m_k, tau_k, phi, w_dist, w_time, weight}`.
- `checkpoint_weights(...)` (Notebook 02 Stage W, if enabled): weights are strictly positive; `weight == w_dist * w_time`; barrier-distance weights respect configured caps; time-discount weights lie in `(0, 1]`.

## 9. Train/Validation/Test Split

### 9.0 Overview
**Purpose:** Define how chronological splits and embargo are applied, including walk-forward CV behavior used for (optional) HPO.  
**Scope:** Splitting happens only in `03_model_training.ipynb` on the fully built dataset; no random shuffling is permitted.  
**Dependencies:** Sections 5–6 (horizon timing), Section 12.4 (feature column contract), Appendix B (TRAIN_FRAC, VAL_FRAC, EMBARGO_K, N_CV_FOLDS).  
**Implementation Location:** `src/utils.py::chronological_split_with_embargo()`, `src/utils.py::walk_forward_cv()`, `notebooks/03_model_training.ipynb`.

**IMPORTANT:** The train/val/test split is performed in `03_model_training.ipynb`, NOT in `02_feature_building.ipynb`.

**Why:**
- The complete dataset is saved as one file (`dataset.parquet`)
- This allows flexibility to experiment with different split ratios
- Feature engineering is decoupled from model training
- Makes it easier to do walk-forward CV with different fold configurations

**Workflow:**
```
02_feature_building.ipynb:
  → Saves: data/model_dataset/dataset.parquet (COMPLETE dataset)
  
03_model_training.ipynb:
  → Loads: dataset.parquet
  → Performs chronological split with embargo
  → Trains on train, tunes on val, evaluates on test
```

**CRITICAL (feature/label column contract):** In `03_model_training.ipynb`, build the model matrix `X` using `data/model_dataset/feature_list.json` exactly. Do **not** infer features as “all columns except `y`”, because `dataset.parquet` may include label diagnostics such as `m_k`/`tau_k` (future-dependent) and identifiers such as `ts`/`k`.

### 9.1 Chronological Requirement

Splits are **strictly chronological**: train < validation < test by timestamp.

```
Time ─────────────────────────────────────────────────────────────────►
      │◄──────── Train ────────►│E│◄──── Val ────►│E│◄──── Test ────►│
```

Where `E` = embargo period.

### 9.2 Embargo Rule (Purging)

Apply an embargo of **`EMBARGO_K` decision intervals** between adjacent splits (Appendix B default: `EMBARGO_K = 60`).

**Why:** Label `y_k` depends on future bars `{n_k+1, ..., n_k+M}`. An embargo prevents adjacent splits from sharing or closely bordering the same horizon-dependent information and reduces leakage risk from overlapping horizon windows and strong short-range autocorrelation.

**Reference:** López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 7.

### 9.3 Split Ratios

| Split | Fraction | Purpose |
|-------|----------|---------|
| Train | 70% | Model fitting |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final evaluation (NEVER peek during development) |

### 9.4 Implementation

```python
def chronological_split_with_embargo(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo_k: int = 1  # Decision intervals
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically with embargo between splits.
    
    Args:
        df: DataFrame sorted by time, with 'k' column
        train_frac: Fraction for training
        val_frac: Fraction for validation
        embargo_k: Number of decision intervals to skip
    
    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    
    train_end_idx = int(train_frac * n)
    val_end_idx = int((train_frac + val_frac) * n)
    
    train_df = df.iloc[:train_end_idx]
    val_df = df.iloc[train_end_idx + embargo_k : val_end_idx]
    test_df = df.iloc[val_end_idx + embargo_k :]
    
    # Verify no overlap
    assert train_df['k'].max() < val_df['k'].min()
    assert val_df['k'].max() < test_df['k'].min()
    
    return train_df, val_df, test_df
```

### 9.5 Walk-Forward Validation for Hyperparameter Tuning (NEW)

To improve confidence that hyperparameters generalize across time:

```python
def walk_forward_cv(
    df: pd.DataFrame,
    n_folds: int = 3,
    embargo_k: int = 1
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/val splits for hyperparameter tuning.
    
    Returns list of (train, val) pairs where each fold is chronologically
    later than the previous.
    """
    n = len(df)
    fold_size = n // (n_folds + 1)
    
    folds = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        val_start = train_end + embargo_k
        val_end = min(val_start + fold_size, n)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]
        
        if len(val_df) > 0:
            folds.append((train_df, val_df))
    
    return folds
```

**Usage:** Average validation metrics across folds; a hyperparameter configuration is selected only if it performs well across multiple time periods.

### 9.6 Validation Criteria
- Splits are chronological and separated by at least `EMBARGO_K` decision steps; `src/utils.py::checkpoint_before_training()` passes.
- Split sizes match configured fractions when accounting for embargo (Section 9.4).

---

## 10. CatBoost Configuration

### 10.0 Overview
**Purpose:** Specify the exact CatBoost training configuration, including ordered/time-aware training, optional Optuna HPO, and how sample weights are handled.  
**Scope:** CPU CatBoostClassifier configuration used by `03_model_training.ipynb`; this is not a generic CatBoost tutorial.  
**Dependencies:** Appendix B (CB_FIXED_PARAMS, seeds, HPO toggles) and Section 9 (split discipline).  
**Implementation Location:** `notebooks/03_model_training.ipynb`, `src/utils.py` (CB_FIXED_PARAMS, split helpers, metrics).

### 10.1 Base Configuration

```python
from catboost import CatBoostClassifier, Pool
from src import utils
import numpy as np

# Fixed CatBoost params live in utils.CB_FIXED_PARAMS (Appendix B).
FIXED_PARAMS = dict(utils.CB_FIXED_PARAMS)

# The training notebook merges FIXED_PARAMS with best hyperparameters
# (either Optuna-selected or a pinned fallback) and trains with time-aware Pools.

def make_pool(df, feature_list, *, weight_col: str | None = "weight") -> Pool:
    X = df[feature_list].to_numpy()
    y = df["y"].astype(int).to_numpy()
    ts = df["k"].to_numpy(dtype=np.uint32)
    w = None
    if weight_col is not None and weight_col in df.columns:
        w = df[weight_col].to_numpy(dtype=float)
    return Pool(data=X, label=y, timestamp=ts, weight=w, feature_names=feature_list)

# Example (see 03_model_training.ipynb for full pipeline)
train_pool = make_pool(train_df, feature_list)
val_pool = make_pool(val_df, feature_list)

best_params = utils.load_json("data/model_dataset/analytics/best_hyperparameters.json")
final_params = {**FIXED_PARAMS, **best_params, "verbose": 100}

model = CatBoostClassifier(**final_params)
model.fit(train_pool, eval_set=val_pool, plot=False)
```

**Implementation Note (discrepancy to review):** `notebooks/03_model_training.ipynb` currently reassigns `best_params` to a hard-coded dictionary in the final training cell *after* the `RUN_HPO`/file-load branch, which overrides both Optuna output and `data/model_dataset/analytics/best_hyperparameters.json`. This effectively pins hyperparameters unless that line is removed.


### 10.2 Hyperparameter Search Space

Hyperparameter optimization is optional and is controlled by `ENABLE_HPO` (Appendix B). When enabled, `03_model_training.ipynb` runs an Optuna NSGA-II multi-objective search over a walk-forward CV procedure on the training split:

- Objectives: minimize mean LogLoss, maximize mean PR-AUC (implemented as minimizing negative PR-AUC).
- CV folds: `walk_forward_cv(train_df, n_folds=N_CV_FOLDS, embargo_k=EMBARGO_K)`.
- Optional speed knob: `HPO_DROP_OLDEST_FRAC` drops the oldest fraction of the training split for HPO only.

**Search space (as implemented in `03_model_training.ipynb`):**
- `learning_rate`: log-uniform in `[0.001, 0.015]`
- `l2_leaf_reg`: log-uniform in `[0.1, 5.0]`
- `depth`: integer in `[5, 10]`
- `rsm`: uniform in `[0.65, 1.0]`
- `subsample`: uniform in `[0.7, 1.0]`
- `mvs_reg`: uniform in `[1.0, 10.0]`
- `diffusion_temperature`: log-uniform in `[5000, 15000]` (only meaningful when `langevin=True`)

**Outputs (when enabled):** `analytics/best_hyperparameters.json` plus Optuna plots under `data/model_dataset/plots/` (e.g., `pareto_frontier.png`, `hp_importance_logloss.png`, `hp_importance_prauc.png`).

### 10.3 Training with Validation Set

```python
# Training uses time-aware Pools (Ordered boosting + timestamps) and early stopping.
model.fit(train_pool, eval_set=val_pool, verbose=100, plot=False)
print("Best iteration:", model.get_best_iteration())
```

### 10.4 Class Imbalance Handling

This project primarily addresses imbalance via (optional) **sample weights** computed during dataset construction (`w_dist`, `w_time`, and combined `weight`). By default, the CatBoost Pool is built with `weight_col='weight'` (see `03_model_training.ipynb`).

If you disable sample weights, you may experiment with CatBoost class weighting (`auto_class_weights` or `scale_pos_weight`), but this is not the default implementation. For barrier-crossing, the base rate varies by volatility regime; barrier-aware features (Group N) provide the model with regime context directly.

### 10.5 Validation Criteria
- Training uses time-aware Pools with `timestamp=k` and produces `data/model_dataset/catboost_model.cbm` plus required metrics/plots (Section 12).
- If HPO is enabled, `data/model_dataset/analytics/best_hyperparameters.json` is written and (intended behavior) used for final training; see the implementation note in Section 10.1.

---

## 11. Evaluation Metrics

### 11.0 Overview
**Purpose:** Define the metrics and plots required to evaluate probability quality, discrimination, and calibration on held-out data.  
**Scope:** Metrics computed on validation and test splits only; regime stratification uses a fixed, feature-derived signal (not model-derived).  
**Dependencies:** Section 9 (splits) and Section 12 (required output files).  
**Implementation Location:** `src/utils.py::compute_all_metrics()`, `src/utils.py::expected_calibration_error()`, `src/utils.py::calibration_by_regime()`, `src/utils.py::threshold_analysis()`, `notebooks/03_model_training.ipynb`.

### 11.1 Probability Quality (Proper Scoring Rules)

**Log Loss:**
```python
from sklearn.metrics import log_loss
logloss = log_loss(y_true, y_pred_proba)
```

**Brier Score:**
```python
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_true, y_pred_proba)
```

### 11.2 Discrimination (Classification Quality)

**ROC-AUC:**
```python
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_true, y_pred_proba)
```

**PR-AUC (Precision-Recall):**
```python
from sklearn.metrics import average_precision_score
pr_auc = average_precision_score(y_true, y_pred_proba)
```

### 11.3 Calibration

**Expected Calibration Error (ECE):**
```python
def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    """
    Compute ECE: weighted average of |accuracy - confidence| per bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        
        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_pred_proba[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece
```

### 11.4 Regime-Stratified Calibration (NEW)

Report calibration separately for volatility regimes:

**Canonical regime signal (unambiguous):** Use `vol_series = vol__rs__f__w240` evaluated at decision boundaries (same rows as `y_true`/`y_pred_proba`). This keeps the regime definition fixed across experiments and prevents accidental use of a model-derived quantity.

```python
def calibration_by_regime(y_true, y_pred_proba, vol_series, n_bins=10):
    """
    Compute ECE and Brier score stratified by volatility terciles.
    """
    vol_terciles = pd.qcut(vol_series, 3, labels=['low', 'med', 'high'])
    
    results = {}
    for regime in ['low', 'med', 'high']:
        mask = vol_terciles == regime
        if mask.sum() < 50:
            continue
        
        results[regime] = {
            'n_samples': mask.sum(),
            'base_rate': y_true[mask].mean(),
            'ece': expected_calibration_error(y_true[mask], y_pred_proba[mask], n_bins),
            'brier': brier_score_loss(y_true[mask], y_pred_proba[mask]),
        }
    
    return results
```

### 11.5 Required Metrics Summary

| Metric | Target | Notes |
|--------|--------|-------|
| ROC-AUC | > 0.55 | Basic discrimination |
| PR-AUC | > base rate | Handles imbalance |
| Log Loss | < null model | Proper scoring |
| Brier Score | < 0.25 | Probability accuracy |
| ECE | < 0.05 | Calibration quality |
| ECE by regime | Report all | Regime stability |

### 11.6 Validation Criteria
- `metrics.json` contains all required scalar metrics for validation and test, and regime-stratified calibration reports all three terciles.
- Calibration plots/curves are computed strictly on held-out splits only (no training leakage).

---

## 12. Output Requirements

### 12.0 Overview
**Purpose:** Define the required saved artifacts and plots that constitute a complete, verifiable run.  
**Scope:** Output file contracts and minimal schemas; does not prescribe plotting aesthetics beyond filenames and content.  
**Dependencies:** Sections 4, 8A, 9–11 (validated inputs, checkpoints, splits, metrics).  
**Implementation Location:** Notebooks under `notebooks/` and helper writers in `src/utils.py`.

### 12.1 Plots

1. **ROC Curve** with AUC annotation
2. **Precision-Recall Curve** with AP annotation
3. **Calibration Curve** (reliability diagram) with ECE annotation
4. **Calibration by Volatility Regime** (3 subplots)
5. **Feature Importance** (top 30 features, horizontal bar)
6. **Feature Importance (PredictionValuesChange)** (top-N, horizontal bar)
7. **Learning Curves** (train/validation loss vs iterations)
8. **Prediction Distribution** (histogram of p̂ by class)
9. **Threshold Analysis** (hit rate, trade rate, precision vs threshold)

**Plot filenames (save all to `data/model_dataset/plots/`):**

| Plot | Filename |
|------|----------|
| ROC Curve | `roc_curve.png` |
| Precision-Recall Curve | `pr_curve.png` |
| Calibration Curve | `calibration_curve.png` |
| Calibration by Volatility Regime | `calibration_by_regime.png` |
| Feature Importance (top 30) | `feature_analysis.png` |
| Feature Importance (PredictionValuesChange) | `feature_importance_prediction.png` |
| Learning Curves | `learning_curves.png` |
| Prediction Distribution | `prediction_distribution.png` |
| Threshold Analysis | `threshold_analysis.png` |

### 12.2 Saved Artifacts

| File | Content |
|------|---------|
| `dataset.parquet` | Final model dataset (features + labels + optional label diagnostics + optional weights; never infer features by exclusion) |
| `dataset_metadata.json` | Dataset build metadata + config snapshot |
| `catboost_model.cbm` | Trained model |
| `feature_list.json` | Ordered feature names |
| `analytics/metrics.json` | All evaluation metrics |
| `analytics/calibration_by_regime.json` | Regime-stratified calibration |
| `analytics/threshold_analysis.csv` | Performance at various thresholds |
| `analytics/best_hyperparameters.json` | Best hyperparameters selected by HPO |
| `plots/pareto_frontier.png` | Optuna Pareto frontier (only if `ENABLE_HPO=True`) |
| `plots/hp_importance_logloss.png` | Optuna hyperparameter importance for LogLoss (only if `ENABLE_HPO=True`) |
| `plots/hp_importance_prauc.png` | Optuna hyperparameter importance for PR-AUC (only if `ENABLE_HPO=True`) |

### 12.3 Metrics JSON Structure

```json
{
  "validation": {
    "n_samples": 12000,
    "base_rate": 0.33,
    "roc_auc": 0.61,
    "pr_auc": 0.44,
    "log_loss": 0.66,
    "brier_score": 0.23,
    "ece": 0.04
  },
  "test": {
    "n_samples": 15000,
    "base_rate": 0.35,
    "roc_auc": 0.62,
    "pr_auc": 0.45,
    "log_loss": 0.65,
    "brier_score": 0.22,
    "ece": 0.03
  },
  "best_hyperparameters": {
    "learning_rate": 0.001,
    "l2_leaf_reg": 0.5,
    "depth": 6
  },
  "best_iteration": 1200,
  "calibration_by_regime": {
    "validation": {
      "low": {"n_samples": 4000, "base_rate": 0.14, "ece": 0.02, "brier": 0.13},
      "med": {"n_samples": 4000, "base_rate": 0.33, "ece": 0.03, "brier": 0.22},
      "high": {"n_samples": 4000, "base_rate": 0.52, "ece": 0.04, "brier": 0.25}
    },
    "test": {
      "low": {"n_samples": 5000, "base_rate": 0.15, "ece": 0.02, "brier": 0.12},
      "med": {"n_samples": 5000, "base_rate": 0.35, "ece": 0.03, "brier": 0.22},
      "high": {"n_samples": 5000, "base_rate": 0.55, "ece": 0.04, "brier": 0.24}
    }
  }
}
```

### 12.4 Dataset & JSON Schemas (Minimal, Implementable)

This section removes ambiguity about what gets saved and what is used as a model input.

**`feature_list.json` (required):**

- Type: JSON array of strings
- Content: ordered list of model input columns
- Must include: all engineered feature columns + all `undef__*` flag columns
- Must exclude: identifiers (`ts`, `k`), all label/label-derived columns (`y`, `m_k`, `tau_k`, `phi`), and any non-feature columns such as sample weights (`w_dist`, `w_time`, `weight`) or raw derivatives join fields (Appendix E base columns).

Example:
```json
[
  "ret__lag1__f__w0",
  "vol__rs__f__w240",
  "time__sin_minute__f__w0",
  "undef__ofi__inst__f__w0"
]
```

**`dataset.parquet` (required):**

- Must contain at least these columns: `ts`, `k`, `y`
- May contain label diagnostics: `m_k`, `tau_k`, `phi` (if saved, they must never be used as features)
- May contain sample weight columns: `w_dist`, `w_time`, `weight` (used as CatBoost weights, never as features)
- May contain raw derivatives join columns (e.g., `close_fut`, `funding_rate`, `oi_usd`, `bvol`) that are inputs to derivatives feature computation but are never model inputs
- Must contain every column listed in `feature_list.json` (exact name match)

**`dataset_metadata.json` (required):**

- Type: JSON object
- Must include at minimum:
  - `SYMBOL`, `INTERVAL`
  - `START_YEAR`, `START_MONTH`, `END_YEAR`, `END_MONTH`
  - `M`, `ETA`, `C`, `PHI`, `EPS`
  - `WINDOWS_F`, `LAGS_F`, `WINDOWS_H`, `VOL_PAIRS`
  - `N_WARMUP`, `K_WARMUP`
  - `label_aux_cols` (e.g., `["m_k","tau_k","phi"]`)
  - `weight_cols` (e.g., `["w_dist","w_time","weight"]`, if present)
  - `non_feature_cols` (e.g., `["ts","k","y","m_k","tau_k","phi","w_dist","w_time","weight", ...]`)

**`download_manifest.json` (required):**

- Type: JSON object
- Must include:
  - `config` (SYMBOL, INTERVAL, START_YEAR/MONTH, END_YEAR/MONTH)
  - `downloads`: list of per-month entries with `data_url`, `checksum_url`, `filename`, `status`, and optional `error`

**`validation_report.json` (required):**

- Type: JSON object
- Must include:
  - `is_valid` boolean
  - `n_rows`, `date_range`
  - `issues`: list of human-readable strings (empty when valid)

### 12.5 Validation Criteria
- All artifacts listed in Section 12.2 exist after a full run and match the documented schemas in Section 12.4.
- `feature_list.json` excludes label diagnostics (`m_k`, `tau_k`, `phi`) and weight columns (`w_dist`, `w_time`, `weight`).

---

## 13. Feature Parsimony Strategy

### 13.0 Overview
**Purpose:** Provide optional research procedures for reducing feature count while preserving out-of-sample probability quality.  
**Scope:** Non-mandatory experimentation guidance; not required for acceptance unless explicitly adopted.  
**Dependencies:** Sections 9–12 (evaluation protocol).  
**Implementation Location:** Not implemented as a fixed pipeline stage; intended for iterative research using the saved dataset and feature list.

### 13.1 Group-Wise Ablation

Train baseline CatBoost, then add feature groups incrementally:

```python
ablation_order = [
    'A',      # Lags
    'B',      # Rolling stats
    'C',      # Volatility estimators
    'N',      # Barrier-aware (NEW)
    'D+O',    # Candle + excursion
    'E',      # Trend/momentum
    'F+P',    # Activity/liquidity
    'G+H',    # Correlations + entropy
    'Q',      # Seasonality (NEW)
    'I+J',    # Block + event
    'K',      # Past-target
]
```

Measure incremental lift in PR-AUC, Brier, and ECE. Drop groups that don't improve metrics.

### 13.2 Window Pruning

After initial training, examine feature importance by window size. Often half the windows can be dropped with no loss:

```python
def importance_by_window(model, feature_names):
    """Group feature importance by window parameter."""
    importance = dict(zip(feature_names, model.feature_importances_))
    
    by_window = defaultdict(float)
    for name, imp in importance.items():
        # Extract window from feature name
        match = re.search(r'__w(\d+)', name)
        if match:
            by_window[int(match.group(1))] += imp
    
    return dict(sorted(by_window.items(), key=lambda x: -x[1]))
```

### 13.3 Stability Selection

A feature is "kept" only if important across multiple time slices:

```python
def stability_selection(df, feature_cols, n_folds=4, importance_threshold=0.001):
    """
    Identify features consistently important across time folds.
    """
    importance_counts = defaultdict(int)
    
    for train_df, val_df in walk_forward_cv(df, n_folds):
        model = train_catboost(train_df, val_df, feature_cols)
        
        for feat, imp in zip(feature_cols, model.feature_importances_):
            if imp > importance_threshold:
                importance_counts[feat] += 1
    
    # Keep features important in majority of folds
    stable_features = [f for f, count in importance_counts.items() 
                       if count >= n_folds // 2]
    
    return stable_features
```

### 13.4 Validation Criteria
- Any feature-parsimony experiment preserves the split + embargo discipline (Section 9) and compares models only on held-out data.
- The reduced feature set is a documented subset of `feature_list.json` and checkpoints in Section 8A still pass.

---

## 14. Implementation Index

### 14.0 Overview
**Purpose:** Provide a compact map from spec sections to notebook stages and `src/utils.py` entrypoints.  
**Scope:** High-level entrypoints only (not full signatures or bodies).  
**Dependencies:** Sections 1?13 and Appendix B.  
**Implementation Location:** `src/utils.py` and `notebooks/*.ipynb`.

### 14.1 Notebook-to-Code Map

| Notebook | Scope | Primary entrypoints |
|----------|-------|---------------------|
| `notebooks/01_data_download.ipynb` | Downloads + validates raw data, writes parquet | `src/utils.py::generate_download_urls()`, `src/utils.py::download_file()`, `src/utils.py::verify_checksum()`, `src/utils.py::convert_timestamps()`, `src/utils.py::validate_klines()`, gap repair via `df.reindex(expected_index)` + deterministic fill (in-notebook), `src/utils.py::checkpoint_raw_data()` (+ derivatives: `src/utils.py::generate_all_derivatives_urls()`, `src/utils.py::load_*()`, `src/utils.py::align_to_1m_grid()`, `src/utils.py::checkpoint_derivatives_data()`) |
| `notebooks/02_feature_building.ipynb` | Feature engineering + labels + imputation, writes model dataset | `src/utils.py::compute_base_series()`, `src/utils.py::compute_*()` (Section 7), `src/utils.py::construct_labels()`, `src/utils.py::compute_past_target_features()`, `src/utils.py::compute_barrier_aware_features()`, `src/utils.py::compute_block_features()`, `src/utils.py::create_undef_flags_and_impute()`, `src/utils.py::checkpoint_*()` |
| `notebooks/03_model_training.ipynb` | Split + train/evaluate CatBoost, write artifacts | `src/utils.py::chronological_split_with_embargo()`, `src/utils.py::checkpoint_before_training()`; training helpers (`create_catboost_pool`, Optuna HPO) are defined in-notebook |

### 14.2 Key Entry Points by Spec Section
- **Section 4 (data acquisition):** `generate_download_urls()`, `download_file()`, `verify_checksum()`, `extract_and_load_csv()`, `convert_timestamps()`, `validate_klines()` (gap repair is implemented in-notebook via `reindex` + deterministic fill).
- **Section 6 (labels):** `construct_labels()`.
- **Section 7 (features):** `compute_base_series()` plus the family of `compute_*` feature builders (spot Groups A?Q; derivatives Groups R?V in Appendix E).
- **Section 8 (missingness):** `get_imputation_value()`, `create_undef_flags_and_impute()`.
- **Section 9 (splits):** `chronological_split_with_embargo()`, `walk_forward_cv()`.
- **Section 10 (model config):** CatBoost/Optuna orchestration lives in `notebooks/03_model_training.ipynb`; fixed defaults and ranges live in Appendix B (`CB_FIXED_PARAMS`, `CB_HP_RANGES`).
- **Section W (weights):** `compute_training_weights()`, `compute_barrier_distance_weight()`, `compute_time_discount_weight()`, `checkpoint_weights()`.

### 14.3 Validation Criteria
- Notebook runs must execute all referenced `checkpoint_*` functions without error.
- Any change to a public entrypoint used by notebooks must be reflected in this index and in the relevant spec section.

## Appendix A: Requirements

### A.0 Overview
**Purpose:** Declare the Python environment dependencies required to reproduce the notebooks.  
**Scope:** Package requirements only (no OS-level provisioning).  
**Dependencies:** None.  
**Implementation Location:** `requirements.txt`.


```
# requirements.txt
pandas>=2.0
numpy>=1.24
pyarrow>=14.0
catboost>=1.2
optuna>=3.6
plotly>=5.18
kaleido>=0.2
scikit-learn>=1.3
matplotlib>=3.8
seaborn>=0.13
requests>=2.31
tqdm>=4.66
jupyter>=1.0
```

### A.1 Validation Criteria
- `pip install -r requirements.txt` succeeds in a clean environment.
- The notebook kernel can import `pandas`, `numpy`, `catboost`, and (optionally) `optuna`.

---

## Appendix B: Configuration Constants

### B.0 Overview
**Purpose:** Centralize all configurable constants and defaults used by the pipeline.  
**Scope:** Defaults only; derived quantities (e.g., `K_WARMUP`) are included for reproducibility.  
**Dependencies:** Sections 4?12.  
**Implementation Location:** `src/utils.py` (constants) and notebook usage.


```python
# Single source of truth for constants: `src/utils.py`.
# This appendix mirrors those values and is expected to match exactly.

# =============================================================================
# Global / core
# =============================================================================

# Numerical stability
EPS = 1e-10

# Symbol and interval
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

# Date range (inclusive months; END_* is the final month fetched)
START_YEAR, START_MONTH = 2023, 1
END_YEAR, END_MONTH = 2025, 11

# =============================================================================
# Time indexing
# =============================================================================

# Decision multiplier: decision every M 1-minute bars
M = 10

# =============================================================================
# Label construction (log-return units)
# =============================================================================

# Net profit target (ETA) and round-trip cost estimate (C)
ETA = 0.0002
C = 0.0023
PHI = C + ETA  # barrier size used in labels and barrier-aware features

# =============================================================================
# Feature windows / lags
# =============================================================================

WINDOWS_F = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 45, 60, 75, 90, 120, 150, 180, 240, 300, 360, 480, 600, 720, 960, 1440, 1920, 2880, 4320, 10080, 20160]
WINDOWS_H = [2, 3, 6, 12, 24, 36, 72, 144]  # in decision blocks (10 min each)
LAGS_F = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 32, 35, 40, 42, 45, 50, 55, 60, 66, 75, 84, 90, 105, 120, 150, 180, 240, 300, 360, 480, 600, 720, 960, 1440, 2880, 4320]
VOL_PAIRS = [(10, 60), (10, 240), (20, 120), (20, 480), (30, 180), (60, 360), (60, 1440), (120, 720), (120, 2880), (240, 1440), (240, 4320), (720, 4320)]

# Per-group window subsets
WINDOWS_B = WINDOWS_F
WINDOWS_BPLUS = [30, 45, 60, 90, 120, 180, 240, 360, 720, 1440, 2880, 4320]
WINDOWS_VOL_OHLC = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 1920, 2880, 4320]
WINDOWS_VOL_DECOMP = [60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320]
WINDOWS_BARRIER = [10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320]
WINDOWS_CANDLE_ROLL = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480]
WINDOWS_BREAKOUT = [20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 1920, 2880, 4320]
WINDOWS_EXCURSION = [10, 20, 30, 60, 120, 240, 480, 960, 1440]
WINDOWS_MAXRET = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
WINDOWS_LOGP_Z = [30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2880, 4320]
WINDOWS_RSI = [7, 10, 14, 20, 30, 45, 60, 90, 120, 180, 240, 360]
WINDOWS_LIQ_AMIHUD = [15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
WINDOWS_LIQ_RPV = [15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
WINDOWS_OFI_IMPULSE = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360]
WINDOWS_CORR = [30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440]
WINDOWS_PENTROPY = [60, 120, 240, 720]
HITRATE_WINDOWS_H = [3, 6, 12, 24, 36, 72, 144]

# =============================================================================
# Warmup & trimming (derived)
# =============================================================================

N_WARMUP = max(max(WINDOWS_F) - 1, max(LAGS_F), M * max(WINDOWS_H))
K_WARMUP = (N_WARMUP + M - 1) // M  # ceil(N_WARMUP / M)

# =============================================================================
# Train/val/test split (at training time)
# =============================================================================

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
EMBARGO_K = 60  # decision steps (k units) between adjacent splits
N_CV_FOLDS = 1  # walk-forward folds (HPO only)

# =============================================================================
# CatBoost configuration (training notebook uses CB_FIXED_PARAMS)
# =============================================================================

CB_SEED = 42

# Legacy/simple CatBoost defaults (kept for reference; the notebook uses CB_FIXED_PARAMS)
CB_ITERATIONS = 2000
CB_LEARNING_RATE = 0.03
CB_DEPTH = 6
CB_EARLY_STOPPING = 100
CB_L2_LEAF_REG = 3.0

CB_FIXED_PARAMS = {
    # Objective / metrics
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "custom_metric": ["Logloss", "AUC", "PRAUC"],

    # Training control
    "iterations": 6000,
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

# Optuna hyperparameter search (optional; see Section 10.2)
ENABLE_HPO = False
OPTUNA_N_TRIALS = 500
OPTUNA_SEED = 42
HPO_DROP_OLDEST_FRAC = 0.6

# Hyperparameter search ranges (mirrors `src/utils.py`; notebooks may further narrow these)
CB_HP_RANGES = {
    "learning_rate": (0.001, 0.05),
    "l2_leaf_reg": (0.1, 5.0),
    "diffusion_temperature": (5000, 15000),
    "depth": (5, 8),
    "rsm": (0.65, 0.9),
    "subsample": (0.8, 1.0),
    "mvs_reg": (1.0, 10.0),
}

# Observation weighting (optional; see Section 8 + 02_feature_building.ipynb)
WEIGHT_USE_BARRIER_DISTANCE = True
WEIGHT_USE_TIME_DISCOUNT = True
WEIGHT_DIST_W_MAX = 2.0
WEIGHT_DIST_Q_TAIL = 0.01
WEIGHT_DIST_USE_POSITIVE = False
WEIGHT_DIST_W_MAX_POS = 2.0
WEIGHT_DIST_Q_TAIL_POS = 0.01
WEIGHT_TIME_R = 0.3
WEIGHT_TIME_DELTA = 0.99996
WEIGHT_NORMALIZE = False

# =============================================================================
# Derivatives configuration (Appendix E)
# =============================================================================

ENABLE_DERIVATIVES_FEATURES = True
ENABLE_FUTURES_KLINES = True
ENABLE_FUNDING_RATE = True
ENABLE_FUTURES_METRICS = True
ENABLE_EOH_SUMMARY = True
ENABLE_BVOL_INDEX = True
ENABLE_DELIVERY_TERM_STRUCTURE = False

# Derivatives feature windows
WINDOWS_BASIS = [0, 5, 60]
WINDOWS_FLOW_CSUM = [5, 10, 20]
WINDOWS_LIQ = [0, 15]
WINDOWS_ACTIVITY_Z = [30]
WINDOWS_OI_CHG = [60, 120]
WINDOWS_FUNDING = [0, 1440, 4320]
WINDOWS_OPTIONS = [0, 1440]
WINDOWS_VOL_IDX = [0, 1440, 43200]

# Default imputation helper (E.9)
MEDIAN_OI_USD = 15e9

# Derivatives coverage anchors (used to bound forward-fill)
FUTURES_METRICS_DATA_START = "2021-12-01"
EOH_DATA_START = "2023-05-18"
EOH_DATA_END = "2023-10-23"
BVOL_DATA_START = "2023-06-20"

# Derivatives data paths
DERIVATIVES_RAW_DIR = "data/raw_data/derivatives"
```

### B.1 Validation Criteria
- Values in this appendix match `src/utils.py` exactly (treat code as authoritative).
- Derived constants (`PHI`, `N_WARMUP`, `K_WARMUP`) equal the expressions shown here when recomputed from the base constants.

---

## Appendix C: Quick Start

### C.0 Overview
**Purpose:** Provide a minimal end-to-end reproduction path for a clean machine.  
**Scope:** One-shot execution (no experiment tracking).  
**Dependencies:** Appendix A (requirements) and Sections 1?12.  
**Implementation Location:** The three notebooks.


```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Create directory structure
mkdir -p data/raw_data/zips data/model_dataset/plots data/model_dataset/analytics

# 3. Run notebooks in order
cd notebooks
jupyter notebook

# Execute sequentially:
# - 01_data_download.ipynb   (downloads ~2GB, takes 10-30 min)
# - 02_feature_building.ipynb (compute features, ~5-15 min)
# - 03_model_training.ipynb  (train and evaluate, ~10-30 min)
```

### C.1 Validation Criteria
- Following the steps produces the artifacts in Section 12 without manual intervention beyond notebook execution.
- The run completes with all checkpoints printing `OK:` messages (Section 8A).

---

## Appendix D: Changelog

### D.0 Overview
**Purpose:** Track spec-level changes that affect reproducibility or implementation alignment.  
**Scope:** High-level deltas only.  
**Dependencies:** None.  
**Implementation Location:** This document.


### v4.0 (Implementation Alignment)

This release updates the specification to match the current repository implementation (as of 2026-01-03) and adds a required executive summary and section-overview scaffolding.

**Significant changes from v3.1:**
- Add Section 0 “Technical Executive Summary” and a top-level Table of Contents.
- Update Appendix B constants to match `src/utils.py` (date range, `ETA/C/PHI`, embargo, HPO toggles, weighting config, derivatives config, CatBoost fixed params).
- Update Section 4 to document deterministic gap repair (reindex + synthetic bars) used by `01_data_download.ipynb` when rare missing timestamps occur.
- Update Section 7 feature count summary to include Appendix E derivatives features and the current `feature_list.json` totals (327 engineered + 103 undef flags = 430 model inputs).
- Update Section 9 embargo rule to the configured `EMBARGO_K` (default 60) and align the pre-training checkpoint logic with the implementation (exclude label diagnostics and weight columns from NaN checks).
- Update Section 10 CatBoost training description to match the implemented ordered/time-aware Pools and optional Optuna NSGA-II HPO workflow; de-emphasize class weighting in favor of sample weights.
- Update Section 12 output requirements to include learning curves and prediction-based feature-importance plots, plus optional Optuna visualization outputs.

### v3.1 (Implementation Clarity)

| Addition | Section | Purpose |
|----------|---------|---------|
| Execution Order | 7A.2 | Stage-by-stage ordering for `02_feature_building.ipynb` |
| Implementation Notes | 7A.4 | Ordering pitfalls and invariants |
| Warmup Explanation | 7A.3 | Why warmup exists, what causes it, how much data is lost |
| Pipeline Checkpoints | 8A.0-8A.5 | Runtime checkpoint contract for each stage |
| Complete Imputation | 8.3 | All new feature patterns covered, lookup function provided |
| Excursion Pseudocode | 7.11.1 | Working Python code for max drawup/drawdown |
| Split Timing | 9.0 | Clarifies dataset saved whole, split at training time |
| Global Constants | Appendix B | EPS, derived values, organized by category |

### v3.0 (Feature Engineering)

| Group | Features Added | Rationale |
|-------|----------------|-----------|
| C+ | Bipower ratio, semivariance, vol-of-vol | Volatility quality/jump detection |
| N | Barrier-normalized z, expected-max ratio, vol ratio | Barrier-aware signal-to-noise |
| O | Max drawup/drawdown, max returns, block excursions | Label-aligned path features |
| P | Amihud illiquidity, range/vol, OFI impulse | Enhanced microstructure proxies |
| Q | Cyclical time encoding (hour, day-of-week) | Crypto seasonality capture |

**Modeling Improvements (v3.0):**

1. **Walk-forward CV:** Hyperparameters validated across multiple time periods (Section 9.5)
2. **Regime-stratified calibration:** ECE and Brier reported by volatility tercile (Section 11.4)
3. **Feature parsimony framework:** Group-wise ablation, window pruning, stability selection (Section 13)

### Feature Count Evolution

| Version | Base Features | With Flags | Notes |
|---------|---------------|------------|-------|
| v2.0 | ~235 | ~285 | Original spec |
| v3.0 | ~292 | ~352 | +57 targeted features |
| v3.1 | ~292 | ~352 | No feature changes, implementation clarity |

---
### D.1 Validation Criteria
- Any change to spec version, constants, feature definitions, outputs, or checkpoints is reflected in this changelog.

---

## Appendix E: BTC/USDT Derivatives-Derived Features

### E.0 Overview
**Purpose:** Define the optional derivatives-derived feature extension and its strict no-lookahead alignment contract.  
**Scope:** BTCUSDT perpetual futures, funding, futures metrics, options EOHSummary, and BVOL index.  
**Dependencies:** Sections 4?8 and Appendix B.  
**Implementation Location:** `notebooks/01_data_download.ipynb` (derivatives download), `notebooks/02_feature_building.ipynb` (Stage 2.5), and `src/utils.py` (loaders, aligner, feature builders).


**Version:** 1.0  
**Status:** Specification for Integration  
**Scope:** Augment the spot-based Barrier Classifier with ~35 derivatives-derived base features (plus `undef__*` flags, yielding ~50–60 additional model columns) from BTC/USDT perpetual futures, funding rates, options sentiment, and implied volatility indices.

**Alignment with Main Spec:** This appendix follows the conventions, notation, and design patterns of the Minimal Barrier-Crossing Classifier Specification v4.0. All features use only information available up to the current minute (no forward leakage), adhere to the `feature__transform__f__w{window}` naming scheme, and integrate into the existing pipeline stages.

---

## E.1 Overview and Rationale

### E.1.1 Why Derivatives Data?

The spot market for BTC/USDT reflects immediate supply and demand, but **derivatives markets** often lead price discovery:

1. **Futures basis** (premium/discount) signals leveraged positioning and arbitrage flows
2. **Funding rates** reveal the cost of holding leveraged longs vs shorts—extreme values precede reversals
3. **Open interest** tracks capital committed to futures—changes indicate trend confirmation or exhaustion
4. **Options put/call ratios** gauge hedging demand and directional sentiment
5. **Implied volatility (BVOL)** captures market expectations for future price swings

By incorporating ~35 derivative-driven features (plus `undef__*` flags), the classifier gains insight into market structure, sentiment, and positioning—complementing the spot-only features in Sections 7.4–7.22.

### E.1.2 Feature Count Summary

| Group | Description | Base Features |
|-------|-------------|--------------|
| R | Basis (perpetual futures vs spot) | 8 |
| S | Futures flow, liquidity, activity | 9 |
| T | Futures open interest + funding | 8 |
| U | Options sentiment (EOHSummary) | 5 |
| V | Implied vol + IV/RV | 5 |
| **Total** | | **35** |

*Note: NaN-capable derivatives features add `undef__*` flags, so total added model columns is higher than 35.*

### E.1.3 Conventions (Critical for Correct Implementation)

1. **Canonical time index:** All series are aligned to the spot 1-minute *bar-complete* UTC index used everywhere in this project:  
   `ts = open_time + 60 seconds` (see `src/utils.py:convert_timestamps`).
2. **No lookahead:** When aligning non-1m sources (funding, metrics, EOHSummary, BVOL), assign to each spot bar timestamp `t` the *last observation with timestamp ≤ t* (backward-asof / forward-fill on the 1m grid).
3. **No extrapolation beyond coverage:** Do **not** forward-fill past the last timestamp available in a source (or before its first). Outside `[min_ts, max_ts]`, values stay `NaN` and are handled by `undef__*` + imputation in Stage 10 (Section 8.5).
4. **Rolling window definition:** For window size `W` minutes, the trailing index set is `I_{n,W} = {n-W+1, …, n}`. Use `min_periods=W`.
5. **Std definition:** All rolling `std()` use `ddof=0` (population std), matching the main spec.
6. **Numerical stability:** Use `EPS = 1e-10` (Appendix B) as the only division guard.

---

## E.2 Data Sources

All derivatives inputs are fetched from Binance public archives (`https://data.binance.vision/`) and validated via SHA256 `.CHECKSUM` (same checksum contract as spot). Each source is parsed to a UTC timestamp index and then aligned onto the spot 1-minute `DatetimeIndex` using `src/utils.py::align_to_1m_grid(df_source, index_1m, method='ffill')`, which forward-fills *within* source coverage and sets values to `NaN` before the first and after the last available source timestamp (no extrapolation).

| Source | Public-data path pattern | Native cadence | Parsed columns (pre-align) | Aligned columns used in Stage 2.5 |
|--------|--------------------------|---------------|----------------------------|-----------------------------------|
| Futures klines (BTCUSDT perpetual) | `data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-YYYY-MM.zip` | 1m | Standard kline fields (Section 4.2) | `close_fut`, `volume_fut`, `quote_volume_fut`, `taker_buy_base_fut`, `num_trades_fut` |
| Funding rate (BTCUSDT) | `data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-YYYY-MM.zip` | 8h | `calc_time`, `last_funding_rate` | `funding_rate` (ffill onto 1m grid) |
| Futures metrics (BTCUSDT) | `data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-YYYY-MM-DD.zip` | 5m | `create_time`, `sum_open_interest_value` (+ ratios if present) | `oi_usd` (and any ratio cols, currently unused by features) |
| Options EOHSummary (BTCUSDT) | `data/option/daily/EOHSummary/BTCUSDT/BTCUSDT-EOHSummary-YYYY-MM-DD.zip` | 1h | `date`, `hour`, `type`, `volume_usdt`, `openinterest_usdt` | `opt_oi`, `put_open_interest`, `call_open_interest`, `opt_volume`, `put_volume`, `call_volume` |
| BVOL index (BTCBVOLUSDT) | `data/option/daily/BVOLIndex/BTCBVOLUSDT/BTCBVOLUSDT-BVOLIndex-YYYY-MM-DD.zip` | irregular/daily | `calc_time`, `index_value` | `bvol` |

**Coverage constraints (must be preserved):** `EOH_DATA_START = 2023-05-18`, `EOH_DATA_END = 2023-10-23`, `BVOL_DATA_START = 2023-06-20` (Appendix B). Outside each source?s coverage window, aligned values remain `NaN` and are handled via `undef__*` + imputation.

## E.3 Feature Definitions – Group R: Term Structure & Basis

These features quantify the relationship between futures and spot prices, capturing **basis** (premium/discount) and term structure signals.

### E.3.1 Instantaneous Basis Features

#### `basis__abs__f__w0` — Absolute Basis (USDT)

**Definition:**
```
basis_abs_n = P^{fut}_n - P^{spot}_n
```
where `P^{fut}_n` is the futures close price and `P^{spot}_n` is the spot close price at minute n.

**Motivation:** The price difference between futures and spot indicates market sentiment and funding pressure. Positive basis (contango) signals bullish speculation; negative basis (backwardation) signals bearish sentiment or hedging demand. Extreme values often precede market tops or bottoms.

**Output Range:** ℝ (typically [-500, +500] USDT for BTC)

**Undefined When:** Either price is missing (exchange outage).

**Imputation:** 0.0 (no premium/discount).

---

#### `basis__pct__f__w0` — Percentage Basis

**Definition:**
```
basis_pct_n = (P^{fut}_n - P^{spot}_n) / (P^{spot}_n + EPS) × 100
```

**Motivation:** Normalizes basis by price level, making it comparable across different price regimes. A +0.5% basis roughly corresponds to +0.5% annualized funding rate (assuming 8h intervals). Directly relates to implied funding costs.

**Output Range:** ℝ (typically [-2%, +2%])

**Undefined When:** Spot price is missing.

**Imputation:** 0.0 (no premium/discount).

---

#### `basis__ann_yield__f__w0` — Annualized Basis Yield

**Definition:**
```
basis_ann_n = (P^{fut}_n - P^{spot}_n) / (P^{spot}_n + EPS) × (365 × 24 × 60 / τ_n) × 100
```
where τ_n is minutes until the next funding timestamp (00:00/08:00/16:00 UTC), clipped to [60, 480] to avoid numerically explosive annualization in the last few minutes before a funding event.

**Motivation:** Converts instantaneous basis to an annualized yield, enabling comparison to interest rates and identifying extreme carry trade opportunities.

**Output Range:** ℝ (typically [-100%, +200%] annualized)

**Undefined When:** Spot price missing.

**Imputation:** 0 (neutral yield).

**Implementation Note:**
```python
def compute_basis_ann_yield(p_fut: pd.Series, p_spot: pd.Series, minutes_to_funding: pd.Series) -> pd.Series:
    """
    Annualized basis yield.
    τ is minutes until next funding; clamp to [60, 480] for numerical stability (Appendix E.3.1).
    """
    tau = minutes_to_funding.clip(lower=60, upper=480)
    basis_pct = (p_fut - p_spot) / (p_spot + EPS)
    return basis_pct * (365 * 24 * 60 / tau) * 100
```

### E.3.2 Basis Dynamics Features

#### `basis__chg__f__w5` — Basis Change (5-minute)

**Definition:**
```
basis_chg_{n,5} = basis_pct_n - basis_pct_{n-5}
```

**Motivation:** Captures the rate of change in basis. Rapidly widening basis indicates accelerating bullish leverage; narrowing or flipping basis can foreshadow reversals as longs close.

**Output Range:** ℝ

**Undefined When:** Warmup (n < 5) or either endpoint missing.

**Imputation:** 0 (no change).

---

#### `basis__mean__f__w5` — Rolling Mean Basis (5-minute)

**Definition:**
```
basis_mean_{n,5} = mean({basis_pct_i : i ∈ I_{n,5}})
```
where `I_{n,5} = {n-4, …, n}`.

**Motivation:** Smooths minute-to-minute basis noise and provides a short-horizon estimate of average premium/discount.

**Output Range:** ℝ

**Undefined When:** Warmup (n < 5).

**Imputation:** 0.0.

---

#### `basis__std__f__w5` — Basis Volatility (5-minute)

**Definition:**
```
basis_std_{n,5} = std_{ddof=0}({basis_pct_i : i ∈ I_{n,5}})
```

**Motivation:** Detects short-lived dislocations (rapid basis expansion/contraction) that may precede squeezes or mean reversion.

**Output Range:** ℝ≥0

**Undefined When:** Warmup (n < 5).

**Imputation:** 0.0.

---

#### `basis__mean__f__w60` — Rolling Mean Basis (1-hour)

**Definition:**
```
basis_mean_{n,60} = mean({basis_pct_i : i ∈ I_{n,60}})
```

**Output Range:** ℝ

**Undefined When:** Warmup (n < 60).

**Imputation:** 0.0.

#### `basis__std__f__w60` - Basis Volatility (1-hour)

**Definition:**
```
basis_std_{n,60} = std_{ddof=0}({basis_pct_i : i ∈ I_{n,60}})
```
where `I_{n,60} = {n-59, ..., n}`.

**Motivation:** Measures how volatile the futures premium has been. High basis volatility suggests dislocation or squeeze events; stable basis indicates tight arbitrage.

**Output Range:** ℝ≥0

**Undefined When:** Warmup (n < 60).

**Imputation:** 0 (no volatility).

### E.3.3 Optional Term Structure Features (Delivery Quarterly Futures)

**Status:** Optional / deferred. These features require downloading and maintaining two delivery quarterly futures series (front and next quarter). If you do not download quarterly delivery futures, omit these features entirely.

**Quarterly futures symbols and URLs (verified):**
- Symbol format: `BTCUSDT_YYYYMMDD` (expiry date), e.g. `BTCUSDT_240329`
- URL pattern (monthly 1m klines):
  ```
  https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT_YYYYMMDD/1m/BTCUSDT_YYYYMMDD-1m-{YEAR}-{MONTH:02d}.zip
  ```
- Schema: same as E.2.1 futures klines (CSV with header row; columns include `count`, `taker_buy_volume`, etc)

**Preprocessing rule (important):** Treat bars with `volume == 0` as missing for term-structure features (delivery contracts can publish trailing zero-volume bars near/after expiry).

#### `term_struct__spread_q__f__w0` — Near vs. Far Quarterly Spread

**Definition:**
```
spread_q_n = P^{Q1}_n - P^{Q2}_n
```
where Q1 is the **front** quarterly delivery contract and Q2 is the **next** quarterly delivery contract, and prices are aligned to the canonical 1m index (E.1.3).

**Undefined When:** Either contract close is missing (including zero-volume bars treated as missing).

**Imputation:** 0.0 (flat term structure).

---

#### `term_struct__ann_slope__f__w0` — Annualized Term Structure Slope (Optional)

**Definition:**
```
ann_slope_n = ((P^{Q2}_n / P^{spot}_n)^{1/T_n} - 1) × 100
```
where `T_n` is time in years until the Q2 expiry timestamp defined as `YYYY-MM-DD 08:00:00Z` (derived from the `BTCUSDT_YYYYMMDD` symbol).

**Undefined When:** Missing Q2, missing spot, or `T_n ≤ 0`.

**Imputation:** 0.0.

---

## E.4 Feature Definitions – Group S: Derivatives Volume & Order Flow

These features leverage futures trading activity—taker buy/sell volumes and trade counts—to gauge order flow imbalance and participation shifts.

### E.4.1 Taker Flow Features

#### `flow__taker_buy_ratio__f__w0` — Futures Taker Buy Ratio (Instantaneous)

**Definition:**
```
tb_ratio^{fut}_n = taker_buy_base^{fut}_n / volume^{fut}_n
```

**Motivation:** Indicates which side was more aggressive in the futures market that minute. Ratio > 0.5 means net buy pressure; ratio < 0.5 means net sell pressure. Futures taker flow often leads spot price moves.

**Output Range:** [0, 1]

**Undefined When:** `volume^{fut}_n = 0` (no trades) or futures data is missing for the minute.

**Imputation:** 0.5 (balanced flow).

---

#### `flow__net_vol_btcs__f__w0` — Net Taker Volume (BTC, Signed)

**Definition:**
```
net_vol_n = taker_buy_base^{fut}_n - (volume^{fut}_n - taker_buy_base^{fut}_n)
          = 2 × taker_buy_base^{fut}_n - volume^{fut}_n
```

**Motivation:** Signed measure of net buying/selling pressure in BTC terms. Preserves magnitude (100 BTC net buy is more significant than 5 BTC). Sudden spikes often coincide with price upticks/downticks.

**Output Range:** ℝ (typically [-5000, +5000] BTC for high-activity minutes)

**Undefined When:** Futures `volume`/`taker_buy_base` is missing for the minute.

**Imputation:** N/A (always defined).

---

#### `flow__net_vol_csum__f__w5`, `flow__net_vol_csum__f__w10`, `flow__net_vol_csum__f__w20` — Cumulative Net Taker Volume

**Definition:**
```
csum_net_vol_{n,W} = Σ_{i=n-W+1}^{n} net_vol_i
```

**Windows:** W ∈ {5, 10, 20}

**Motivation:** Aggregates net flow to measure accumulated buying or selling pressure. Consistently positive cumulative flow indicates ongoing accumulation; flip from positive to negative signals momentum reversal.

**Output Range:** ℝ

**Undefined When:** Warmup (n < W).

**Imputation:** 0 (no accumulation).

---

### E.4.2 Futures vs. Spot Volume Features

#### `liq__fut_vs_spot_vol__f__w0` — Futures/Spot Volume Ratio

**Definition:**
```
vol_ratio_n = quote_volume^{fut}_n / quote_volume^{spot}_n
```

**Motivation:** Indicates which market is "in the driver's seat." High ratio (futures >> spot) suggests leveraged traders are leading; low ratio suggests organic spot demand.

**Output Range:** ℝ≥0 (typically [0.5, 10])

**Undefined When:** `quote_volume^{spot}_n = 0` or either quote volume is missing.

**Imputation:** 1.0 (neutral futures/spot volume).

---

#### `liq__avg_trade_size__f__w0` — Average Futures Trade Size (BTC)

**Definition:**
```
avg_trade_n = volume^{fut}_n / num_trades^{fut}_n
```

**Motivation:** Larger average trade size indicates institutional participation or whale activity. Spikes can precede large moves.

**Output Range:** ℝ≥0

**Undefined When:** `num_trades = 0`.

**Imputation:** 0.0 (no trades / no size signal).

---

#### `liq__avg_trade_size__f__w15` — Average Trade Size (15-min Rolling)

**Definition:**
```
avg_trade_{n,15} = Σ_{i∈I_{n,15}} volume^{fut}_i / Σ_{i∈I_{n,15}} num_trades^{fut}_i
```

**Motivation:** Smooths single-minute noise to identify trends in participant size.

**Output Range:** ℝ≥0

**Undefined When:** Warmup.

**Imputation:** 0.

---

### E.4.3 Activity Features

#### `activity__trades_zscore__f__w30` — Trade Count Z-Score

**Definition:**
```
z_trades_{n,30} = (N^{fut}_n - μ_{N,30}) / σ_{N,30}
```
where μ and σ are rolling mean and std of `num_trades^{fut}` over 30 minutes.

**Motivation:** Identifies unusual trading activity bursts—high z-score means "something is happening" even if volume hasn't spiked proportionally (e.g., many small bot trades).

**Output Range:** ℝ

**Undefined When:** Warmup (n < 30) or σ = 0.

**Imputation:** 0 (at-mean).

---

## E.5 Feature Definitions – Group T: Open Interest & Funding

These features incorporate **open interest** (total outstanding contracts) and **funding rates** to gauge positioning and the cost of holding leveraged positions.

### E.5.1 Open Interest Features

#### `oi__total_usd__f__w0` — Total Futures Open Interest (USD Notional)

**Definition:**
```
OI_n = oi_usd_n   (from futures metrics `sum_open_interest_value`, aligned to 1m via backward-asof / forward-fill)
```

**Motivation:** Open interest represents capital committed to futures. Rising OI with rising price = bullish continuation; rising OI with falling price = bearish buildup. Extremely high OI relative to history implies crowded positioning (shake-out risk).

**Output Range:** ℝ≥0 (typically 10B–30B USD for BTCUSDT)

**Undefined When:** Futures metrics data is not available for the timestamp (outside the source’s coverage, or a missing archive day).

**Imputation:** Remaining `NaN` values after alignment use `undef__*` + imputation per Section E.9 (default level impute: `median_oi_usd`).

**Note:** Futures metrics are 5-minute snapshots (E.2.3). Alignment forward-fills *within* source coverage only (E.1.3).

---

#### `oi__chg__f__w60` — Open Interest Change (1-Hour)

**Definition:**
```
ΔOI_{n,60} = OI_n - OI_{n-60}
```

**Motivation:** Captures the trend in position-building. Sharp rise in OI during a rally = new longs entering (continuation signal). Sharp fall in OI during rally = shorts covering (potentially weaker).

**Output Range:** ℝ

**Undefined When:** Warmup (n < 60) or OI missing.

**Imputation:** 0 (no change).

---

#### `oi__chg_pct__f__w60` — Open Interest Change (%, 1-Hour)

**Definition:**
```
ΔOI%_{n,60} = (OI_n - OI_{n-60}) / (OI_{n-60} + EPS) × 100
```

**Motivation:** Percentage change normalizes across different OI levels.

**Output Range:** ℝ (typically [-5%, +5%])

**Undefined When:** Warmup or OI missing.

**Imputation:** 0.

---

#### `oi__vol_ratio__f__w60` — OI / Volume Ratio (1-Hour)

**Definition:**
```
oi_vol_ratio_{n,60} = OI_n / Σ_{i∈I_{n,60}} quote_volume^{fut}_i
```

**Motivation:** High OI relative to recent volume = dormant positions (spring coiled). Low ratio = high turnover (positions changing rapidly).

**Output Range:** ℝ≥0

**Undefined When:** Volume = 0 or OI missing.

**Imputation:** 0.0 (no turnover signal).

---

### E.5.2 Funding Rate Features

#### `funding__rate__f__w0` — Current Funding Rate (per 8-Hour Interval, %)

**Definition:**
```
funding_rate_pct_n = last_announced_funding_rate_n × 100
```

**Motivation:** Funding rate is the cost longs pay shorts (if positive) or vice versa. High positive funding = expensive to be long (market bullish); high negative = expensive to be short (market bearish). Extreme funding often precedes mean reversion.

**Output Range:** ℝ (typically [-0.5%, +0.5%] per 8h)

**Undefined When:** Before funding data available.

**Imputation:** 0 (neutral).

**Note:** Funding is piecewise constant (updates every 8h). Forward-fill between events.

---

#### `funding__trend__f__w4320` — Funding Rate Trend (3-Day)

**Definition:**
```
funding_trend_{n,4320} = funding_rate_pct_n - mean({funding_rate_pct_i : i ∈ I_{n,4320}})
```
where 4320 = 3 days × 1440 minutes/day.

**Motivation:** Captures whether funding is rising or falling over multiple days. Rising funding = accelerating bullish leverage. Falling funding = cooling or flip to bearish.

**Output Range:** ℝ

**Undefined When:** Insufficient funding history.

**Imputation:** 0 (no trend).

---

#### `funding__ewma__f__w1440` — Funding Rate EWMA (24h)

**Definition:**
```
funding_ewma_{n,1440} = EMA(funding_rate_pct, span=1440, adjust=False)
```

**Motivation:** Smoothed funding level to identify sustained regimes vs. temporary spikes.

**Output Range:** ℝ

**Undefined When:** Warmup.

**Imputation:** 0.

---

### E.5.3 OI-Price Relationship Features

#### `oi__price_corr__f__w120` — OI-Price Correlation (2-Hour)

**Definition:**
```
corr_{n,120} = Pearson({ΔOI_i}, {r_i}) over i ∈ I_{n,120}
```
where `ΔOI_i = OI_i - OI_{i-1}` and `r_i` is the 1-minute spot log return already defined in the main spec.

**Motivation:** Positive correlation = new positions reinforcing price direction (continuation). Negative correlation = position unwinding against price direction (reversal signal).

**Output Range:** [-1, 1]

**Undefined When:** Insufficient data or zero variance.

**Imputation:** 0 (no correlation).

---

## E.6 Feature Definitions – Group U: Options Market Sentiment

These features exploit BTCUSDT **options data** (from EOHSummary) to gauge sentiment via put/call ratios and aggregate positioning.

### E.6.1 Put/Call Ratio Features

#### `opt_pcr__oi__f__w0` — Put/Call Open Interest Ratio

**Definition:**
```
PCR_OI_n = put_open_interest_n / call_open_interest_n
```

**Motivation:** Classic sentiment indicator. PCR > 0.7 = bearish tilt (more puts than calls); PCR < 0.7 = bullish. Extreme values can be contrarian signals (peak fear = potential bounce).

**Output Range:** ℝ≥0 (typically [0.3, 2.0])

**Undefined When:** EOHSummary options data is unavailable for the timestamp (outside the EOHSummary coverage window, a missing archive day, or `call_open_interest_n = 0`). As of validation (2026-01-03), EOHSummary coverage is `2023-05-18` through `2023-10-23` (E.2.4).

**Imputation:** 1.0 (neutral).

---

#### `opt_pcr__vol__f__w0` — Put/Call Volume Ratio (24h)

**Definition:**
```
PCR_vol_n = put_volume_n / call_volume_n
```

**Motivation:** Volume-based PCR captures more immediate sentiment (trading flow) vs. outstanding positions.

**Output Range:** ℝ≥0

**Undefined When:** Options data missing or `call_volume_n = 0`.

**Imputation:** 1.0.

---

#### `opt_pcr__oi_chg__f__w1440` — Put/Call OI Ratio Change (1-Day)

**Definition:**
```
ΔPCR_{n,1440} = PCR_OI_n - PCR_OI_{n-1440}
```

**Motivation:** Trend in put/call positioning. Rising PCR = bearish sentiment building. Falling PCR = bullish sentiment building.

**Output Range:** ℝ

**Undefined When:** Warmup or data missing.

**Imputation:** 0.

---

### E.6.2 Options Activity Features

#### `opt_oi__total_usd__f__w0` — Total Options Open Interest (USD)

**Definition:**
```
opt_OI_n = opt_oi_n   (EOHSummary aggregate: puts + calls notional, USDT)
```

**Motivation:** Indicates options market size and hedging activity. Rising options OI = more positioning for volatility or direction.

**Output Range:** ℝ≥0

**Undefined When:** Options data missing.

**Imputation:** 0.

---

#### `opt_vol__24h_usd__f__w0` — Options Volume (24h, USD)

**Definition:**
```
opt_vol_n = opt_volume_n   (EOHSummary aggregate: trailing 24h options volume snapshot, USDT)
```

**Motivation:** Spikes in options volume can presage large moves as traders position for events.

**Output Range:** ℝ≥0

**Undefined When:** Options data missing.

**Imputation:** 0.

---

## E.7 Feature Definitions – Group V: Implied Volatility & Risk Premium

These features leverage the **Binance Volatility Index (BVOL)** and compare implied vs. realized volatility.

### E.7.1 Implied Volatility Features

#### `vol_idx__bvol30d__f__w0` — BVOL Index (30-Day Implied, %)

**Definition:**
```
BVOL_n = bvol_index_value   (annualized %)
```

**Motivation:** BVOL is the "crypto VIX"—market expectation of volatility. High BVOL = fear/uncertainty; low BVOL = complacency. Historically, extremely low vol precedes breakouts; extremely high vol coincides with capitulation lows.

**Output Range:** ℝ≥0 (typically [30%, 150%])

**Undefined When:** BVOL data not available.

**Imputation:** 60% (typical BTC vol).

---

#### `vol_idx__bvol_chg__f__w1440` — BVOL Change (24h)

**Definition:**
```
ΔBVOL_{n,1440} = BVOL_n - BVOL_{n-1440}
```

**Motivation:** Rising BVOL = market bracing for moves. Falling BVOL = calming.

**Output Range:** ℝ

**Undefined When:** Warmup or data missing.

**Imputation:** 0.

---

### E.7.2 Realized Volatility (from Spot)

#### `vol_realized__30d__f__w43200` — Realized Volatility (30-Day, Annualized)

**Definition:**
```
σ_realized_{n,43200} = std_{ddof=0}({r_i : i ∈ I_{n,43200}}) × √(525600) × 100
```
where 43200 = 30 days × 1440 min/day, and 525600 = minutes per year.

**Motivation:** Baseline for comparing to implied vol. Derived from spot data (not new derivative input), included here for completeness.

**Output Range:** ℝ≥0

**Undefined When:** Warmup.

**Imputation:** 60%.

---

### E.7.3 Volatility Risk Premium Features

#### `vol_risk_premium__diff__f__w0` — Volatility Risk Premium (IV − RV)

**Definition:**
```
VRP_n = BVOL_n - σ_realized_{n,43200}
```

**Motivation:** Positive VRP = market pricing more vol than observed (fear premium or event risk). Negative VRP = complacency. Large positive VRP often precedes mean reversion in vol.

**Output Range:** ℝ

**Undefined When:** Either component missing.

**Imputation:** 0 (IV = RV).

---

#### `vol_risk_premium__ratio__f__w0` — VRP Ratio

**Definition:**
```
VRP_ratio_n = BVOL_n / (σ_realized_{n,43200} + EPS)
```

**Motivation:** Ratio > 1 means implied > realized (vol premium). Historical mean ~1.1 for BTC.

**Output Range:** ℝ≥0

**Undefined When:** Either component missing.

**Imputation:** 1.0.

---

## E.8 Feature Count and Naming Summary

| Group | Pattern | Windows | Count |
|-------|---------|---------|-------|
| R: Basis | `basis__*__f__w*` | 0, 5, 60 | 8 |
| R: Term (optional) | `term_struct__*__f__w0` | 0 | 2 |
| S: Flow | `flow__*__f__w*` | 0, 5, 10, 20 | 5 |
| S: Liq | `liq__*__f__w*` | 0, 15 | 3 |
| S: Activity | `activity__*__f__w*` | 30 | 1 |
| T: OI | `oi__*__f__w*` | 0, 60, 120 | 5 |
| T: Funding | `funding__*__f__w*` | 0, 1440, 4320 | 3 |
| U: Options | `opt_*__f__w*` | 0, 1440 | 5 |
| V: Vol | `vol_idx__*`, `vol_realized__*`, `vol_risk_premium__*` | 0, 1440, 43200 | 5 |
| **Total (baseline)** | | | **35** |

*With `undef__*` flags for NaN-capable features, total added model columns may reach ~50–60.*

---

## E.9 Imputation Rules (Derivatives Features)

Extends Section 8.3 of the main specification.

| Feature Pattern | Impute Value | Rationale |
|-----------------|--------------|-----------|
| **Basis** | | |
| `basis__abs*` | 0.0 | No premium/discount |
| `basis__pct*` | 0.0 | No premium/discount |
| `basis__ann_yield*` | 0.0 | Neutral yield |
| `basis__chg*` | 0.0 | No change |
| `basis__mean*` | 0.0 | Neutral mean |
| `basis__std*` | 0.0 | No volatility |
| **Term Structure** | | |
| `term_struct__spread_q*` | 0.0 | Flat term structure |
| `term_struct__ann_slope*` | 0.0 | Flat term structure |
| **Flow** | | |
| `flow__taker_buy_ratio*` | 0.5 | Balanced flow |
| `flow__net_vol_btcs*` | 0.0 | No imbalance |
| `flow__net_vol_csum*` | 0.0 | No accumulation |
| **Liquidity** | | |
| `liq__fut_vs_spot_vol*` | 1.0 | Equal futures/spot volume |
| `liq__avg_trade_size*` | 0.0 | No trades |
| **Activity** | | |
| `activity__trades_zscore*` | 0.0 | At-mean activity |
| **Open Interest** | | |
| `oi__total_usd*` | `median_oi_usd` | Historical median level |
| `oi__chg*`, `oi__chg_pct*` | 0.0 | No change |
| `oi__vol_ratio*` | 0.0 | No turnover signal |
| **Funding** | | |
| `funding__rate*` | 0.0 | Neutral funding |
| `funding__trend*` | 0.0 | No trend |
| `funding__ewma*` | 0.0 | Neutral |
| `oi__price_corr*` | 0.0 | No correlation |
| **Options** | | |
| `opt_pcr__oi*` | 1.0 | Balanced puts/calls |
| `opt_pcr__vol*` | 1.0 | Balanced puts/calls |
| `opt_pcr__oi_chg*` | 0.0 | No change |
| `opt_oi__total_usd*` | 0.0 | No options OI |
| `opt_vol__24h_usd*` | 0.0 | No options volume |
| **Volatility** | | |
| `vol_idx__bvol30d*` | 60.0 | Typical BTC vol (60%) |
| `vol_idx__bvol_chg*` | 0.0 | No change |
| `vol_realized__30d*` | 60.0 | Typical BTC vol |
| `vol_risk_premium__diff*` | 0.0 | IV = RV |
| `vol_risk_premium__ratio*` | 1.0 | IV = RV |

**Implementation Note (utils.py):** Extend the existing `get_imputation_value()` by inserting derivatives-specific patterns **before** the final catch-all `(r'.*', 0.0)`. Use a constant default `median_oi_usd = 15e9` (or compute a median from your training range and pass it into the function).

---

## E.10 Computation Order and Pipeline Integration

### E.10.0 Overview
**Purpose:** Define where derivatives data enters the main pipeline and how it is aligned without leakage.  
**Scope:** Data acquisition outputs, alignment rule, and notebook integration points (not the per-feature formulas).  
**Dependencies:** Section 4 (downloads + checksums), Section 5 (time index), Section 8 (missingness contract), Appendix B (derivatives toggles + windows).  
**Implementation Location:** `notebooks/01_data_download.ipynb` (derivatives download block), `notebooks/02_feature_building.ipynb` (Stage 2.5), and `src/utils.py` loaders/aligners.

### E.10.1 Integration Points (Authoritative)
1. **Download + validate (Notebook 01):** If `ENABLE_DERIVATIVES_FEATURES=True`, the notebook downloads the derivatives ZIP archives, verifies checksums, parses each file via `src/utils.py::load_*()`, aligns each series to the spot 1-minute grid via `src/utils.py::align_to_1m_grid(..., method='ffill')`, writes `data/raw_data/derivatives/*_1m.parquet`, and records `data/raw_data/derivatives/derivatives_validation.json`.
2. **Join + featurize (Notebook 02 Stage 2.5):** The aligned derivative parquets are left-joined onto the spot 1-minute DataFrame `df` by timestamp. Derivatives base series are computed (`compute_derivatives_base_series()`), then Groups R?V features are built via the `compute_*_features()` functions.
3. **Boundary dataset contract (Notebook 02 Stage 4):** After boundary sampling, raw/base/derivatives-join columns are dropped; only engineered feature columns remain eligible for `feature_list.json` (Section 12.4).
4. **Missingness handling (Notebook 02 Stages 9?10):** Derivatives NaNs caused by limited historical coverage are expected. After warmup trimming, `undef__*` flags are created and remaining NaNs are imputed deterministically per Section 8.

### E.10.2 No-Lookahead Alignment Rule
For all non-1m sources (funding, metrics, EOHSummary, BVOL), the value attached to a spot bar timestamp `t` is the most recent observation at or before `t` (piecewise-constant backward-asof). `align_to_1m_grid()` enforces **no extrapolation** beyond source coverage by masking values outside `[min_ts, max_ts]` to `NaN`.

## E.11 Implementation Location (Non-Normative)

This appendix is fully implemented in `src/utils.py` (no external scripts required):
- URL generation: `generate_futures_klines_urls()`, `generate_funding_rate_urls()`, `generate_futures_metrics_urls()`, `generate_eoh_summary_urls()`, `generate_bvol_index_urls()`, `generate_all_derivatives_urls()`.
- Loading/parsing: `load_futures_klines()`, `load_funding_rate()`, `load_futures_metrics()`, `load_eoh_summary()`, `load_bvol_index()`.
- Alignment: `align_to_1m_grid()`.
- Feature engineering: `compute_derivatives_base_series()`, `compute_basis_features()`, `compute_flow_features()`, `compute_oi_features()`, `compute_funding_features()`, `compute_options_features()`, `compute_vol_index_features()`.

## E.12 Validation Criteria (Derivatives)

### E.12.1 Checkpoint: After Derivatives Data Loading

```python
def checkpoint_derivatives_data(
    df_futures: pd.DataFrame,
    df_funding: pd.DataFrame,
    df_metrics: pd.DataFrame,
    df_eoh: pd.DataFrame,
    df_bvol: pd.DataFrame,
    df_spot: pd.DataFrame,
) -> dict:
    """
    Validate derivatives data before feature computation.
    """
    results = {}
    
    # 1. Index alignment
    futures_aligned = df_futures.index.equals(df_spot.index)
    results['futures_index_aligned'] = {'ok': futures_aligned}
    
    # 2. Futures price correlation with spot (sanity check)
    corr = df_futures['close'].corr(df_spot['close'])
    results['futures_spot_corr'] = {'value': corr, 'ok': corr > 0.99}
    
    # 3. Funding rate range
    funding_range_ok = df_funding['funding_rate'].between(-0.1, 0.1).all()
    results['funding_range'] = {'ok': funding_range_ok}
    
    # 4. Futures metrics sanity (OI should be >= 0 when present)
    oi_ok = df_metrics['oi_usd'].dropna().ge(0).all()
    results['oi_non_negative'] = {'ok': bool(oi_ok)}

    # 5. EOH coverage (EOHSummary is limited; gaps expected outside its archive window)
    eoh_coverage = df_eoh.notna().mean().mean()
    results['eoh_coverage'] = {'value': eoh_coverage}
    
    # 6. BVOL coverage and range
    bvol_valid = df_bvol['bvol'].between(10, 300).mean()
    results['bvol_valid_frac'] = {'value': bvol_valid, 'ok': bvol_valid > 0.95}
    
    all_ok = all(r.get('ok', True) for r in results.values() if isinstance(r, dict))
    results['all_ok'] = all_ok
    
    if all_ok:
        print("✓ Derivatives data validation passed")
    else:
        print("✗ Derivatives data validation FAILED")
        for k, v in results.items():
            if isinstance(v, dict) and not v.get('ok', True):
                print(f"  - {k}: {v}")
    
    return results
```

### E.12.2 Checkpoint: After Derivatives Features

```python
def checkpoint_derivatives_features(df: pd.DataFrame) -> dict:
    """
    Validate derivatives features after computation.
    """
    results = {}
    
    # 1. Expected features exist
    expected = [
        'basis__pct__f__w0',
        'flow__taker_buy_ratio__f__w0',
        'funding__rate__f__w0',
        'opt_pcr__oi__f__w0',
        'vol_idx__bvol30d__f__w0',
    ]
    missing = [f for f in expected if f not in df.columns]
    results['features_exist'] = {'missing': missing, 'ok': len(missing) == 0}
    
    # 2. Basis reasonableness
    basis_pct = df['basis__pct__f__w0'].dropna()
    results['basis_range'] = {
        'min': basis_pct.min(),
        'max': basis_pct.max(),
        'ok': basis_pct.between(-5, 5).mean() > 0.99,  # 99% within ±5%
    }
    
    # 3. PCR reasonableness
    if 'opt_pcr__oi__f__w0' in df.columns:
        pcr = df['opt_pcr__oi__f__w0'].dropna()
        results['pcr_range'] = {
            'min': pcr.min(),
            'max': pcr.max(),
            'ok': pcr.between(0.1, 10).mean() > 0.95,
        }
    
    all_ok = all(r.get('ok', True) for r in results.values() if isinstance(r, dict))
    results['all_ok'] = all_ok
    
    if all_ok:
        print("✓ Derivatives features validation passed")
    else:
        print("✗ Derivatives features validation FAILED")
    
    return results
```

---

### E.1 Validation Criteria
- With derivatives enabled, Notebook 01 produces `data/raw_data/derivatives/*_1m.parquet` aligned to the spot 1-minute index without extrapolation (Section E.2).
- Notebook 02 Stage 2.5 produces the derivatives feature columns and `src/utils.py::checkpoint_derivatives_features()` reports `all_ok=True` under normal market conditions.

---

*Derivatives constants (toggles, windows, paths, and coverage anchors) are defined in Appendix B and must match `src/utils.py`. References are consolidated in Appendix F.*

## Appendix F: References

### F.0 Overview
**Purpose:** List external references used for API formats and financial/statistical definitions.  
**Scope:** Authoritative sources only.  
**Dependencies:** None.  
**Implementation Location:** This appendix.


### Data
- Binance Public Data: https://github.com/binance/binance-public-data
- Kline Schema: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints

### Volatility Estimators
- Parkinson (1980): DOI:10.1086/296071
- Garman-Klass (1980): DOI:10.1086/296072
- Rogers-Satchell (1991): DOI:10.1214/aoap/1177005835

### Jump Detection
- Barndorff-Nielsen & Shephard (2004): "Power and Bipower Variation" DOI:10.1111/j.1468-0262.2004.00515.x

### Permutation Entropy
- Bandt & Pompe (2002): DOI:10.1103/PhysRevLett.88.174102

### Extreme Value Theory
- Embrechts et al. (1997): *Modelling Extremal Events*. Springer.

### Liquidity Measures
- Amihud (2002): "Illiquidity and stock returns" DOI:10.1016/S1386-4181(01)00024-6

### Financial ML
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Purged Cross-Validation: https://en.wikipedia.org/wiki/Purged_cross-validation

### CatBoost
- Documentation: https://catboost.ai/docs/
- Paper: Prokhorenkova et al. (2018). "CatBoost: unbiased boosting with categorical features." NeurIPS.

---

*End of Specification*
