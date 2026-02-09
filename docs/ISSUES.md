# Issues Log

## Volatility Semivariance Ratio Definition
- Problem: Section 7.8.2 defines `semivar_ratio` as `sqrt(SV_down) / (sqrt(SV_up) + EPS)`, but the table lists `vol__semivar_ratio__f__w{W}` as `SV_down / (SV_up + EPS)`. This changes the scale of `vol__semivar_ratio__f__w{W}` materially.
- Options:
  1) Use variance ratio: `SV_down / (SV_up + EPS)`.
  2) Use semivolatility ratio: `sqrt(SV_down) / (sqrt(SV_up) + EPS)`.
- Chosen resolution: Option 2 (semivolatility ratio).
- Rationale: The preceding formula block explicitly defines `semivar_ratio` with square roots and aligns with the separately reported `vol__semivar_down__*` and `vol__semivar_up__*` (which are sqrt of semivariances). This keeps the ratio in consistent "volatility" units.
- Status: Confirmed (spec table updated to Option 2).

## Hyperparameter Search Scope (Runtime vs Exhaustive Grid)
- Problem: Section 10.2 defines a parameter grid but does not specify whether hyperparameter selection must exhaustively evaluate the full Cartesian product, which can be prohibitively slow.
- Options:
  1) Exhaustive search over the full grid with walk-forward CV.
  2) Deterministic subset of grid combinations (seeded) with walk-forward CV.
- Chosen resolution: Option 2 by default (deterministic subset), with an explicit notebook knob to force exhaustive search.
- Rationale: Keeps `03_model_training.ipynb` within the spec’s expected runtime envelope while still performing walk-forward CV selection within the declared search space.
- Status: Implemented in `notebooks/03_model_training.ipynb` (set `MAX_PARAM_COMBOS = len(grid)` to run exhaustive search).

## Binance Data Gap Repair (March 2023 Missing Bars)
- Problem: Binance monthly klines can contain rare timestamp anomalies and missing 1-minute bars (observed in `BTCUSDT-1m-2023-03.zip`: one bar with abnormal `close_time` and an ~81-minute jump in `open_time`, implying 80 missing minutes). This violates the spec’s “no gaps” validation requirement and breaks the intended 1-minute cadence.
- Options:
  1) Fail hard and require external data remediation.
  2) Enforce a complete 1-minute grid by reindexing to the expected minute index and inserting synthetic missing bars (flat OHLC at previous close, zero volumes/trades), then validate the repaired series.
- Chosen resolution: Option 2.
- Rationale: Preserves the spec’s cadence assumptions for features/labels while making the pipeline robust to rare upstream data defects. The repair is deterministic and localized to missing timestamps only.
- Status: Implemented in `notebooks/01_data_download.ipynb` after timestamp conversion; `src/utils.py` `convert_timestamps()` uses `open_time + 60s` as the canonical timestamp (equivalent under the Binance schema, more robust to rare `close_time` anomalies).

## Pre-Training NaN Check vs Optional Label Diagnostics
- Problem: The spec allows `dataset.parquet` to include label diagnostics (`m_k`, `tau_k`, `phi`) and defines `tau_k = None` when `y_k = 0` (Section 6.6). However, the Section 8A.7 `checkpoint_before_training` snippet checks NaNs after dropping only `['k','ts','y']`, which incorrectly fails whenever `tau_k` is present and NaN for negative labels.
- Options:
  1) Impute/fill `tau_k` (and/or omit label diagnostics from `dataset.parquet`), forcing “no NaNs” globally.
  2) Treat label diagnostics as non-features and exclude them from NaN checks, verifying NaNs only across model input columns.
- Chosen resolution: Option 2.
- Rationale: This matches the feature/label column contract (Section 12.4): label diagnostics are future-dependent and must never be treated as model inputs; their structural undefinedness should not block training validation.
- Status: Implemented by updating `src/utils.py` `checkpoint_before_training()` to exclude `['m_k','tau_k','phi']` from the NaN check.
