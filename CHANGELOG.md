## 2026-07-10 — Live trading engine (`src/engine/`) + repo consolidation

The researched 1-min P1+P3 strategy is now servable end-to-end: stream bars
→ build features live → predict → decide → execute → persist, with
event-time scheduled retraining. `docs/ENGINE.md` is the architectural
contract; `python -m src.engine --help` is the front door.

- **New package `src/engine/`**: `domain` (frozen types: Bar → FeatureVector
  → Prediction → Decision → Order/Fill/Trade), `guards` (grid gap-repair per
  spec §4.5, OHLC schema repair, warmup), `buffer` (preallocated M-grid
  phase-aligned trailing window), `sources` (`DataSource` protocol,
  `ReplaySource` simulated historical stream, `CallbackSource` push queue
  for exchange adapters), `features` (`FeatureContract` + batch/rolling
  serving over the research pipeline), `model` (versioned `ModelRegistry`
  with atomic ACTIVE pointer; `import-model` packages the research
  artifacts as v0001), `strategy` (`LiveTrader` — the simulator's boundary
  step, incremental; `LiveProbFeed` binds the researched let-winners-run
  exits to live predictions), `execution` (`Broker` protocol +
  `PaperBroker`), `store` (SQLite WAL working set with event-time retention),
  `retrain` (nb02+nb03 automated: asymmetric barrier-distance × uniqueness
  weights, purged split with 1,200-row embargo, research CatBoost params,
  early stopping kept ON as a diagnostic, champion/challenger gate on the
  challenger's val split, train-frozen top-q threshold refresh, atomic
  publish + hot swap), `engine` (orchestrator + `EngineConfig`/TOML + CLI).
- **`src/features/pipeline.py`**: stages extracted into shared helpers;
  new `run_inference_pipeline` — identical stages, keeps unlabeled tail
  rows, warmup-guard instead of warmup-trim, optional bounded boundary-stage
  tail slice. `run_pipeline` behavior unchanged (existing tests pass
  untouched).
- **`src/strategy/simulator.py`**: `get_intra_bars` /
  `resolve_intra_path_exits` / `resolve_expiries` promoted to public API
  (the live engine is a second consumer); underscore aliases retained.
- **Parity guarantees, tested** (`tests/engine/`, themes `engine` +
  `engine_slow`): training/inference feature equality; rolling/batch
  serving prediction equality; live-trader/backtest ledger equality
  (production spec, monotonic variant, cluster-aware spec); degraded-bar
  fail-safe exits; synthetic end-to-end with scheduled retrain + hot swap.
- **Removed dead code**: `legacy/` (boundary-cadence pipeline + 5-fold
  ensemble; archived, zero consumers), one-shot scripts
  (`update_nb05_to_winning_spec.py`, `validate_label_horizon.py`,
  `smoke_test_relabel.py`, `quick_ev_analysis.py` — horizon-sweep outputs
  remain under `data/model_dataset/horizon_sweep/`), and unused `utils.py`
  members (`walk_forward_cv`, `threshold_analysis`, `calibration_by_regime`,
  the §12.1 plotting helpers, `CB_HP_RANGES`). `compute_all_metrics` stays
  (parity oracle for `analytics.metrics`).
- **`requirements.txt`**: added `river>=0.21` (already a hard runtime dep of
  `src/strategy/simulator.py` via `DriftADWIN`).
- **`.gitignore`**: `models/`, `runtime/` (engine outputs).
- **Pre-existing test fixes**: `test_bootstrap_metric_propagates_metric_errors`
  rewritten for sklearn ≥ 1.8 (single-class ROC-AUC now returns NaN instead of
  raising — the propagation contract is tested with an explicitly raising
  metric, plus a new NaN-propagation test);
  `test_excursion_rolling_is_strictly_causal_under_masking` no longer treats
  NaN==NaN (both sides in warmup for w7200 at the probe row) as a causality
  violation.
- **Real-data validation** (`scripts/validate_engine_replay.py`, 52-day val
  window): engine predictions bit-exact vs `research_predictions_1min.parquet`
  (74,613/74,613); closed-trade ledger identical to a fresh `simulate()` of
  the winning spec (287/287 trades, realized +4.4549%, Δ=0); rolling live
  path bit-equal to batch (≈20 s features + 3 ms predict per bar).

## 2026-05-11 — Legacy consolidation

Boundary-cadence pipeline + 5-model CatBoost ensemble retired. Single
1-minute pipeline + single CatBoost model is now the only active path.

- Moved to `legacy/`: notebooks `02_feature_building.ipynb`, `02_feature_building_legacy.ipynb`, `03_model_training.ipynb`, `04_offline_study.ipynb`; scripts `analyze_wait_for_tp.py`, `production_run.py`, `sweep_thresholds_and_sizing.py`, `sweep_base_frequency.py`, `generate_offline_study_notebook.py`.
- Deleted from `data/model_dataset/`: `dataset.parquet`, `dataset_metadata.json`, `feature_list.json`, `research_predictions.parquet`, `catboost_model.cbm` + `.0..4.cbm` (5-fold ensemble).
- Deleted from `src/utils.py`: `CatBoostEnsemble`, `checkpoint_weights`, `N_ENSEMBLE_MODELS`, `ENABLE_HPO`, `OPTUNA_N_TRIALS`, `OPTUNA_SEED`, `HPO_DROP_OLDEST_FRAC` — no remaining consumers.
- Deleted: `docs/ISSUES.md` (all entries marked Confirmed/Implemented).
- `README.md`: rewritten from scratch around the single 1-min pipeline. No more LEGACY banner; the README is the source-of-truth quick-start again.
- Canonical run order is now: `01_data_download.ipynb` → `scripts/build_1min_dataset.py` → `scripts/train_1min_model.py` → `scripts/precompute_train_scores_and_unc.py` → `scripts/run_strategy_1min.py` → `scripts/generate_strategy_calibration_notebook.py` → `notebooks/05_strategy_calibration.ipynb`.

## 2026-05-11 — Audit cleanup

Swarm-audit driven cleanup landed alongside the overlapping-target refactor.

- `pytest.ini`: replaced legacy `step2`…`step15` / `analytics_phaseN` / `strategy_v1` markers with theme markers (`framework`, `features_primitives`, `features_families`, `features_pipeline`, `analytics_bootstrap`, `analytics_metrics`, `analytics_audits`, `analytics_cohorts`, `analytics_uncertainty`, `analytics_fast_train`, `strategy`); silenced `PytestUnknownMarkWarning` for the legacy `pytestmark` lines still in test files.
- `tests/conftest.py`: rewrote partition logic. Themes are stamped on items by a path map (`_THEME_BY_PATH`) inside `pytest_collection_modifyitems` *before* filtering, so `pytest -m <theme>` works regardless of the legacy `pytestmark` line. Default behaviour: malformed `tests/current_step.txt` (comment-only / empty after parsing) raises `pytest.UsageError` instead of silently running everything; same for an empty `CURRENT_STEP` env var. Missing file still falls back to "run all". Explicit `-m` bypasses the file filter.
- `tests/current_step.txt`: replaced the long `analytics_phase0..6, strategy_v1` list with `framework` only — the default daily loop now runs framework invariants; full coverage is `pytest --all`.
- `requirements.txt`: added `scipy>=1.10` (used by `src/analytics/audits.py`, `src/analytics/degradation.py`, `src/strategy/reporting.py`).
- `.gitignore`: added `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `*.egg-info/`.
- `.github/workflows/ci.yml`: new minimal CI — push + pull_request on every branch, Python 3.11, install from `requirements.txt`, run `pytest --all -q`. No deploy step, no caching.
- `docs/MINIMAL_PROJECT_SPEC_v2.md`: bumped to v4.1; removed the legacy banner; updated every `M = 10` / "10-minute" reference to `M = 20` / "20-minute" to match `src/utils.py:M`; added Section 6.9 "1-Minute Overlapping-Target Cadence" covering `barrier_source`, `bar_stride`, label-uniqueness weights, purged splits + embargo, block bootstrap, simulator ordering, and cluster-aware sizing — pointing to `src/features/boundary.py:construct_labels_pl`, `src/analytics/sampling.py`, `src/analytics/bootstrap.py`, and `src/strategy/simulator.py` as canonical implementations.
- `docs/ISSUES.md`: deleted — every recorded issue (semivariance ratio, HPO scope, Binance gap repair, pre-training NaN check) carried a "Confirmed/Implemented" status and is fully reflected in `src/utils.py` and the notebooks.
- `src/features/__init__.py`: dropped the `Validator` re-export — it has no remaining caller in the repo and `src/features/observability.py` no longer defines it.

---

> **LEGACY — DO NOT USE AS SOURCE OF TRUTH.** This document predates the ongoing refactor and may not reflect current code. Kept for historical reference only.

# Changelog

## 2026-01-01
- Add optional HPO train truncation (drop oldest fraction) and explicit per-trial fold count in Optuna walk-forward CV, plus a tqdm fallback progress bar.
- Set CatBoost `border_count=128`, `thread_count=-1`, and `allow_writing_files=False` to reduce HPO runtime (notably on OneDrive-backed paths).
- Add Optuna NSGA-II multi-objective search (logloss, PR-AUC) with ordered CatBoost Pools and per-trial seed variation.
- Add Optuna Pareto/importance visualizations and learning curve plotting outputs.
- Add prediction-based feature importance plot and Pool-based evaluation outputs with best params/iteration saved.
- Add Optuna/plotly/kaleido dependencies and shared hyperparameter constants in utils.
- Add post-save target visualizations (return distribution and time series markers) in the feature building notebook.
