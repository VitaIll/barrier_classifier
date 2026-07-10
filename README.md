# Barrier-Crossing Classifier + Live Trading Engine (BTCUSDT 1m)

Binary classification on Binance 1-minute klines — predict whether price
crosses an upward log-return barrier (`phi = 0.0025`) within the next
`M = 20` minutes — plus a **production engine** that serves the researched
strategy live: stream bars → build features → predict → decide → execute →
persist, with scheduled retraining.

Two layers, one set of contracts:

- **Research** (`notebooks/` + `src/features`, `src/analytics`, `src/strategy`)
  — the offline pipeline: dataset build, single CatBoost with
  label-uniqueness weights and purged splits, block-bootstrap evaluation,
  cluster-aware strategy calibration. `docs/MINIMAL_PROJECT_SPEC_v2.md`
  (v4.1) is the canonical spec.
- **Engine** (`src/engine`) — productionizes exactly what the research
  validated. Same feature pipeline (batch/rolling parity-tested), same
  simulator semantics (ledger parity-tested), same training procedure
  (automated with a champion/challenger gate). `docs/ENGINE.md` is the
  architectural contract.

The traded configuration is the researched **P1+P3 spec**: top-1% selective
entry (`p ≥ train-frozen q99`), let-winners-run exit (hold the TP zone while
conviction stays above the threshold), 2% lots, ≤ 50 concurrent, 5 bp cost,
no stop-loss, no time expiry.

## Quick start — research pipeline

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/01_data_download.ipynb        # raw klines + derivatives
jupyter notebook notebooks/02_build_features.ipynb       # 1-min feature dataset
jupyter notebook notebooks/03_train_model.ipynb          # single CatBoost + thresholds
jupyter notebook notebooks/04_offline_evaluation.ipynb   # scoring + VE uncertainty
jupyter notebook notebooks/05_strategy_calibration.ipynb # winning strategy + reports
```

## Quick start — engine

```bash
# 1. Package the research artifacts as model version v0001
python -m src.engine import-model

# 2. Replay a historical slice as a simulated live stream (fast batch path)
python -m src.engine replay --start 2025-09-01 --end 2025-10-01

# 3. Same stream through the true per-bar live path (rolling recompute)
python -m src.engine run --start 2025-09-01 --max-bars 60

# 4. Enable event-time scheduled retraining inside a replay
python -m src.engine replay --start 2025-09-01 --retrain-every-days 7

python -m src.engine status
```

Or programmatically:

```python
from src.engine import Engine, EngineConfig, ReplaySource

cfg = EngineConfig(feature_mode="batch")
src = ReplaySource("data/raw_data/klines_1m.parquet", start="2025-09-01")
report = Engine(cfg, source=src).run()
print(report.summary())
```

A live deployment implements the `DataSource` protocol (see
`CallbackSource` for the push-queue shape a websocket adapter targets) and
optionally a real `Broker`; everything else is unchanged. Configuration
can also come from TOML (`EngineConfig.from_toml`).

## Outputs

- Raw data: `data/raw_data/klines_1m.parquet`, `data/raw_data/derivatives/*_1m.parquet`
- Feature dataset: `data/model_dataset/dataset_1min.parquet`, `feature_list_1min.json`, `dataset_metadata_1min.json`
- Model: `data/model_dataset/catboost_model_1min.cbm` + `predictions_metadata_1min.json` (train-frozen quantiles)
- Strategy artifacts: `data/model_dataset/strategy/<spec>/…`
- Engine: `models/` (versioned registry: model + contract + thresholds + metrics per version),
  `runtime/engine.db` (SQLite session store: bars, predictions, decisions,
  orders, fills, trades, equity, matured labels, guard events, retrain runs)

## Tests

```bash
pytest              # default loop: framework invariants (see tests/current_step.txt)
pytest --all        # full suite, including slow engine parity + synthetic e2e
pytest -m engine    # fast engine tests (guards, buffer, store, ledger parity)
pytest -m engine_slow  # full-pipeline parity + synthetic end-to-end
```

The engine's correctness rests on three parity guarantees, each enforced
by tests: training/inference feature equality, rolling/batch serving
equality, and live-trader/backtest ledger equality
(`tests/engine/test_parity_simulator.py`).

## Layout

```
src/
  features/        # polars feature engine + run_pipeline / run_inference_pipeline
    families/      # per-family Feature classes
  analytics/       # bootstrap, sampling, metrics, thresholds, audits, uncertainty
  strategy/        # policy (specs/gates/sizers/exits), simulator, online stats
  engine/          # live engine: domain, guards, buffer, sources, features,
                   # model registry, strategy binding, execution, store, retrain,
                   # orchestrator + CLI (python -m src.engine)
  utils.py         # constants, Binance acquisition, splits, legacy parity oracles
notebooks/         # research orchestration (01–05)
scripts/           # strategy recalibration tools (variants sweep + winning charts)
tests/             # features/, analytics/, strategy/, engine/
docs/              # MINIMAL_PROJECT_SPEC_v2.md (research), ENGINE.md (engine)
```
