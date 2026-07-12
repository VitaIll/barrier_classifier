---
name: debugging
description: Debugging playbook for this repo — the typed error taxonomy, where evidence lives (SQLite audit store, guard events, session provenance), deterministic reproduction via replay, and the catalog of known landmines (grid phase, ULP wobble, embargo gap, cross-OS libm, polars gotchas). Use this whenever investigating a failure, exception, wrong number, parity mismatch, engine halt, weird test behavior, or any "why is this happening" question about the pipeline or engine.
---

# Debugging

Every deliberate failure is TYPED (`src/engine/errors.py`, all derive from
`EngineError`) and most leave evidence in the store. Start from the type,
then the store, then reproduce deterministically.

## Error taxonomy → first move

| Error | Meaning | Check first |
|---|---|---|
| `ConfigError` | invalid/inconsistent `EngineConfig` | the TOML/flags; `EngineConfig.validate()` message says which field |
| `EnvironmentDriftError` | numeric stack ≠ validated | `python -m src.engine status` version table; `environment` skill |
| `GridError` | 1-min grid broken beyond repair (out-of-order ts, gap > `max_repair_gap`) | upstream feed continuity; `bars` table timestamps |
| `BarSchemaError` | OHLCV sanity violated, unrepairable | the offending bar in the feed store |
| `PhaseAlignmentError` | buffer phase ≠ model `grid_anchor_ts` | always a BUG (never repair at runtime): buffer construction / `window_frame` head-trim |
| `FeatureContractError` | feature row irreconcilable with the model's list | which column (message); undeclared column = membership contract; non-finite = imputation |
| `ModelArtifactError` | version dir missing/invalid | `models/` layout, `ACTIVE` pointer |
| `StoreError` | persistence failure | disk/WAL; store path collisions |
| `RetrainError` | retrain inputs broken (empty window, non-finite) | `retrain_runs` table; NOTE gate rejection is NOT an error — it's a recorded outcome |
| `ExchangeError` | transport/API failed beyond bounded retries | network, Binance status, rate limits |
| `ExecutionError` | order could not execute → engine halts entries | RUN THE RUNBOOK: `live-ops` skill failure-recovery §; reconcile before restart |

## Where evidence lives

The SQLite store (`--store`, default `runtime/engine.db`) is the audit
trail: `bars`, `predictions`, `decisions`, `orders`, `fills`, `trades`,
`equity`, `labels`, `guard_events`, `model_versions`, `sessions`,
`retrain_runs`, `meta`, `open_positions`.

- `guard_events` with severity `error` are the incident timeline.
- `sessions` records git sha + full config JSON per engine start — any trade
  is traceable to a build and configuration.
- `python -m src.engine status` prints stack table, registry versions,
  store row counts.
- Degraded bars (feature failure): engine serves `p=NaN` for the bar —
  entries suppressed, exits still resolve (fail-safe `tp_market` on TP
  touch). A streak ≥ `halt_after_feature_errors` (default 25) halts entries
  and alerts. Look for the exception in logs + `guard_events`.

## Deterministic reproduction

Replay IS the debugger — event-time stamping makes runs reproducible,
including scheduled retrains:

```bash
python -m src.engine replay --start 2025-09-01 --end 2025-10-01   # fast batch path
python -m src.engine run --start 2025-09-01 --max-bars 60          # true rolling path (~18 s/bar!)
```

Narrow with `--start/--end/--max-bars`; keep rolling-mode bar counts tiny.
Resume behavior is pinned by test: a dirty store without `--resume` is a
`ConfigError`, never a silent overwrite.

## Landmine catalog (verified, will bite again)

- **M-grid phase**: quantile-family features key off `row_index % M` of the
  frame they see. A phase-shifted buffer silently feeds the model a
  distribution it never saw. Symptom: predictions look plausible but differ
  from research. Check anchor alignment first on any serve≠train mismatch.
- **Sliding-sum ULP wobble**: tail-sliced rolling recompute ≠ full-frame by
  ~1 ULP (accumulation start differs). Not a bug; assert `rtol=1e-9`.
- **Embargo-gap artifact**: `production_1min_P1P3/closed_trades.parquet` is
  NOT a live-parity target — the research cache has a 2×1,200-row embargo
  hole where `p_map` returns NaN. Canonical target =
  `scripts/validate_engine_replay.py`.
- **Cross-OS libm**: glibc vs Windows round `log1p`-family values 1 ULP
  apart on identical package versions. Linux-only test failures on
  transcendentals are platform-scoped, not regressions (see `testing`
  skill for the assertion rule).
- **polars silent-divergence defaults**: joins don't validate cardinality,
  NaN ≠ null semantics differ from pandas, `Enum` vs `Categorical` matters
  for new Feature classes. When a number is silently wrong, diff the frame
  against the pandas oracle at each stage boundary.
- **Feed-chain warmup**: `min_ready_rows` must be `n_warmup + 1` (guard is
  strictly `>`); getting this wrong stalls the first prediction forever.
- **CatBoost early stopping firing early** is a data-prep diagnostic (bad
  window/weights), not a config bug to suppress — keep
  `early_stopping_rounds` on.

## Update triggers — edit THIS skill when

- An error class is added/removed/renamed in `src/engine/errors.py`
  (`tests/test_skills.py` checks every class is covered here).
- A new landmine is confirmed (add it — this catalog is the institutional
  memory that keeps the next debugging session short).
- Store schema changes (table list above goes stale).
