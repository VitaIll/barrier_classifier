# Live Trading Engine (`src/engine/`)

**Status:** implemented — this document is the architectural contract.
**Scope:** productionize the researched 1-minute overlapping-target strategy into a
live engine: stream bars → build features → predict → decide → execute → persist,
with scheduled retraining. The research pipeline (notebooks + `src/features`,
`src/analytics`, `src/strategy`) stays the source of truth for *what* is computed;
the engine adds *when/how* it runs continuously.

---

## 1. Domain model

One vocabulary, used everywhere (types in `src/engine/domain.py`, all frozen):

```
MarketUpdate ─▶ Bar (+ DerivSnapshot) ─▶ FeatureVector ─▶ Prediction ─▶ Decision
                                                                          │
   Trade ◀─ Fill ◀─ Order ◀──────────────────────────────────────────────┘
```

- `Bar` — one *closed* 1-minute kline. `ts` is the bar-complete timestamp
  (`open_time + 60s`), matching the research convention. Bars are facts;
  they are never mutated.
- `DerivSnapshot` — last-known derivatives state as of `ts` (futures kline,
  funding, OI, options, BVOL). The engine forward-fills internally, mirroring
  `utils.align_to_1m_grid`. All fields optional (`None` = source dark).
- `MarketUpdate` — `Bar` + optional `DerivSnapshot`: the unit a `DataSource` emits.
- `FeatureVector` — `ts` + float64 array aligned to a `FeatureContract` (exact
  ordered `feature_list` of the active model version). Never a dict — order is
  part of the contract.
- `Prediction` — `ts, p, model_version, latency_ms`.
- `Decision` — what the strategy chose (`enter/hold/skip` + size + gate trace).
- `Order / Fill / Trade` — execution lifecycle. `Trade` mirrors the offline
  simulator's closed-position ledger row so research reporting works unchanged.

`EngineEvent`s (`BarIngested`, `GuardTripped`, `PredictionMade`, `OrderFilled`,
`TradeClosed`, `RetrainCompleted`, …) are emitted on an observer bus for
dashboards/logging; they are not control flow.

## 2. Public interfaces (ports)

Three `typing.Protocol`s define the engine boundary (`runtime_checkable`):

- `DataSource` — `stream() -> Iterator[MarketUpdate]` plus `bootstrap(n) ->
  pl.DataFrame` (historical prefix to pre-fill the warmup buffer).
  Implementations: `ReplaySource` (historical parquet replayed as a simulated
  live stream — the definition-of-done source), `CallbackSource` (push-based
  shape for a websocket adapter; a Binance adapter only has to translate
  exchange payloads into `MarketUpdate`s).
- `Broker` — `submit(order) -> Fill | None`, `resolve_exits(bar, ctx) ->
  list[Trade]`, `positions() -> tuple[Position, ...]`. Implementation:
  `PaperBroker`, whose fill semantics are **defined as** "identical to
  `src/strategy/simulator.py`" and enforced by a parity test.
- `Store` — typed `record_*` methods + `flush()`. Implementation:
  `SQLiteStore` (WAL, batched writes, event-time retention pruning) — the
  "temporary fast database". `:memory:` works for tests.

## 3. Dataflow per closed bar (the only loop)

```
source ─▶ GridGuard ─▶ BarSchemaGuard ─▶ BarBuffer.append
                                             │ (warm?)
                     FeatureService.latest ◀─┘
                     ContractGuard ─▶ Model.predict ─▶ p(t)
        Broker.resolve_exits(bar t, p(t))          # exits FIRST (simulator ordering)
        LiveStrategy.decide(State(t))              # gates → sizer → Decision
        Broker.submit(entry order)                 # fills at close[t]
        Store.record_*(…) ; matured-label bookkeeping (y(t−M))
        Retrainer.on_bar(ts)                       # event-time schedule
```

Ordering contract (same as the simulator): exits for the just-closed bar are
resolved **before** the entry decision, and both use the prediction computed at
that bar's close. Matured labels (`y(t−M)`, computable from the buffer alone)
feed monitoring and online stats strictly after the decision.

## 4. Feature computation: batch/rolling duality

The research pipeline (`src/features/pipeline.py`) is the single implementation.
It gains an inference entry point (`run_inference_pipeline`) that reuses every
stage but (a) keeps tail rows whose labels are not yet mature, (b) replaces the
warmup *trim* with a warmup *guard*, and (c) optionally restricts the
boundary-stage computation to a tail slice (all boundary-stage lookbacks are
≤ ~2,920 rows; default slice 6,000).

- `BatchFeatureService` — precomputes the full frame once (replay/retrain path;
  bit-identical to the research dataset by construction).
- `RollingFeatureService` — recomputes on the trailing `BarBuffer` each bar and
  takes the last row (true-live path).

Anti-skew guards (why this is safe):

1. **Warmup**: no prediction until the buffer holds `capacity_rows`
   (default 64,800 = 45 days: covers the 20,160-bar spot windows, the
   43,200-bar BVOL window, and EWMA burn-in; see §8).
2. **M-grid phase**: boundary-sparse kernels (quantile family) key off
   `row_index % M` of the frame they see. The buffer only trims in multiples
   of `M` and anchors its phase to the model's `grid_anchor_ts` (recorded in
   the `FeatureContract` at training time). Misalignment is a hard error.
3. **Contract reconciliation**: after imputation, the live row is reconciled
   against `feature_list` — the contract's features are selected *in order*;
   a missing contract feature or a non-finite value after imputation is a
   hard `FeatureContractError` (pipeline/contract version drift, never a
   runtime repair). In the rolling path any pipeline exception is wrapped
   into the same typed error, so one bad window degrades that bar (exits
   still resolve, entry suppressed) instead of crashing the session;
   repeated failures trip the halt kill-switch.
4. **Recursive features** (EWMA/RSI) converge rather than truncate exactly;
   buffer burn-in bounds the error (documented in §8). The batch/rolling
   parity test quantifies residual skew (`tests/engine/test_feature_inference.py`).

## 5. Model service and scheduled retraining

- `ModelRegistry` — versioned directory store (`models/v{N}/`): `model.cbm`,
  `contract.json` (FeatureContract incl. grid anchor), `thresholds.json`
  (`p_threshold` = train-frozen top-q, `train_p_quantiles`), `metrics.json`,
  `training_meta.json`. `registry.active()` returns the live `ModelHandle`;
  publishing is atomic (write dir + repoint `ACTIVE` file). The existing
  research artifacts import as `v1` via `import_research_artifacts()`.
- `Retrainer` — event-time scheduled (`every` bars/days on the *stream clock*,
  so replay and live behave identically and deterministically). A retrain run
  reproduces the researched procedure exactly:
  1. rebuild dataset via `run_pipeline` (training mode) on the trailing window;
  2. `chronological_split_with_embargo` (70/15/15, embargo 1,200 rows);
  3. weights = barrier-distance asymmetric × label-uniqueness (`normalize=False`);
  4. single CatBoost, research hyperparameters, `early_stopping_rounds` **on**
     (early firing is logged as a data-quality diagnostic, never suppressed);
  5. validation gate vs the incumbent (block-bootstrap logloss/PR-AUC,
     `block_size=M`) — fail ⇒ keep incumbent, record the run;
  6. re-derive `p_threshold` from the *new* training slice (the walk-forward
     threshold refresh nb05 called out as missing);
  7. publish; the engine hot-swaps at the next bar boundary — after a
     serving-compatibility check (same M, same raw schema, anchor congruent
     with the buffer grid, warmup fits the session's ready window; batch
     mode additionally requires an identical feature list). An incompatible
     (manually published) version is guard-evented and the incumbent kept.
  Runs on a worker thread; the trading loop never blocks. Retraining reads
  its window from the store's `bars` table (spot columns) — the
  derivatives-enabled contract would need a parquet-backed window instead,
  which is deliberately out of scope while the production contract is
  spot-only.

## 6. Strategy (the researched "strict" spec, live)

`LiveStrategy` composes the same `State` the simulator composes (same online
stats: `RollingQuantileRank`, `FastVolEWMA`, `RollingRegimeBaseRate` from
`src/strategy/online.py`) and evaluates an unmodified `StrategySpec`.

Default spec = the winning **P1+P3** configuration
(`production_1min_P1P3`): entry `p ≥ p_threshold` (train-frozen top-1%
quantile), exit `make_exit_let_winners_run(hold_threshold=p_threshold,
sl_log_return=None)`, constant 0.02 lots, ≤ 50 concurrent, 5 bp round-trip
cost, no stop-loss, no time expiry. The live `p_map` is a `LiveProbFeed`
updated each bar before exit resolution — semantics identical to the
precomputed series the backtest used. The stricter monotonic exit variant is
available as `exit_variant="monotonic"`.

## 7. Persistence (`SQLiteStore`)

Tables: `bars`, `predictions`, `decisions`, `orders`, `fills`, `trades`,
`equity`, `labels` (matured), `model_versions`, `retrain_runs`,
`guard_events`, `open_positions` (live-inventory snapshot for resume),
`meta`. Single writer (engine thread), WAL, `synchronous=NORMAL`,
`busy_timeout=5s`, `executemany` batches flushed per bar and on close;
readers (retrain thread, dashboards) use separate connections. Flushes are
atomic: on failure the transaction rolls back and buffered rows are retained
for retry (`StoreError`) — never partially committed or silently dropped.
Event-time retention pruning (`retention_days`) keeps the DB a bounded, fast
working set (audit tables — trades, orders, fills, guard events — are never
pruned); long-term truth stays in parquet + the model registry.

## 8. Numerics & performance (measured, real 2025 data)

- Hot path is the existing vectorized polars/numpy pipeline — no per-feature
  Python loops were added. Measured on the production contract
  (1,511 spot features, 40,320-row buffer, `scripts/validate_engine_replay.py`):
  **rolling live path ≈ 20.4 s features + 2.7 ms predict per bar** (max 21.4 s)
  — a third of the 60 s bar interval; **batch replay ≈ 3.3 ms/bar** end to end
  including the one-off precompute (74,613 bars in 248 s).
- EWMA truncation on the 40,320-row buffer: worst production half-life 480 ⇒
  relative error ~0.5^84 (zero at float64); the trend family's span-4,320 EMA
  ⇒ ~8e-9 of the seed delta. Measured end-to-end on real data:
  **rolling vs batch max|Δp| = 0.0** — no CatBoost split ever flipped.
- Guards are O(1) float checks per bar; the buffer is a preallocated numpy
  sliding array with O(1) amortized append.

## 9. Configuration & ergonomics

```python
from src.engine import Engine, EngineConfig, ReplaySource

cfg = EngineConfig(model_dir="models", store_path="runtime/engine.db")
src = ReplaySource.from_parquet("data/raw_data", start="2025-09-01", end="2025-10-01")
report = Engine(cfg, source=src).run()
print(report.summary())
```

`EngineConfig` is a plain dataclass with `from_toml()` (stdlib `tomllib`);
every field and the `[retrain]` table are validated on load (unknown keys,
bad types, and out-of-range values are `ConfigError`s). CLI:
`python -m src.engine import-model|replay|run|feed|live|status`. Every knob
has a research-faithful default; nothing requires editing source to operate.

The numeric stack is part of the serving contract:
`src/engine/environment.py` holds the validated versions
(mirroring the `requirements.txt` pins, tied together by test), `status`
reports installed-vs-validated, replay/dry-run warn on drift, and
`live --execute` refuses to arm on a drifted host unless
`--allow-stack-drift` is passed deliberately (docs/PRODUCTION.md §7).

### Operations: kill-switch, shutdown, resume

- **Risk kill-switch** — `max_drawdown` (peak-to-now total-equity drop,
  log-return units) and `max_cumulative_loss` (total-equity floor) trip a
  session **halt**: entries are suppressed for the rest of the session while
  open positions keep resolving through the researched exit policy. The same
  halt fires after `halt_after_feature_errors` consecutive-session feature
  failures. Every halt is logged, guard-evented, and carried in
  `SessionReport.halt_reason`. Both risk limits default to `None`
  (disabled) so the engine reproduces the offline simulator exactly.
- **Clean shutdown** — `Engine` is a context manager (`close()` is
  idempotent); the CLI translates SIGTERM/Ctrl-C into a drained store and
  exit code 130.
- **Crash-safe resume** — the store keeps an `open_positions` snapshot
  (rewritten atomically whenever inventory changes) plus the equity curve
  and counters. Restarting with `resume=True` (CLI `--resume`) restores the
  open portfolio, realized P&L, trade/order counters, and the drawdown peak;
  a fresh engine over a non-empty store *without* `resume` is a
  `ConfigError` (protects the audit trail from silent overwrite). Online
  monitoring stats (quantile ranks, base rates) re-warm from the stream —
  the production spec's decisions do not consume them. Downtime between the
  last stored bar and the first streamed bar is guard-evented (exits during
  the gap were not observed).
- **Determinism** — replays are byte-reproducible: bookkeeping rows are
  stamped with event time (never wall clock), retrains carry a fixed
  `random_seed` (its presence is asserted), and event-time scheduling makes
  retrain triggers replay-identical.

## 10. Definition of done (verified 2026-07-10)

1. `pytest --all` green, including: training/inference feature parity,
   rolling/batch serving parity, live-trader vs `simulate()` ledger parity
   (three specs), store/registry/retrainer unit tests, and a synthetic
   end-to-end replay with a scheduled retrain + hot swap.
2. Real-data validation (`scripts/validate_engine_replay.py`, full 52-day
   val window): **74,613/74,613 predictions bit-exact** vs
   `research_predictions_1min.parquet`; **287/287 closed trades identical**
   to a fresh `simulate()` of the winning spec (realized +4.4549% log-return,
   Δ = 0.0); rolling live path bit-equal to batch; zero guard events.
   (The published `production_1min_P1P3` ledger is not the comparison target:
   it spans the val|test embargo gap where the research cache has no
   predictions — an evaluation artifact a live engine correctly doesn't have.)
3. A replay session exercises the full loop: guards, features, predictions,
   fills, persistence, event-time retraining, and model hot-swap.
