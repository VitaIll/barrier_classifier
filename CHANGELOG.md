## 2026-07-11 — Live trading: Binance adapter, execution containment, production ops

The engine's two ports get their exchange implementations, and the
operational surface for running them safely. Safety ladder is explicit
and default-on: testnet + dry-run unless `--mainnet` and `--execute` are
BOTH passed, credentials via env vars only.

- **`src/engine/binance.py`**:
  - `BinanceClient` — signed/unsigned REST with bounded retries
    (5xx/429 with Retry-After honored, exponential backoff), typed
    `ExchangeError` on rejection, injectable transport (every path
    hermetically tested; no test touches the network).
  - `BinanceKlineSource` (`DataSource`) — REST-paginated buffer backfill
    (28 days ≈ 30 requests) + closed-candle polling with automatic
    in-order gap catch-up; refuses silent catch-up beyond
    `max_backfill_bars`. Timestamp convention identical to research
    (bar-complete UTC = open_time + 60s).
  - `BinanceBroker` (`Broker`) — MARKET orders matching the researched
    strategy semantics exactly: entries buy by quote amount
    (`size × trade_capital`), closes sell the recorded base quantity
    snapped to the LOT_SIZE grid; NOTIONAL/minQty validated against the
    live exchangeInfo filters; idempotent `bc-<session>-<n>` client
    order ids (a duplicate-id rejection fetches the landed order —
    double-sends cannot double-fill); actual fill prices reported so
    slippage is measured, not assumed. `reconcile()` compares ledger
    exposure to the exchange balance.
- **Execution containment** (`engine.py`): a failed order after bounded
  retries (`ExecutionError`) records a guard event, alerts, and halts
  new entries (exits keep resolving) — ledger/exchange divergence is an
  operator decision, per the runbook. Resume now runs broker
  reconciliation before trading continues.
- **`src/engine/alerts.py`** — `AlertSink` protocol, `WebhookAlerter`
  (Slack-compatible payload, best-effort with retry, never raises into
  the loop), `NullAlerter` default. Wired at the halt choke-point
  (covers kill-switch + feature-error halts), execution failures,
  reconcile mismatches, retrain outcomes. `EngineConfig.alert_webhook_url`.
- **CLI**: `python -m src.engine live` with the three-gate posture
  (`--mainnet`, `--execute`, env credentials), `--trade-capital`,
  `--alert-webhook`.
- **docs/PRODUCTION.md** — the runbook: bring-up ladder (testnet-execute
  is mandatory), parallel-retraining operations, monitoring, failure
  modes with recovery procedures (crash/resume, execution failure, feed
  stall, kill-switch, bad publish rollback), and an explicit
  residual-risk register (strategy risk, fill quality, exchange risk,
  latency margin) — enumerated and bounded, not hand-waved away.
- 26 new hermetic adapter tests (signing, retries, pagination,
  closed-candle discipline, filter arithmetic, dry-run/live order paths,
  duplicate handling, reconciliation, alert sinks).

## 2026-07-11 — Ownership pass: behaviors attach to the objects that own them

Continuation of the self-describing-Feature principle across the other
domain objects — a behavior lives at the lowest level that naturally owns
it (`matrix.rank()`-style), and outside code acts through that contract.

- **`BarrierSpec.label(bars)` / `.uniqueness_weights(n)`**: labeling and
  overlap-weighting are behaviors of the label definition; the free
  functions remain as the kernels underneath.
- **`SimResult.composition() / .entry_local_min_rank() / .sample_trades()`**:
  the ~75 lines of trade forensics notebook 05 hand-rolled (open-book MTM
  composition, entry-quality local-min rank, representative trade
  sampling) become methods on the result object, with the market context
  (raw bars) injected by the caller. Kernels in
  `strategy/reporting.py`; composition loop vectorized with
  `searchsorted`/`np.add.at`. Tested on a golden-scenario run
  (`tests/strategy/test_result_forensics.py`).
- **`src/market/bars.py` — `RawBars`**: the 1-min bar series owns its
  integrity: `.validate()` (kline sanity report — port of
  `utils.validate_klines`, reporting instead of raising) and
  `.fill_gaps()` (deterministic flat-bar grid repair from the previous
  close, spec §4.5 — port of the notebook-01 inline cell). First-ever
  test coverage for both (they were untested despite being
  causality-relevant), including the leading-gap refusal and the
  `GapFillReport.synthetic_ts` hand-off into the retrain
  synthetic-exclusion path (review N11's research-side source of truth).

## 2026-07-11 — Self-describing Feature contract; pipeline as pure aggregator

Everything feature-level now lives ON the Feature class; the pipeline
aggregates declarations instead of owning per-feature knowledge.

- **`Domain` value objects** (`base.py`): a feature's TYPE as one
  declaration — `REAL`, `NONNEGATIVE`, `FRACTION`, `SIGNED_FRACTION`,
  `RATIO`, `BINARY`, `OSCILLATOR_0_100` — deriving the imputation fill
  (domain neutral), the expected range (domain bounds), and the legal
  operations (`LEGAL_OPS_BY_DOMAIN`: e.g. logit for fractions, log for
  ratios, winsorize for unbounded tails). 16 of the 21 explicit impute
  declarations became domain declarations; true sentinels (e.g.
  `extreme__dist_z -> 5.0`, `oi__total_usd -> median`) stay explicit.
  The imputation bridge suite pins values unchanged.
- **`depends_on(w)` — true predecessors.** Tier-2 features that read
  earlier-tier columns via f-strings now DECLARE them (equilibrium
  residual/pair interactions, pivot, extreme z-distances, flow
  absorption/exhaustion). `FeatureEngine.validate_dependencies()` runs at
  construction: a same-/later-tier reference — the polars
  `with_columns` forward-reference landmine — is now a hard error with
  the offending edge named, instead of a silent missing-column failure
  mid-transform.
- **`tags` / `effective_tags`**: categorical labels (family, tier,
  domain, windowed-ness structural; class enrichment additive) for
  selection and reporting.
- **`describe()` + `FeatureEngine.catalog()`**: the metadata surface —
  one row per emitted column with domain, range, imputation, warmup,
  predecessors, tags, legal ops.
- **Positive feature membership** (`pipeline._impute_stage`): a column is
  a feature because the engine emitted it or a boundary prefix declares
  it — never because it failed to appear on an exclusion list. A column
  with NO declared role is now a `ContractError` (under the old
  subtraction, a stray column silently became a model feature). Frame
  order — the frozen serving contract — is preserved; the bridge suite
  asserts positive == legacy subtraction exactly on a real pipeline
  frame.

## 2026-07-11 — Re-architecture Phase 3.3 + Phase 4: imputation on classes, evaluation consolidation

**Imputation lives with the features now.** The 140-line order-sensitive
regex table (`utils.get_imputation_value`) is no longer consulted by the
pipeline: registry features declare `impute_default` on their class (21
non-zero declarations added), boundary-stage columns declare theirs in
`boundary.BOUNDARY_IMPUTE_PREFIXES` next to their constructors, and
`FeatureEngine.imputation_map()` threads the registry half through the
pipeline. An UNDECLARED column is now a hard `ContractError` — the silent
`.* -> 0.0` catch-all is gone. `tests/features/test_imputation_bridge.py`
(28 tests) pinned the new resolution equal to the legacy registry for
EVERY produced column before the switch — and surfaced one latent legacy
bug preserved deliberately: `opt_pcr__oi_chg` was intended to fill 0.0 but
the `^opt_pcr__oi` pattern matched first and returned 1.0; v0001 trained
with 1.0, so parity wins until a deliberate retrain (documented on the
class).

**Phase 4 — evaluation consolidation:**
- `bootstrap.choose_indices` is public and THE resampling-precedence rule;
  the three private copies in `curves`/`edge`/`degradation` are aliases.
  `bootstrap_apply` is the canonical NaN-tolerant resample loop
  (`bootstrap_metric` now runs on it — identical draws, outputs pinned by
  the 288-test analytics suite). `wilson_interval` moved to `bootstrap`
  (generic statistic, not a drift concept); `degradation` re-exports.
- `analytics/schema.py`: the decision-cache column contract declared once;
  `edge.bootstrap_threshold_sweep` and `degradation.conditional_precision`
  validate up front with errors that name the missing columns and the
  augmenter that adds them (was a bare `KeyError` deep inside pandas).
- `SimResult.summary()` — headline metrics (Sharpe/Calmar/max-DD/
  utilization, cadence-aware annualization) as one call on the result
  object; the run's effective cost is now recorded on `SimResult.config`.

**Live-serving profile + interim optimization (Phase 3.5 groundwork):**
profiling the 40,320-row rolling call: 22.4s total = 18.4s polars
expression evaluation (intrinsic to full recompute) + ~3.5s imputation
orchestration waste (`df.schema` rebuilt per column × 3,059, per-column
null-count queries). The impute stage now snapshots the schema once and
scans null/NaN/inf counts in single engine passes: **22.4s -> 18.0s per
rolling call**, behavior identical. The remaining 18s is the
full-recompute floor — O(depth)-per-bar streaming state is the designed
Phase-3b fix.

## 2026-07-11 — Re-architecture Phase 3.2: sample-weight blocks (`src/weights/`)

`BarrierDistanceWeight`, `TimeDiscountWeight`, `TrainingWeights` — faithful
ports of the legacy `utils.compute_*_weight` functions with the
module-global `WEIGHT_*` defaults replaced by explicit frozen
configuration objects; `UniquenessWeight` bridges
`analytics.sampling.compute_uniqueness_weights` to a `BarrierSpec` so the
weighting horizon can never drift from the label definition. Numerics
bit-exact against the legacy implementations (29-test parity suite, theme
`weights`, covering every configuration axis + degenerate inputs +
argument non-mutation). The utils originals remain the oracle until their
Phase-5 retirement.

## 2026-07-11 — Re-architecture Phase 3.1: FeatureConfig injection

The feature layer's shared parameters (horizon ``M``, barrier ``phi``, all
window families) are now an immutable, validated ``FeatureConfig`` value
object injected at registry-build time — not module globals frozen at
import. Two configurations coexist in one process; a custom horizon or
window grid is a constructor call. ``DEFAULT_CONFIG`` mirrors the legacy
``src.utils`` constants exactly (bit-parity asserted at import and by the
feature oracle suites — all 301 green, values unchanged).

- **`src/features/config.py`**: `FeatureConfig` (frozen dataclass, ~40
  fields) with derived `phi`/`n_warmup`/`k_warmup` properties and
  construction-time validation, including cross-field checks (eq pair
  windows must exist in `windows_eq` — was a deep polars missing-column
  crash, now a `ConfigError` naming the inconsistency). Legacy constant
  re-exports remain for unthreaded modules (retire in Phase 5).
- **`Feature` base**: instances bind a config at construction
  (`cls(config)` from `FeatureEngine`); windows resolve via
  `windows_field = "<config field>"` with class-attribute shadowing so
  class-local static tuples keep working unchanged.
- **All 18 family modules migrated** off module-global constants
  (`_SQRT_M`-style import-time constants become per-instance
  `self.cfg.m` reads; identical floats). The six import-time-generated
  equilibrium pair classes are ONE config-driven `EqPairInteractions`
  yielding (pullback, above) interleaved per pair — preserving the frozen
  v0001 feature-list COLUMN ORDER exactly (the serving contract compares
  ordered tuples; verified explicitly).
- **`run_pipeline`/`run_inference_pipeline` accept `config=`** (default =
  production config, outputs byte-identical); `_PipelinePlan` carries it;
  every boundary-stage constant read now flows from the plan's config.
- New `tests/features/test_feature_config.py`: validation, derived
  properties, and coexistence proofs (two engines with different
  windows/M side by side, no cross-talk; sqrt(M) denominators tracking
  the injected M; pair grids from config).

## 2026-07-11 — Re-architecture Phase 2: one BoundaryStep, typed decision domain

The per-boundary trading loop now exists ONCE. ``simulate()`` (batch) and
``LiveTrader`` (streaming) are thin drivers over the shared
``src/strategy/step.BoundaryStep`` — live≡offline parity holds by
construction; the parity suite and a new recorded golden-ledger corpus pin
the extraction.

- **Golden ledger corpus** (`tests/strategy/golden/`, generated by
  `scripts/generate_strategy_goldens.py` from the pre-refactor simulator):
  4 scenarios exercising every exit path (tp, sl, expiry, tp_market,
  bulk_cluster_loss — 327 trades, 196 clusters). `test_golden_ledgers.py`
  asserts exact reproduction (floats at 1e-9 for cross-platform libm).
- **`src/strategy/step.py`**: `BoundaryStep` (the 10-step sequence,
  verbatim semantics), `TradingState` (the run's single mutable
  accumulator, LAW 5), `ClusterTracker` (replaces 5-7 loose locals
  duplicated across two files; preserves the historical
  accrue-after-entry ordering, documented), `PathIndex` (searchsorted
  span extraction). `resolve_intra_path_exits`/`resolve_expiries` moved
  here; simulator re-exports.
- **`simulate()` rewired**: columnar row access + O(log n) path spans
  replace per-boundary full-frame masks and `.iloc` — **49.9s → 6.7s
  (7.4x) on a 60k-row production-shaped scenario, identical ledgers**.
- **`LiveTrader` rewired** onto the same step; public surface unchanged
  (`portfolio`/`realized_cum` now properties over `TradingState`).
- **`ExitReason` StrEnum** (`inventory.py`): the closed reason vocabulary
  (incl. the subtle `tp_or_sl` both-barriers case), normalized at the
  `close_position` choke point — typos now raise instead of flowing into
  reports. `Portfolio.close_one` uses identity-based removal (value
  equality could match the wrong twin).
- **`src/strategy/definitions.py`**: declarative `StrategyDefinition` —
  gates/sizers/exits/bulks as `ComponentRef`s resolved through registries;
  printable, JSON round-trip, hashable `key()` for sweeps; runtime
  probability feeds injected at `build(prob_feed=...)`.
  `production_definition()` is P1+P3 as data (ledger-equivalence pinned
  against the hand-wired spec).
- **N10 fixed**: monotonic let-winners-run exit state is released on
  EVERY close path via the `on_position_closed` hook the BoundaryStep
  invokes (bulk/SL closes used to leak `p_max` entries forever).
- **N11 fixed**: gap-repair synthetic bars no longer train models —
  `bars_frame(with_synthetic=True)` surfaces the flag, the retrain job
  excludes labeled rows anchored on fabricated bars (features still see
  the contiguous grid), logged with counts;
  `RetrainPolicy.exclude_synthetic_bars` (default True) opts out.

## 2026-07-11 — Re-architecture Phase 1: kernel + labels slice + numerical guards

Full-depth review landed as `docs/REVIEW_2026-07-11.md` (evidence-ranked
findings); the binding target design as `docs/TARGET_ARCHITECTURE.md`
(domain model, 8 architectural laws, 5-phase migration plan). This change
is Phase 1 of that plan.

- **New package `src/core/`** — the block kernel: `errors` (typed taxonomy
  `CoreError` → `ConfigError`/`ContractError`/`StateError`/`BlockError`
  with block attribution), `contracts` (`ColumnSpec`/`FrameSchema` with
  structure/data validation levels; `assert_unmutated` no-mutation checker),
  `block` (`Block` template — validate in, attribute failures, verify
  provides, log one INFO line; `Pipeline` with per-stage attribution),
  `log` (namespaced `bc.*` loggers + `timed` spans), `num` (shared guards:
  `assert_all_finite`, `safe_div`, `stable_sigmoid`, `clip_exp`,
  `require_finite_scalar`, `shifted_variance`), `rng` (Generator-injection
  policy). 59 tests, theme `framework`.
- **New package `src/labels/`** — the label domain: `BarrierSpec` (frozen
  value object owning `M`/`phi`/source/stride; derives `maturity_shift` and
  `label_intervals` so weights/splits/serving stop re-deriving them),
  `barrier_label_arrays` (vectorized kernel, **bit-exact** with the legacy
  loop — same divide→log→max operation order — and memory-bounded via row
  chunks), `label_frame` + `BarrierLabeler` block. 32 tests (theme
  `labels`) including a frozen copy of the legacy loop as parity oracle,
  causality properties, and chunk-boundary invariance.
- **`construct_labels_pl` now delegates to the kernel**: 3.40s → 0.115s on
  527k 1-min rows (~30x), identical outputs (existing
  `features_pipeline` suite passes untouched). New frame-alignment guard:
  boundary/raw frames whose `ts` columns disagree at `k*bar_stride` now
  raise `ContractError` instead of silently mislabeling.
- **Numerical guards** (review §2 findings, regression-tested in
  `tests/test_numerical_guards.py`):
  - `Position`/`close_position`/`mtm_log_return` reject non-finite
    prices/sizes (`NaN <= 0` is False — a NaN take-profit used to
    construct a position that could never exit); simulator and live trader
    refuse entries when the cache `phi` is non-finite (N4).
  - `bootstrap_threshold_sweep` fails with a diagnosis on single-class
    splits instead of a NaN→int crash (N1); its Sharpe variance now uses
    shifted cumulative sums (cancellation-stable) (N5).
  - New tie-robust `quantile_buckets` replaces bare `pd.qcut` in
    `bootstrap_metrics_by_regime` and `conditional_precision` — point-mass
    regime distributions no longer crash (N2).
  - `psi` rejects empty inputs (N3); `virtual_ensemble_predictions` uses
    overflow-safe `expit` (N6); `bootstrap_brier_decomposition` and
    `bootstrap_shap_diff` now follow the package-wide NaN-tolerant
    aggregation contract (`nanquantile` + `B_effective`) (N7).

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
