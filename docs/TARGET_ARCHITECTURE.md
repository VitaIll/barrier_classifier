# Target Architecture

Binding design for the re-architecture. Companion to
[REVIEW_2026-07-11.md](REVIEW_2026-07-11.md), which records the evidence this
design answers. Legacy documents (MINIMAL_PROJECT_SPEC_v2, ENGINE.md) describe
the system we are migrating **from**; where they conflict with this document,
this document wins.

## 1. Goals

1. **A domain model ergonomic for the trader and the developer.** Every concept a
   trader reasons about — instrument, bar grid, barrier, label, signal, position,
   cluster, fill, ledger, strategy — is a named object with invariants, not a
   DataFrame column convention.
2. **Decomposable, interpretable blocks.** Feature preparation, labeling,
   weighting, model fitting, evaluation, and decisioning are composable units in
   the style of sklearn / nixtla / river: constructed from parameters, applied to
   data, introspectable (`repr`, `params()`), and serializable.
3. **Maximum speed of the core execution path.** Vectorized kernels; one
   boundary-step engine shared by backtest and live; no full-pipeline recompute
   per bar as the long-term serving model.
4. **Extensibility.** Adding a feature family, a label definition, a weighting
   scheme, a gate/sizer/exit, or a model wrapper is a single-file, single-registry
   change with a declared contract.

## 2. Architectural laws

These are the enforceable rules every module in the new tree follows. They
restate the project owner's principles as concrete mechanisms.

**LAW 1 — Strict method boundaries.** A block does exactly what its contract
declares: declared input schema → declared output schema. Undeclared columns
pass through untouched; a block never reads columns outside its declared inputs
(`Block.requires`) and never emits columns outside its declared outputs
(`Block.provides`). Validation is mechanical, at the boundary.

**LAW 2 — Failure isolation with attribution.** A failure inside a block
surfaces as `BlockError(block=<name>, stage=<i/n>, cause=<original>)`; the
composition layer never lets one block's exception escape anonymously. Errors
are typed (see §5) so callers can catch by meaning.

**LAW 3 — Contracts on every data hand-off.** Frames crossing block boundaries
are validated against `FrameSchema` objects: column presence, dtype, and (at
configurable strictness) finiteness/bounds. Schemas are declared once and
imported, never restated as string lists.

**LAW 4 — Logged key actions.** Every block application logs one
human-interpretable line at INFO through `core.log`: what ran, over how many
rows, key parameters, elapsed time, and anything dropped/trimmed/imputed with
counts. Silence is a bug; noise is a bug.

**LAW 5 — Known mutation points.** All mutation is enumerated:

| Kind | Objects | Who may mutate | How |
|---|---|---|---|
| Pass-through data (frames, arrays) | any `pl.DataFrame`/`np.ndarray` crossing a block boundary | **nobody** | transforms return new frames; test suites assert argument non-mutation |
| Accumulators | `Portfolio`, `Cluster`, `BarBuffer`, online stats (`FastVolEWMA`, …), `Ledger` | exactly one owning driver | named methods only (`update`, `open_one`, `append`) |
| Fitted state | model handles, fitted transforms | **nobody after fit** | `fit()` returns a new immutable artifact; it never mutates the learner |
| External state | stores, files, registries | store/registry objects | atomic write patterns (temp + rename), retained-on-failure buffers |

**LAW 6 — Static operations do not mutate arguments.** Any function not on an
accumulator is pure with respect to its inputs. The kernel ships
`core.contracts.assert_unmutated`, and the test convention is that every
transform's test suite includes a non-mutation check.

**LAW 7 — Stateless methods, injected state.** No module-level mutable state, no
import-time configuration reads inside compute paths, no fit-mutates-self.
Configuration objects are frozen dataclasses passed at construction. RNG enters
as `numpy.random.Generator` (or an explicit seed), never global.

**LAW 8 — Intentional resource lifetimes.** Objects owning resources (DB
connections, buffers, model handles) declare their lifetime: who creates them,
who closes them, what happens on failure. Context-manager protocol where a scope
exists; documented ownership where it spans scopes (engine session).

## 3. Domain model

The vocabulary, its invariants, and its lifetime. Objects marked ✅ exist today
and are kept (possibly moved); 🔧 exist but are reworked; ⭐ are new.

### Market layer (`src/market/`) — Phase 2+
- ⭐ `Instrument` — symbol, quote/base, fee model, tick size. Owns
  `cost_per_trade`; today that float travels loose through four configs.
- ⭐ `Clock` / `MGrid` — bar cadence, M, grid anchor, phase arithmetic, tz policy
  (UTC everywhere; naive timestamps forbidden past the ingest boundary). Replaces
  the five copies of tz-stripping/grid math.
- ✅ `Bar`, `DerivSnapshot`, `MarketUpdate` (from `engine/domain.py`) — frozen facts.
- 🔧 `BarFrame` schema — the canonical OHLCV frame contract (`FrameSchema`),
  imported by everything that consumes bars.

### Label layer (`src/labels/`) — Phase 1 (this change)
- ⭐ `BarrierSpec` — the label *definition* as a value object: horizon `M`,
  barrier `phi` (log-return), `source` (`high`/`close`), stride, optional
  downside diagnostics. Serializable; hashable; the single source of truth that
  weights, splits (label intervals), and serving contracts derive from.
- ⭐ `label_barriers(bars, spec)` — vectorized kernel (bit-exact with the legacy
  loop, 72× faster). Stateless function; no global config reads.
- ⭐ `maturity_shift(spec)` — the causal shift rule (`M // stride`), one home.

### Sample-weight layer (`src/weights/`) — Phase 3
- 🔧 `BarrierDistanceWeight`, `TimeDiscountWeight` (from `utils.py` — numerics
  kept, module-global defaults replaced by explicit config), ✅ `UniquenessWeight`
  (from `analytics/sampling.py`). All become blocks: `weight(frame) -> frame`
  with a `weight_info` result object instead of loose dicts.

### Feature layer (`src/features/`) — Phase 3 rework in place
- ✅ `Feature` classes + registry + tiered polars engine — kept.
- 🔧 Config injection: family windows/constants resolved at **registry-build
  time** from a `FeatureConfig` object, not at class-definition time from module
  globals. Two configs can coexist in one process.
- 🔧 Imputation policy moves **onto the Feature class** (`impute_value(ctx)`),
  deleting the 140-line regex table; the catch-all becomes a hard error for
  unregistered columns.
- 🔧 The frame's column roles (feature / label / raw / base) become a
  `ColumnRole` tag carried by schema metadata, deleting the private
  `_RAW_COLS/_BASE_COLS/...` set-subtraction contract.
- 🔧 Boundary stages become polars/numpy only (no pandas round-trips).

### Decision layer (`src/strategy/` + `src/engine/`) — Phase 2
- ✅ `Position`, `ClosedPosition`, `Portfolio` — kept, with finiteness invariants
  added (a NaN price/size/tp is a construction error).
- ⭐ `Cluster` — first-class overlapping-exposure bookkeeping object (today: 5-7
  loose locals duplicated in two files).
- ⭐ `ExitReason` enum (today: unchecked strings across ≥5 sites).
- ⭐ `DecisionRow` / decision-cache `FrameSchema` — the typed contract for what
  the simulator/live trader consumes (today: implicit DataFrame columns).
- 🔧 `StrategySpec` becomes **declarative**: gates/sizers/exits are parameterized
  spec objects (`GateSpec("score_above", threshold=0.545)`) resolved through a
  registry, so a spec can be printed, hashed, persisted, swept, and diffed.
  Closures disappear; per-position exit state (e.g. monotonic `p_max`) moves
  onto the position/step state where bulk-close can see it.
- ⭐ **`BoundaryStep` — the single most important new object.** One
  implementation of the per-boundary sequence (resolve path exits → bulk-close →
  expiries → compose state → gate/size → open → cluster/ledger update), consumed
  by two thin drivers: `Backtest` (vectorized pre-indexed path lookups) and
  `LiveTrader` (streaming). Live≡offline parity becomes **by construction**;
  the existing parity tests stay as regression insurance.

### Model layer (`src/models/`) — Phase 4
- 🔧 `BarrierClassifier` — CatBoost wrapper with sklearn-style surface
  (`fit(dataset) -> FittedModel`), owning Pool construction, early-stopping
  diagnostics (logged, never suppressed), and provenance capture.
  `CB_FIXED_PARAMS` and friends become a versioned `ModelParams` object.
- ✅ `ModelRegistry`, `FeatureContract` (engine) — kept; `FeatureContract`
  gains derivation from `BarrierSpec` + `FeatureConfig` instead of loose floats.

### Evaluation layer (`src/eval/`) — Phase 4
- 🔧 One bootstrap engine (`bootstrap_apply`) replacing the ~8 copied loops; a
  single `_choose_indices`; uniform NaN contract (`nanquantile` + `B_effective`).
- ⭐ Report objects (skore-style): `EvaluationReport`, `BacktestReport` with
  `.summary()`, `.to_json()`, `.plot_*()`; plotting split from computation.
- 🔧 Cache schema validated on entry everywhere (`fast_train.CACHE_REQUIRED_COLS`
  promoted to a shared `FrameSchema`).

### Data layer (`src/data/`) — Phase 5
- 🔧 Binance acquisition, checksum, grid alignment, gap repair from
  `utils.py`/notebook 01 as `Source` objects driven by one loop; gap-repaired
  bars carry their `synthetic` flag **through** retraining filters.
- ⭐ `Run`/`ArtifactStore` — typed artifact hand-off between workflow stages
  (dataset, feature list, model, prediction cache, thresholds) replacing
  filename-by-convention. Notebooks become thin: construct blocks, call, render
  report objects.

## 4. The kernel (`src/core/`) — Phase 1 (this change)

- `core/errors.py` — taxonomy: `CoreError` → `ContractError`, `ConfigError`,
  `BlockError` (wraps any failure with block attribution), `StateError`
  (lifecycle misuse). Domain packages subclass from these.
- `core/contracts.py` — `ColumnSpec` (name, dtype, nullable, finite, bounds),
  `FrameSchema` (validate at STRUCTURE or DATA strictness; compose via
  `.extend()`), `assert_unmutated` helper for the no-mutation law.
- `core/block.py` — `Block` base: frozen config, declared
  `requires`/`provides`, `apply(frame)` template that validates in, times,
  logs, attributes failures, validates out. `Pipeline` for sequential
  composition with per-stage isolation.
- `core/log.py` — `get_logger(component)`, `timed(logger, msg)` span helper;
  human-readable single-line INFO records.
- `core/num.py` — shared numerical guards: `assert_all_finite`,
  `safe_div`, `stable_sigmoid`, `clip_exp` — the standard fixes for the crash
  edges catalogued in the review.
- `core/rng.py` — `resolve_rng(seed_or_generator)`, the only sanctioned RNG entry.

## 5. Performance targets

| Path | Today | Target | Mechanism |
|---|---|---|---|
| Labels, 527k rows | 3.4 s | **< 0.1 s** (measured 0.047 s) | vectorized kernel (Phase 1) |
| Batch features, 527k rows | ~4.5 min | < 2 min | drop pandas round-trips, lazy/streaming polars where profitable (Phase 3) |
| Backtest driver overhead | O(boundaries × raw) path lookups | O(raw) total | pre-indexed intra-path spans, numpy column access (Phase 2) |
| Live serving | ~20.4 s/bar (full recompute) | < 1 s/bar interim; O(depth)/bar eventual | bounded-tail recompute now; incremental feature state (river-style) as the Phase 3 exit criterion |

## 6. Migration plan

Each phase lands green: the full test suite passes, parity gates hold, and the
phase adds its own tests. No big-bang cutover; old entry points become thin
adapters over new kernels until their consumers migrate, then die.

- **Phase 1 — kernel + labels (this change).** `src/core/`, `src/labels/`,
  `construct_labels_pl` delegating to the vectorized kernel (bit-exact),
  numerical hot-fixes for the concrete crash/corruption bugs (review §2:
  N1-N4, N6, N7), tests for all of the above.
  *Gate:* full suite green; label parity property tests; measured speedup.
- **Phase 2 — one boundary step.** `BoundaryStep` extracted; simulator and
  live trader become drivers; `Cluster`, `ExitReason`, decision-cache schema,
  declarative `StrategySpec`; O(raw) path lookup; N10/N11 fixes land here.
  *Gate:* `test_parity_simulator` + `test_engine_e2e` untouched and green;
  ledger bit-parity vs pre-refactor on a recorded scenario corpus.
- **Phase 3 — features on the kernel.** `FeatureConfig` injection, imputation
  on the class, role-tagged schemas, pandas-free boundary stages, weight blocks.
  *Gate:* feature-value parity suite (existing oracle tests) green; two
  configs coexist in one test process.
- **Phase 4 — evaluation consolidation.** One bootstrap engine, schema-checked
  caches, report objects, viz split.
  *Gate:* analytics tests green; notebook 04 renders from report objects only.
- **Phase 5 — workflow & data.** `Run`/`ArtifactStore`, `src/data/`, thin
  notebooks, `utils.py` reduced to a deprecation shim and then removed.
  *Gate:* notebooks 01-05 re-run end-to-end from the new API.

## 7. Testing strategy

- **Parity gates** (non-negotiable): live≡offline ledgers, serve≡train
  features, resume≡uninterrupted, research-cache bit-equality scripts.
- **Kernel property tests**: causality (perturbing bars beyond a label's window
  never changes it; perturbing inside does), NaN-poisoning semantics, stride
  equivalences, no-mutation assertions on every transform.
- **Numerical edge corpus**: single-class splits, tied regimes, zero/NaN/inf
  prices, empty frames — every block must fail typed or produce documented
  output, never crash bare or corrupt silently.
- CI gains ruff + mypy (Phase 2) and keeps the full suite.
