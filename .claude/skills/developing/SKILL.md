---
name: developing
description: Contribution and development guide for this repo — architecture map, frozen serving invariants, coding conventions, and the definition of done. Use this BEFORE writing or refactoring any code in src/, adding a feature, changing the engine, touching the research pipeline, or when the user asks to implement, extend, restructure, or clean up anything in this codebase. Also consult it when unsure which module owns a behavior or whether something is safe to change.
---

# Developing in this repo

Two layers, one set of contracts. **Research** (`notebooks/` + `src/features`,
`src/analytics`, `src/strategy`) built and validated the model; **Engine**
(`src/engine`) serves exactly what research validated. The engine's value is
bit-exact parity with research — most "harmless" changes are not harmless here.

## Architecture map (where things live)

| Module | Owns |
|---|---|
| `src/core/` | block kernel, numerical guards — the compute substrate |
| `src/labels/` | barrier labels: spec, vectorized kernel (bit-exact vs legacy) |
| `src/market/` | `RawBars.validate()` + `fill_gaps()` (gap repair, synthetic bars) |
| `src/features/` | self-describing `Feature` classes; pipeline is a pure aggregator |
| `src/weights/` | barrier-distance, time-discount, uniqueness weight blocks |
| `src/strategy/` | `BoundaryStep`, simulator, declarative `StrategyDefinition` |
| `src/analytics/` | bootstrap, thresholds, curves, audits, fast-train |
| `src/data/` | EXTERNAL feed service: `BinanceClient`, `FeedStore`, `FeedWriter` |
| `src/engine/` | live serving: buffer, features, model registry, risk, execution, store |
| `src/utils.py` | legacy god-module — do not add to it; bodies are parity oracles |

Authoritative docs: `docs/TARGET_ARCHITECTURE.md` (phase plan + status §6),
`docs/ENGINE.md` (engine contract), `docs/PRODUCTION.md` (ops runbook),
`docs/MINIMAL_PROJECT_SPEC_v2.md` (research spec).

## Frozen invariants — changing these is a retrain event, not a refactor

The deployed model (v0001) was trained and validated against these exact
behaviors. Changing any of them silently changes model inputs; the change may
be "more correct" and still be wrong, because the model learned the old one.

1. **Feature column ORDER** of the serving frame is frozen (contract.json).
   Never reorder; new feature classes must preserve emission order.
2. **`opt_pcr__oi_chg` imputes 1.0** — a preserved legacy order bug. v0001
   trained with it. Fix only alongside a deliberate retrain.
3. **Feature primitive bodies** (`src/features/primitives.py`) — e.g.
   `log1p_vol` is `(x+1).log()`, NOT polars `log1p()`. Real-data validation
   ran with these exact kernels.
4. **pandas rolling semantics** for `ret__std` and block sums are pinned
   bitwise by oracle suites — replacing pandas with polars/numpy there is a
   conscious behavior change (deferred Phase 3.4), not a cleanup.
5. **M-grid phase**: quantile-family features emit at `row_index % M == 0`
   of the frame they see; everything must stay aligned to the model's
   `grid_anchor_ts`. See `src/engine/buffer.py` (`window_frame` head-trim).
6. **Engine ↔ simulator ledger parity by construction**: `LiveTrader`
   reuses `Portfolio` and the simulator's exit resolvers. Never fork that
   logic into a second implementation.

## Standing principles

- **Ownership**: attach data/behavior at the lowest level that naturally owns
  it (`BarrierSpec.label()`, `SimResult.summary()`, `RawBars.validate()`
  style). Outside code acts through that contract.
- **Fail typed**: raise the specific `EngineError`/`ContractError` subclass;
  never silently coerce or fill. Guard violations must be actionable from a
  log line alone.
- **polars in hot paths** (except the pinned pandas stages above); no
  per-row Python loops in feature/label kernels.
- Match the file's existing idiom, comment density, and naming. Single-char
  math names (`l`, `I`) are intentional (ruff E741 is ignored for this).

## Definition of done for any change

1. Focused theme green during the loop (see the `testing` skill), then the
   FULL suite: `python -m pytest --all -q` (~10 min, must be 100% green).
2. `ruff check src scripts` clean (CI gates on it; config in `ruff.toml`).
3. Parity gates untouched or consciously revalidated — engine changes that
   move numbers need `scripts/validate_engine_replay.py` on real data.
4. `CHANGELOG.md` entry: dated `## YYYY-MM-DD — title` section, prose intro,
   bullets naming files and the why. Match existing entries' voice.
5. Impacted skills updated per the routing table in `CLAUDE.md`
   (`tests/test_skills.py` enforces the machine-checkable part).
6. Commit style: `feat(scope): ...` / `fix(scope): ...` with an explanatory
   body; push only when the suite is green.

## Update triggers — edit THIS skill when

- A module is added/renamed/retired in `src/` (table above goes stale).
- A frozen invariant is deliberately changed (retrain event) or a new one
  is created.
- The definition-of-done steps change (new gate, new lint, new doc).
