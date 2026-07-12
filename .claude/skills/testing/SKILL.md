---
name: testing
description: How to run, write, and reason about tests in this repo — the pytest theme system, parity/oracle suites, numeric assertion rules, and real-data validation gates. Use this whenever running tests, adding or modifying a test file, interpreting a test failure, choosing tolerances for numeric assertions, or verifying a change didn't break parity. Also read it before claiming any change is "tested".
---

# Testing

~1,140 tests guard bit-exact parity between research and the live engine.
The suite is partitioned by *theme* so the dev loop stays fast.

## Running

```bash
python -m pytest                 # framework tests + theme(s) in tests/current_step.txt
python -m pytest --all -q       # FULL suite (~10 min) — the pre-commit/CI gate
python -m pytest -m engine      # one theme explicitly (bypasses current_step.txt)
CURRENT_STEP=features_pipeline python -m pytest   # env-var override (bash)
$env:CURRENT_STEP="features_pipeline"; python -m pytest  # PowerShell
```

- `tests/current_step.txt` holds the active theme(s): one per line or
  comma-separated, `#` comments. Malformed/empty file raises UsageError
  by design (a broken filter must never silently run everything).
- `framework`-marked tests ALWAYS run regardless of the filter.

## Themes (declared in `pytest.ini`, routed in `tests/conftest.py`)

`framework`, `labels`, `weights`, `market`, `data`, `features_primitives`,
`features_families`, `features_pipeline`, `analytics_bootstrap`,
`analytics_metrics`, `analytics_audits`, `analytics_cohorts`,
`analytics_uncertainty`, `analytics_fast_train`, `strategy`, `engine`,
`engine_slow`.

**Adding a test file**: the path→theme map `_THEME_BY_PATH` in
`tests/conftest.py` is the source of truth — add your file there (a
`pytestmark = pytest.mark.<theme>` line in the file is tolerated legacy,
not the router). Root-level `tests/test_*.py` files use
`pytestmark = pytest.mark.framework`.

## Suites of record (what protects what)

- **Feature oracles** (`tests/features/`): every ported feature family is
  pinned against the legacy `src/utils.py` implementations.
- **Golden ledgers** (`tests/strategy/test_golden_ledgers.py`): simulator
  output pinned on a recorded corpus.
- **Engine parity trio** (`tests/engine/`): `test_parity_simulator`
  (LiveTrader ≡ simulate ledgers), `test_feature_inference` (serve ≡ train
  values), `test_engine_e2e` (whole loop, `engine_slow`).
- **Hardening set** (`tests/engine/test_hardening.py`): resume≡uninterrupted,
  kill-switch, degraded bars, hot-swap guard — fast and hermetic.
- **Binance adapter** (`tests/engine/test_binance.py`): hermetic (injected
  transport) — signing, retries, pagination, filters, idempotency.
- **Environment triad** (`tests/engine/test_environment.py`): requirements
  pins ≡ `VALIDATED_STACK` ≡ running interpreter; CLI refusal wiring.
- **Skills freshness** (`tests/test_skills.py`): `.claude/skills/` content
  is machine-checked against the repo — see `CLAUDE.md` routing table.

## Numeric assertion rules (hard-won)

- **Bitwise equality (`rtol=0, atol=0`) ONLY when both sides share the same
  accumulation path in the same library.** Two different libm code paths are
  never bitwise-comparable across platforms: glibc rounds `log1p`-family
  values 1 ULP differently than the validated Windows libm (this exact
  mistake made CI red on 2026-07-12). Cross-implementation transcendentals:
  `np.testing.assert_array_max_ulp(..., maxulp=2)`.
- **Sliding-window recompute vs full-frame**: polars rolling sums accumulate
  from array start, so tail-sliced recompute wobbles ~1 ULP → slice-invariance
  asserts `rtol=1e-9`, never exact.
- NaN semantics are contracts: poisoned windows must PROPAGATE NaN, imputed
  columns must be finite — assert both directions explicitly.
- Property tests over examples where possible: causality (perturbing bars
  outside a label's window never changes it), no-mutation of inputs.

## Real-data gates (not in CI — data lives only on the dev machine)

CI shows `2 skipped`: `tests/strategy/test_cache.py` and
`tests/analytics/test_audits.py` skip when the ~4.5 GB research artifacts are
absent. That is by design. Before claiming ENGINE-level parity after a
numerics-adjacent change, run on the machine that has `data/`:

```bash
python scripts/validate_engine_replay.py   # research-cache bit-equality + ledger parity
python scripts/validate_feed_chain.py      # FeedStore->FeedSource ≡ ReplaySource
```

Note: feed-chain testing needs `min_ready_rows = n_warmup + 1` (the guard
requires `> n_warmup`), and rolling mode costs ~18 s/bar — keep bar counts
tiny in anything iterative.

## Performance floors (re-measure when touching hot paths)

labels 0.115 s @ 527k rows · simulate 0.112 ms/row · rolling serve ~18 s/bar
(40,320-row buffer) · batch serve ~3.3 ms/bar.

## Update triggers — edit THIS skill when

- A theme is added/renamed in `pytest.ini` (the theme list here is
  machine-checked against it by `tests/test_skills.py`).
- A new suite of record lands (new parity gate, new golden corpus).
- A numeric-assertion convention changes or a new tolerance rule is learned.
- The real-data gate scripts change name or scope.
