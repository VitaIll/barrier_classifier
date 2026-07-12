---
name: environment
description: Installation, the frozen numeric stack, and the dependency-migration procedure. Use this whenever installing or setting up the project, adding/upgrading/removing ANY dependency, seeing EnvironmentDriftError or a stack-drift warning, moving to a new machine or OS, or when pip/resolver/version questions come up. Dependency changes here are revalidation events, not routine maintenance — consult this skill BEFORE touching requirements.txt.
---

# Environment, dependencies, migration

The serving contract is bit-exact ONLY on the validated environment.
`pip install` resolving something newer is how CI stayed red for days and
how a production host would silently serve the model inputs it was never
validated on. The environment is therefore enforced, not advisory.

## Install (Python 3.11 — validated on 3.11.9)

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.engine status         # first line block = validated -> installed table
python -m pytest --all -q           # full gate on a new machine
```

The DevOps skills install with the project: they live version-controlled
in `.claude/skills/` and are discovered in place (cloning IS the install
for Claude Code). For other agent surfaces,
`python scripts/package_skills.py` exports `.skill` bundles, or
`--install --target <skills-dir>` copies them; copies are snapshots — the
in-repo skills are the source of truth.

## The frozen stack (single sources of truth)

Serving-critical, mirrored in `src/engine/environment.py::VALIDATED_STACK`
— the versions the v0001 real-data validation (74,613/74,613 bit-exact)
ran under:

```
numpy==1.26.4  pandas==2.2.3  polars==1.39.3  pyarrow==19.0.1  catboost==1.2.8
```

Research/analytics determinism pins: `scipy==1.17.1`,
`scikit-learn==1.5.2`, `river==0.21.0`. Toolchain: `pytest==8.4.2`,
`ruff==0.6.9` (must equal the version the CI lint job installs).
Ranged (`>=`) is ONLY for packages with no effect on numerics
(plotly, matplotlib, jupyter, tqdm, requests, ...).

Three enforcement layers (never fight them — they are the point):

1. **Install time**: exact pins in `requirements.txt`.
2. **Test time**: `tests/engine/test_environment.py` pins requirements ↔
   `VALIDATED_STACK` ↔ the running interpreter; CI additionally fail-fast
   verifies the resolved stack before the suite.
3. **Run time**: `live --execute` raises `EnvironmentDriftError` on a
   drifted host; `--allow-stack-drift` is the deliberate override;
   dry-run/replay warn per package; `status` prints the table.

**The OS libm is part of the validated environment.** Identical package
versions still round transcendentals (log/exp family) 1 ULP apart across
platforms (Windows vs glibc, measured). Deploying on a different OS than
the one that produced the parity evidence = the same revalidation event as
a version bump.

## Migration procedure (any pinned-package change, or new OS)

1. Bump `requirements.txt` pin AND `VALIDATED_STACK` in
   `src/engine/environment.py` **together** (a test fails if they diverge).
2. Fresh install, then `python -m pytest --all -q` — 100% green required.
3. Real-data gates on the machine holding `data/`:
   `python scripts/validate_engine_replay.py` and
   `python scripts/validate_feed_chain.py`.
4. If ANY parity gate moves (bit-equality broken, ledgers differ): STOP.
   That is a model-retrain decision — the deployed model was trained and
   threshold-calibrated under the old numerics. Escalate; do not ship the
   bump alone.
5. Update this skill's version table + `CHANGELOG.md`
   (`tests/test_skills.py` fails until the skill matches the new pins).
6. `docs/PRODUCTION.md` §7 documents this contract — amend it if the
   procedure itself changes.

## Adding a NEW dependency — pick its tier first

- Touches feature values, model scoring, artifact I/O, or the live path →
  exact pin + add to `VALIDATED_STACK` + extend
  `tests/engine/test_environment.py`.
- Affects research numbers only → exact pin in the research block.
- Pure tooling/viz → ranged is fine.

Never re-loosen a pin to "fix" a resolver conflict — resolve the conflict
at a pinned version and revalidate. `pip check` must stay clean.

## Update triggers — edit THIS skill when

- Any pin in `requirements.txt` or entry in `VALIDATED_STACK` changes
  (machine-enforced: the version table above must match both).
- The enforcement mechanics change (new guard location, new CLI flag).
- The migration procedure gains/loses a step (keep §7 of PRODUCTION.md,
  this skill, and reality in agreement).
