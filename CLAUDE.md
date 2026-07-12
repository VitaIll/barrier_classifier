# CLAUDE.md — repo contract for AI agents

Barrier-crossing classifier (BTCUSDT 1m) + live trading engine. Research
(`notebooks/`, `src/features|analytics|strategy`) validated the model;
the engine (`src/engine`) serves it with bit-exact parity. Parity is the
product — read `.claude/skills/developing/SKILL.md` before changing code.

Canonical docs: `docs/TARGET_ARCHITECTURE.md` (plan+status §6),
`docs/ENGINE.md` (engine contract), `docs/PRODUCTION.md` (ops runbook),
`docs/MINIMAL_PROJECT_SPEC_v2.md` (research spec), `CHANGELOG.md`.

Fast commands: `python -m pytest --all -q` (full gate) ·
`ruff check src scripts` · `python -m src.engine status`.

## DevOps skill registry — the maintenance contract

The project's operating knowledge lives in six skills. **A major change is
not done until the impacted skills are updated** — same standard as tests
and CHANGELOG. `tests/test_skills.py` (framework theme, runs in every CI
build) machine-checks skill content against the repo: referenced paths must
exist, version pins must match `requirements.txt` and `VALIDATED_STACK`,
CLI commands/flags must exist in `src/engine/__main__.py`, error classes
must match `src/engine/errors.py`, themes must match `pytest.ini`, and this
registry must list exactly the skills that exist. If that test is red, a
skill went stale — fix the skill, never weaken the test.

| Skill | Governs | Update when you change |
|---|---|---|
| `developing` | architecture map, frozen invariants, definition of done | `src/` module layout, a frozen invariant, the dev workflow |
| `testing` | theme system, suites of record, numeric assertion rules, real-data gates | `pytest.ini` themes, `tests/conftest.py` routing, parity suites, validation scripts |
| `ci-cd` | workflow anatomy, red-run triage, change discipline | `.github/workflows/ci.yml`, CI gates, action/tool versions |
| `debugging` | error taxonomy, store forensics, landmine catalog | `src/engine/errors.py`, store schema, a newly confirmed landmine |
| `live-ops` | safety ladder, bring-up, config, monitoring, recovery | `src/engine/__main__.py` CLI, `EngineConfig`/risk defaults, `docs/PRODUCTION.md` procedures |
| `environment` | install, frozen stack, dependency migration | `requirements.txt` pins, `VALIDATED_STACK`, enforcement mechanics |

Skill hygiene: each skill ends with an "Update triggers" section — honor
it. Keep `.claude/skills/` free of anything but the skill directories
above (the test rejects strays). When adding a skill: create
`.claude/skills/<name>/SKILL.md`, add it to this registry, and extend
`tests/test_skills.py` with its freshness checks.

Distribution: the skills ship with the repo (cloning installs them —
Claude Code discovers them in place). `scripts/package_skills.py` exports
portable `.skill` bundles or copies them into another agent's skills
directory; such copies are snapshots and the in-repo skills remain the
source of truth.
