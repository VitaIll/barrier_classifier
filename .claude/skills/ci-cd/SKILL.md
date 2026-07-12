---
name: ci-cd
description: The GitHub Actions CI pipeline — what each job does, why the numeric stack is verified before the suite, and how to diagnose and fix a red run. Use this whenever CI fails or must be inspected, before editing .github/workflows/ci.yml, after pushing (to verify the run), when a dependency change might affect CI, or when the user mentions pipelines, workflows, actions, red builds, or "make CI green".
---

# CI/CD

One workflow: `.github/workflows/ci.yml`, on every push and PR. Two jobs;
both must be green. Runner is ubuntu (Linux) — the repo's validated dev
platform is Windows, so platform-scoped numeric caveats apply (below).

## Job anatomy

1. **`ruff (static analysis)`** — installs `ruff==0.6.9` (same pin as
   `requirements.txt`; keep the two in lockstep) and runs
   `ruff check src scripts`. Rule policy lives in `ruff.toml`: real-defect
   classes only (F, E7, E9, B, PLE), E741 ignored (quant single-char math),
   E402 allowed in `scripts/` and `src/utils.py`.
2. **`pytest (Python 3.11)`** — pip cache keyed on `requirements.txt`;
   installs the FROZEN stack; `pip check`; then a **fail-fast stack
   verification** (imports `src.engine.environment`, asserts the resolved
   environment matches `VALIDATED_STACK`, prints the version table); then
   `python -m pytest --all -q` (~6–11 min).

Hygiene already configured: per-ref `concurrency` with cancel-in-progress,
job timeouts (10/30 min), `actions/checkout@v5` + `actions/setup-python@v6`.

## Diagnosing a red run

```bash
gh run list --limit 5                                  # find the run
gh run view <run-id>                                   # which job/step failed
gh run view <run-id> --log-failed | grep -E "FAILED|Error|assert" | tail -40
gh run watch <run-id> --exit-status                    # block until done
```

Do NOT pipe `gh run watch --exit-status` through `tail`/`head` — the pipe
masks the exit code and a failure reads as success (this happened here).
Capture it explicitly: `gh run watch <id> --exit-status; echo $?`.

Failure triage by step:

| Failing step | Meaning | Fix path |
|---|---|---|
| Install dependencies | resolver conflict or yanked wheel | `environment` skill — never loosen pins to "fix" this |
| Verify installed stack | resolver produced ≠ `VALIDATED_STACK` | same — the guard did its job |
| pytest, `test_environment` | pins/manifest/interpreter disagree | bump the three together (`environment` skill) |
| pytest, `test_skills` | a skill went stale vs the repo | update the impacted skill(s), not the test |
| pytest, anything else | real regression OR platform-scoped numerics | reproduce locally; if Linux-only, suspect cross-libm ULP (see `testing` skill) — never "fix" by loosening a same-path bitwise gate |

## Change discipline

- **Dependency changes** are revalidation events — follow the `environment`
  skill procedure; CI going green is necessary, not sufficient.
- **Workflow edits**: keep the fail-fast stack verification ahead of the
  suite (a drifted resolver should fail in seconds with a version table,
  not as a buried parity cascade). Update this skill in the same change.
- **Action major bumps**: watch one full run after bumping; a bad tag fails
  at job start with "unable to resolve action".
- CI has no Binance access and no real data: exchange tests are hermetic by
  construction and the two real-data tests skip (see `testing` skill). Never
  add a test that needs network or the 4.5 GB dataset to pass on CI.

## Update triggers — edit THIS skill when

- `.github/workflows/ci.yml` changes (jobs, steps, versions, triggers) —
  `tests/test_skills.py` pins the ruff version and python version quoted
  here to the workflow file.
- The triage table gains a new failure class worth recording.
- A new quality gate joins CI (e.g. mypy per TARGET_ARCHITECTURE §7).
