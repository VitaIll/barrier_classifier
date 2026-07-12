"""Pytest test partitioning by theme.

Tests are grouped by *theme* (a coarse subsystem partition) rather than
by individual files. Each test file gets a theme marker automatically
based on its path, via the ``pytest_collection_modifyitems`` hook below.
Legacy ``pytestmark = pytest.mark.<step>`` lines inside individual test
files are tolerated (they still attach a marker) but are no longer the
source of truth for selection — the path-based map is.

By default ``pytest`` runs only:
  - tests marked ``framework`` (always — invariants of the engine itself)
  - tests matching the theme(s) in ``tests/current_step.txt``

This keeps the per-iteration test loop focused on one subsystem. To run
the full suite (verify nothing regressed elsewhere)::

    pytest --all

To override the current theme from the CLI without editing the file::

    CURRENT_STEP=features_pipeline pytest          # bash / zsh
    $env:CURRENT_STEP="features_pipeline"; pytest  # PowerShell

``current_step.txt`` accepts one theme per line OR comma-separated, with
``#`` for comments. Example contents to validate two themes together::

    features_pipeline,analytics_bootstrap

If ``current_step.txt`` does not exist, every test runs (acts like
``--all``). If ``current_step.txt`` exists but parses to empty (only
comments / blank), pytest raises ``UsageError`` rather than silently
running everything — this prevents a malformed file from masking a
failed filter. The same applies if the ``CURRENT_STEP`` env var is set
to an empty / comment-only value.

Explicit ``-m <expr>`` invocations bypass the ``current_step.txt``
filter entirely, deferring to pytest's own marker selector. This makes
``pytest -m features_pipeline`` ergonomic for focused runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path -> theme mapping. Keys are POSIX-style paths under ``tests/``.
# Anything not mapped here falls through with whatever marker its module
# declared via ``pytestmark`` (legacy fallback). Cross-cutting framework
# tests are intentionally absent — they keep their explicit
# ``pytestmark = pytest.mark.framework`` declaration.
# ---------------------------------------------------------------------------

_THEME_BY_PATH: dict[str, str] = {
    # features_primitives
    "features/test_primitives.py": "features_primitives",
    "features/test_rolling_primitives.py": "features_primitives",
    "features/test_composite_primitives.py": "features_primitives",
    "features/test_custom_primitives.py": "features_primitives",
    "features/test_specialized_primitives.py": "features_primitives",
    # features_families
    "features/test_family_lag_rolling.py": "features_families",
    "features/test_family_volatility.py": "features_families",
    "features/test_family_step9.py": "features_families",
    "features/test_family_step10.py": "features_families",
    "features/test_family_step11.py": "features_families",
    "features/test_family_step12.py": "features_families",
    "features/test_family_round1.py": "features_families",
    "features/test_family_equilibrium.py": "features_families",
    "features/test_round1234_correctness.py": "features_families",
    # labels (barrier-label domain: spec + vectorized kernel + block)
    "labels/test_barrier.py": "labels",
    # weights (sample-weight blocks)
    "weights/test_blocks.py": "weights",
    # market (bar-series domain)
    "market/test_bars.py": "market",
    # data (external feed service)
    "data/test_feed.py": "data",
    # features_pipeline
    "features/test_boundary.py": "features_pipeline",
    "features/test_quality.py": "features_pipeline",
    "features/test_pipeline.py": "features_pipeline",
    "features/test_cadence_helpers.py": "features_pipeline",
    "features/test_feature_config.py": "features_pipeline",
    "features/test_imputation_bridge.py": "features_pipeline",
    "features/test_self_description.py": "features_pipeline",
    # analytics_bootstrap
    "analytics/test_bootstrap.py": "analytics_bootstrap",
    "analytics/test_block_bootstrap_propagation.py": "analytics_bootstrap",
    "analytics/test_metrics.py": "analytics_bootstrap",
    "analytics/test_sampling.py": "analytics_bootstrap",
    # analytics_metrics
    "analytics/test_curves.py": "analytics_metrics",
    "analytics/test_degradation.py": "analytics_metrics",
    "analytics/test_edge.py": "analytics_metrics",
    "analytics/test_thresholds.py": "analytics_metrics",
    # analytics_audits
    "analytics/test_audits.py": "analytics_audits",
    # analytics_cohorts
    "analytics/test_cohorts.py": "analytics_cohorts",
    # analytics_uncertainty
    "analytics/test_uncertainty.py": "analytics_uncertainty",
    # analytics_fast_train
    "analytics/test_fast_train.py": "analytics_fast_train",
    # strategy (entire directory)
    "strategy/test_cache.py": "strategy",
    "strategy/test_diagnostics.py": "strategy",
    "strategy/test_inventory.py": "strategy",
    "strategy/test_online.py": "strategy",
    "strategy/test_policy.py": "strategy",
    "strategy/test_reporting.py": "strategy",
    "strategy/test_simulator.py": "strategy",
    "strategy/test_golden_ledgers.py": "strategy",
    "strategy/test_definitions.py": "strategy",
    "strategy/test_result_forensics.py": "strategy",
    # engine (live engine)
    "engine/test_domain_buffer_guards.py": "engine",
    "engine/test_sources.py": "engine",
    "engine/test_store.py": "engine",
    "engine/test_registry.py": "engine",
    "engine/test_parity_simulator.py": "engine",
    "engine/test_hardening.py": "engine",
    "engine/test_synthetic_exclusion.py": "engine",
    "engine/test_binance.py": "engine",
    "engine/test_risk_controls.py": "engine",
    "engine/test_environment.py": "engine",
    "engine/test_feature_inference.py": "engine_slow",
    "engine/test_engine_e2e.py": "engine_slow",
}

_TESTS_DIR = Path(__file__).parent


def _theme_for_item(item: pytest.Item) -> str | None:
    """Return the theme marker name for ``item`` based on file path."""
    try:
        rel = Path(item.fspath).resolve().relative_to(_TESTS_DIR.resolve())
    except (ValueError, AttributeError):
        return None
    key = rel.as_posix()
    return _THEME_BY_PATH.get(key)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run the full test suite, ignoring tests/current_step.txt",
    )


def _parse_theme_text(raw: str) -> set[str]:
    parts: set[str] = set()
    for line in raw.replace(",", "\n").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            parts.add(line)
    return parts


def _read_current_themes() -> set[str]:
    """Parse the current-theme list.

    Sources, in priority order:
      1. ``CURRENT_STEP`` env var (if non-empty after stripping).
      2. ``tests/current_step.txt`` (if it exists).

    A malformed source (env var or file is *set* but parses to empty
    after stripping comments / blanks) raises ``UsageError`` — silent
    fall-through to "run everything" used to mask broken filters.

    If the env var is unset AND the file does not exist, the function
    returns an empty set (the caller treats this as "no filter — run
    everything").
    """
    env_raw = os.environ.get("CURRENT_STEP")
    if env_raw is not None:
        themes = _parse_theme_text(env_raw)
        if not themes:
            raise pytest.UsageError(
                "CURRENT_STEP env var is set but empty after parsing "
                "(comment-only or whitespace). Either unset it, or list "
                "at least one theme. Use `pytest --all` to skip the filter."
            )
        return themes

    path = _TESTS_DIR / "current_step.txt"
    if not path.exists():
        return set()

    text = path.read_text(encoding="utf-8")
    themes = _parse_theme_text(text)
    if not themes:
        raise pytest.UsageError(
            f"malformed {path.as_posix()} — comment-only or empty after "
            "parsing. Add at least one theme, delete the file to run "
            "everything by default, or use `pytest --all`."
        )
    return themes


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    # Step 1: stamp each item with its path-based theme marker BEFORE any
    # filtering. This ensures `pytest -m <theme>` works regardless of
    # what (legacy) ``pytestmark`` the individual file declared.
    for item in items:
        theme = _theme_for_item(item)
        if theme is not None:
            item.add_marker(getattr(pytest.mark, theme))

    # Step 2: optional theme filter via current_step.txt / CURRENT_STEP.
    if config.getoption("--all"):
        return

    # If the user passed an explicit ``-m`` marker expression, defer to
    # pytest's own marker filter — don't double-apply current_step.txt.
    if config.getoption("-m"):
        return

    current = _read_current_themes()
    if not current:
        return  # no filter configured -> behave like --all

    keep = current | {"framework"}
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        marks = {m.name for m in item.iter_markers()}
        if marks & keep:
            selected.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
