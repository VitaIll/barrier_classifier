"""Pytest test partitioning by step.

Each test module declares which delivery step it belongs to via a
module-level marker, e.g.::

    pytestmark = pytest.mark.step4

By default ``pytest`` runs only:
  - tests marked ``framework`` (always — invariants of the engine itself)
  - tests matching the step(s) in ``tests/current_step.txt``

This keeps the per-iteration test loop fast and focused on the delivery
in flight. To run the full suite (verify nothing earlier regressed):

    pytest --all

To override the current step from the CLI without editing the file::

    CURRENT_STEP=step3 pytest          # bash / zsh
    $env:CURRENT_STEP="step3"; pytest  # PowerShell

``current_step.txt`` accepts one step per line OR comma-separated, with
``#`` for comments. Example contents to validate two steps together::

    step3,step4
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Run the full test suite, ignoring tests/current_step.txt",
    )


def _read_current_steps() -> set[str]:
    raw = os.environ.get("CURRENT_STEP", "")
    if not raw:
        path = Path(__file__).parent / "current_step.txt"
        if path.exists():
            raw = path.read_text(encoding="utf-8")
    parts: set[str] = set()
    for line in raw.replace(",", "\n").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            parts.add(line)
    return parts


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--all"):
        return

    current = _read_current_steps()
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
