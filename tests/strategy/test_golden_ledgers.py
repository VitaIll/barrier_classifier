"""Golden-ledger pin: simulate() must reproduce the recorded corpus.

The parquet goldens under ``tests/strategy/golden/`` were recorded from the
pre-BoundaryStep simulator (2026-07-11, commit c5ca4b6) by
``scripts/generate_strategy_goldens.py``. Any implementation change that
alters a ledger row, an equity value, or a cluster record fails here.

Float columns compare at ``atol=1e-9`` (not bit-exact) so the pin holds
across platforms/libm builds; integers, strings, timestamps, and row
counts compare exactly. 1e-9 log-return units is ~5 orders of magnitude
below one basis point — far tighter than any behavioral difference.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.strategy.simulator import simulate
from tests.strategy.golden_scenarios import SCENARIOS

pytestmark = pytest.mark.strategy

GOLDEN_DIR = Path(__file__).parent / "golden"


def _assert_matches_golden(got: pd.DataFrame, golden: pd.DataFrame, name: str) -> None:
    assert list(got.columns) == list(golden.columns), (
        f"{name}: column set/order changed"
    )
    assert len(got) == len(golden), f"{name}: row count {len(got)} != {len(golden)}"
    if len(golden) == 0:
        return
    got = got.reset_index(drop=True)
    golden = golden.reset_index(drop=True)
    for col in golden.columns:
        g = golden[col]
        a = got[col]
        if pd.api.types.is_float_dtype(g):
            pd.testing.assert_series_equal(
                a, g, check_exact=False, atol=1e-9, rtol=0.0, obj=f"{name}.{col}"
            )
        else:
            # object columns may mix str and None — compare positionally.
            pd.testing.assert_series_equal(
                a.astype(object), g.astype(object), obj=f"{name}.{col}"
            )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_simulator_reproduces_golden_ledgers(scenario):
    golden_dir = GOLDEN_DIR / scenario.name
    if not golden_dir.exists():
        pytest.fail(
            f"golden corpus missing for {scenario.name!r} — run "
            "scripts/generate_strategy_goldens.py and commit the outputs"
        )
    cache, raw_bars, spec, cfg = scenario.build()
    result = simulate(cache, raw_bars, spec, config=cfg)

    _assert_matches_golden(
        result.closed, pd.read_parquet(golden_dir / "closed.parquet"), "closed"
    )
    _assert_matches_golden(
        result.equity, pd.read_parquet(golden_dir / "equity.parquet"), "equity"
    )
    _assert_matches_golden(
        result.cluster_log,
        pd.read_parquet(golden_dir / "clusters.parquet"),
        "clusters",
    )
