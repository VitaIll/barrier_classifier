"""Record golden simulator ledgers for the BoundaryStep re-architecture.

Runs every scenario in ``tests/strategy/golden_scenarios.py`` through the
CURRENT ``simulate()`` and writes closed/equity/cluster frames plus a
manifest to ``tests/strategy/golden/<scenario>/``. The committed outputs
are the pre-refactor truth: ``tests/strategy/test_golden_ledgers.py``
asserts any later implementation reproduces them.

Run from the repo root::

    python scripts/generate_strategy_goldens.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.strategy.simulator import simulate  # noqa: E402
from tests.strategy.golden_scenarios import SCENARIOS  # noqa: E402

GOLDEN_DIR = ROOT / "tests" / "strategy" / "golden"


def main() -> None:
    for scenario in SCENARIOS:
        cache, raw_bars, spec, cfg = scenario.build()
        result = simulate(cache, raw_bars, spec, config=cfg)
        out_dir = GOLDEN_DIR / scenario.name
        out_dir.mkdir(parents=True, exist_ok=True)
        result.closed.to_parquet(out_dir / "closed.parquet", index=False)
        result.equity.to_parquet(out_dir / "equity.parquet", index=False)
        result.cluster_log.to_parquet(out_dir / "clusters.parquet", index=False)
        manifest = {
            "scenario": scenario.name,
            "spec": spec.name,
            "n_boundaries": int(len(cache)),
            "n_closed": int(len(result.closed)),
            "n_clusters": int(len(result.cluster_log)),
            "realized_cum_final": (
                float(result.equity["realized_cum"].iloc[-1])
                if len(result.equity)
                else 0.0
            ),
            "exit_reason_counts": (
                result.closed["exit_reason"].value_counts().to_dict()
                if len(result.closed)
                else {}
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(
            f"{scenario.name}: {manifest['n_closed']} trades, "
            f"{manifest['n_clusters']} clusters, "
            f"realized={manifest['realized_cum_final']:+.6f}, "
            f"reasons={manifest['exit_reason_counts']}"
        )


if __name__ == "__main__":
    main()
