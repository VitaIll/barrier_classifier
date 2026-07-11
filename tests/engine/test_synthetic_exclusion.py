"""N11: gap-repair synthetic bars must not become training rows.

The store persists the ``synthetic`` flag; ``bars_frame(with_synthetic=True)``
surfaces it; ``exclude_synthetic_rows`` drops labeled rows anchored on
fabricated bars while leaving the (grid-contiguous) feature window intact.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from src.engine.domain import Bar
from src.engine.retrain import exclude_synthetic_rows
from src.engine.store import SQLiteStore

pytestmark = pytest.mark.engine

TS = pd.Timestamp("2025-06-01 12:00:00", tz="UTC")


def _bar(i: int, synthetic: bool = False) -> Bar:
    return Bar(
        ts=TS + pd.Timedelta(minutes=i), open=100.0, high=101.0,
        low=99.0, close=100.0, volume=2.0, quote_volume=200.0,
        num_trades=5.0, taker_buy_base=1.0, taker_buy_quote=100.0,
        synthetic=synthetic,
    )


class TestBarsFrameSyntheticFlag:
    def test_default_drops_flag_but_opt_in_keeps_it(self):
        store = SQLiteStore(":memory:")
        try:
            store.record_bar(_bar(0))
            store.record_bar(_bar(1, synthetic=True))
            store.record_bar(_bar(2))
            store.flush()
            default = store.bars_frame()
            assert "synthetic" not in default.columns
            kept = store.bars_frame(with_synthetic=True)
            assert "synthetic" in kept.columns
            assert kept["synthetic"].tolist() == [0, 1, 0]
        finally:
            store.close()


class TestExcludeSyntheticRows:
    def _ds(self, n: int = 6) -> pl.DataFrame:
        ts = pd.date_range("2025-06-01 12:00", periods=n, freq="1min")
        return pl.DataFrame({"ts": ts.to_numpy(), "y": [0.0] * n})

    def test_none_and_empty_are_noops(self):
        ds = self._ds()
        out, dropped = exclude_synthetic_rows(ds, None)
        assert dropped == 0 and out.height == ds.height
        out, dropped = exclude_synthetic_rows(
            ds, pd.DatetimeIndex([], tz="UTC")
        )
        assert dropped == 0 and out.height == ds.height

    def test_drops_exactly_the_synthetic_timestamps(self):
        ds = self._ds(6)
        syn = pd.DatetimeIndex(
            [
                pd.Timestamp("2025-06-01 12:01", tz="UTC"),
                pd.Timestamp("2025-06-01 12:04", tz="UTC"),
                pd.Timestamp("2025-06-01 13:00", tz="UTC"),  # outside ds
            ]
        )
        out, dropped = exclude_synthetic_rows(ds, syn)
        assert dropped == 2
        remaining = pd.DatetimeIndex(out["ts"].to_pandas())
        assert pd.Timestamp("2025-06-01 12:01") not in remaining
        assert pd.Timestamp("2025-06-01 12:04") not in remaining
        assert out.height == 4

    def test_naive_synthetic_index_also_works(self):
        ds = self._ds(4)
        syn = pd.DatetimeIndex([pd.Timestamp("2025-06-01 12:02")])
        out, dropped = exclude_synthetic_rows(ds, syn)
        assert dropped == 1 and out.height == 3
