"""RawBars: first-ever coverage for kline validation and flat-bar gap repair.

These behaviors ran untested as a utils function + notebook cells despite
being causality-relevant (fabricated bars entering training was review
finding N11).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.errors import ContractError
from src.market import RawBars

pytestmark = pytest.mark.market


def _clean(n: int = 120, start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    volume = rng.gamma(2, 5, n)
    return pd.DataFrame(
        {
            "open": close + 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "taker_buy_base": volume * rng.uniform(0.0, 1.0, n),
            "taker_buy_quote": volume * close * 0.5,
            "num_trades": rng.integers(1, 100, n).astype(float),
        },
        index=idx,
    )


class TestConstruction:
    def test_requires_utc_tz_aware_index(self):
        df = _clean()
        with pytest.raises(ContractError, match="tz-aware"):
            RawBars(df.tz_localize(None))
        with pytest.raises(ContractError, match="UTC"):
            RawBars(df.tz_convert("Europe/Prague"))

    def test_requires_ohlc_columns(self):
        with pytest.raises(ContractError, match="close"):
            RawBars(_clean().drop(columns=["close"]))


class TestValidate:
    def test_clean_frame_is_valid(self):
        report = RawBars(_clean()).validate()
        assert report.is_valid and report.issues == ()
        assert report.n_rows == 120

    @pytest.mark.parametrize(
        "mutation,needle",
        [
            (lambda df: df.iloc[5, df.columns.get_loc("high")] * 0 + df["low"].iloc[5] - 1, "High < Low"),
            (lambda df: -1.0, "Negative volume"),
        ],
    )
    def test_issue_detection(self, mutation, needle):
        df = _clean()
        if needle == "High < Low":
            df.iloc[5, df.columns.get_loc("high")] = df["low"].iloc[5] - 1.0
        else:
            df.iloc[5, df.columns.get_loc("volume")] = -1.0
        report = RawBars(df).validate()
        assert not report.is_valid
        assert any(needle in issue for issue in report.issues)

    def test_gap_reported_with_locations(self):
        df = _clean().drop(index=_clean().index[40:45])
        report = RawBars(df).validate()
        assert any("Gaps detected" in i for i in report.issues)
        assert report.gap_locations

    def test_taker_exceeding_volume_flagged(self):
        df = _clean()
        df.iloc[7, df.columns.get_loc("taker_buy_base")] = (
            df["volume"].iloc[7] + 1.0
        )
        report = RawBars(df).validate()
        assert any("Taker > Volume" in i for i in report.issues)


class TestFillGaps:
    def test_flat_bars_from_previous_close_and_zero_volumes(self):
        full = _clean()
        gappy = full.drop(index=full.index[40:45])
        before = gappy.copy()
        bars, report = RawBars(gappy).fill_gaps()

        pd.testing.assert_frame_equal(gappy, before)  # input not mutated
        assert report.n_filled == 5
        assert list(report.synthetic_ts) == list(full.index[40:45])
        prev_close = full["close"].iloc[39]
        filled = bars.frame.loc[report.synthetic_ts]
        for col in ("open", "high", "low", "close"):
            assert (filled[col] == prev_close).all()
        assert (filled["volume"] == 0.0).all()
        assert (filled["num_trades"] == 0).all()
        # Repaired frame passes validation (grid complete again).
        assert bars.validate().is_valid

    def test_no_gaps_is_a_noop_with_empty_report(self):
        bars, report = RawBars(_clean()).fill_gaps()
        assert not report.any_filled
        assert len(report.synthetic_ts) == 0

    def test_leading_gap_refuses_to_fabricate(self):
        full = _clean()
        headless = full.iloc[10:]
        with pytest.raises(ContractError, match="no previous close"):
            RawBars(headless).fill_gaps(expected_index=full.index)

    def test_synthetic_ts_feed_training_exclusion(self):
        # The report's timestamps plug straight into the retrain exclusion.
        from src.engine.retrain import exclude_synthetic_rows
        import polars as pl

        full = _clean()
        gappy = full.drop(index=full.index[40:42])
        _, report = RawBars(gappy).fill_gaps()
        ds = pl.DataFrame({"ts": full.index.tz_localize(None).to_numpy()})
        out, dropped = exclude_synthetic_rows(ds, report.synthetic_ts)
        assert dropped == 2 and out.height == len(full) - 2
