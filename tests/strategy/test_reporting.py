"""Tests for reporting (headline table, deflated Sharpe, regime attribution)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.strategy.reporting import (
    _annualization_factor_from_cadence_minutes,
    _max_drawdown_log,
    _sharpe,
    cluster_summary,
    deflated_sharpe_ratio,
    headline_row,
    headline_table,
    regime_attribution,
)
from src.strategy.simulator import SimResult

pytestmark = pytest.mark.strategy_v1


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _mk_closed(n: int = 10, base_ret: float = 0.001, hit_frac: float = 0.6) -> pd.DataFrame:
    """Build a minimal ledger DataFrame for testing."""
    n_hits = int(hit_frac * n)
    n_misses = n - n_hits
    rows = []
    for i in range(n_hits):
        rows.append({
            "k_entry": i, "k_exit": i + 1,
            "size": 0.5,
            "gross_log_return": base_ret,
            "exit_reason": "tp",
            "regime_quantile_at_entry": 0.8,
            "p_at_entry": 0.4,
            "knowledge_unc_at_entry": 0.2,
        })
    for i in range(n_hits, n):
        rows.append({
            "k_entry": i, "k_exit": i + 1,
            "size": 0.5,
            "gross_log_return": -base_ret,
            "exit_reason": "expiry",
            "regime_quantile_at_entry": 0.5,
            "p_at_entry": 0.4,
            "knowledge_unc_at_entry": 0.5,
        })
    return pd.DataFrame(rows)


def _mk_equity(n: int = 10, slope: float = 0.0005, noise: float = 0.0002, seed: int = 0) -> pd.DataFrame:
    """Equity curve with constant drift + iid noise so Sharpe is finite."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(slope, noise, size=n)
    eq = np.cumsum(steps)
    return pd.DataFrame({
        "ts": pd.date_range("2025-06-01", periods=n, freq="20min"),
        "k": np.arange(n),
        "equity": eq,
    })


# ---------------------------------------------------------------------------
# Annualization
# ---------------------------------------------------------------------------


def test_annualization_factor_for_20min_cadence():
    expected = (365 * 24 * 60) / 20  # ≈ 26280
    assert _annualization_factor_from_cadence_minutes(20.0) == pytest.approx(expected)


def test_annualization_factor_for_1min_cadence():
    """1-min cadence sets the bar count = 365 * 24 * 60 = 525_600 per year.
    A buggy default of 20.0 would under-annualize a 1-min Sharpe by sqrt(20)."""
    expected = 365 * 24 * 60  # 525_600
    assert _annualization_factor_from_cadence_minutes(1.0) == pytest.approx(expected)


def test_annualization_factor_zero_cadence_is_one():
    assert _annualization_factor_from_cadence_minutes(0) == 1.0


def test_headline_table_reads_cadence_from_simresult_config():
    """When cadence_minutes is not passed to headline_table, it must read
    the value from each SimResult.config so 1-min and 20-min runs in the
    same table get correctly-annualized Sharpes."""
    r1 = SimResult(
        spec_name="20min_spec",
        closed=_mk_closed(n=20, base_ret=0.001, hit_frac=0.6),
        equity=_mk_equity(n=200, slope=0.0005, noise=0.0002, seed=0),
        cluster_log=pd.DataFrame(),
        diagnostics_used={},
        config={"cost_per_trade_override": 0.0005, "cadence_minutes": 20.0},
    )
    r2 = SimResult(
        spec_name="1min_spec",
        closed=_mk_closed(n=20, base_ret=0.001, hit_frac=0.6),
        equity=_mk_equity(n=200, slope=0.0005, noise=0.0002, seed=1),
        cluster_log=pd.DataFrame(),
        diagnostics_used={},
        config={"cost_per_trade_override": 0.0005, "cadence_minutes": 1.0},
    )
    df = headline_table([r1, r2])  # no cadence_minutes passed
    # Per-spec annualization should differ — 1-min run has sqrt(20)x more
    # bars per year, so its annualized Sharpe is ~ sqrt(20)x higher for
    # the same per-step distribution.
    sharpe_20 = float(df[df["spec"] == "20min_spec"]["ann_sharpe"].iloc[0])
    sharpe_1 = float(df[df["spec"] == "1min_spec"]["ann_sharpe"].iloc[0])
    # Same equity-step distribution, different annualization factor
    ratio = sharpe_1 / sharpe_20
    assert ratio == pytest.approx(math.sqrt(20.0), rel=0.10), (
        f"1-min Sharpe should be ~sqrt(20)x the 20-min Sharpe; got ratio={ratio}"
    )


# ---------------------------------------------------------------------------
# Sharpe + drawdown
# ---------------------------------------------------------------------------


def test_sharpe_zero_for_constant_returns():
    """Zero variance → NaN."""
    assert math.isnan(_sharpe(np.array([0.001, 0.001, 0.001]), annualization=252))


def test_sharpe_positive_for_positive_mean():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.001, size=1000)
    s = _sharpe(r, annualization=252)
    assert s > 0


def test_max_drawdown_zero_for_monotone_up():
    eq = np.array([0.0, 0.001, 0.002, 0.003])
    assert _max_drawdown_log(eq) == pytest.approx(0.0)


def test_max_drawdown_captures_peak_to_trough():
    eq = np.array([0.0, 0.005, 0.002, 0.004, 0.001])
    # peak = 0.005, trough = 0.001 → DD = 0.004
    assert _max_drawdown_log(eq) == pytest.approx(0.004)


# ---------------------------------------------------------------------------
# headline_row
# ---------------------------------------------------------------------------


def test_headline_row_empty_ledger():
    row = headline_row(
        "noop", pd.DataFrame(), pd.DataFrame(),
        cost_per_trade=0.0005, cadence_minutes=20.0,
    )
    assert row["n_trades"] == 0
    assert math.isnan(row["hit_rate"])
    assert row["total_log_pnl"] == 0.0


def test_headline_row_reports_expected_fields():
    closed = _mk_closed(n=10, hit_frac=0.6)
    eq = _mk_equity(n=10, slope=0.0005)
    row = headline_row(
        "test_spec", closed, eq,
        cost_per_trade=0.0001, cadence_minutes=20.0,
    )
    assert row["spec"] == "test_spec"
    assert row["n_trades"] == 10
    assert row["hit_rate"] == pytest.approx(0.6)
    assert row["tp_rate"] == pytest.approx(0.6)
    assert row["expiry_rate"] == pytest.approx(0.4)
    assert row["bulk_close_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------


def test_deflated_sharpe_handles_degenerate():
    assert math.isnan(deflated_sharpe_ratio(float("nan"), 100, n_trials=5))
    assert math.isnan(deflated_sharpe_ratio(1.0, 1, n_trials=5))


def test_deflated_sharpe_in_unit_interval_for_finite_inputs():
    p = deflated_sharpe_ratio(2.0, 1000, n_trials=10)
    assert 0.0 <= p <= 1.0


def test_deflated_sharpe_decreases_with_more_trials():
    """Smaller observed SR + more trials => DSR should drop sharply.
    Use a modest SR so the threshold matters."""
    p_5 = deflated_sharpe_ratio(0.3, 200, n_trials=5)
    p_100 = deflated_sharpe_ratio(0.3, 200, n_trials=100)
    assert p_100 < p_5


# ---------------------------------------------------------------------------
# Regime attribution
# ---------------------------------------------------------------------------


def test_regime_attribution_empty():
    out = regime_attribution(pd.DataFrame(), cost_per_trade=0.0005)
    assert out.empty


def test_regime_attribution_reports_per_tercile():
    closed = _mk_closed(n=30, hit_frac=0.5)
    closed["regime_quantile_at_entry"] = np.tile([0.1, 0.5, 0.9], 10)
    out = regime_attribution(closed, cost_per_trade=0.0001)
    # Should have 3 terciles
    assert len(out) == 3
    assert set(out["regime_tercile"].astype(str)) == {"low", "med", "high"}


# ---------------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------------


def test_cluster_summary_empty():
    assert cluster_summary(pd.DataFrame())["n_clusters"] == 0


def test_cluster_summary_counts_by_end_reason():
    df = pd.DataFrame({
        "duration_boundaries": [3, 5, 2, 4, 1],
        "n_entries": [3, 5, 2, 4, 1],
        "cluster_pnl": [0.01, -0.005, 0.002, 0.003, -0.001],
        "end_reason": ["bulk_regime", "tp", "bulk_unc", "tp", "expiry_flat"],
    })
    s = cluster_summary(df)
    assert s["n_clusters"] == 5
    assert s["n_ending_bulk_regime"] == 1
    assert s["n_ending_bulk_unc"] == 1
    assert s["n_ending_tp"] == 2
    assert s["n_ending_expiry_flat"] == 1


# ---------------------------------------------------------------------------
# headline_table (over multiple SimResults)
# ---------------------------------------------------------------------------


def test_headline_table_sorted_by_sharpe_desc():
    r1 = SimResult(
        spec_name="bad",
        closed=_mk_closed(n=20, base_ret=0.0001, hit_frac=0.3),
        equity=_mk_equity(n=200, slope=-0.00005, noise=0.0002, seed=0),
        cluster_log=pd.DataFrame(),
        diagnostics_used={},
        config={"cost_per_trade_override": 0.0005},
    )
    r2 = SimResult(
        spec_name="good",
        closed=_mk_closed(n=20, base_ret=0.002, hit_frac=0.7),
        equity=_mk_equity(n=200, slope=0.001, noise=0.0002, seed=1),
        cluster_log=pd.DataFrame(),
        diagnostics_used={},
        config={"cost_per_trade_override": 0.0005},
    )
    df = headline_table([r1, r2])
    assert df.iloc[0]["spec"] == "good"
    assert df.iloc[1]["spec"] == "bad"
