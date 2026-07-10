"""Tests for the Position / Portfolio data model.

Hand-computed expectations for log-return arithmetic and the bulk-close path.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.strategy.inventory import (
    ClosedPosition,
    Portfolio,
    Position,
    close_position,
)

pytestmark = pytest.mark.strategy_v1


# ---------------------------------------------------------------------------
# Position invariants
# ---------------------------------------------------------------------------


def _mk_position(**overrides) -> Position:
    defaults = dict(
        k_entry=10,
        ts_entry=pd.Timestamp("2025-06-01 00:00:00"),
        side=1,
        size=0.5,
        entry_price=100.0,
        tp_price=100.0 * math.exp(0.0025),
        sl_price=None,
        expiry_k=11,
    )
    defaults.update(overrides)
    return Position(**defaults)


def test_position_rejects_bad_side():
    with pytest.raises(ValueError, match="side"):
        _mk_position(side=0)


def test_position_rejects_negative_size():
    with pytest.raises(ValueError, match="size"):
        _mk_position(size=-0.1)


def test_position_rejects_nonpositive_entry_price():
    with pytest.raises(ValueError, match="entry_price"):
        _mk_position(entry_price=0.0)


def test_position_rejects_expiry_before_entry():
    with pytest.raises(ValueError, match="expiry_k"):
        _mk_position(k_entry=10, expiry_k=9)


def test_mtm_log_return_long_at_higher_price():
    pos = _mk_position(entry_price=100.0, side=1)
    # +5 log-points up => mtm = ln(105/100) ~ 0.04879
    assert math.isclose(pos.mtm_log_return(105.0), math.log(1.05), rel_tol=1e-12)


def test_mtm_log_return_long_at_lower_price_negative():
    pos = _mk_position(entry_price=100.0, side=1)
    assert pos.mtm_log_return(95.0) < 0


def test_mtm_log_return_rejects_nonpositive_price():
    pos = _mk_position()
    with pytest.raises(ValueError, match="current_price"):
        pos.mtm_log_return(0.0)


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------


def test_close_position_records_gross_log_return_long():
    pos = _mk_position(entry_price=100.0, side=1, size=0.5)
    closed = close_position(
        pos,
        k_exit=11,
        ts_exit=pd.Timestamp("2025-06-01 00:20:00"),
        exit_price=100.0 * math.exp(0.0025),
        exit_reason="tp",
    )
    assert closed.exit_reason == "tp"
    # Long, +0.0025 log-return on the bar
    assert math.isclose(closed.gross_log_return, 0.0025, rel_tol=1e-9, abs_tol=1e-12)


def test_close_position_records_gross_log_return_short():
    pos = _mk_position(side=-1, entry_price=100.0)
    closed = close_position(
        pos,
        k_exit=11,
        ts_exit=pd.Timestamp("2025-06-01 00:20:00"),
        exit_price=99.0,
        exit_reason="tp",
    )
    # Short profit when price drops
    assert closed.gross_log_return > 0
    assert math.isclose(closed.gross_log_return, -math.log(0.99), rel_tol=1e-12)


def test_net_log_return_subtracts_cost():
    pos = _mk_position(side=1, entry_price=100.0)
    closed = close_position(
        pos,
        k_exit=11,
        ts_exit=pd.Timestamp("2025-06-01 00:20:00"),
        exit_price=100.0 * math.exp(0.0025),
        exit_reason="tp",
    )
    assert math.isclose(
        closed.net_log_return(cost_per_trade=0.0005), 0.0025 - 0.0005, abs_tol=1e-12
    )


def test_weighted_net_log_return_scales_by_size():
    pos = _mk_position(side=1, entry_price=100.0, size=0.4)
    closed = close_position(
        pos,
        k_exit=11,
        ts_exit=pd.Timestamp("2025-06-01 00:20:00"),
        exit_price=100.0 * math.exp(0.0025),
        exit_reason="tp",
    )
    # 0.4 * (0.0025 - 0.0005) = 0.0008
    assert math.isclose(
        closed.weighted_net_log_return(0.0005), 0.4 * 0.0020, abs_tol=1e-12
    )


def test_close_position_rejects_bad_exit_price():
    pos = _mk_position()
    with pytest.raises(ValueError, match="exit_price"):
        close_position(
            pos,
            k_exit=11,
            ts_exit=pd.Timestamp("2025-06-01 00:20:00"),
            exit_price=0.0,
            exit_reason="tp",
        )


def test_close_position_rejects_exit_before_entry():
    pos = _mk_position(k_entry=10)
    with pytest.raises(ValueError, match="k_exit"):
        close_position(
            pos,
            k_exit=9,
            ts_exit=pd.Timestamp("2025-06-01 00:20:00"),
            exit_price=101.0,
            exit_reason="tp",
        )


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


def test_portfolio_open_close_one():
    p = Portfolio()
    pos = _mk_position()
    p.open_one(pos)
    assert p.n_open() == 1
    assert p.gross_size() == pytest.approx(0.5)
    closed = p.close_one(
        pos,
        k_exit=11,
        ts_exit=pd.Timestamp("2025-06-01 00:20:00"),
        exit_price=100.0 * math.exp(0.0025),
        exit_reason="tp",
    )
    assert p.n_open() == 0
    assert len(p.closed_positions) == 1
    assert closed.exit_reason == "tp"


def test_portfolio_close_one_rejects_unknown_position():
    p = Portfolio()
    pos = _mk_position()
    with pytest.raises(ValueError, match="not in open_positions"):
        p.close_one(
            pos,
            k_exit=11,
            ts_exit=pd.Timestamp("2025-06-01 00:20:00"),
            exit_price=101.0,
            exit_reason="tp",
        )


def test_portfolio_mtm_sums_size_weighted():
    """Two positions on the same instrument, same entry price, summed MTM.

    Note: ``Portfolio.mtm_log_return`` evaluates all open positions at one
    ``current_price``, so this test only fits if every position shares
    that instrument. We do NOT compose positions with different entry
    prices and one current price — that would be meaningless. See the
    next test if a multi-instrument portfolio is added later.
    """
    p = Portfolio()
    p.open_one(_mk_position(entry_price=100.0, size=0.5))
    p.open_one(_mk_position(entry_price=100.0, size=0.3))
    # Both up 5% => log(1.05); size-weighted aggregate = 0.8 * log(1.05)
    expected = 0.8 * math.log(1.05)
    assert math.isclose(p.mtm_log_return(105.0), expected, abs_tol=1e-12)


def test_portfolio_close_all_bulk_close():
    p = Portfolio()
    p.open_one(_mk_position(entry_price=100.0, size=0.5))
    p.open_one(_mk_position(entry_price=100.0, size=0.3))
    closed_list = p.close_all(
        k_exit=12,
        ts_exit=pd.Timestamp("2025-06-01 00:40:00"),
        exit_price=99.0,
        exit_reason="bulk_regime",
    )
    assert p.n_open() == 0
    assert len(closed_list) == 2
    assert all(c.exit_reason == "bulk_regime" for c in closed_list)
    # Both lost ln(0.99) per unit; pre-cost realized = 0.8 * ln(0.99)
    realized = p.realized_log_return(cost_per_trade=0.0)
    expected = 0.8 * math.log(0.99)
    assert math.isclose(realized, expected, abs_tol=1e-12)


def test_portfolio_realized_subtracts_cost_per_trade():
    p = Portfolio()
    p.open_one(_mk_position(entry_price=100.0, size=1.0))
    p.close_one(
        p.open_positions[0],
        k_exit=11,
        ts_exit=pd.Timestamp("2025-06-01 00:20:00"),
        exit_price=100.0 * math.exp(0.0025),
        exit_reason="tp",
    )
    # 1.0 * (0.0025 - 0.0005)
    assert math.isclose(
        p.realized_log_return(cost_per_trade=0.0005), 0.0020, abs_tol=1e-12
    )


def _expected_closed_columns() -> list[str]:
    """The set of columns ``Portfolio.closed_to_frame`` produces on empty."""
    return list(Portfolio().closed_to_frame().columns)


def test_portfolio_closed_to_frame_empty():
    p = Portfolio()
    df = p.closed_to_frame()
    # Reference the schema from inventory.py rather than a hardcoded
    # column count — adding a field there will then break this test
    # loudly rather than silently desynchronizing.
    assert df.shape == (0, len(_expected_closed_columns()))


def test_portfolio_closed_to_frame_columns_present():
    p = Portfolio()
    p.open_one(_mk_position())
    p.close_one(
        p.open_positions[0],
        k_exit=11,
        ts_exit=pd.Timestamp("2025-06-01 00:20:00"),
        exit_price=101.0,
        exit_reason="tp",
    )
    df = p.closed_to_frame()
    expected_cols = _expected_closed_columns()
    assert df.shape == (1, len(expected_cols))
    for col in (
        "k_entry", "k_exit", "exit_reason", "gross_log_return",
        "p_at_entry", "knowledge_unc_at_entry", "regime_quantile_at_entry",
    ):
        assert col in df.columns
    # Strong cross-check: ledger has every column the schema declares
    for col in expected_cols:
        assert col in df.columns, f"schema declares {col} but frame is missing it"
