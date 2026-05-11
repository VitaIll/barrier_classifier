"""Tests for cadence-aware split / weight helpers in ``src/utils.py``.

These guard the train/val/test split and the time-discount weight against
silent leakage when switching the dataset frequency from boundary to 1-min
cadence. The default embargo and delta values are designed for boundary
cadence; without these helpers a copy-paste at 1-min cadence is a leak.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import utils
from src.features.config import M

pytestmark = pytest.mark.step15


# ---------------------------------------------------------------------------
# recommended_embargo_for_cadence
# ---------------------------------------------------------------------------


def test_embargo_boundary_returns_base():
    assert utils.recommended_embargo_for_cadence("boundary", base_embargo=60) == 60
    assert utils.recommended_embargo_for_cadence("boundary", base_embargo=1) == 1


def test_embargo_1min_scales_by_M():
    # 60 boundaries calendar-equivalent = 60 * M rows at 1-min cadence
    assert utils.recommended_embargo_for_cadence("1min", base_embargo=60) == 60 * M
    assert utils.recommended_embargo_for_cadence("1min", base_embargo=10) == 10 * M


def test_embargo_1min_floor_is_M():
    """At least M rows of embargo are required at 1-min cadence to prevent
    label-overlap between train's last row and val's first row."""
    # base_embargo=0 should still yield at least M
    assert utils.recommended_embargo_for_cadence("1min", base_embargo=0) == M
    # base_embargo=1 should also yield at least M (1*M = M, fine)
    assert utils.recommended_embargo_for_cadence("1min", base_embargo=1) == M


def test_embargo_rejects_bad_cadence():
    with pytest.raises(ValueError, match="label_cadence"):
        utils.recommended_embargo_for_cadence("hourly")


def test_embargo_split_at_1min_prevents_label_overlap():
    """Real-data integration check: with the recommended 1-min embargo,
    train's last future-bar index does not overlap val's first row's
    prediction window."""
    # Synthetic 1-min cadence frame — large enough that 1-min embargo of
    # 60*M = 1200 rows leaves non-empty val and test windows after the
    # 70/15/15 split.
    n = 50_000
    df = pd.DataFrame(
        {
            "k": np.arange(n, dtype=int),
            "ts": pd.date_range("2025-01-01", periods=n, freq="1min"),
            "y": np.random.default_rng(0).integers(0, 2, size=n),
        }
    )
    embargo = utils.recommended_embargo_for_cadence("1min", base_embargo=60)
    train, val, test = utils.chronological_split_with_embargo(
        df, train_frac=0.7, val_frac=0.15, embargo_k=embargo
    )
    # Train's last row at index k_t looks forward to k_t+M.
    # Val's first row at index k_v looks forward to k_v+M.
    # For non-overlap of future-bar coverage we need k_v >= k_t + M.
    k_train_last = int(train["k"].max())
    k_val_first = int(val["k"].min())
    assert k_val_first - k_train_last >= M, (
        f"label-overlap risk: val starts only {k_val_first - k_train_last} "
        f"rows after train ends (need >= {M})"
    )
    # Same between val and test
    k_val_last = int(val["k"].max())
    k_test_first = int(test["k"].min())
    assert k_test_first - k_val_last >= M


# ---------------------------------------------------------------------------
# recommended_time_discount_delta_for_cadence
# ---------------------------------------------------------------------------


def test_delta_boundary_returns_base():
    assert utils.recommended_time_discount_delta_for_cadence(
        "boundary", base_delta=0.5
    ) == 0.5


def test_delta_1min_preserves_calendar_decay():
    """delta_1min^M should equal delta_boundary so M minutes of 1-min decay
    equals one boundary's decay at boundary cadence."""
    base = 0.5
    d = utils.recommended_time_discount_delta_for_cadence("1min", base_delta=base)
    assert math.isclose(d ** M, base, rel_tol=1e-12, abs_tol=1e-15)


def test_delta_1min_lt_base_for_base_lt_1():
    """Per-row decay must shrink (closer to 1) when each row covers less time."""
    d = utils.recommended_time_discount_delta_for_cadence("1min", base_delta=0.5)
    assert 0.0 < d < 1.0
    assert d > 0.5  # closer to 1 than the per-boundary delta


def test_delta_rejects_bad_base():
    with pytest.raises(ValueError, match="base_delta"):
        utils.recommended_time_discount_delta_for_cadence("boundary", base_delta=0.0)
    with pytest.raises(ValueError, match="base_delta"):
        utils.recommended_time_discount_delta_for_cadence("1min", base_delta=1.5)


def test_delta_rejects_bad_cadence():
    with pytest.raises(ValueError, match="label_cadence"):
        utils.recommended_time_discount_delta_for_cadence("hourly", base_delta=0.5)
