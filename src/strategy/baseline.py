"""Strategy baselines and shared cost constants.

Replaces the old ``scripts/_strategy_helpers.py`` sidekick — the
philosophy is "tooling lives in src/, experiments live in notebooks".

Exports:

- ``COST_PER_TRADE`` — round-trip transaction cost (5 bp) used by every
  strategy-reporting consumer.
- ``normalize_raw_index(raw_bars)`` — strip tz from the raw-bars
  DatetimeIndex so it can be compared to the simulator's tz-naive
  equity timestamps without a tz mismatch.
- ``compute_btc_buy_and_hold(raw_bars, ts_start, ts_end, span_days=...)`` —
  BTC buy-and-hold log-return baseline over a deployment span, plus the
  annualized version when ``span_days`` is supplied.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# Standardized round-trip transaction cost (5 bp).
COST_PER_TRADE: float = 0.0005


def normalize_raw_index(raw_bars: pd.DataFrame) -> pd.Index:
    """Strip tz from the raw-bars DatetimeIndex so it can be compared to
    the simulator's tz-naive equity timestamps without a tz mismatch.

    Returns a *view* of the index (or the original if already tz-naive);
    does NOT mutate ``raw_bars`` in place.
    """
    idx = raw_bars.index
    return idx.tz_localize(None) if idx.tz is not None else idx


def compute_btc_buy_and_hold(
    raw_bars: pd.DataFrame,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
    *,
    span_days: float | None = None,
) -> Tuple[float, float]:
    """Compute BTC buy-and-hold log-return over ``[ts_start, ts_end]``.

    Returns ``(btc_log_return, btc_annualized)``. ``btc_annualized`` is
    only meaningful when ``span_days`` is supplied; otherwise it is NaN.
    """
    raw_idx = normalize_raw_index(raw_bars)
    n0 = int(np.searchsorted(raw_idx, np.datetime64(ts_start), side="left"))
    n1 = int(np.searchsorted(raw_idx, np.datetime64(ts_end), side="left"))
    btc_start = float(raw_bars["close"].iloc[n0])
    btc_end = float(raw_bars["close"].iloc[n1])
    btc_log_return = float(np.log(btc_end / btc_start))
    if span_days is None or span_days <= 0:
        btc_annualized = float("nan")
    else:
        btc_annualized = btc_log_return * 365.0 / span_days
    return btc_log_return, btc_annualized
