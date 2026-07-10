"""Selective-classification policy + honest order-by-order backtest.

This module is the trading skin on a generic problem: given a calibrated
classifier, an optional posterior over its score (virtual ensemble), and a
cost structure for acting on its predictions, decide *when* to act, *how
much* to commit, and *when* to abstain or unwind.

Same primitives recur in fraud detection, medical triage, ad serving, and
selective classification with reject option. We use the trading vocabulary
(positions, P&L, Sharpe) but the design is domain-neutral: each piece is a
small swappable function so the spec list at the top of the calibration
notebook composes a strategy variant in one line.

Public surface (kept minimal until cell-level tests prove out the core):

- ``inventory`` — Position dataclass + Portfolio (open / close / mark-to-market)
- ``policy``    — composable score / gate / sizer / exit / bulk-close primitives
                  + ``StrategySpec`` that bundles them. Risk caps, halts, and
                  cluster-loss accounting live on ``RiskConfig`` inside this
                  module (there is no separate ``risk`` submodule).
- ``simulator`` — bar-by-bar driver: ``simulate(cache, raw_bars, spec) -> SimResult``
- ``online``    — River wrappers: streaming quantile, EWVar, online isotonic, ADWIN
- ``diagnostics`` — pre-flight checks that gate which specs are eligible
- ``reporting`` — equity ladder, deflated-Sharpe table, regime attribution
- ``cache``     — augment a research cache with boundary OHLC and r_realized

Causality contract:
- All decision functions consume only ``State`` snapshots built from data
  available *at or before* the current boundary's close.
- ``simulate`` walks boundaries in chronological order; entry fills happen
  at the boundary-close price (information set at that ts), TP/SL detection
  uses subsequent intra-horizon 1-min OHLC bars.
- No function in this module reads ``y_k`` until after the position closes.
"""
