"""Composable policy primitives + StrategySpec.

This module is the rehaul-friendly heart of the strategy work. Every
decision component (score, gate, sizer, exit, bulk-close) is a small pure
function. A ``StrategySpec`` is a frozen dataclass that bundles them; a
new variant is one line at the top of the calibration notebook.

Causality: every primitive consumes only a ``State`` snapshot built from
data available at or before the current boundary's close. No primitive
peeks at ``y_k`` or any future bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.strategy.inventory import Position


# ---------------------------------------------------------------------------
# Decision-time state snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntraBar:
    """The OHLC for one 1-min bar inside the position's horizon."""

    n: int               # 1-min bar index (n = k * M + j for j in 1..M)
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class State:
    """All info available at boundary k's close — the strategy's input.

    Mandatory fields are populated by the simulator on every step. Optional
    fields (``mean_p_ve``, ``knowledge_unc``, ``p_ve_samples``) are NaN /
    empty when the virtual-ensemble cache columns are absent — primitives
    that depend on them should declare the requirement in the spec's
    ``requires`` tuple.
    """

    k: int
    ts: pd.Timestamp
    p: float                      # raw model prob (for class 1)
    p_calibrated: float           # post online recalibration; falls back to p
    bar_close: float              # close[n_k] — boundary close price
    bar_high: float               # high[n_k]
    bar_low: float                # low[n_k]
    regime_value: float           # raw regime signal value (e.g. vol__rs__f__w240)
    regime_quantile: float        # streaming quantile rank in [0, 1]; NaN until warm
    fast_sigma: float             # fast vol estimate (EWMA on log-returns); NaN until warm
    n_open_positions: int
    cluster_pnl: float            # cumulative MTM of currently-open positions
    cluster_streak: int           # boundary count since the most recent flat-state
    # Sum of sizes across currently-open positions (i.e. the gross size
    # already deployed in the active cluster). Used by cluster-aware
    # marginal sizers so overlapping high-score rows do not stack to
    # uncapped correlated exposure.
    inventory_gross_size: float = 0.0
    # Optional virtual-ensemble pieces ----------------------------------------
    mean_p_ve: float = float("nan")
    knowledge_unc: float = float("nan")
    knowledge_unc_quantile: float = float("nan")
    p_ve_samples: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Score is computed by the spec's score_fn and cached here for downstream
    # primitives (gates, sizer) that share it.
    score: float = float("nan")


# ---------------------------------------------------------------------------
# Score functions: (state, **opts) -> float
# ---------------------------------------------------------------------------


def score_raw_p(state: State) -> float:
    """Trade on the raw (or calibrated) model probability."""
    return float(state.p_calibrated)


def score_residualized(state: State, regime_base_rate: float) -> float:
    """Trade on excess probability over the regime-conditional base rate.

    ``regime_base_rate`` is ``E[y | regime_quantile_t]`` — supplied by the
    simulator from a calibration table fitted on a prior window.
    """
    return float(state.p_calibrated - regime_base_rate)


def score_ve_mean(state: State) -> float:
    """Use the VE mean directly (defaults to NaN if VE not populated)."""
    return float(state.mean_p_ve)


# ---------------------------------------------------------------------------
# Entry gates: each (state, threshold) -> bool. Spec AND-composes them.
# ---------------------------------------------------------------------------


def gate_score_above(state: State, threshold: float) -> bool:
    """Open only if the cached score is at or above ``threshold``."""
    s = state.score
    if not np.isfinite(s):
        return False
    return bool(s >= float(threshold))


def gate_regime_high(state: State, q_min: float) -> bool:
    """Open only if the slow regime is in the top fraction of its trailing distribution."""
    q = state.regime_quantile
    if not np.isfinite(q):
        return False
    return bool(q >= float(q_min))


def gate_unc_below(state: State, q_max: float) -> bool:
    """Open only if knowledge uncertainty is below the trailing-distribution cutoff.

    Cutoff is given as a quantile (e.g. 0.9 = drop the top decile of MI).
    Falls open (returns True) when MI is missing — the pipeline downstream
    of the spec's ``requires`` decides whether to schedule the spec at all.
    """
    qu = state.knowledge_unc_quantile
    if not np.isfinite(qu):
        return True
    return bool(qu <= float(q_max))


def gate_knowledge_unc_cap(state: State, cap: float) -> bool:
    """Open only if raw knowledge uncertainty is at or below ``cap``.

    Unlike ``gate_unc_below`` (which compares the streaming-quantile rank
    of MI to a quantile threshold), this gate uses the raw MI value
    directly. Use it with a cap frozen from a training-period reference
    distribution — e.g. the median of MI among training rows that pass the
    score gate. Fail-closed on NaN: if the cache doesn't carry MI, this
    gate rejects every row rather than silently waving them through.
    """
    qu = state.knowledge_unc
    if not np.isfinite(qu):
        return False
    return bool(qu <= float(cap))


def gate_no_concurrent_loss_cluster(state: State, max_drawdown: float) -> bool:
    """Open only if the current cluster's mark-to-market loss is shallower than
    ``max_drawdown`` (in log-return units, magnitude). When deeper, we sit out
    new entries and let the existing inventory unwind through normal exits."""
    return bool(state.cluster_pnl >= -float(max_drawdown))


# ---------------------------------------------------------------------------
# Sizers: (state, **outcome_params) -> float in [0, max_size]
# ---------------------------------------------------------------------------


def size_constant(state: State, *, default: float = 0.1) -> float:
    """Trade a fixed unit per signal — the dumbest baseline."""
    return float(default)


def _kelly(p: float, b_ratio: float) -> float:
    """Standard Kelly fraction for a binary bet with payoff ratio ``b_ratio``."""
    if not np.isfinite(p) or not (0.0 < b_ratio < math.inf):
        return 0.0
    f = (b_ratio * p - (1.0 - p)) / b_ratio
    return max(0.0, float(f))


def size_kelly_point(state: State, *, b_ratio: float, fraction: float = 0.25) -> float:
    """Fractional-Kelly on the point estimate of p (calibrated)."""
    return float(fraction) * _kelly(state.p_calibrated, b_ratio)


def size_bayesian_kelly(
    state: State,
    *,
    b_ratio: float,
    percentile: float = 0.25,
    fraction: float = 0.25,
) -> float:
    """Kelly per VE replicate, then take the ``percentile`` quantile.

    Falls back to point Kelly on ``mean_p_ve`` (or ``p_calibrated`` if the
    VE samples aren't populated). The lower the percentile, the more
    conservative the size — high knowledge uncertainty widens the
    posterior, which automatically shrinks the lower-quantile Kelly.
    """
    samples = state.p_ve_samples
    if samples.size == 0:
        # No VE — degrade gracefully to point-Kelly on best available p.
        p_best = state.mean_p_ve if np.isfinite(state.mean_p_ve) else state.p_calibrated
        return float(fraction) * _kelly(float(p_best), b_ratio)
    f_samples = (b_ratio * samples - (1.0 - samples)) / b_ratio
    q = float(np.quantile(f_samples, float(percentile)))
    return max(0.0, float(fraction) * q)


def size_voltarget_overlay(
    base_size: float,
    *,
    fast_sigma: float,
    sigma_target: float,
    sigma_floor: float = 1e-6,
    max_multiplier: float = 4.0,
) -> float:
    """Multiplier ``sigma_target / max(fast_sigma, sigma_floor)`` capped at
    ``max_multiplier``. Holds gross ex-ante vol roughly invariant lot-by-lot.
    Pass ``base_size`` from any other sizer through this overlay."""
    if not np.isfinite(fast_sigma):
        return float(base_size)
    denom = max(float(fast_sigma), float(sigma_floor))
    multiplier = min(float(max_multiplier), float(sigma_target) / denom)
    return float(base_size) * float(max(0.0, multiplier))


def size_clip(size: float, *, max_size: float) -> float:
    """Final clip at the per-lot cap. Always pass through this before opening."""
    return float(min(max(0.0, float(size)), float(max_size)))


def size_cluster_marginal(
    state: State,
    *,
    target_cluster_size: float,
    base_size: float | None = None,
) -> float:
    """Marginal lot size relative to a per-cluster target.

    A signal cluster (a run of consecutive boundaries with at least one
    open position) is one economic opportunity, not many. Overlapping
    high-score rows would otherwise stack to ``n * base_size`` of
    correlated exposure within the same underlying event. This sizer
    enforces a single per-cluster cap by returning only the marginal
    headroom:

        size_n = max(0, target_cluster_size - inventory_gross_size_n)

    If ``base_size`` is provided it is used as the size for the *first*
    entry into a flat-state cluster (when ``inventory_gross_size == 0``),
    so this sizer composes with point-Kelly or constant sizers without
    forcing the cluster's opening lot to equal the cluster cap.
    """
    target = float(target_cluster_size)
    inv = float(getattr(state, "inventory_gross_size", 0.0))
    if inv <= 0.0:
        first_lot = float(base_size) if base_size is not None else target
        return max(0.0, min(first_lot, target))
    return max(0.0, target - inv)


# ---------------------------------------------------------------------------
# Exit policies: (position, intra_bar, k_now) -> Optional[ExitReason]
# ---------------------------------------------------------------------------


def exit_tp_or_expiry(
    position: Position, intra_bar: Optional[IntraBar], k_now: int
) -> Optional[str]:
    """Two-phase exit policy.

    - With ``intra_bar`` (path-walk phase): returns ``"tp"`` if the bar's
      range crosses the upper barrier (long) or lower barrier (short),
      else ``None``. Crucially does NOT return ``"expiry"`` here — that
      would prematurely close a position on its first non-TP path-bar.
    - With ``intra_bar=None`` (end-of-path phase): returns ``"expiry"``
      iff ``k_now >= position.expiry_k``.

    The simulator calls this once per intra-bar, then once with
    ``intra_bar=None`` after the loop. This split keeps the price-driven
    exits (TP/SL) cleanly separate from the time-driven exit (expiry).
    """
    if intra_bar is not None:
        # Long: high crosses tp. Short: low crosses tp (which is below entry).
        if position.side == 1 and intra_bar.high >= position.tp_price:
            return "tp"
        if position.side == -1 and intra_bar.low <= position.tp_price:
            return "tp"
        return None
    # intra_bar is None → end-of-path expiry check
    if k_now >= position.expiry_k:
        return "expiry"
    return None


def exit_tp_sl_or_expiry(
    position: Position,
    intra_bar: Optional[IntraBar],
    k_now: int,
    *,
    bar_resolution: str = "pessimistic_sl",
) -> Optional[str]:
    """Triple-barrier two-phase exit. Same split as ``exit_tp_or_expiry``:
    intra-bar phase returns TP/SL only, end-of-path phase returns expiry.

    Tie-break when both TP and SL would have fired in the same bar — note
    that this is a *train/exec mismatch artifact*: training labels treat the
    upper barrier as hit on intrabar high (TP-on-high label), but the
    execution simulator pessimistically takes SL here. The choices are:

    - ``"pessimistic_sl"`` (default; current behavior): assume the path
      hit SL before TP. Lower bound on realized return — what a risk-
      averse audit wants to see.
    - ``"neutral"``: return ``"tp_or_sl"`` to surface the ambiguity to
      the caller; the caller decides at policy level (e.g. take the
      midpoint or flag for hand review). The simulator's ledger will
      record this as ``"tp_or_sl"`` so analytics can audit how often
      this fires.
    - ``"optimistic_tp"``: assume the path hit TP before SL. Upper
      bound; only honest if the label source is also high-source.

    Finer 1-second/tick paths would resolve the order; bars don't carry
    that info.
    """
    if bar_resolution not in ("pessimistic_sl", "neutral", "optimistic_tp"):
        raise ValueError(
            f"bar_resolution must be 'pessimistic_sl', 'neutral', or "
            f"'optimistic_tp'; got {bar_resolution!r}"
        )
    if intra_bar is not None:
        sl = position.sl_price
        if position.side == 1:
            tp_hit = intra_bar.high >= position.tp_price
            sl_hit = sl is not None and intra_bar.low <= sl
        else:
            tp_hit = intra_bar.low <= position.tp_price
            sl_hit = sl is not None and intra_bar.high >= sl
        if tp_hit and sl_hit:
            if bar_resolution == "pessimistic_sl":
                return "sl"
            if bar_resolution == "optimistic_tp":
                return "tp"
            return "tp_or_sl"
        if sl_hit:
            return "sl"
        if tp_hit:
            return "tp"
        return None
    # End-of-path
    if k_now >= position.expiry_k:
        return "expiry"
    return None


# ---------------------------------------------------------------------------
# "Let winners run" exit policy
#
# Architecture note: exit_policy's signature is (pos, intra_bar, k_now) — no
# access to the current model score p_t. To condition the exit on the model's
# continued conviction we build a closure that captures a {ts -> p_t}
# mapping at strategy-construction time; the closure looks up p_now keyed on
# intra_bar.ts every step.
#
# Behavior contract:
#   - If MTM <= -sl_log_return (when sl_log_return is provided): close "sl"
#     at the pre-set sl_price (or bar.close if no sl_price was set on the
#     position).
#   - Otherwise, if price has reached the original tp_price (in-the-money):
#       * If model still bullish (p_now >= hold_threshold): SKIP the TP fill
#         — keep holding. The position rides any further upside.
#       * If model has lost conviction (p_now < hold_threshold): close at
#         bar.close (market exit) with reason "tp_market". Note this fills
#         at the *actual* path price, not the static tp_price, so any extra
#         upside the position rode while we waited is captured.
#   - Otherwise (price hasn't crossed tp_price yet): hold (None).
#   - End-of-path phase: standard expiry check at k_now >= expiry_k.
#
# Failure mode to know: if p_now is NaN (not in the map) the policy treats
# it as "no signal" and falls through to the standard TP behavior — i.e.
# closes at tp_price. The cache is dense at 1-min cadence so this should
# never fire in normal operation, but it's the safe fallback.
# ---------------------------------------------------------------------------


def make_exit_let_winners_run(
    p_map: dict[pd.Timestamp, float] | "pd.Series[float]",
    *,
    hold_threshold: float,
    sl_log_return: float | None = None,
) -> Callable[[Position, Optional[IntraBar], int], Optional[str]]:
    """Build a let-winners-run exit policy bound to a {ts -> p} mapping.

    Parameters
    ----------
    p_map : mapping pd.Timestamp -> float
        Lookup table of model probability at each 1-min boundary. Built
        once at strategy-construction time from the prediction cache.
        Accessed as ``p_map[intra_bar.ts]`` (pandas Series with a
        DatetimeIndex satisfies this interface).
    hold_threshold : float
        Probability cut-off below which we stop holding. When the path
        crosses the original TP target AND ``p_now >= hold_threshold``,
        the TP fill is skipped (we ride upside). When ``p_now <
        hold_threshold``, we close at bar.close (market exit).
    sl_log_return : float or None, optional
        If provided, close at "sl" whenever the position's MTM (log-return
        from entry to bar.close) falls below ``-sl_log_return``. Use a
        positive number, e.g. ``2 * PHI = 0.005`` for a 2x phi stop.
        ``None`` (default) matches the existing patient-wait baseline:
        no stop, lots can sit underwater indefinitely.
    """
    if not np.isfinite(hold_threshold):
        raise ValueError(f"hold_threshold must be finite; got {hold_threshold!r}")
    if sl_log_return is not None and (sl_log_return <= 0 or not np.isfinite(sl_log_return)):
        raise ValueError(
            f"sl_log_return must be positive and finite when set; got {sl_log_return!r}"
        )
    hold_threshold = float(hold_threshold)
    sl_log_return_f: float | None = float(sl_log_return) if sl_log_return is not None else None

    # pandas Series .at[] is the fastest scalar lookup on a DatetimeIndex.
    # Support both dict-like and Series-like lookups via try/except.
    def _lookup_p(ts: pd.Timestamp) -> float:
        try:
            return float(p_map[ts])  # Series with DatetimeIndex, or dict
        except (KeyError, IndexError):
            return float("nan")

    def exit_let_winners_run(
        position: Position, intra_bar: Optional[IntraBar], k_now: int
    ) -> Optional[str]:
        if intra_bar is None:
            # End-of-path expiry check (mirrors exit_tp_or_expiry).
            if k_now >= position.expiry_k:
                return "expiry"
            return None

        side = position.side
        entry = float(position.entry_price)
        close_now = float(intra_bar.close)

        # MTM in log-return space (positive for in-the-money).
        if side == 1:
            mtm_log = float(np.log(close_now / entry)) if (close_now > 0 and entry > 0) else float("nan")
        else:
            mtm_log = float(np.log(entry / close_now)) if (close_now > 0 and entry > 0) else float("nan")

        # Stop loss check (if enabled and finite MTM).
        if sl_log_return_f is not None and np.isfinite(mtm_log) and mtm_log <= -sl_log_return_f:
            return "sl"

        # TP trigger: bar's range touched the original TP barrier.
        if side == 1:
            tp_hit = intra_bar.high >= position.tp_price
        else:
            tp_hit = intra_bar.low <= position.tp_price

        if not tp_hit:
            return None

        # In-the-money this bar. Check model conviction.
        p_now = _lookup_p(intra_bar.ts)
        if np.isfinite(p_now) and p_now >= hold_threshold:
            # Model still says go: skip the TP fill, ride upside.
            return None

        # Conviction dropped (or NaN p_now treated as "no signal"):
        # close at market. The simulator fills "tp_market" at bar.close
        # via its default branch, capturing whatever price the position
        # rode to.
        return "tp_market"

    return exit_let_winners_run


def make_exit_let_winners_run_monotonic(
    p_map: dict[pd.Timestamp, float] | "pd.Series[float]",
) -> Callable[[Position, Optional[IntraBar], int], Optional[str]]:
    """Strict-monotonic let-winners-run exit policy.

    Stricter cousin of :func:`make_exit_let_winners_run`:

      * Tracks, per-position, the maximum probability ``p`` observed since
        entry (call it ``p_max``).
      * **Hold contract**: at each bar, if the bar crosses the original
        TP barrier AND ``p_now > p_max``, **skip the TP fill** and update
        ``p_max``. The position rides any further upside.
      * **Exit contract**: at each bar that crosses TP AND ``p_now <=
        p_max`` (conviction has stopped growing), close at ``bar.close``
        with reason ``"tp_market"``.
      * **Don't release into loss**: if the bar does NOT cross the TP
        barrier (price < entry+phi), do nothing. Even if conviction has
        clearly stopped growing, we wait for price to recover into the TP
        zone before considering exit. This matches the baseline patient-
        wait behavior: never exit at a loss; only ever close +TP.

    There is no SL. Underwater lots sit open indefinitely, just like in
    the baseline. The "winner-running" gain comes from capturing realized
    upside above +phi when the model's conviction keeps growing.

    Per-position state is stored in a closure-local dict keyed by the
    position's ``entry_k`` (unique across the simulator's lifetime, since
    each 1-min boundary fires at most one entry). The state is cleaned
    up on close/expiry.
    """
    # Per-position state. Key = position.k_entry → p_max so far.
    p_max_per_position: dict[int, float] = {}

    def _lookup_p(ts: pd.Timestamp) -> float:
        try:
            return float(p_map[ts])
        except (KeyError, IndexError):
            return float("nan")

    def exit_let_winners_run_monotonic(
        position: Position, intra_bar: Optional[IntraBar], k_now: int
    ) -> Optional[str]:
        # End-of-path expiry phase.
        if intra_bar is None:
            if k_now >= position.expiry_k:
                p_max_per_position.pop(position.k_entry, None)
                return "expiry"
            return None

        # Decide if this bar is in the TP zone.
        if position.side == 1:
            tp_hit = intra_bar.high >= position.tp_price
        else:
            tp_hit = intra_bar.low <= position.tp_price

        # Update p_max tracker on every bar (whether or not TP hits).
        p_now = _lookup_p(intra_bar.ts)
        if np.isfinite(p_now):
            p_max = p_max_per_position.get(position.k_entry, float("-inf"))
            still_growing = p_now > p_max
            if still_growing:
                p_max_per_position[position.k_entry] = p_now
        else:
            # No probability data for this ts: treat as "not growing"
            # (conservative — defaults to taking TP if it hits).
            still_growing = False

        if not tp_hit:
            # Below TP zone: never release into loss. Wait.
            return None

        # In TP zone.
        if still_growing:
            # Conviction making a new high → skip TP, ride upside.
            return None

        # Conviction has flattened or dropped from peak. Before closing
        # at market, verify bar.close is on the profit side of entry — the
        # user's "don't release into loss" rule applies at the actual fill
        # price, not just the TP-touched indicator. If the bar's high
        # crossed TP but the close has reverted below entry, defer: wait
        # for a future bar whose close lands above entry.
        if position.side == 1 and intra_bar.close < position.entry_price:
            return None
        if position.side == -1 and intra_bar.close > position.entry_price:
            return None

        p_max_per_position.pop(position.k_entry, None)
        return "tp_market"

    def _on_position_closed(closed) -> None:
        """Release per-position state for ANY close path.

        The policy's own pops above only cover exits it returns itself
        ("expiry", "tp_market"). Bulk-close and SL paths close positions
        WITHOUT consulting the exit policy, which used to leak their
        ``p_max`` entries forever. The BoundaryStep calls this hook for
        every ClosedPosition; ``k_entry`` is unique per run, so releasing
        an already-popped key is a no-op.
        """
        p_max_per_position.pop(closed.k_entry, None)

    exit_let_winners_run_monotonic.on_position_closed = _on_position_closed
    return exit_let_winners_run_monotonic


# ---------------------------------------------------------------------------
# Bulk-close triggers: (state) -> Optional[reason]
# ---------------------------------------------------------------------------


def bulk_on_regime_drop(state: State, exit_q: float = 0.5) -> Optional[str]:
    """Flatten if the slow regime drops below ``exit_q`` (asymmetric to entry)."""
    q = state.regime_quantile
    if np.isfinite(q) and q < float(exit_q):
        return "bulk_regime"
    return None


def bulk_on_unc_spike(state: State, spike_q: float = 0.95) -> Optional[str]:
    """Flatten if knowledge uncertainty has spiked above ``spike_q`` quantile.

    Zero-lag drift detector: triggers on the input distribution alone,
    *before* labels mature. Replaces / complements label-based ADWIN.
    """
    qu = state.knowledge_unc_quantile
    if np.isfinite(qu) and qu >= float(spike_q):
        return "bulk_unc"
    return None


def bulk_on_cluster_loss(state: State, cap_log_return: float) -> Optional[str]:
    """Flatten if the open cluster's MTM is below ``-cap_log_return``.

    Hard cap on per-cluster loss; pairs with ``gate_no_concurrent_loss_cluster``
    (one prevents new entries while underwater, the other liquidates if it
    deepens past the cap)."""
    if state.cluster_pnl <= -float(cap_log_return):
        return "bulk_cluster_loss"
    return None


# ---------------------------------------------------------------------------
# Risk config + StrategySpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskConfig:
    max_size_per_lot: float = 1.0
    max_open_positions: int = 5
    max_gross_size: float = 3.0
    cost_per_trade: float = 0.0005     # 5 bps round-trip default
    cluster_loss_cap: float = 0.0075   # 3 * phi default; consumed by bulk_on_cluster_loss
    # ----- position horizon + per-position stop -------------------------------
    # ``max_horizon_boundaries`` controls how many boundaries (each = M bars)
    # a position can stay open before forced expiry. Setting it to 1 mirrors
    # the model's training horizon exactly — useful as a label-aligned
    # baseline. Setting it higher lets TP have more time to materialize at
    # the cost of capital tied up; with bulk-close triggers handling the
    # regime-break risk, the upper bound becomes a backstop rather than the
    # primary exit driver. Reference: with a fixed-φ barrier and σ_T scaling
    # like √T, P(hit +φ within T·M bars) approaches 1 as T grows — extending
    # horizon trades drawdown variance for hit rate.
    max_horizon_boundaries: int = 1
    # Per-position MTM floor in log-return units (e.g., -0.0075 = -3·φ).
    # When set, the simulator opens each lot with sl_price set so the spec's
    # exit_policy can fire SL on a path-bar breach. ``None`` disables the
    # per-position stop entirely — bulk-close on cluster loss is the only
    # downside containment.
    position_mtm_floor_log_return: Optional[float] = None


@dataclass(frozen=True)
class StrategySpec:
    """Bundle of primitives + the diagnostic flags it requires.

    The simulator filters specs by ``requires`` against the diagnostics
    bundle produced upstream — so a spec that needs VE will be silently
    skipped if the VE diagnostic failed.
    """

    name: str
    requires: tuple[str, ...] = ()
    score_fn: Callable[[State], float] = score_raw_p
    entry_gates: tuple[Callable[[State], bool], ...] = ()
    sizer: Callable[[State], float] = size_constant
    exit_policy: Callable[[Position, Optional[IntraBar], int], Optional[str]] = exit_tp_or_expiry
    bulk_close: Callable[[State], Optional[str]] = lambda s: None
    risk: RiskConfig = field(default_factory=RiskConfig)
    description: str = ""

    def evaluate_entry(self, state: State) -> bool:
        """AND-compose every entry gate. Empty tuple => no entries (a no-op spec)."""
        if not self.entry_gates:
            return False
        for g in self.entry_gates:
            if not g(state):
                return False
        return True


# ---------------------------------------------------------------------------
# Spec primitives — common compositions ready for use by the calibration NB
# ---------------------------------------------------------------------------


def make_baseline_spec(
    *,
    threshold: float = 0.30,
    size: float = 0.5,
    cost_per_trade: float = 0.0005,
    max_horizon_boundaries: int = 1,
) -> StrategySpec:
    """Label-aligned baseline: TP within M bars or expire at boundary close.

    Horizon = 1 boundary by default — matches the model's training horizon
    exactly. This is the *honest reference* number, not the strategy we'd
    deploy. Comparing against the patient variants quantifies what label-
    alignment costs us in realized P&L.
    """
    return StrategySpec(
        name="baseline_label_aligned",
        requires=(),
        score_fn=score_raw_p,
        entry_gates=(lambda s, t=threshold: gate_score_above(s, t),),
        sizer=lambda s, sz=size: size_clip(size_constant(s, default=sz), max_size=1.0),
        exit_policy=exit_tp_or_expiry,
        bulk_close=lambda s: None,
        risk=RiskConfig(
            cost_per_trade=cost_per_trade,
            max_horizon_boundaries=max_horizon_boundaries,
        ),
        description=(
            "long-only at p>=tau, fixed size, label-aligned 1-boundary horizon, "
            "no bulk-close — the reference number, not a deployable strategy"
        ),
    )


def make_patient_spec(
    *,
    threshold: float = 0.30,
    size: float = 0.5,
    cost_per_trade: float = 0.0005,
    max_horizon_boundaries: int = 360,  # ~5 days at M=20 min
    regime_exit_q: float = 0.4,
    use_position_stop: bool = False,
    position_mtm_floor: float = -0.0075,  # -3·φ if enabled
) -> StrategySpec:
    """The "wait-for-level" design.

    Open when the model fires (single threshold on `p`). Each position is
    a TP limit order at ``entry · exp(+φ)`` that stays live for up to
    ``max_horizon_boundaries`` boundaries. The only forced exits are:
    - TP fills whenever the path crosses +φ (deterministic);
    - bulk-close when the slow regime drops below ``regime_exit_q``;
    - optional per-position MTM floor (``use_position_stop=True``);
    - hard timeout at the horizon (backstop only).

    Rationale: the model isn't a 20-minute price oracle — it scores the
    favorability of the +φ level being reached *from here*. Under any
    positive-vol process, that level materializes with probability → 1 as
    holding time grows. The strategy aligns to the level, not the model's
    label horizon. Bulk-close is the only "real" risk exit; it pays for
    the unrealized leftover when the regime fundamentally shifts.
    """
    return StrategySpec(
        name="patient_wait_for_level",
        requires=(),  # no diagnostic gate — this is the "make-it-work" design
        score_fn=score_raw_p,
        entry_gates=(lambda s, t=threshold: gate_score_above(s, t),),
        sizer=lambda s, sz=size: size_clip(size_constant(s, default=sz), max_size=1.0),
        exit_policy=exit_tp_sl_or_expiry if use_position_stop else exit_tp_or_expiry,
        bulk_close=lambda s, q=regime_exit_q: bulk_on_regime_drop(s, exit_q=q),
        risk=RiskConfig(
            cost_per_trade=cost_per_trade,
            max_horizon_boundaries=max_horizon_boundaries,
            position_mtm_floor_log_return=(
                position_mtm_floor if use_position_stop else None
            ),
        ),
        description=(
            f"patient TP-limit-order strategy: open at p>={threshold}, hold up to "
            f"{max_horizon_boundaries} boundaries waiting for +φ, "
            f"bulk-close on regime drop (q<{regime_exit_q}), "
            f"{'with' if use_position_stop else 'no'} per-position MTM floor"
        ),
    )


def make_regime_gated_spec(
    *,
    threshold: float = 0.30,
    regime_entry_q: float = 0.7,
    regime_exit_q: float = 0.4,
    size: float = 0.5,
    cost_per_trade: float = 0.0005,
    max_horizon_boundaries: int = 360,
) -> StrategySpec:
    """Patient strategy + binary vol gate (entry only).

    Same wait-for-level design as ``make_patient_spec``, but adds an entry
    filter: only open when slow vol is in the top ``regime_entry_q`` of its
    trailing distribution. Hysteresis between entry-q and exit-q prevents
    chatter at the boundary."""
    return StrategySpec(
        name="regime_gated_patient",
        requires=("vol_gate",),
        score_fn=score_raw_p,
        entry_gates=(
            lambda s, t=threshold: gate_score_above(s, t),
            lambda s, q=regime_entry_q: gate_regime_high(s, q),
        ),
        sizer=lambda s, sz=size: size_clip(size_constant(s, default=sz), max_size=1.0),
        exit_policy=exit_tp_or_expiry,
        bulk_close=lambda s, q=regime_exit_q: bulk_on_regime_drop(s, exit_q=q),
        risk=RiskConfig(
            cost_per_trade=cost_per_trade,
            max_horizon_boundaries=max_horizon_boundaries,
        ),
        description=(
            "patient wait-for-level + vol-gated entry; bulk-flatten on regime drop"
        ),
    )


def make_1min_cluster_aware_spec(
    *,
    M: int = 20,
    threshold: float = 0.30,
    cluster_target_size: float = 1.0,
    first_lot_size: float = 0.5,
    cost_per_trade: float = 0.0005,
    max_horizon_boundaries: int | None = None,
    cluster_loss_cap: float = 0.020,
    gate_drawdown_threshold: float | None = None,
    regime_exit_q: float | None = None,
) -> StrategySpec:
    """1-min-cadence spec aligned to the high-source overlapping label.

    Production parameters at 1-min cadence with M=20:

    - Horizon defaults to ``M`` rows = one full barrier horizon (20 min).
      Pass a larger value for a patient wait-for-level variant: e.g.
      ``max_horizon_boundaries = 360 * M`` for ~5 days.
    - **Cluster-marginal sizing**: overlapping high-score rows would
      otherwise stack to ``n * lot_size`` of correlated exposure within
      one underlying event. ``size_cluster_marginal`` caps the cluster's
      total deployed size at ``cluster_target_size``; the first lot
      uses ``first_lot_size`` and subsequent entries within the same
      cluster only add the marginal headroom.
    - **Cluster-loss circuit breaker** at ``-cluster_loss_cap`` log-
      return on the open inventory.
    - **Drawdown-gate** suppresses NEW entries when cluster MTM is
      already underwater by ``gate_drawdown_threshold``. Distinct from
      ``cluster_loss_cap``: the gate prevents stacking into a losing
      cluster (suppressing entries) while the cap liquidates the
      cluster outright. Defaults to ``0.5 * cluster_loss_cap`` so the
      gate fires at half the liquidation level — gives existing
      inventory room to recover before slamming the brakes. If the
      gate and the cap were equal, the cap would always liquidate
      before the gate could matter (the gate would be a no-op).
    - Optional ``regime_exit_q`` bulk-close on regime drop. Set to
      ``None`` for an unconditional spec; set to e.g. 0.4 for a
      regime-gated patient variant.

    Pairs with ``barrier_source="high"`` labels from
    ``run_pipeline(label_cadence="1min")``: the label trains exactly the
    event this spec trades (long TP fills on intrabar high crossing).
    """
    horizon = int(max_horizon_boundaries) if max_horizon_boundaries is not None else int(M)
    gate_drawdown = (
        float(gate_drawdown_threshold)
        if gate_drawdown_threshold is not None
        else 0.5 * float(cluster_loss_cap)
    )

    def _sizer(state: State) -> float:
        return size_clip(
            size_cluster_marginal(
                state,
                target_cluster_size=cluster_target_size,
                base_size=first_lot_size,
            ),
            max_size=cluster_target_size,
        )

    def _bulk(state: State) -> Optional[str]:
        if regime_exit_q is not None:
            r = bulk_on_regime_drop(state, exit_q=regime_exit_q)
            if r is not None:
                return r
        return bulk_on_cluster_loss(state, cap_log_return=cluster_loss_cap)

    return StrategySpec(
        name="1min_cluster_aware",
        requires=(),
        score_fn=score_raw_p,
        entry_gates=(
            lambda s, t=threshold: gate_score_above(s, t),
            lambda s, cap=gate_drawdown: gate_no_concurrent_loss_cluster(s, cap),
        ),
        sizer=_sizer,
        exit_policy=exit_tp_or_expiry,
        bulk_close=_bulk,
        risk=RiskConfig(
            cost_per_trade=cost_per_trade,
            max_size_per_lot=cluster_target_size,
            max_gross_size=cluster_target_size,
            cluster_loss_cap=cluster_loss_cap,
            max_horizon_boundaries=horizon,
        ),
        description=(
            f"1-min cluster-aware long-TP: enter at p>={threshold}, "
            f"cluster cap {cluster_target_size} (first lot {first_lot_size}, "
            f"subsequent rows add marginal headroom), horizon={horizon} rows, "
            f"drawdown-gate at -{gate_drawdown}, "
            f"bulk-close on cluster loss>={cluster_loss_cap}"
            + (f" or regime drop<{regime_exit_q}" if regime_exit_q is not None else "")
        ),
    )


def make_bayesian_kelly_spec(
    *,
    threshold: float = 0.30,
    regime_entry_q: float = 0.7,
    regime_exit_q: float = 0.4,
    unc_q_max: float = 0.9,
    unc_spike_q: float = 0.95,
    cluster_loss_cap: float = 0.020,        # 8·φ — the cluster-level circuit breaker
    gate_drawdown_threshold: float | None = None,
    max_horizon_boundaries: int = 360,
    fraction_kelly: float = 0.25,
    kelly_percentile: float = 0.25,
    b_ratio: float = 1.0,
    cost_per_trade: float = 0.0005,
    max_size_per_lot: float = 1.0,
) -> StrategySpec:
    """Full stack: vol gate + MI gate + Bayesian-Kelly sizing + cluster controls.

    Same wait-for-level horizon as ``make_patient_spec``. The MI gate and the
    Bayesian-Kelly sizer require the virtual-ensemble cache columns; only
    ``ve_diag`` is a hard requirement (``vol_gate`` is advisory rather than
    blocking — the regime gate here primarily controls *when* to enter, not
    whether the gate improves point precision in the data).

    ``gate_drawdown_threshold`` controls the entry-suppression level for
    ``gate_no_concurrent_loss_cluster``; defaults to ``0.5 *
    cluster_loss_cap`` so the gate fires at half the liquidation level,
    suppressing new entries before the bulk-close cap triggers. Coupling
    them (gate fires at the same level as liquidation) would make the
    gate a no-op because the bulk-close would have already liquidated.
    """
    gate_drawdown = (
        float(gate_drawdown_threshold)
        if gate_drawdown_threshold is not None
        else 0.5 * float(cluster_loss_cap)
    )

    def _sizer(state: State) -> float:
        f = size_bayesian_kelly(
            state,
            b_ratio=b_ratio,
            percentile=kelly_percentile,
            fraction=fraction_kelly,
        )
        return size_clip(f, max_size=max_size_per_lot)

    def _bulk_close(state: State) -> Optional[str]:
        # Order: regime drop > MI spike > cluster loss; whichever fires first.
        return (
            bulk_on_regime_drop(state, exit_q=regime_exit_q)
            or bulk_on_unc_spike(state, spike_q=unc_spike_q)
            or bulk_on_cluster_loss(state, cap_log_return=cluster_loss_cap)
        )

    return StrategySpec(
        name="bayesian_kelly_patient",
        requires=("ve_diag",),  # vol_gate de-elevated to advisory
        score_fn=score_raw_p,
        entry_gates=(
            lambda s, t=threshold: gate_score_above(s, t),
            lambda s, q=regime_entry_q: gate_regime_high(s, q),
            lambda s, qu=unc_q_max: gate_unc_below(s, qu),
            lambda s, cap=gate_drawdown: gate_no_concurrent_loss_cluster(s, cap),
        ),
        sizer=_sizer,
        exit_policy=exit_tp_or_expiry,
        bulk_close=_bulk_close,
        risk=RiskConfig(
            cost_per_trade=cost_per_trade,
            max_size_per_lot=max_size_per_lot,
            cluster_loss_cap=cluster_loss_cap,
            max_horizon_boundaries=max_horizon_boundaries,
        ),
        description=(
            "patient wait-for-level + vol-gated + MI-gated entry + Bayesian-Kelly sizing; "
            f"drawdown-gate at -{gate_drawdown}; "
            "bulk-close on regime drop OR MI spike OR cluster-loss"
        ),
    )
