"""Tests for the policy primitives + StrategySpec composition."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.strategy.inventory import Position
from src.strategy.policy import (
    IntraBar,
    RiskConfig,
    State,
    StrategySpec,
    bulk_on_cluster_loss,
    bulk_on_regime_drop,
    bulk_on_unc_spike,
    exit_tp_or_expiry,
    exit_tp_sl_or_expiry,
    gate_no_concurrent_loss_cluster,
    gate_regime_high,
    gate_score_above,
    gate_unc_below,
    make_baseline_spec,
    make_bayesian_kelly_spec,
    make_regime_gated_spec,
    score_raw_p,
    score_residualized,
    size_bayesian_kelly,
    size_clip,
    size_constant,
    size_kelly_point,
    size_voltarget_overlay,
)

pytestmark = pytest.mark.strategy_v1


# ---------------------------------------------------------------------------
# State construction helper
# ---------------------------------------------------------------------------


def _mk_state(**overrides) -> State:
    base = dict(
        k=10,
        ts=pd.Timestamp("2025-06-01 00:00:00"),
        p=0.4,
        p_calibrated=0.4,
        bar_close=100.0,
        bar_high=100.5,
        bar_low=99.5,
        regime_value=0.0005,
        regime_quantile=0.8,
        fast_sigma=0.001,
        n_open_positions=0,
        cluster_pnl=0.0,
        cluster_streak=0,
    )
    base.update(overrides)
    return State(**base)


def _mk_position(**overrides) -> Position:
    base = dict(
        k_entry=5,
        ts_entry=pd.Timestamp("2025-06-01 00:00:00"),
        side=1,
        size=0.5,
        entry_price=100.0,
        tp_price=100.0 * math.exp(0.0025),
        sl_price=None,
        expiry_k=6,
    )
    base.update(overrides)
    return Position(**base)


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------


def test_score_raw_p_returns_calibrated_p():
    s = _mk_state(p=0.3, p_calibrated=0.35)
    assert score_raw_p(s) == pytest.approx(0.35)


def test_score_residualized_subtracts_base_rate():
    s = _mk_state(p=0.4, p_calibrated=0.4)
    assert score_residualized(s, regime_base_rate=0.2) == pytest.approx(0.2)


def test_score_residualized_negative_for_underperformer():
    s = _mk_state(p_calibrated=0.1)
    assert score_residualized(s, regime_base_rate=0.3) == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# Entry gates
# ---------------------------------------------------------------------------


def test_gate_score_above_uses_state_score_field():
    s = _mk_state()
    s2 = State(**{**vars(s), "score": 0.5})
    assert gate_score_above(s2, threshold=0.3) is True
    s3 = State(**{**vars(s), "score": 0.2})
    assert gate_score_above(s3, threshold=0.3) is False


def test_gate_score_above_returns_false_when_score_nan():
    s = _mk_state()  # score defaults to NaN
    assert gate_score_above(s, threshold=0.3) is False


def test_gate_regime_high_passes_at_or_above_qmin():
    assert gate_regime_high(_mk_state(regime_quantile=0.7), q_min=0.7) is True
    assert gate_regime_high(_mk_state(regime_quantile=0.71), q_min=0.7) is True
    assert gate_regime_high(_mk_state(regime_quantile=0.69), q_min=0.7) is False


def test_gate_regime_high_returns_false_when_quantile_nan():
    assert gate_regime_high(_mk_state(regime_quantile=float("nan")), q_min=0.7) is False


def test_gate_unc_below_passes_when_quantile_at_or_below_qmax():
    s_lo = _mk_state(knowledge_unc=0.1, knowledge_unc_quantile=0.5)
    s_hi = _mk_state(knowledge_unc=0.9, knowledge_unc_quantile=0.95)
    assert gate_unc_below(s_lo, q_max=0.9) is True
    assert gate_unc_below(s_hi, q_max=0.9) is False


def test_gate_unc_below_falls_open_when_quantile_nan():
    """When VE not populated, the gate degrades gracefully (returns True);
    the StrategySpec ``requires`` field is what gates the spec out, not the
    primitive."""
    s = _mk_state()  # default knowledge_unc_quantile is NaN
    assert gate_unc_below(s, q_max=0.9) is True


def test_gate_no_concurrent_loss_cluster_blocks_when_underwater():
    deep = _mk_state(cluster_pnl=-0.01)
    shallow = _mk_state(cluster_pnl=-0.002)
    assert gate_no_concurrent_loss_cluster(deep, max_drawdown=0.0075) is False
    assert gate_no_concurrent_loss_cluster(shallow, max_drawdown=0.0075) is True


# ---------------------------------------------------------------------------
# Sizers
# ---------------------------------------------------------------------------


def test_size_constant_returns_default():
    assert size_constant(_mk_state(), default=0.4) == pytest.approx(0.4)


def test_size_kelly_point_zero_when_p_below_breakeven():
    """At b_ratio=1, Kelly = 2p - 1; zero or negative below p=0.5."""
    s = _mk_state(p_calibrated=0.4)
    assert size_kelly_point(s, b_ratio=1.0, fraction=1.0) == pytest.approx(0.0)


def test_size_kelly_point_positive_when_p_above_breakeven():
    s = _mk_state(p_calibrated=0.6)
    # f = 2*0.6 - 1 = 0.2; quarter-Kelly = 0.05
    assert size_kelly_point(s, b_ratio=1.0, fraction=0.25) == pytest.approx(0.05)


def test_size_bayesian_kelly_with_samples_uses_quantile():
    s = _mk_state(p_calibrated=0.55)
    # 11 ensemble draws spanning 0.4..0.7
    samples = np.linspace(0.4, 0.7, 11)
    s2 = State(**{**vars(s), "p_ve_samples": samples})
    f = size_bayesian_kelly(s2, b_ratio=1.0, percentile=0.25, fraction=1.0)
    # At b_ratio=1, Kelly per replicate = 2p - 1. Samples span
    # [0.4, 0.7] in 11 evenly-spaced steps; f_samples spans
    # [-0.2, 0.4]. ``np.quantile(..., 0.25)`` on those 11 evenly-spaced
    # values returns -0.05 exactly (the 25th-percentile linear interp
    # between -0.2 and 0.4 is -0.05). The negative quantile clips to 0.
    expected_q25 = np.quantile(2.0 * samples - 1.0, 0.25)
    assert expected_q25 < 0.0  # sanity: q25 is negative
    assert f == pytest.approx(0.0, abs=1e-12), (
        f"expected Kelly clamp to 0 when q25={expected_q25:.4f} < 0; got {f}"
    )


def test_size_bayesian_kelly_falls_back_to_point_kelly_when_no_samples():
    s = _mk_state(p_calibrated=0.6)
    # No VE samples -> identical to size_kelly_point
    f = size_bayesian_kelly(s, b_ratio=1.0, percentile=0.25, fraction=0.25)
    assert f == pytest.approx(0.05)


def test_size_bayesian_kelly_shrinks_with_wider_posterior():
    """Two states with identical mean p but different posterior spread; the
    wider one (higher implicit knowledge uncertainty) must size smaller."""
    s = _mk_state(p_calibrated=0.6)
    narrow = State(
        **{**vars(s), "p_ve_samples": np.array([0.59, 0.60, 0.60, 0.60, 0.61])}
    )
    wide = State(
        **{**vars(s), "p_ve_samples": np.array([0.40, 0.50, 0.60, 0.70, 0.80])}
    )
    f_narrow = size_bayesian_kelly(narrow, b_ratio=1.0, percentile=0.25, fraction=1.0)
    f_wide = size_bayesian_kelly(wide, b_ratio=1.0, percentile=0.25, fraction=1.0)
    assert f_wide < f_narrow


def test_size_voltarget_overlay_doubles_when_sigma_half():
    out = size_voltarget_overlay(
        base_size=0.5, fast_sigma=0.0005, sigma_target=0.001, max_multiplier=4.0
    )
    assert out == pytest.approx(1.0)


def test_size_voltarget_overlay_capped_at_max_multiplier():
    out = size_voltarget_overlay(
        base_size=0.5, fast_sigma=1e-9, sigma_target=0.001, max_multiplier=4.0
    )
    assert out == pytest.approx(0.5 * 4.0)


def test_size_voltarget_overlay_passthrough_when_sigma_nan():
    out = size_voltarget_overlay(
        base_size=0.5, fast_sigma=float("nan"), sigma_target=0.001
    )
    assert out == pytest.approx(0.5)


def test_size_clip_lower_zero_upper_max():
    assert size_clip(-0.5, max_size=1.0) == 0.0
    assert size_clip(2.0, max_size=1.0) == 1.0
    assert size_clip(0.4, max_size=1.0) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Exit policies
# ---------------------------------------------------------------------------


def test_exit_tp_or_expiry_tp_long():
    pos = _mk_position(side=1, entry_price=100.0, tp_price=100.0 * math.exp(0.0025))
    bar = IntraBar(
        n=101,
        ts=pd.Timestamp("2025-06-01 00:01:00"),
        open=100.05,
        high=100.30,
        low=99.95,
        close=100.20,
    )
    assert exit_tp_or_expiry(pos, bar, k_now=5) == "tp"


def test_exit_tp_or_expiry_expiry_when_no_path_data():
    pos = _mk_position(k_entry=5, expiry_k=6)
    assert exit_tp_or_expiry(pos, intra_bar=None, k_now=6) == "expiry"
    assert exit_tp_or_expiry(pos, intra_bar=None, k_now=5) is None


def test_exit_tp_sl_or_expiry_sl_takes_precedence_when_both_in_same_bar():
    pos = _mk_position(
        side=1, entry_price=100.0,
        tp_price=100.0 * math.exp(0.0025),
        sl_price=100.0 * math.exp(-0.0025),
    )
    # High and low both cross
    bar = IntraBar(
        n=101,
        ts=pd.Timestamp("2025-06-01 00:01:00"),
        open=100.0,
        high=100.30,
        low=99.70,
        close=100.0,
    )
    # Default is pessimistic_sl
    assert exit_tp_sl_or_expiry(pos, bar, k_now=5) == "sl"


def test_exit_tp_sl_or_expiry_pessimistic_sl_explicit():
    """Explicit pessimistic_sl param matches default."""
    pos = _mk_position(
        side=1, entry_price=100.0,
        tp_price=100.0 * math.exp(0.0025),
        sl_price=100.0 * math.exp(-0.0025),
    )
    bar = IntraBar(
        n=101, ts=pd.Timestamp("2025-06-01 00:01:00"),
        open=100.0, high=100.30, low=99.70, close=100.0,
    )
    assert exit_tp_sl_or_expiry(pos, bar, k_now=5, bar_resolution="pessimistic_sl") == "sl"


def test_exit_tp_sl_or_expiry_optimistic_tp_takes_tp_when_both_hit():
    """The optimistic_tp branch resolves TP-and-SL-in-same-bar as TP."""
    pos = _mk_position(
        side=1, entry_price=100.0,
        tp_price=100.0 * math.exp(0.0025),
        sl_price=100.0 * math.exp(-0.0025),
    )
    bar = IntraBar(
        n=101, ts=pd.Timestamp("2025-06-01 00:01:00"),
        open=100.0, high=100.30, low=99.70, close=100.0,
    )
    assert exit_tp_sl_or_expiry(pos, bar, k_now=5, bar_resolution="optimistic_tp") == "tp"


def test_exit_tp_sl_or_expiry_neutral_returns_ambiguous_label():
    """The neutral branch surfaces the ambiguity to the caller."""
    pos = _mk_position(
        side=1, entry_price=100.0,
        tp_price=100.0 * math.exp(0.0025),
        sl_price=100.0 * math.exp(-0.0025),
    )
    bar = IntraBar(
        n=101, ts=pd.Timestamp("2025-06-01 00:01:00"),
        open=100.0, high=100.30, low=99.70, close=100.0,
    )
    assert exit_tp_sl_or_expiry(pos, bar, k_now=5, bar_resolution="neutral") == "tp_or_sl"


def test_exit_tp_sl_or_expiry_unrecognized_resolution_raises():
    """Unknown bar_resolution must raise ValueError, not silently fall through."""
    pos = _mk_position(
        side=1, entry_price=100.0,
        tp_price=100.0 * math.exp(0.0025),
        sl_price=100.0 * math.exp(-0.0025),
    )
    bar = IntraBar(
        n=101, ts=pd.Timestamp("2025-06-01 00:01:00"),
        open=100.0, high=100.30, low=99.70, close=100.0,
    )
    with pytest.raises(ValueError, match="bar_resolution"):
        exit_tp_sl_or_expiry(pos, bar, k_now=5, bar_resolution="bogus")


def test_exit_tp_sl_or_expiry_returns_none_when_no_event():
    pos = _mk_position(side=1, entry_price=100.0)
    bar = IntraBar(
        n=101,
        ts=pd.Timestamp("2025-06-01 00:01:00"),
        open=100.0,
        high=100.05,
        low=99.95,
        close=100.0,
    )
    assert exit_tp_sl_or_expiry(pos, bar, k_now=5) is None


# ---------------------------------------------------------------------------
# Bulk-close triggers
# ---------------------------------------------------------------------------


def test_bulk_on_regime_drop_fires_below_threshold():
    assert bulk_on_regime_drop(_mk_state(regime_quantile=0.4), exit_q=0.5) == "bulk_regime"
    assert bulk_on_regime_drop(_mk_state(regime_quantile=0.6), exit_q=0.5) is None


def test_bulk_on_unc_spike_fires_above_threshold():
    s = _mk_state(knowledge_unc_quantile=0.97)
    assert bulk_on_unc_spike(s, spike_q=0.95) == "bulk_unc"


def test_bulk_on_unc_spike_silent_when_quantile_nan():
    s = _mk_state()
    assert bulk_on_unc_spike(s, spike_q=0.95) is None


def test_bulk_on_cluster_loss_fires_below_negative_cap():
    s = _mk_state(cluster_pnl=-0.01)
    assert bulk_on_cluster_loss(s, cap_log_return=0.0075) == "bulk_cluster_loss"


def test_bulk_on_cluster_loss_silent_when_shallow():
    s = _mk_state(cluster_pnl=-0.005)
    assert bulk_on_cluster_loss(s, cap_log_return=0.0075) is None


# ---------------------------------------------------------------------------
# StrategySpec composition
# ---------------------------------------------------------------------------


def test_strategy_spec_evaluate_entry_and_composed():
    spec = StrategySpec(
        name="t",
        score_fn=score_raw_p,
        entry_gates=(
            lambda s: gate_score_above(s, threshold=0.3),
            lambda s: gate_regime_high(s, q_min=0.7),
        ),
    )
    s_pass = State(**{**vars(_mk_state(regime_quantile=0.8)), "score": 0.4})
    s_fail = State(**{**vars(_mk_state(regime_quantile=0.4)), "score": 0.4})
    assert spec.evaluate_entry(s_pass) is True
    assert spec.evaluate_entry(s_fail) is False


def test_strategy_spec_evaluate_entry_empty_gates_returns_false():
    spec = StrategySpec(name="noop", entry_gates=())
    assert spec.evaluate_entry(_mk_state()) is False


def test_make_baseline_spec_smoke():
    spec = make_baseline_spec(threshold=0.3, size=0.4, cost_per_trade=0.0005)
    assert spec.name == "baseline_label_aligned"
    assert spec.requires == ()
    s = State(**{**vars(_mk_state()), "score": 0.4})
    assert spec.evaluate_entry(s) is True
    assert spec.sizer(s) == pytest.approx(0.4)
    # Label-aligned default: horizon = 1 boundary
    assert spec.risk.max_horizon_boundaries == 1


def test_make_patient_spec_uses_long_horizon_by_default():
    from src.strategy.policy import make_patient_spec
    spec = make_patient_spec()
    assert spec.name == "patient_wait_for_level"
    assert spec.requires == ()
    assert spec.risk.max_horizon_boundaries >= 100  # patient
    assert spec.risk.position_mtm_floor_log_return is None  # no per-position stop by default


def test_make_patient_spec_with_position_stop_sets_mtm_floor():
    from src.strategy.policy import make_patient_spec
    spec = make_patient_spec(use_position_stop=True, position_mtm_floor=-0.005)
    assert spec.risk.position_mtm_floor_log_return == pytest.approx(-0.005)


def test_make_regime_gated_spec_requires_vol_gate_diag():
    spec = make_regime_gated_spec(threshold=0.3, regime_entry_q=0.7)
    assert "vol_gate" in spec.requires


def test_make_bayesian_kelly_spec_requires_ve_diag():
    """vol_gate was de-elevated to advisory in the patient redesign: the
    regime gate's job here is *when* to enter, not point-precision uplift."""
    spec = make_bayesian_kelly_spec()
    assert "ve_diag" in spec.requires
    # vol_gate is NOT a hard requirement
    assert "vol_gate" not in spec.requires


def test_risk_config_defaults_match_phi_constants():
    """3 * PHI is the cluster-loss cap default. Reference ``utils.PHI``
    directly so the test follows constant changes."""
    from src.utils import PHI
    rc = RiskConfig()
    assert rc.cluster_loss_cap == pytest.approx(3 * PHI)
    assert rc.cost_per_trade == pytest.approx(0.0005)


def test_make_bayesian_kelly_spec_gate_drawdown_decoupled_from_cap():
    """``gate_no_concurrent_loss_cluster`` must use ``gate_drawdown_threshold``,
    NOT ``cluster_loss_cap``. If the gate and the cap were equal, the cap
    would always liquidate before the gate could matter — making the gate
    a no-op.

    Verify by: build a spec with cap=0.020 and the default gate
    (which should be 0.5 * cap = 0.010). At cluster_pnl = -0.012:
    - the bulk-close cap (0.020) hasn't been hit (pnl > -0.020),
    - but the gate threshold (0.010) HAS (pnl < -0.010),
    => the gate must reject new entries while bulk_close stays silent.
    """
    spec = make_bayesian_kelly_spec(
        threshold=0.30, cluster_loss_cap=0.020,
        # gate_drawdown_threshold defaults to 0.5 * cluster_loss_cap = 0.010
    )
    # State with cluster_pnl = -0.012: between the gate's 0.010 and the cap's 0.020
    s = _mk_state(
        p=0.5, p_calibrated=0.5,
        cluster_pnl=-0.012,
        regime_quantile=0.8,
        knowledge_unc=0.01,
        knowledge_unc_quantile=0.1,
    )
    s = State(**{**vars(s), "score": 0.5})  # plant a passing score

    # The entry-gate composition AND-chains every gate. With cluster_pnl
    # at -0.012, ``gate_no_concurrent_loss_cluster(s, 0.010)`` should reject.
    assert spec.evaluate_entry(s) is False, (
        "the drawdown-gate should fire at half the cap, suppressing entries"
    )
    # And bulk_close should NOT trigger at the same cluster_pnl (cap is 0.020)
    assert spec.bulk_close(s) is None, (
        "bulk_close should NOT fire at cluster_pnl=-0.012 with cap=0.020"
    )


def test_make_bayesian_kelly_spec_explicit_gate_threshold():
    """An explicit ``gate_drawdown_threshold`` overrides the default."""
    spec = make_bayesian_kelly_spec(
        threshold=0.30, cluster_loss_cap=0.020,
        gate_drawdown_threshold=0.005,  # tighter than 0.5*0.020=0.010
    )
    # cluster_pnl = -0.006 is past the explicit gate (0.005) but well above the cap
    s = _mk_state(
        p=0.5, p_calibrated=0.5,
        cluster_pnl=-0.006,
        regime_quantile=0.8,
        knowledge_unc=0.01,
        knowledge_unc_quantile=0.1,
    )
    s = State(**{**vars(s), "score": 0.5})
    assert spec.evaluate_entry(s) is False
    assert spec.bulk_close(s) is None
