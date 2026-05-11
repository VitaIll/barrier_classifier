"""Tests for the diagnostics module.

Use synthetic data with known structure so each pass/fail outcome is
hand-traceable."""

from __future__ import annotations

import numpy as np
import pytest

from src.strategy.diagnostics import (
    DiagnosticResult,
    cluster_persistence_diagnostic,
    passed_flags,
    run_all_diagnostics,
    ve_diagnostic,
    vol_gate_diagnostic,
    within_regime_signal_diagnostic,
)

pytestmark = pytest.mark.strategy_v1


# ---------------------------------------------------------------------------
# DiagnosticResult basics
# ---------------------------------------------------------------------------


def test_diagnostic_result_to_dict_roundtrips_fields():
    r = DiagnosticResult(
        name="x", passed=True, value=0.5, threshold=0.3,
        message="ok", details={"k": 1},
    )
    d = r.to_dict()
    assert d["passed"] is True
    assert d["value"] == 0.5
    assert d["threshold"] == 0.3
    assert d["details"] == {"k": 1}


# ---------------------------------------------------------------------------
# ve_diagnostic
# ---------------------------------------------------------------------------


def test_ve_diagnostic_passes_when_unc_predicts_mistakes():
    """High MI → much lower hit rate at same p. Built so the high-MI half
    of the top-p subset has near-zero precision."""
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.1, 0.9, size=n)
    unc = rng.uniform(0.0, 1.0, size=n)
    # When unc low: hit rate matches p; when high: hit rate ≈ 0
    p_true = np.where(unc < 0.5, p, p * 0.05)
    y = (rng.uniform(size=n) < p_true).astype(int)
    res = ve_diagnostic(y, mean_p=p, knowledge_unc=unc)
    assert res.passed, res.message
    assert res.value > 0.04


def test_ve_diagnostic_fails_when_unc_uninformative():
    """MI independent of y → low/high MI subsets have indistinguishable precision."""
    rng = np.random.default_rng(1)
    n = 5000
    p = rng.uniform(0.1, 0.9, size=n)
    unc = rng.uniform(size=n)  # independent of y
    y = (rng.uniform(size=n) < p).astype(int)
    res = ve_diagnostic(y, mean_p=p, knowledge_unc=unc)
    assert not res.passed
    # uplift should be small (< pass threshold of 0.04)
    assert abs(res.value) < 0.04


def test_ve_diagnostic_handles_constant_unc():
    rng = np.random.default_rng(2)
    n = 1000
    p = rng.uniform(size=n)
    unc = np.full(n, 0.5)
    y = (rng.uniform(size=n) < p).astype(int)
    res = ve_diagnostic(y, mean_p=p, knowledge_unc=unc)
    assert not res.passed
    assert "constant" in res.message.lower()


def test_ve_diagnostic_handles_too_few_rows():
    res = ve_diagnostic(
        y=np.array([0, 1, 0, 1]),
        mean_p=np.array([0.1, 0.5, 0.3, 0.7]),
        knowledge_unc=np.array([0.1, 0.2, 0.3, 0.4]),
    )
    assert not res.passed
    assert "insufficient" in res.message.lower() or "too few" in res.message.lower()


# ---------------------------------------------------------------------------
# vol_gate_diagnostic
# ---------------------------------------------------------------------------


def test_vol_gate_passes_when_high_vol_lifts_precision():
    """Construct so high-regime rows are more accurate at top-p."""
    rng = np.random.default_rng(0)
    n = 4000
    p = rng.uniform(size=n)
    regime = rng.uniform(size=n)
    # Hit probability increases with regime: p_true = p * (0.5 + regime)
    p_true = np.clip(p * (0.5 + regime), 0, 1)
    y = (rng.uniform(size=n) < p_true).astype(int)
    res = vol_gate_diagnostic(y, p, regime, pass_uplift=0.05)
    assert res.passed, res.message


def test_vol_gate_fails_when_regime_uninformative():
    rng = np.random.default_rng(1)
    n = 4000
    p = rng.uniform(size=n)
    regime = rng.uniform(size=n)  # independent
    y = (rng.uniform(size=n) < p).astype(int)
    res = vol_gate_diagnostic(y, p, regime, pass_uplift=0.05)
    assert not res.passed


def test_vol_gate_handles_too_few_candidates():
    y = np.array([0, 1, 0, 1] * 5)
    p = np.linspace(0, 1, 20)
    regime = np.linspace(0, 1, 20)
    res = vol_gate_diagnostic(y, p, regime, min_trades=200)
    assert not res.passed


# ---------------------------------------------------------------------------
# cluster_persistence_diagnostic
# ---------------------------------------------------------------------------


def test_cluster_persistence_passes_when_p_autocorrelated():
    """Build an AR(1) on p with strong autocorrelation."""
    rng = np.random.default_rng(0)
    n = 2000
    p = np.zeros(n)
    p[0] = rng.uniform()
    rho = 0.9
    for t in range(1, n):
        p[t] = rho * p[t - 1] + (1 - rho) * rng.uniform()
    res = cluster_persistence_diagnostic(p, pass_lift=1.5)
    assert res.passed, res.message
    assert res.value > 1.5


def test_cluster_persistence_fails_when_p_iid():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)
    res = cluster_persistence_diagnostic(p, pass_lift=1.5)
    assert not res.passed
    # Lift should be ~ 1.0 for iid
    assert 0.7 < res.value < 1.3


def test_cluster_persistence_handles_short_input():
    p = np.linspace(0, 1, 50)
    res = cluster_persistence_diagnostic(p)
    assert not res.passed
    assert "too few" in res.message.lower()


# ---------------------------------------------------------------------------
# within_regime_signal_diagnostic
# ---------------------------------------------------------------------------


def test_within_regime_signal_passes_when_p_discriminates_within():
    """p has signal at every regime level."""
    rng = np.random.default_rng(0)
    n = 3000
    regime = rng.uniform(size=n)
    # Within each tercile, p strongly correlates with y
    p = rng.uniform(size=n)
    y = (rng.uniform(size=n) < (0.05 + 0.7 * p)).astype(int)
    res = within_regime_signal_diagnostic(y, p, regime, pass_threshold=0.55)
    assert res.passed, res.message


def test_within_regime_signal_fails_when_p_random():
    rng = np.random.default_rng(0)
    n = 3000
    regime = rng.uniform(size=n)
    p = rng.uniform(size=n)
    y = rng.binomial(1, 0.3, size=n).astype(int)
    res = within_regime_signal_diagnostic(y, p, regime, pass_threshold=0.55)
    assert not res.passed


# ---------------------------------------------------------------------------
# bundle helpers
# ---------------------------------------------------------------------------


def test_run_all_diagnostics_skips_ve_when_no_unc_supplied():
    rng = np.random.default_rng(0)
    n = 3000
    p = rng.uniform(size=n)
    regime = rng.uniform(size=n)
    y = rng.binomial(1, 0.2, size=n).astype(int)
    out = run_all_diagnostics(y, p, regime)
    assert "ve_diag" in out
    assert out["ve_diag"].passed is False
    assert "not provided" in out["ve_diag"].message.lower()


def test_run_all_diagnostics_includes_ve_when_unc_supplied():
    rng = np.random.default_rng(0)
    n = 3000
    p = rng.uniform(size=n)
    regime = rng.uniform(size=n)
    unc = rng.uniform(size=n)
    y = rng.binomial(1, 0.2, size=n).astype(int)
    out = run_all_diagnostics(y, p, regime, mean_p_ve=p, knowledge_unc=unc)
    assert set(out.keys()) >= {"ve_diag", "vol_gate", "cluster_persistence", "within_regime_signal"}


def test_passed_flags_returns_bool_dict():
    rng = np.random.default_rng(0)
    n = 3000
    p = rng.uniform(size=n)
    regime = rng.uniform(size=n)
    y = rng.binomial(1, 0.2, size=n).astype(int)
    out = run_all_diagnostics(y, p, regime)
    flags = passed_flags(out)
    assert all(isinstance(v, bool) for v in flags.values())
    assert set(flags.keys()) == set(out.keys())
