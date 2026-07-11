"""Regression tests for the 2026-07-11 numerical-guard fixes.

One test class per finding from docs/REVIEW_2026-07-11.md §2:

- N4: NaN prices/sizes must not construct valid positions or open trades
- N1: single-class threshold sweep fails typed, not NaN->int crash
- N5: sweep Sharpe variance is cancellation-stable
- N2: quantile bucketing survives tied/point-mass regimes
- N3: psi rejects empty inputs
- N7: brier-decomposition bootstrap follows the NaN-tolerant contract
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.framework


# ---------------------------------------------------------------------------
# N4 — inventory + simulator NaN guards
# ---------------------------------------------------------------------------


class TestPositionFinitenessInvariants:
    def _pos_kwargs(self, **over):
        base = dict(
            k_entry=1,
            ts_entry=pd.Timestamp("2025-01-01 00:20"),
            side=1,
            size=0.1,
            entry_price=100.0,
            tp_price=100.25,
            sl_price=None,
            expiry_k=2,
        )
        base.update(over)
        return base

    def test_nan_tp_price_rejected(self):
        from src.strategy.inventory import Position

        # The original bug: `NaN <= 0` is False, so this constructed a
        # position whose take-profit could never fire.
        with pytest.raises(ValueError, match="tp_price"):
            Position(**self._pos_kwargs(tp_price=float("nan")))

    def test_nan_entry_price_and_size_rejected(self):
        from src.strategy.inventory import Position

        with pytest.raises(ValueError, match="entry_price"):
            Position(**self._pos_kwargs(entry_price=float("nan")))
        with pytest.raises(ValueError, match="size"):
            Position(**self._pos_kwargs(size=float("nan")))

    def test_nan_and_nonpositive_sl_rejected_but_none_allowed(self):
        from src.strategy.inventory import Position

        Position(**self._pos_kwargs(sl_price=None))
        Position(**self._pos_kwargs(sl_price=99.0))
        with pytest.raises(ValueError, match="sl_price"):
            Position(**self._pos_kwargs(sl_price=float("nan")))
        with pytest.raises(ValueError, match="sl_price"):
            Position(**self._pos_kwargs(sl_price=0.0))

    def test_mtm_and_close_reject_nan_prices(self):
        from src.strategy.inventory import Position, close_position

        pos = Position(**self._pos_kwargs())
        with pytest.raises(ValueError, match="current_price"):
            pos.mtm_log_return(float("nan"))
        with pytest.raises(ValueError, match="exit_price"):
            close_position(
                pos,
                k_exit=2,
                ts_exit=pd.Timestamp("2025-01-01 00:40"),
                exit_price=float("nan"),
                exit_reason="tp",
            )


class TestSimulatorPhiGuard:
    def _run(self, phi_values):
        from src.strategy.policy import (
            RiskConfig,
            StrategySpec,
            exit_tp_or_expiry,
            score_raw_p,
        )
        from src.strategy.simulator import SimConfig, simulate

        n = len(phi_values)
        ts = pd.date_range("2025-01-01", periods=n, freq="20min")
        cache = pd.DataFrame(
            {
                "ts": ts,
                "k": np.arange(n),
                "p": 0.9,
                "regime": 1.0,
                "phi": phi_values,
                "close": 100.0,
                "high": 100.1,
                "low": 99.9,
                "y": 0.0,
                "m_k": 0.0,
            }
        )
        raw = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
            },
            index=pd.date_range("2025-01-01", periods=n * 20, freq="1min"),
        )
        spec = StrategySpec(
            name="always_enter",
            entry_gates=(lambda s: True,),
            score_fn=score_raw_p,
            exit_policy=exit_tp_or_expiry,
            risk=RiskConfig(max_open_positions=10, max_gross_size=10.0),
        )
        return simulate(cache, raw, spec, config=SimConfig(M=20))

    def test_finite_phi_runs(self):
        result = self._run([0.0025, 0.0025, 0.0025])
        assert len(result.equity) == 3

    def test_nan_phi_at_entry_raises_with_diagnosis(self):
        with pytest.raises(ValueError, match="phi"):
            self._run([0.0025, float("nan"), 0.0025])


# ---------------------------------------------------------------------------
# N1 + N5 — edge.py threshold sweep
# ---------------------------------------------------------------------------


def _sweep_cache(n=400, seed=0, single_class=False, r_shift=0.0, r_std=1e-9):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.99, n)
    y = np.zeros(n, dtype=int) if single_class else (rng.uniform(0, 1, n) < p).astype(int)
    ts = pd.date_range("2025-01-01", periods=n, freq="20min")
    return pd.DataFrame(
        {
            "split": "test",
            "ts": ts,
            "y": y,
            "p": p,
            "r_realized": r_shift + rng.normal(0.0, r_std, n),
        }
    )


class TestThresholdSweepGuards:
    def test_single_class_split_raises_typed_diagnosis(self):
        from src.analytics.edge import bootstrap_threshold_sweep

        with pytest.raises(ValueError, match="single-class"):
            bootstrap_threshold_sweep(
                _sweep_cache(single_class=True), split="test", B=8
            )

    def test_sharpe_variance_is_cancellation_stable(self):
        from src.analytics.edge import OutcomeModel, bootstrap_threshold_sweep

        # Returns with mean 1.0 and std 1e-7 (true var 1e-14, well above the
        # 1e-18 degenerate-variance guard): the naive E[X^2]-E[X]^2 on raw
        # cumulative sums carries ~1e-13 cancellation error here — larger
        # than the variance itself — while the shifted form is exact.
        cache = _sweep_cache(r_shift=1.0, r_std=1e-7)
        om = OutcomeModel(use_realized_return=True, cost_per_trade=0.0)
        out = bootstrap_threshold_sweep(
            cache,
            split="test",
            thresholds=np.array([0.05]),
            outcome_model=om,
            B=4,
            stratify=False,
        )
        sel = cache[cache["p"] >= 0.05]["r_realized"].to_numpy()
        expected = sel.mean() / sel.std(ddof=0)
        got = float(out["sharpe_per_trade"].iloc[0])
        assert np.isfinite(got)
        assert np.isclose(got, expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# N2 — tie-robust quantile buckets
# ---------------------------------------------------------------------------


class TestQuantileBuckets:
    def test_constant_regime_no_crash_single_bucket(self):
        from src.analytics.metrics import quantile_buckets

        out = quantile_buckets(np.full(50, 1.0), ["low", "med", "high"])
        assert set(out.dropna().unique()) == {"low"}

    def test_normal_case_three_buckets(self):
        from src.analytics.metrics import quantile_buckets

        out = quantile_buckets(np.arange(300, dtype=float), ["low", "med", "high"])
        assert set(out.unique()) == {"low", "med", "high"}

    def test_nan_maps_to_nan(self):
        from src.analytics.metrics import quantile_buckets

        vals = np.array([1.0, np.nan, 2.0, 3.0])
        out = quantile_buckets(vals, ["low", "med", "high"])
        assert pd.isna(out.iloc[1])

    def test_bootstrap_metrics_by_regime_survives_point_mass(self):
        from src.analytics.metrics import bootstrap_metrics_by_regime

        rng = np.random.default_rng(1)
        n = 300
        y = rng.integers(0, 2, n)
        p = np.clip(rng.normal(0.5, 0.2, n), 0.01, 0.99)
        regime = np.full(n, 2.5)  # point mass: legacy qcut raised here
        out = bootstrap_metrics_by_regime(y, p, regime, B=8)
        assert isinstance(out, dict)

    def test_conditional_precision_survives_point_mass(self):
        from src.analytics.degradation import conditional_precision

        rng = np.random.default_rng(2)
        n = 300
        cache = pd.DataFrame(
            {
                "split": "test",
                "ts": pd.date_range("2025-01-01", periods=n, freq="20min"),
                "y": rng.integers(0, 2, n),
                "p": np.clip(rng.normal(0.5, 0.2, n), 0.01, 0.99),
                "regime": np.full(n, 7.0),  # point mass
            }
        )
        out = conditional_precision(cache, split="test", threshold=0.5)
        assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# N3 + N7 — psi guard, brier bootstrap NaN contract
# ---------------------------------------------------------------------------


class TestPsiAndBrierContracts:
    def test_psi_empty_inputs_rejected(self):
        from src.analytics.degradation import psi

        with pytest.raises(ValueError, match="non-empty"):
            psi(np.array([]), np.array([0.5]))
        with pytest.raises(ValueError, match="non-empty"):
            psi(np.array([0.5]), np.array([]))

    def test_brier_bootstrap_reports_b_effective_and_uses_nanquantile(self):
        from src.analytics.degradation import bootstrap_brier_decomposition

        rng = np.random.default_rng(3)
        n = 240
        y = rng.integers(0, 2, n)
        p = np.clip(rng.normal(0.5, 0.2, n), 0.01, 0.99)
        out = bootstrap_brier_decomposition(y, p, B=16, seed=0)
        for k, res in out.items():
            assert res.B_effective == int(np.isfinite(res.samples).sum()), k
            assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high), k


# ---------------------------------------------------------------------------
# cohorts nanquantile (N7 companion)
# ---------------------------------------------------------------------------


class TestCohortShapDiffNaNTolerance:
    def test_ci_finite_despite_nan_shap_rows(self):
        from src.analytics.cohorts import bootstrap_shap_diff

        rng = np.random.default_rng(4)
        n, f = 120, 3
        shap = rng.normal(0, 1, (n, f))
        shap[5, :] = np.nan  # one poisoned row -> some NaN replicates
        cohorts = np.array(["FP", "FN"] * (n // 2))
        out = bootstrap_shap_diff(
            shap, cohorts, [f"f{i}" for i in range(f)], B=32, seed=0
        )
        assert np.isfinite(out["shap_diff_ci_low"]).all()
        assert np.isfinite(out["shap_diff_ci_high"]).all()
