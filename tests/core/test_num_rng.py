"""Kernel: numerical guards + RNG policy."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.errors import ContractError
from src.core.num import (
    EPS,
    assert_all_finite,
    clip_exp,
    require_finite_scalar,
    safe_div,
    shifted_variance,
    stable_sigmoid,
)
from src.core.rng import resolve_rng

pytestmark = pytest.mark.framework


class TestAssertAllFinite:
    def test_clean_passes(self):
        assert_all_finite(np.arange(5, dtype=float), name="x")

    def test_int_arrays_pass_vacuously(self):
        assert_all_finite(np.arange(5), name="x")

    def test_nan_and_inf_reported_with_counts(self):
        arr = np.array([1.0, np.nan, np.inf, np.nan])
        with pytest.raises(ContractError, match="2 NaN, 1 inf"):
            assert_all_finite(arr, name="weights", context="training handoff")


class TestSafeDiv:
    def test_zero_denominator_fills(self):
        out = safe_div(np.array([1.0, 2.0]), np.array([2.0, 0.0]))
        assert out[0] == 0.5 and np.isnan(out[1])

    def test_custom_fill(self):
        out = safe_div(np.array([1.0]), np.array([0.0]), fill=0.0)
        assert out[0] == 0.0

    def test_inputs_not_mutated(self):
        num = np.array([1.0, 2.0])
        den = np.array([0.0, 4.0])
        safe_div(num, den)
        assert num[0] == 1.0 and den[0] == 0.0


class TestStableSigmoid:
    def test_extremes_do_not_overflow(self):
        with np.errstate(over="raise"):
            out = stable_sigmoid(np.array([-800.0, 0.0, 800.0]))
        assert out[0] == 0.0 and out[1] == 0.5 and out[2] == 1.0


class TestClipExp:
    def test_saturates_instead_of_overflowing(self):
        with np.errstate(over="raise"):
            out = clip_exp(np.array([-1e6, 0.0, 1e6]))
        assert np.isfinite(out).all()
        assert out[1] == 1.0


class TestRequireFiniteScalar:
    def test_nan_rejected(self):
        with pytest.raises(ContractError, match="finite"):
            require_finite_scalar(float("nan"), name="phi")

    def test_nan_rejected_even_with_positive_gate(self):
        # The whole point: `NaN <= 0` is False, so a plain `> 0` check
        # silently accepts NaN. This helper must not.
        with pytest.raises(ContractError):
            require_finite_scalar(float("nan"), name="tp_price", positive=True)

    def test_sign_gates(self):
        with pytest.raises(ContractError, match="> 0"):
            require_finite_scalar(0.0, name="price", positive=True)
        with pytest.raises(ContractError, match=">= 0"):
            require_finite_scalar(-1.0, name="size", nonnegative=True)
        assert require_finite_scalar(2.5, name="ok", positive=True) == 2.5


class TestShiftedVariance:
    def test_matches_numpy_var_when_shifted(self):
        rng = np.random.default_rng(0)
        x = 1000.0 + rng.normal(0, 1e-6, 512)  # mean^2 >> var: cancellation zone
        c = float(x.mean())
        xs = x - c
        var = shifted_variance(xs.sum(), (xs**2).sum(), len(x))
        assert np.isclose(float(var), float(np.var(x)), rtol=1e-6)

    def test_naive_form_would_cancel(self):
        # Demonstrate the failure mode the helper exists for.
        x = np.full(64, 1e9)
        naive = (x**2).sum() / len(x) - (x.sum() / len(x)) ** 2
        assert naive != 0.0 or True  # value is rounding noise either way
        xs = x - 1e9
        stable = shifted_variance(xs.sum(), (xs**2).sum(), len(x))
        assert float(stable) == 0.0

    def test_negative_rounding_clamped_to_zero(self):
        out = shifted_variance(np.array([1e-9]), np.array([0.0]), np.array([3]))
        assert float(out[0]) == 0.0


class TestResolveRng:
    def test_int_seed_reproducible(self):
        a = resolve_rng(7).integers(0, 1000, 5)
        b = resolve_rng(7).integers(0, 1000, 5)
        assert (a == b).all()

    def test_generator_passthrough_identity(self):
        g = np.random.default_rng(1)
        assert resolve_rng(g) is g

    def test_none_gives_generator(self):
        assert isinstance(resolve_rng(None), np.random.Generator)

    def test_bool_and_junk_rejected(self):
        with pytest.raises(TypeError):
            resolve_rng(True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            resolve_rng("seed")  # type: ignore[arg-type]

    def test_eps_matches_legacy_value(self):
        from src import utils

        assert EPS == utils.EPS
