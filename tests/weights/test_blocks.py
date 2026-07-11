"""Weight blocks: bit-exact parity with the legacy utils implementations.

The legacy functions remain the oracle until their Phase-5 retirement —
every configuration axis is exercised against them over randomized and
degenerate inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import utils
from src.core.errors import ConfigError
from src.labels.barrier import BarrierSpec
from src.weights import (
    BarrierDistanceWeight,
    TimeDiscountWeight,
    TrainingWeights,
    UniquenessWeight,
)

pytestmark = pytest.mark.weights

PHI = 0.0025


def _m_k(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(PHI * 0.6, PHI * 0.8, n)  # mixed classes around the barrier


class TestDefaultsMatchLegacyConstants:
    def test_distance_defaults(self):
        d = BarrierDistanceWeight()
        assert d.w_max == utils.WEIGHT_DIST_W_MAX
        assert d.q_tail == utils.WEIGHT_DIST_Q_TAIL
        assert d.w_max_pos == utils.WEIGHT_DIST_W_MAX_POS
        assert d.q_tail_pos == utils.WEIGHT_DIST_Q_TAIL_POS
        assert d.use_pos == utils.WEIGHT_DIST_USE_POSITIVE

    def test_time_defaults(self):
        t = TimeDiscountWeight()
        assert t.r == utils.WEIGHT_TIME_R
        assert t.delta == utils.WEIGHT_TIME_DELTA

    def test_combined_defaults(self):
        c = TrainingWeights()
        assert c.distance.enabled == utils.WEIGHT_USE_BARRIER_DISTANCE
        assert c.time.enabled == utils.WEIGHT_USE_TIME_DISCOUNT
        assert c.normalize == utils.WEIGHT_NORMALIZE


class TestBarrierDistanceParity:
    @pytest.mark.parametrize("use_pos", [False, True])
    @pytest.mark.parametrize("enabled", [True, False])
    @pytest.mark.parametrize("seed", [0, 7])
    def test_bit_exact_vs_legacy(self, use_pos, enabled, seed):
        m_k = _m_k(4_000, seed)
        block = BarrierDistanceWeight(
            w_max=5.0, q_tail=0.001, w_max_pos=2.0, q_tail_pos=0.01,
            use_pos=use_pos, enabled=enabled,
        )
        got = block.compute(m_k, phi=PHI)
        exp_w, exp_info = utils.compute_barrier_distance_weight(
            m_k, PHI, w_max=5.0, q_tail=0.001, w_max_pos=2.0, q_tail_pos=0.01,
            use_pos=use_pos, enabled=enabled,
        )
        assert np.array_equal(got.values, exp_w)
        assert got.info == exp_info

    @pytest.mark.parametrize("m_k", [
        np.array([]),                        # empty
        np.full(64, PHI * 2.0),              # all positive
        np.full(64, -PHI),                   # all negative
        np.zeros(64),                        # constant at zero distance... below barrier
    ])
    def test_degenerate_inputs_match_legacy(self, m_k):
        got = BarrierDistanceWeight().compute(m_k, phi=PHI)
        exp_w, exp_info = utils.compute_barrier_distance_weight(m_k, PHI)
        assert np.array_equal(got.values, exp_w)
        assert got.info == exp_info

    def test_input_not_mutated(self):
        m_k = _m_k(512, 3)
        before = m_k.copy()
        BarrierDistanceWeight().compute(m_k, phi=PHI)
        assert np.array_equal(m_k, before)

    def test_config_validation(self):
        with pytest.raises(ConfigError, match="q_tail"):
            BarrierDistanceWeight(q_tail=0.0)
        with pytest.raises(ConfigError, match="w_max"):
            BarrierDistanceWeight(w_max=0.5)
        # Disabled blocks skip validation, matching the legacy behavior of
        # only validating when enabled.
        BarrierDistanceWeight(q_tail=0.0, enabled=False)


class TestTimeDiscountParity:
    @pytest.mark.parametrize("r,delta", [(0.0, 0.99999), (0.5, 0.999), (1.0, 0.9)])
    @pytest.mark.parametrize("shuffled", [False, True])
    def test_bit_exact_vs_legacy(self, r, delta, shuffled):
        n = 2_000
        rng = np.random.default_rng(11)
        k_index = rng.permutation(n) if shuffled else None
        got = TimeDiscountWeight(r=r, delta=delta).compute(n, k_index=k_index)
        exp_w, exp_info = utils.compute_time_discount_weight(
            n, r=r, delta=delta, k_index=k_index, enabled=True
        )
        assert np.array_equal(got.values, exp_w)
        assert got.info == exp_info

    def test_empty_and_disabled(self):
        got = TimeDiscountWeight().compute(0)
        exp_w, exp_info = utils.compute_time_discount_weight(0, enabled=True)
        assert np.array_equal(got.values, exp_w)
        assert got.info == exp_info
        got_off = TimeDiscountWeight(enabled=False).compute(128)
        exp_w_off, _ = utils.compute_time_discount_weight(128, enabled=False)
        assert np.array_equal(got_off.values, exp_w_off)


class TestTrainingWeightsParity:
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("use_time", [False, True])
    def test_bit_exact_vs_legacy(self, normalize, use_time):
        m_k = _m_k(3_000, 5)
        block = TrainingWeights(
            distance=BarrierDistanceWeight(use_pos=True),
            time=TimeDiscountWeight(r=0.3, delta=0.999, enabled=use_time),
            normalize=normalize,
        )
        got = block.compute(m_k, phi=PHI)
        exp_c, exp_d, exp_t, exp_info = utils.compute_training_weights(
            m_k, PHI, use_dist=True, use_time=use_time, use_pos=True,
            r=0.3, delta=0.999, normalize=normalize,
        )
        assert np.array_equal(got.combined, exp_c)
        assert np.array_equal(got.distance, exp_d)
        assert np.array_equal(got.time, exp_t)
        assert got.info == exp_info


class TestUniquenessWeight:
    def test_derives_from_barrier_spec(self):
        from src.analytics.sampling import compute_uniqueness_weights

        spec = BarrierSpec(horizon=20, upper=PHI, stride=1)
        got = UniquenessWeight(normalize=False).compute(500, spec=spec)
        exp = compute_uniqueness_weights(500, 20, bar_stride=1, normalize=False)
        assert np.array_equal(got, exp)
