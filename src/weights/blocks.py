"""Training-weight blocks — configuration as objects, computation stateless.

Faithful ports of the legacy ``utils.compute_barrier_distance_weight`` /
``compute_time_discount_weight`` / ``compute_training_weights`` (numerics
identical, pinned by the cross-implementation parity suite) with the
module-global default constants replaced by explicit frozen configuration:

    weights = TrainingWeights(distance=BarrierDistanceWeight(w_max=5.0))
    result = weights.compute(m_k, phi=spec.upper)
    pool_weight = result.combined * uniqueness

Every ``compute`` is pure — inputs are never mutated, results are new
arrays plus an ``info`` dict matching the legacy diagnostic payload
(notebook metadata compatibility).

``UniquenessWeight`` bridges the label-overlap weighting in
``src.analytics.sampling`` to a :class:`~src.labels.barrier.BarrierSpec`,
so the horizon/stride used for weights can never drift from the label
definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.core.errors import ConfigError

# Defaults mirror the legacy ``utils.WEIGHT_*`` constants (asserted equal
# in the parity suite; duplicated here because utils must stay importable
# without this package during the migration).
_DEF_W_MAX = 5.0
_DEF_Q_TAIL = 0.001
_DEF_W_MAX_POS = 2.0
_DEF_Q_TAIL_POS = 0.01
_DEF_USE_POSITIVE = False
_DEF_TIME_R = 0.0
_DEF_TIME_DELTA = 0.99999


@dataclass(frozen=True)
class WeightResult:
    """One weighting stage's output: values plus the diagnostic payload."""

    values: np.ndarray
    info: dict[str, Any]


@dataclass(frozen=True)
class CombinedWeightResult:
    combined: np.ndarray
    distance: np.ndarray
    time: np.ndarray
    info: dict[str, Any]


@dataclass(frozen=True)
class BarrierDistanceWeight:
    """Continuous-capped exponential up-weighting by barrier distance.

    Negative-class samples (``m_k < phi``) are up-weighted by how far the
    excursion fell short of the barrier; optionally positive-class samples
    by how far beyond it they ran. Lambdas derive as ``log(w_max)/d*`` for
    continuity at the cap.
    """

    w_max: float = _DEF_W_MAX
    q_tail: float = _DEF_Q_TAIL
    w_max_pos: float = _DEF_W_MAX_POS
    q_tail_pos: float = _DEF_Q_TAIL_POS
    use_pos: bool = _DEF_USE_POSITIVE
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.enabled:
            if not (0.0 < self.q_tail < 1.0):
                raise ConfigError(f"q_tail must be in (0, 1), got {self.q_tail}")
            if self.w_max < 1.0:
                raise ConfigError(f"w_max must be >= 1.0, got {self.w_max}")
            if self.use_pos:
                if not (0.0 < self.q_tail_pos < 1.0):
                    raise ConfigError(
                        f"q_tail_pos must be in (0, 1), got {self.q_tail_pos}"
                    )
                if self.w_max_pos < 1.0:
                    raise ConfigError(
                        f"w_max_pos must be >= 1.0, got {self.w_max_pos}"
                    )

    def compute(self, m_k: np.ndarray, *, phi: float) -> WeightResult:
        m = np.asarray(m_k, dtype=float)
        n = len(m)
        w_dist = np.ones(n, dtype=float)

        pos_mask = m >= phi
        neg_mask = ~pos_mask
        n_positive = int(pos_mask.sum())
        n_negative = int(neg_mask.sum())

        d_k = np.maximum(0.0, phi - m)
        g_k = np.maximum(0.0, m - phi)

        d_star = d_max = lam_neg = 0.0
        n_capped_neg = 0
        g_star = g_max = lam_pos = 0.0
        n_capped_pos = 0

        if self.enabled:
            if n_negative > 0:
                d_neg = d_k[neg_mask]
                d_star = float(np.quantile(d_neg, 1.0 - self.q_tail))
                d_max = float(d_neg.max())
                if d_star > 0.0 and self.w_max > 1.0:
                    lam_neg = math.log(self.w_max) / d_star
                    log_cap = math.log(self.w_max)
                    exp_arg = np.minimum(lam_neg * d_neg, log_cap)
                    w_neg = np.exp(exp_arg)
                    n_capped_neg = int((d_neg >= d_star).sum())
                else:
                    w_neg = np.ones_like(d_neg)
                w_dist[neg_mask] = w_neg

            if self.use_pos and n_positive > 0:
                g_pos = g_k[pos_mask]
                g_star = float(np.quantile(g_pos, 1.0 - self.q_tail_pos))
                g_max = float(g_pos.max())
                if g_star > 0.0 and self.w_max_pos > 1.0:
                    lam_pos = math.log(self.w_max_pos) / g_star
                    log_cap = math.log(self.w_max_pos)
                    exp_arg = np.minimum(lam_pos * g_pos, log_cap)
                    w_pos = np.exp(exp_arg)
                    n_capped_pos = int((g_pos >= g_star).sum())
                else:
                    w_pos = np.ones_like(g_pos)
                w_dist[pos_mask] = w_pos

        max_weight_neg = 1.0 if n_negative == 0 else float(w_dist[neg_mask].max())
        max_weight_pos = 1.0 if n_positive == 0 else float(w_dist[pos_mask].max())
        weight_range = (
            (1.0, 1.0) if n == 0 else (float(w_dist.min()), float(w_dist.max()))
        )
        info = {
            "enabled": bool(self.enabled),
            "enabled_pos": bool(self.enabled and self.use_pos),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "d_star": d_star,
            "g_star": g_star,
            "lambda": lam_neg,
            "lambda_pos": lam_pos,
            "d_max": d_max,
            "g_max": g_max,
            "n_capped": n_capped_neg,
            "n_capped_pos": n_capped_pos,
            "max_weight_neg": max_weight_neg,
            "max_weight_pos": max_weight_pos,
            "weight_range": weight_range,
            "params": {
                "phi": float(phi),
                "w_max": float(self.w_max),
                "q_tail": float(self.q_tail),
                "w_max_neg": float(self.w_max),
                "q_tail_neg": float(self.q_tail),
                "w_max_pos": float(self.w_max_pos),
                "q_tail_pos": float(self.q_tail_pos),
                "use_pos": bool(self.use_pos),
            },
        }
        return WeightResult(values=w_dist, info=info)


@dataclass(frozen=True)
class TimeDiscountWeight:
    """Geometric decay into the past over the oldest ``r`` fraction of rows."""

    r: float = _DEF_TIME_R
    delta: float = _DEF_TIME_DELTA
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.enabled:
            if not (0.0 <= self.r <= 1.0):
                raise ConfigError(f"r must be in [0, 1], got {self.r}")
            if not (0.0 < self.delta <= 1.0):
                raise ConfigError(f"delta must be in (0, 1], got {self.delta}")

    def compute(
        self, n: int, *, k_index: Optional[np.ndarray] = None
    ) -> WeightResult:
        if n < 0:
            raise ConfigError(f"N must be >= 0, got {n}")

        w_time = np.ones(n, dtype=float)
        if n == 0:
            info = {
                "enabled": bool(self.enabled),
                "k0_rank": -1,
                "n_discounted": 0,
                "n_undiscounted": 0,
                "oldest_weight": 1.0,
                "weight_range": (1.0, 1.0),
                "params": {"N": 0, "r": float(self.r), "delta": float(self.delta)},
            }
            return WeightResult(values=w_time, info=info)

        if k_index is None:
            rank = np.arange(n, dtype=int)
        else:
            k = np.asarray(k_index)
            if len(k) != n:
                raise ConfigError(f"k_index length {len(k)} != N={n}")
            order = np.argsort(k)
            rank = np.empty(n, dtype=int)
            rank[order] = np.arange(n, dtype=int)

        k0_rank = int(math.ceil((1.0 - self.r) * n)) - 1 if self.enabled else -1
        older_mask = np.zeros(n, dtype=bool)
        n_clipped = 0
        min_weight_floor = float(np.finfo(float).tiny)

        if self.enabled and k0_rank >= 0:
            older_mask = rank <= k0_rank
            exponents = (k0_rank - rank[older_mask]).astype(float)
            log_delta = float(math.log(self.delta))
            log_w = exponents * log_delta
            log_floor = float(math.log(min_weight_floor))
            n_clipped = int((log_w < log_floor).sum())
            log_w = np.maximum(log_w, log_floor)
            w_time[older_mask] = np.exp(log_w)

        n_discounted = int(older_mask.sum())
        oldest_idx = int(np.argmin(rank))
        info = {
            "enabled": bool(self.enabled),
            "k0_rank": k0_rank,
            "n_discounted": n_discounted,
            "n_undiscounted": int(n - n_discounted),
            "oldest_weight": float(w_time[oldest_idx]),
            "weight_range": (float(w_time.min()), float(w_time.max())),
            "n_clipped": n_clipped,
            "min_weight_floor": min_weight_floor,
            "params": {"N": int(n), "r": float(self.r), "delta": float(self.delta)},
        }
        return WeightResult(values=w_time, info=info)


@dataclass(frozen=True)
class TrainingWeights:
    """Combined per-sample training weights: ``distance × time``.

    ``distance.enabled`` / ``time.enabled`` toggle each stage (a disabled
    stage contributes all-ones, matching the legacy ``use_dist``/
    ``use_time`` flags). ``normalize=True`` rescales to mean 1.
    """

    distance: BarrierDistanceWeight = BarrierDistanceWeight()
    time: TimeDiscountWeight = TimeDiscountWeight(enabled=False)
    normalize: bool = False

    def compute(
        self,
        m_k: np.ndarray,
        *,
        phi: float,
        k_index: Optional[np.ndarray] = None,
    ) -> CombinedWeightResult:
        m = np.asarray(m_k, dtype=float)
        n = len(m)

        dist = self.distance.compute(m, phi=phi)
        time = self.time.compute(n, k_index=k_index)

        w_combined = dist.values * time.values
        if self.normalize and n > 0:
            w_sum = float(w_combined.sum())
            if w_sum > 0.0:
                w_combined = w_combined * (float(n) / w_sum)

        if n > 0:
            w_mean = float(w_combined.mean())
            w_std = float(w_combined.std())
            denom = float((w_combined**2).sum())
            effective_n = (
                float((w_combined.sum() ** 2) / denom) if denom > 0.0 else 0.0
            )
            weight_range = (float(w_combined.min()), float(w_combined.max()))
        else:
            w_mean = 0.0
            w_std = 0.0
            effective_n = 0.0
            weight_range = (1.0, 1.0)

        info = {
            "barrier_distance": dist.info,
            "time_discount": time.info,
            "combined": {
                "weight_range": weight_range,
                "weight_mean": w_mean,
                "weight_std": w_std,
                "effective_n": effective_n,
                "normalized": bool(self.normalize),
            },
            "config": {
                "use_dist": bool(self.distance.enabled),
                "use_time": bool(self.time.enabled),
                "use_pos": bool(self.distance.use_pos),
                "normalize": bool(self.normalize),
            },
        }
        return CombinedWeightResult(
            combined=w_combined, distance=dist.values, time=time.values, info=info
        )


@dataclass(frozen=True)
class UniquenessWeight:
    """López-de-Prado label-uniqueness weights, bound to a BarrierSpec.

    Thin bridge over ``src.analytics.sampling.compute_uniqueness_weights``
    so the horizon/stride can never drift from the label definition the
    weights are correcting for.
    """

    normalize: bool = True

    def compute(self, n_rows: int, *, spec) -> np.ndarray:
        from src.analytics.sampling import compute_uniqueness_weights

        return compute_uniqueness_weights(
            n_rows,
            int(spec.horizon),
            bar_stride=int(spec.stride),
            normalize=self.normalize,
        )
