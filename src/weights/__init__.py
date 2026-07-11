"""Sample-weight blocks: barrier-distance, time-discount, uniqueness.

Canonical home of the training-weight computations going forward
(docs/TARGET_ARCHITECTURE.md §3). The legacy ``src.utils.compute_*_weight``
functions remain until Phase 5 retirement; the parity suite pins the two
implementations equal.
"""

from src.weights.blocks import (
    BarrierDistanceWeight,
    CombinedWeightResult,
    TimeDiscountWeight,
    TrainingWeights,
    UniquenessWeight,
    WeightResult,
)

__all__ = [
    "BarrierDistanceWeight",
    "CombinedWeightResult",
    "TimeDiscountWeight",
    "TrainingWeights",
    "UniquenessWeight",
    "WeightResult",
]
