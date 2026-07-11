"""Label domain: barrier-label definition, vectorized kernel, and block.

The label *definition* lives here as :class:`~src.labels.barrier.BarrierSpec`;
weights, purged splits, and serving contracts should derive their ``M``/``phi``
from a spec instance rather than loose floats.
"""

from src.labels.barrier import (
    LABEL_SCHEMA,
    BarrierLabeler,
    BarrierSpec,
    LabelArrays,
    barrier_label_arrays,
    label_frame,
)

__all__ = [
    "LABEL_SCHEMA",
    "BarrierLabeler",
    "BarrierSpec",
    "LabelArrays",
    "barrier_label_arrays",
    "label_frame",
]
