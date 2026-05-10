"""Feature engine package.

Polars-based, registry-driven feature engine that replaces the monolithic
compute_* functions in src/utils.py family by family.

Public exports:
- Feature, get_registry  (from .base)
- FeatureEngine, EngineResult  (from .engine)
- FeatureReport, Validator  (from .observability)

Importing this package triggers auto-discovery of family modules in
src/features/families/, which populates the registry as a side effect.
"""

from src.features.base import Feature, get_registry
from src.features.engine import EngineResult, FeatureEngine
from src.features.observability import (
    FeatureReport,
    Validator,
    compute_feature_health,
    flag_issues,
    monthly_target_balance,
    summarize_by_family,
)

# Auto-import families so registration fires on package import.
from src.features import families  # noqa: F401

__all__ = [
    "EngineResult",
    "Feature",
    "FeatureEngine",
    "FeatureReport",
    "Validator",
    "compute_feature_health",
    "flag_issues",
    "get_registry",
    "monthly_target_balance",
    "summarize_by_family",
]
