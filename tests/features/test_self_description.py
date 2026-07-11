"""The self-describing Feature contract: domains, tags, predecessors, catalog.

A Feature declares everything about itself — value domain (deriving
imputation neutral, expected range, legal operations), categorical tags,
column-level predecessors — and the engine aggregates: catalog for
introspection, dependency-graph validation at construction.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.features import base as base_mod
from src.features.base import (
    FRACTION,
    OSCILLATOR_0_100,
    RATIO,
    REAL,
    Domain,
    Feature,
)
from src.features.engine import FeatureEngine

pytestmark = pytest.mark.features_pipeline


@pytest.fixture()
def scratch_registry():
    """Snapshot/restore the global registry around synthetic-class tests."""
    saved = list(base_mod._REGISTRY)
    yield
    base_mod._REGISTRY[:] = saved


class TestDomains:
    def test_neutrals(self):
        assert REAL.neutral == 0.0
        assert FRACTION.neutral == 0.5
        assert RATIO.neutral == 1.0
        assert OSCILLATOR_0_100.neutral == 50.0

    def test_impute_derives_from_domain(self, scratch_registry):
        class FracFeat(Feature):
            family = "synth"
            inputs = ("x",)
            domain = FRACTION

            def compute(self, w=None):
                return pl.col("x")

        assert FracFeat().impute_value() == 0.5

    def test_explicit_impute_overrides_domain(self, scratch_registry):
        class SentinelFeat(Feature):
            family = "synth"
            inputs = ("x",)
            domain = FRACTION
            impute_default = 5.0  # deliberate sentinel, not the neutral

            def compute(self, w=None):
                return pl.col("x")

        assert SentinelFeat().impute_value() == 5.0

    def test_value_range_from_domain_and_override(self, scratch_registry):
        class A(Feature):
            family = "synth"
            inputs = ("x",)
            domain = FRACTION

            def compute(self, w=None):
                return pl.col("x")

        class B(A):
            expected_range = (0.0, 0.25)

        assert A().value_range == (0.0, 1.0)
        assert B().value_range == (0.0, 0.25)

    def test_legal_ops_by_domain(self, scratch_registry):
        class R(Feature):
            family = "synth"
            inputs = ("x",)
            domain = RATIO

            def compute(self, w=None):
                return pl.col("x")

        assert "log" in R().legal_ops
        assert "logit" not in R().legal_ops


class TestTagsAndDescribe:
    def test_effective_tags_structural_plus_enrichment(self, scratch_registry):
        class Tagged(Feature):
            family = "synth"
            inputs = ("x",)
            tags = frozenset({"momentum"})

            def compute(self, w=None):
                return pl.col("x")

        tags = Tagged().effective_tags
        assert {"synth", "tier:1", "domain:real", "momentum"} <= tags

    def test_describe_carries_the_full_contract(self):
        # A real production feature: RSI.
        from src.features.families.trend import TrendRsi

        d = TrendRsi().describe(14)
        assert d["domain"] == "oscillator_0_100"
        assert d["impute"] == 50.0
        assert d["range_lo"] == 0.0 and d["range_hi"] == 100.0
        assert d["column"] == "ret__rsi__f__w14"
        assert "trend" in d["tags"]


class TestCatalog:
    def test_catalog_is_the_full_plan(self):
        engine = FeatureEngine(tiers=(1, 2), families=("eq",))
        cat = engine.catalog()
        assert cat.height == len(engine.plan())
        assert {"column", "domain", "impute", "warmup_rows", "depends_on"} <= set(
            cat.columns
        )
        pair_rows = cat.filter(
            pl.col("column").str.contains("pullback_rising_eq")
        )
        deps = pair_rows["depends_on"].to_list()[0]
        assert any("eq__mu_mean" in d for d in deps)


class TestDependencyValidation:
    def test_same_tier_reference_rejected_at_construction(self, scratch_registry):
        class Upstream(Feature):
            family = "synthdep"
            name = "up"
            inputs = ("x",)
            tier = 2

            def compute(self, w=None):
                return pl.col("x")

        class Downstream(Feature):
            family = "synthdep"
            name = "down"
            inputs = ("x",)
            tier = 2  # SAME tier as its dependency — the landmine

            def depends_on(self, w=None):
                return ("synthdep__up",)

            def compute(self, w=None):
                return pl.col("synthdep__up")

        with pytest.raises(ValueError, match="strictly earlier tier"):
            FeatureEngine(tiers=(1, 2), families=("synthdep",))

    def test_earlier_tier_reference_accepted(self, scratch_registry):
        class Up1(Feature):
            family = "synthok"
            name = "up"
            inputs = ("x",)
            tier = 1

            def compute(self, w=None):
                return pl.col("x")

        class Down2(Feature):
            family = "synthok"
            name = "down"
            inputs = ("x",)
            tier = 2

            def depends_on(self, w=None):
                return ("synthok__up",)

            def compute(self, w=None):
                return pl.col("synthok__up")

        FeatureEngine(tiers=(1, 2), families=("synthok",))  # no raise

    def test_production_registry_validates(self):
        # The full default registry must satisfy its own declared graph.
        FeatureEngine(tiers=(1, 2)).validate_dependencies()
