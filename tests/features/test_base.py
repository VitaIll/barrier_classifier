"""Step 1 tests: Feature ABC + registry behavior + engine plumbing.

Structural tests only — they verify the framework, not any feature math.
Per-primitive and per-feature value/parity tests land in steps 2-15.

Marked ``framework`` so they run regardless of the active delivery step.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.features import FeatureEngine, get_registry
from src.features.base import _REGISTRY, Feature, _clear_registry_for_tests

pytestmark = pytest.mark.framework


@pytest.fixture(autouse=True)
def _isolated_registry():
    snapshot = list(_REGISTRY)
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()
    _REGISTRY.extend(snapshot)


@pytest.fixture
def bars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open":   [1.00, 1.10, 1.20, 1.30, 1.40],
            "high":   [1.05, 1.15, 1.25, 1.35, 1.45],
            "low":    [0.95, 1.05, 1.15, 1.25, 1.35],
            "close":  [1.02, 1.12, 1.22, 1.32, 1.42],
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


# --- Auto-registration --------------------------------------------------------

def test_subclass_auto_registers():
    class Foo(Feature):
        family = "test"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

    assert Foo in get_registry()
    assert Foo in get_registry(families=["test"])


def test_abstract_subclass_does_not_register():
    class AbstractRoot(Feature):
        __abstract__ = True
        family = "test"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

    assert AbstractRoot not in get_registry()


def test_concrete_descendant_of_abstract_registers():
    class AbstractRoot(Feature):
        __abstract__ = True
        family = "test"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

    class Concrete(AbstractRoot):
        pass

    assert Concrete in get_registry()
    assert AbstractRoot not in get_registry()


def test_missing_family_raises():
    with pytest.raises(ValueError, match="family"):

        class _Bad(Feature):
            inputs = ("close",)

            def compute(self, w=None):
                return pl.col("close")


def test_missing_inputs_raises():
    with pytest.raises(ValueError, match="inputs"):

        class _Bad(Feature):
            family = "test"

            def compute(self, w=None):
                return pl.col("close")


def test_default_name_is_lowercased_classname():
    class GarmanKlass(Feature):
        family = "vol"
        inputs = ("open", "high", "low", "close")

        def compute(self, w=None):
            return pl.col("close")

    assert GarmanKlass.name == "garmanklass"


def test_explicit_name_overrides_default():
    class MyFeature(Feature):
        family = "trend"
        name = "rsi"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

    assert MyFeature.name == "rsi"


# --- Naming -------------------------------------------------------------------

def test_column_name_unwindowed():
    class Logret(Feature):
        family = "ret"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close").log().diff()

    assert Logret().column_name() == "ret__logret"


def test_column_name_windowed():
    class RMean(Feature):
        family = "ret"
        inputs = ("close",)
        windows = (5, 60)

        def compute(self, w=None):
            return pl.col("close").rolling_mean(w)

    spec = RMean()
    assert spec.column_name(5) == "ret__rmean__w5"
    assert spec.column_name(60) == "ret__rmean__w60"


def test_column_name_override_escapes_convention():
    class Custom(Feature):
        family = "trend"
        inputs = ("close",)
        windows = (14,)

        def compute(self, w=None):
            return pl.col("close")

        def column_name(self, w=None):
            return f"rsi__w{w}"

    assert Custom().column_name(14) == "rsi__w14"


# --- expanded() ---------------------------------------------------------------

def test_expanded_unwindowed_yields_one():
    class X(Feature):
        family = "test"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

    assert list(X().expanded()) == [(None, "test__x")]


def test_expanded_windowed_yields_one_per_window():
    class X(Feature):
        family = "test"
        inputs = ("close",)
        windows = (5, 15)

        def compute(self, w=None):
            return pl.col("close")

    assert list(X().expanded()) == [(5, "test__x__w5"), (15, "test__x__w15")]


# --- warmup_for ---------------------------------------------------------------

def test_warmup_default_is_w_minus_one_for_windowed():
    class X(Feature):
        family = "test"
        inputs = ("close",)
        windows = (5,)

        def compute(self, w=None):
            return pl.col("close")

    assert X().warmup_for(5) == 4


def test_warmup_default_is_zero_for_unwindowed():
    class X(Feature):
        family = "test"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

    assert X().warmup_for(None) == 0


# --- Engine plumbing ----------------------------------------------------------

def test_engine_runs_single_feature(bars):
    class Identity(Feature):
        family = "test"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

    engine = FeatureEngine(tiers=(1,))
    result = engine.transform(bars)
    assert "test__identity" in result.data.columns
    assert result.data["test__identity"].to_list() == bars["close"].to_list()


def test_engine_expands_windows_in_one_pass(bars):
    class Multi(Feature):
        family = "test"
        inputs = ("close",)
        windows = (2, 3)

        def compute(self, w=None):
            return pl.col("close").rolling_mean(w, min_samples=w)

    engine = FeatureEngine(tiers=(1,))
    result = engine.transform(bars)
    assert "test__multi__w2" in result.data.columns
    assert "test__multi__w3" in result.data.columns


def test_engine_filters_by_family(bars):
    class A(Feature):
        family = "alpha"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

    class B(Feature):
        family = "beta"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

    engine = FeatureEngine(tiers=(1,), families=("alpha",))
    result = engine.transform(bars)
    assert "alpha__a" in result.data.columns
    assert "beta__b" not in result.data.columns


def test_engine_trims_warmup(bars):
    class W3(Feature):
        family = "test"
        inputs = ("close",)
        windows = (3,)

        def compute(self, w=None):
            return pl.col("close").rolling_mean(w, min_samples=w)

    engine = FeatureEngine(tiers=(1,))
    result = engine.transform(bars)
    assert result.warmup_trimmed == 2
    assert len(result.data) == len(bars) - 2
    assert result.data["test__w3__w3"].null_count() == 0


def test_engine_trims_tail_when_declared(bars):
    class ForwardLooking(Feature):
        family = "label"
        inputs = ("close",)
        null_tail_bars = 2

        def compute(self, w=None):
            return pl.col("close").shift(-2)

    engine = FeatureEngine(tiers=(1,))
    result = engine.transform(bars)
    assert result.tail_trimmed == 2
    assert len(result.data) == len(bars) - 2


def test_engine_coerces_input_nan_to_null():
    bars = pl.DataFrame({"close": [1.0, float("nan"), 3.0]})

    class Identity(Feature):
        family = "test"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

    engine = FeatureEngine(tiers=(1,))
    result = engine.transform(bars)
    assert result.data["test__identity"].to_list() == [1.0, None, 3.0]


def test_empty_registry_returns_input_unchanged(bars):
    engine = FeatureEngine(tiers=(1,))
    result = engine.transform(bars)
    assert result.warmup_trimmed == 0
    assert result.tail_trimmed == 0
    assert len(result.data) == len(bars)


def test_engine_rejects_intra_tier_column_name_collision():
    """Two Feature subclasses emitting the same column name in the same
    tier would silently overwrite each other inside polars'
    ``with_columns`` call. The engine must raise at construction so the
    collision surfaces at registration time, not as a wrong-column bug.
    """

    class CollisionLeft(Feature):
        family = "test"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close")

        def column_name(self, w=None):
            return "test__collision"

    class CollisionRight(Feature):
        family = "test"
        inputs = ("close",)

        def compute(self, w=None):
            return pl.col("close") * 2

        def column_name(self, w=None):
            return "test__collision"

    with pytest.raises(ValueError, match="collision"):
        FeatureEngine(tiers=(1,))
