"""Kernel: error taxonomy + frame contracts + no-mutation machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.core.contracts import (
    ColumnSpec,
    FrameSchema,
    assert_unmutated,
    require_columns,
    snapshot,
)
from src.core.errors import (
    BlockError,
    ConfigError,
    ContractError,
    CoreError,
    StateError,
)

pytestmark = pytest.mark.framework


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class TestErrorTaxonomy:
    def test_config_and_contract_errors_are_value_errors(self):
        # Legacy call sites catch ValueError; the taxonomy must not break them.
        assert issubclass(ConfigError, ValueError)
        assert issubclass(ContractError, ValueError)

    def test_state_error_is_runtime_error(self):
        assert issubclass(StateError, RuntimeError)

    def test_all_catchable_as_core_error(self):
        for exc in (ConfigError, ContractError, StateError, BlockError):
            assert issubclass(exc, CoreError)

    def test_block_error_attribution_in_message(self):
        e = BlockError("boom", block="Labeler")
        assert "[Labeler]" in str(e) and "boom" in str(e)
        e2 = BlockError("boom", block="Pipe", stage="2/3:Imputer")
        assert "[Pipe @ 2/3:Imputer]" in str(e2)
        assert e2.block == "Pipe" and e2.stage == "2/3:Imputer"


# ---------------------------------------------------------------------------
# ColumnSpec / FrameSchema
# ---------------------------------------------------------------------------


def _frame(**overrides) -> pl.DataFrame:
    base = {
        "ts": pl.Series(
            "ts",
            pd.date_range("2025-01-01", periods=4, freq="1min").to_numpy(),
        ),
        "close": pl.Series("close", [1.0, 2.0, 3.0, 4.0]),
        "n": pl.Series("n", [1, 2, 3, 4], dtype=pl.Int32),
    }
    base.update(overrides)
    return pl.DataFrame(base)


class TestColumnSpec:
    def test_empty_name_rejected(self):
        with pytest.raises(ConfigError):
            ColumnSpec("")

    def test_inverted_bounds_rejected(self):
        with pytest.raises(ConfigError):
            ColumnSpec("x", bounds=(1.0, 0.0))


class TestFrameSchemaStructure:
    def test_duplicate_columns_rejected(self):
        with pytest.raises(ConfigError):
            FrameSchema.of("s", ColumnSpec("a"), ColumnSpec("a"))

    def test_ok_frame_passes_and_returns_same_object(self):
        schema = FrameSchema.of(
            "bars", ColumnSpec("close", pl.Float64), ColumnSpec("n", pl.Int32)
        )
        df = _frame()
        assert schema.validate(df) is df

    def test_missing_column_reported(self):
        schema = FrameSchema.of("bars", ColumnSpec("volume", pl.Float64))
        with pytest.raises(ContractError, match="missing column 'volume'"):
            schema.validate(_frame())

    def test_wrong_dtype_reported(self):
        schema = FrameSchema.of("bars", ColumnSpec("close", pl.Int64))
        with pytest.raises(ContractError, match="dtype"):
            schema.validate(_frame())

    def test_class_dtype_matches_any_parametrization(self):
        # pl.Datetime class form must accept Datetime of any unit/tz.
        schema = FrameSchema.of("bars", ColumnSpec("ts", pl.Datetime))
        schema.validate(_frame())

    def test_any_of_dtype_tuple(self):
        schema = FrameSchema.of(
            "bars", ColumnSpec("n", (pl.Int32, pl.Int64))
        )
        schema.validate(_frame())

    def test_undeclared_columns_pass_through(self):
        schema = FrameSchema.of("bars", ColumnSpec("close", pl.Float64))
        schema.validate(_frame(extra=pl.Series("extra", ["a", "b", "c", "d"])))

    def test_multiple_violations_collected_into_one_error(self):
        schema = FrameSchema.of(
            "bars",
            ColumnSpec("volume", pl.Float64),
            ColumnSpec("close", pl.Int64),
        )
        with pytest.raises(ContractError, match="2 violation"):
            schema.validate(_frame())


class TestFrameSchemaData:
    def test_non_nullable_violation(self):
        schema = FrameSchema.of(
            "s", ColumnSpec("close", pl.Float64, nullable=False)
        )
        df = _frame(close=pl.Series("close", [1.0, None, 3.0, 4.0]))
        schema.validate(df)  # structure level ignores nulls
        with pytest.raises(ContractError, match="non-nullable"):
            schema.validate(df, level="data")

    def test_finite_violation_nan_and_inf(self):
        schema = FrameSchema.of("s", ColumnSpec("close", pl.Float64, finite=True))
        with pytest.raises(ContractError, match="NaN"):
            schema.validate(
                _frame(close=pl.Series("close", [1.0, float("nan"), 3.0, 4.0])),
                level="data",
            )
        with pytest.raises(ContractError, match="inf"):
            schema.validate(
                _frame(close=pl.Series("close", [1.0, float("inf"), 3.0, 4.0])),
                level="data",
            )

    def test_nan_allowed_when_finite_false(self):
        schema = FrameSchema.of("s", ColumnSpec("close", pl.Float64, finite=False))
        schema.validate(
            _frame(close=pl.Series("close", [1.0, float("nan"), 3.0, 4.0])),
            level="data",
        )

    def test_bounds_checked_ignoring_nulls(self):
        schema = FrameSchema.of(
            "s", ColumnSpec("close", pl.Float64, bounds=(0.0, 3.5))
        )
        with pytest.raises(ContractError, match="max"):
            schema.validate(_frame(), level="data")  # max is 4.0
        ok = _frame(close=pl.Series("close", [0.5, None, 3.0, 3.5]))
        schema.validate(ok, level="data")

    def test_unknown_level_rejected(self):
        schema = FrameSchema.of("s", ColumnSpec("close"))
        with pytest.raises(ConfigError):
            schema.validate(_frame(), level="everything")  # type: ignore[arg-type]


class TestRequireColumns:
    def test_missing_lists_all(self):
        with pytest.raises(ContractError, match=r"\['a', 'b'\]"):
            require_columns(_frame(), ["a", "close", "b"], context="unit")


# ---------------------------------------------------------------------------
# LAW 6 machinery
# ---------------------------------------------------------------------------


class TestAssertUnmutated:
    def test_numpy_mutation_detected(self):
        arr = np.arange(5, dtype=float)
        with pytest.raises(ContractError, match="argument #0"):
            with assert_unmutated(arr):
                arr[2] = 99.0

    def test_numpy_nan_stable(self):
        arr = np.array([1.0, np.nan, 3.0])
        with assert_unmutated(arr):
            _ = arr * 2  # pure op

    def test_pandas_mutation_detected(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ContractError):
            with assert_unmutated(df):
                df.loc[0, "a"] = -1.0

    def test_polars_clean_pass(self):
        df = _frame()
        with assert_unmutated(df):
            df.with_columns(pl.col("close") * 2)

    def test_body_exception_not_masked(self):
        arr = np.arange(3, dtype=float)
        with pytest.raises(KeyError):
            with assert_unmutated(arr):
                arr[0] = 5.0  # mutation AND exception: exception wins
                raise KeyError("body failure")

    def test_snapshot_rejects_unknown_types(self):
        with pytest.raises(TypeError):
            snapshot(object())
