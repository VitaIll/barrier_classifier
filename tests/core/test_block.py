"""Kernel: Block template method + Pipeline composition and attribution."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl
import pytest

from src.core.block import Block, Pipeline
from src.core.errors import BlockError, ConfigError, ContractError

pytestmark = pytest.mark.framework


@dataclass(frozen=True)
class AddReturns(Block):
    scale: float = 1.0

    requires = ("close",)
    provides = ("r",)

    def _apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            (pl.col("close").log().diff() * self.scale).alias("r")
        )


@dataclass(frozen=True)
class Exploder(Block):
    requires = ("close",)
    provides = ("boom",)

    def _apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        raise ZeroDivisionError("synthetic failure")


@dataclass(frozen=True)
class Reneger(Block):
    """Declares 'promised' but never emits it."""

    provides = ("promised",)

    def _apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame


@dataclass(frozen=True)
class Dropper(Block):
    """Silently drops a pass-through column — a LAW 1 violation."""

    def _apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.drop("close")


def _bars() -> pl.DataFrame:
    return pl.DataFrame({"close": [1.0, 2.0, 4.0], "keepme": ["a", "b", "c"]})


class TestBlockApply:
    def test_happy_path_adds_column_and_passes_through(self):
        out = AddReturns().apply(_bars())
        assert "r" in out.columns and "keepme" in out.columns

    def test_missing_required_column_is_contract_error(self):
        with pytest.raises(ContractError, match="close"):
            AddReturns().apply(pl.DataFrame({"x": [1.0]}))

    def test_inner_failure_wrapped_with_attribution_and_cause(self):
        with pytest.raises(BlockError, match=r"\[Exploder\]") as ei:
            Exploder().apply(_bars())
        assert isinstance(ei.value.__cause__, ZeroDivisionError)

    def test_missing_provides_is_block_error(self):
        with pytest.raises(BlockError, match="promised"):
            Reneger().apply(_bars())

    def test_dropping_passthrough_column_is_block_error(self):
        with pytest.raises(BlockError, match="dropped"):
            Dropper().apply(_bars())

    def test_input_frame_not_mutated(self):
        df = _bars()
        before = df.clone()
        AddReturns().apply(df)
        assert df.equals(before)

    def test_params_reflect_dataclass_config(self):
        assert AddReturns(scale=2.5).params() == {"scale": 2.5}

    def test_apply_logs_one_info_line(self, caplog):
        with caplog.at_level(logging.INFO, logger="bc.AddReturns"):
            AddReturns().apply(_bars())
        lines = [r for r in caplog.records if "AddReturns" in r.message]
        assert len(lines) == 1
        assert "3 rows" in lines[0].message


class TestPipeline:
    def test_sequential_composition(self):
        @dataclass(frozen=True)
        class UsesR(Block):
            requires = ("r",)
            provides = ("r2",)

            def _apply(self, frame: pl.DataFrame) -> pl.DataFrame:
                return frame.with_columns((pl.col("r") * 2).alias("r2"))

        pipe = Pipeline(AddReturns(), UsesR(), name="toy")
        out = pipe.apply(_bars())
        assert {"r", "r2"} <= set(out.columns)

    def test_requires_provides_rollup(self):
        @dataclass(frozen=True)
        class UsesR(Block):
            requires = ("r", "external")
            provides = ("r2",)

            def _apply(self, frame: pl.DataFrame) -> pl.DataFrame:
                return frame.with_columns((pl.col("r") * 2).alias("r2"))

        pipe = Pipeline(AddReturns(), UsesR())
        # 'r' is satisfied internally; 'close' and 'external' are not.
        assert set(pipe.requires) == {"close", "external"}
        assert set(pipe.provides) == {"r", "r2"}

    def test_stage_attribution_on_failure(self):
        pipe = Pipeline(AddReturns(), Exploder(), name="research")
        with pytest.raises(BlockError, match=r"\[research @ 2/2:Exploder\]") as ei:
            pipe.apply(_bars())
        assert isinstance(ei.value.__cause__, ZeroDivisionError)

    def test_empty_pipeline_rejected(self):
        with pytest.raises(ConfigError):
            Pipeline()

    def test_non_block_rejected(self):
        with pytest.raises(ConfigError, match="function"):
            Pipeline(lambda df: df)  # type: ignore[arg-type]

    def test_params_lists_stages(self):
        p = Pipeline(AddReturns(scale=3.0), name="toy").params()
        assert p["stages"][0] == {"block": "AddReturns", "params": {"scale": 3.0}}
