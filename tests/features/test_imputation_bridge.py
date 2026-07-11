"""Imputation bridge: class-declared fills == the legacy regex registry.

The 140-line order-sensitive regex table (``utils.get_imputation_value``)
was replaced by declarations on the Feature classes plus a boundary-stage
prefix table. This suite pins the two resolutions equal for EVERY column
the pipeline can produce — the license for the switch — and pins the new
hard-error behavior for undeclared columns (the silent ``.* -> 0.0``
catch-all is gone by design).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import utils
from src.core.errors import ContractError
from src.features.boundary import boundary_imputation_entries
from src.features.engine import FeatureEngine
from src.features.quality import resolve_imputation_value

pytestmark = pytest.mark.features_pipeline

P_HIT_PRIOR = 0.5
CAP_H_BLOCKS = 2880


def _resolve(col: str) -> float:
    return resolve_imputation_value(
        col,
        impute_map=FeatureEngine(tiers=(1, 2)).imputation_map(),
        boundary_entries=boundary_imputation_entries(
            p_hit_prior=P_HIT_PRIOR, cap_h_blocks=CAP_H_BLOCKS
        ),
    )


class TestRegistryBridge:
    def test_every_registered_column_matches_legacy(self):
        """All families (incl. derivatives), straight from the plan."""
        engine = FeatureEngine(tiers=(1, 2))
        impute_map = engine.imputation_map()
        assert len(impute_map) > 1_000  # sanity: the full surface
        mismatches = []
        for name, got in impute_map.items():
            expected = utils.get_imputation_value(
                name, p_hit_prior=P_HIT_PRIOR, cap_h_blocks=CAP_H_BLOCKS
            )
            if got != expected:
                mismatches.append((name, got, expected))
        assert not mismatches, (
            f"{len(mismatches)} registry column(s) diverge from the legacy "
            f"registry, e.g. {mismatches[:8]}"
        )


class TestBoundaryBridge:
    @pytest.mark.parametrize(
        "col",
        [
            "barrier__z_tight__f__w60",
            "barrier__emax_ratio__f__w240",
            "barrier__p_hit_drifted__f__w60",
            "vol__ratio__f__ws60__wl240",
            "hit__rate__h__w1440",
            "hit__since__h__w0",
            "hit__prev__h__w0",
            "target__mature_m_mean__h__w1440",
            "target__mature_m_pos_mean__h__w1440",
            "target__mature_tau_pos_mean__h__w1440",
            "target__mature_near_miss_up__h__w1440",
            "target__mature_near_miss_dn__h__w1440",
            "target__autocorr_lag1__h__w1440",
            "block__close_to_high__h__w0",
            "block__maxret__h__w0",
            "block__minret__h__w0",
            "ret__inst__h__w0",
            "range__inst__h__w0",
            "logvol__inst__h__w0",
            "ofi__inst__h__w0",
            "ret__std__h__w1440",
            "data__bad_ohlc__f__w0",
            "data__gap__f__w0",
        ],
    )
    def test_boundary_columns_match_legacy(self, col):
        assert _resolve(col) == utils.get_imputation_value(
            col, p_hit_prior=P_HIT_PRIOR, cap_h_blocks=CAP_H_BLOCKS
        )

    @pytest.mark.parametrize("col", ["cost__c__h__w0", "barrier__phi__h__w0"])
    def test_never_missing_columns_raise_like_legacy(self, col):
        # Legacy raised ValueError for these; the new path raises the
        # (ValueError-compatible) ContractError with the diagnosis.
        with pytest.raises(ValueError):
            utils.get_imputation_value(col)
        with pytest.raises(ContractError, match="never-missing"):
            _resolve(col)


class TestHardErrorReplacesCatchAll:
    def test_undeclared_column_raises_instead_of_silent_zero(self):
        # Legacy: `.*` catch-all silently returned 0.0.
        assert utils.get_imputation_value("mystery__column__f__w5") == 0.0
        with pytest.raises(ContractError, match="no imputation declared"):
            _resolve("mystery__column__f__w5")


class TestEndToEndFrameBridge:
    def test_every_pipeline_feature_column_resolves_to_legacy_value(self):
        """The decisive check: run the real (inference) pipeline and compare
        the resolution of every feature column it actually produces."""
        from src.features.pipeline import (
            _BASE_COLS,
            _DERIV_BASE_COLS,
            _LABEL_AUX_COLS,
            _RAW_COLS,
            run_inference_pipeline,
        )

        rng = np.random.default_rng(9)
        n = 7_000
        r = rng.normal(0, 0.0008, n)
        close = 40_000 * np.exp(np.cumsum(r))
        idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
        raw = pd.DataFrame(
            {
                "open": np.roll(close, 1),
                "high": close * np.exp(np.abs(rng.normal(0, 0.0004, n))),
                "low": close * np.exp(-np.abs(rng.normal(0, 0.0004, n))),
                "close": close,
                "volume": rng.gamma(2, 5, n),
                "quote_volume": rng.gamma(2, 250_000, n),
                "num_trades": rng.integers(50, 5000, n).astype(float),
                "taker_buy_base": rng.gamma(2, 2.5, n),
                "taker_buy_quote": rng.gamma(2, 125_000, n),
            },
            index=idx,
        )
        out = run_inference_pipeline(
            raw, label_cadence="1min", boundary_tail_rows=None
        )
        non_feature = set(
            _LABEL_AUX_COLS + _RAW_COLS + _BASE_COLS + _DERIV_BASE_COLS
        )
        feature_cols = [
            c for c in out.columns
            if c not in non_feature and not c.startswith("undef__")
        ]
        assert len(feature_cols) > 1_400
        mismatches = []
        for col in feature_cols:
            try:
                expected = utils.get_imputation_value(
                    col, p_hit_prior=P_HIT_PRIOR, cap_h_blocks=CAP_H_BLOCKS
                )
            except ValueError:
                # Never-missing constants (cost__c, barrier__phi): the legacy
                # registry raises — the new path must raise too.
                with pytest.raises(ContractError):
                    _resolve(col)
                continue
            try:
                got = _resolve(col)
            except ContractError as exc:
                mismatches.append((col, f"UNRESOLVED: {exc}"))
                continue
            if got != expected:
                mismatches.append((col, got, expected))
        assert not mismatches, (
            f"{len(mismatches)} produced column(s) diverge, "
            f"e.g. {mismatches[:10]}"
        )
