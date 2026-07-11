"""FeatureConfig: validation, derived properties, and config coexistence.

The point of the injection migration: window/horizon parameters are bound
at engine construction from a value object, not at import from module
globals — so two configurations can run in one process, and the default
configuration is bit-identical to the legacy constants (asserted at import
in ``src/features/config.py`` and by the oracle suites).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.core.errors import ConfigError
from src.features.config import DEFAULT_CONFIG, FeatureConfig
from src.features.engine import FeatureEngine

pytestmark = pytest.mark.features_pipeline


def _bars(n: int = 1_200, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.001, n)
    close = 100.0 * np.exp(np.cumsum(r))
    ts = pd.date_range("2025-01-01", periods=n, freq="1min").to_numpy()
    p = np.log(close)
    return pl.DataFrame(
        {
            "ts": ts,
            "p": p,
            "r": np.concatenate([[np.nan], np.diff(p)]),
            "high": close * 1.0005,
            "low": close * 0.9995,
            "vwap": close,
            "volume": rng.gamma(2.0, 5.0, n),
            "close": close,
        }
    )


class TestValidation:
    def test_bad_m_rejected(self):
        with pytest.raises(ConfigError, match="m must be"):
            FeatureConfig(m=0)

    def test_negative_window_rejected(self):
        with pytest.raises(ConfigError, match="windows_eq"):
            FeatureConfig(windows_eq=(30, -60))

    def test_empty_window_family_rejected(self):
        with pytest.raises(ConfigError, match="non-empty"):
            FeatureConfig(windows_corr=())

    def test_bad_pairs_rejected(self):
        with pytest.raises(ConfigError, match="short, long"):
            FeatureConfig(windows_eq_pairs=((240, 30),))

    def test_zero_barrier_rejected(self):
        with pytest.raises(ConfigError, match="phi"):
            FeatureConfig(eta=0.0, c=0.0)

    def test_pair_windows_must_exist_in_windows_eq(self):
        # Caught at construction, not deep inside polars as a
        # missing-column error (the failure mode this validation replaced).
        with pytest.raises(ConfigError, match="windows_eq_pairs"):
            FeatureConfig(windows_eq=(32,), windows_eq_pairs=((30, 240),))


class TestDerivedProperties:
    def test_defaults_match_legacy_constants(self):
        assert DEFAULT_CONFIG.phi == utils.PHI
        assert DEFAULT_CONFIG.n_warmup == utils.N_WARMUP
        assert DEFAULT_CONFIG.k_warmup == utils.K_WARMUP
        assert DEFAULT_CONFIG.m == utils.M

    def test_derived_track_custom_fields(self):
        cfg = FeatureConfig(m=45, windows_h=(2, 4), windows_f=(10, 100),
                            lags_f=(1, 5))
        assert cfg.phi == cfg.c + cfg.eta
        assert cfg.n_warmup == max(100 - 1, 5, 45 * 4)
        assert cfg.k_warmup == (cfg.n_warmup + 45 - 1) // 45

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEFAULT_CONFIG.m = 45  # type: ignore[misc]


class TestCoexistence:
    def test_two_configs_one_process(self):
        """Two engines with different configs run side by side, no cross-talk."""
        bars = _bars()
        default_engine = FeatureEngine(tiers=(1,), families=("eq",))
        custom = FeatureConfig(
            windows_eq=(16, 32), halflives_eq=(8,), windows_eq_pairs=((16, 32),)
        )
        custom_engine = FeatureEngine(tiers=(1,), families=("eq",), config=custom)

        out_default_before = default_engine.transform(bars, trim=False).data
        out_custom = custom_engine.transform(bars, trim=False).data
        out_default_after = default_engine.transform(bars, trim=False).data

        # Different window grids -> different column sets.
        assert "eq__mu_mean__f__w16" in out_custom.columns
        assert "eq__mu_mean__f__w16" not in out_default_before.columns
        assert "eq__mu_mean__f__w30" in out_default_before.columns
        # Using the custom engine did not perturb the default one.
        assert out_default_before.equals(out_default_after)

    def test_m_flows_into_compute(self):
        """sqrt(M) denominators must read the injected config, not a global."""
        bars = _bars()
        cfg20 = FeatureConfig(
            m=20, windows_eq=(16, 32), halflives_eq=(8,),
            windows_eq_pairs=((16, 32),),
        )
        cfg45 = FeatureConfig(
            m=45, windows_eq=(16, 32), halflives_eq=(8,),
            windows_eq_pairs=((16, 32),),
        )
        out20 = FeatureEngine(tiers=(1, 2), families=("eq",), config=cfg20).transform(
            bars, trim=False
        ).data
        out45 = FeatureEngine(tiers=(1, 2), families=("eq",), config=cfg45).transform(
            bars, trim=False
        ).data
        # barrier_vs_eq_hz = (mu - (p + phi)) / (sigma * sqrt(M) + EPS):
        # explicitly horizon-normalized, so M must change its values.
        col = "eq__barrier_vs_eq_hz__via_trend__f__w32"
        assert col in out20.columns, sorted(
            c for c in out20.columns if c.startswith("eq__")
        )
        a = out20[col].to_numpy()
        b = out45[col].to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        assert finite.any()
        # sigma*sqrt(45) vs sigma*sqrt(20): values must differ materially.
        assert not np.allclose(a[finite], b[finite])

    def test_pair_grid_from_config(self):
        bars = _bars()
        custom = FeatureConfig(
            windows_eq=(16, 32, 64),
            halflives_eq=(8,),
            windows_eq_pairs=((16, 64),),
        )
        out = FeatureEngine(tiers=(1, 2), families=("eq",), config=custom).transform(
            bars, trim=False
        ).data
        assert "eq__pullback_rising_eq__f__w16__l64" in out.columns
        assert "eq__above_falling_eq__f__w16__l64" in out.columns
        # Default pairs are NOT emitted under the custom grid.
        assert "eq__pullback_rising_eq__f__w30__l240" not in out.columns
