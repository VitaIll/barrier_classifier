"""Feature-layer configuration: the injectable ``FeatureConfig`` object.

Historically every window/lag constant was a module-level global consumed
at class-definition time, so one process could hold exactly one
configuration and changing ``M`` meant editing source. ``FeatureConfig``
makes the whole parameter surface an immutable value object injected at
registry-build time (``FeatureEngine(config=...)``,
``run_pipeline(config=...)``): two configurations can coexist in one
process, and a custom horizon/window set is a constructor call.

``DEFAULT_CONFIG`` is built from the legacy ``src.utils`` constants, so
default-config outputs are bit-identical to the historical pipeline (the
feature oracle suites enforce this). The legacy constant re-exports below
remain for modules not yet threaded through the config (they will retire
with Phase 5).

Derived quantities (``phi``, ``n_warmup``, ``k_warmup``) are properties —
they can never drift from the fields they derive from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from src.core.errors import ConfigError
from src.utils import (
    C,
    EPS,
    ETA,
    HALFLIVES_EQ,
    HITRATE_WINDOWS_H,
    K_WARMUP,
    LAGS_F,
    M,
    N_WARMUP,
    PHI,
    PIVOT_Q_VALUES,
    VOL_PAIRS,
    WINDOWS_B,
    WINDOWS_BARRIER,
    WINDOWS_BASIS,
    WINDOWS_BPLUS,
    WINDOWS_BREAKOUT,
    WINDOWS_CANDLE_ROLL,
    WINDOWS_CORR,
    WINDOWS_EQ,
    WINDOWS_EQ_PAIRS,
    WINDOWS_EXCURSION,
    WINDOWS_EXTREME,
    WINDOWS_F,
    WINDOWS_FLOW_CSUM,
    WINDOWS_FLOW_PRESSURE,
    WINDOWS_FUNDING,
    WINDOWS_H,
    WINDOWS_LIQ_AMIHUD,
    WINDOWS_LIQ_RPV,
    WINDOWS_LOGP_Z,
    WINDOWS_MAXRET,
    WINDOWS_OFI_IMPULSE,
    WINDOWS_OI_CHG,
    WINDOWS_OI_REGIME,
    WINDOWS_OPTIONS,
    WINDOWS_PENTROPY,
    WINDOWS_PIVOT,
    WINDOWS_QUAD_TREND,
    WINDOWS_RSI,
    WINDOWS_VOL_DECOMP,
    WINDOWS_VOL_IDX,
    WINDOWS_VOL_JUMP,
    WINDOWS_VOL_OHLC,
    WINDOWS_VOL_SIGNED,
)


def _t(xs) -> tuple[int, ...]:
    return tuple(int(x) for x in xs)


def _pairs(xs) -> tuple[tuple[int, int], ...]:
    return tuple((int(a), int(b)) for a, b in xs)


@dataclass(frozen=True)
class FeatureConfig:
    """Every shared parameter of the feature layer, as one value object.

    Class-local constants (a family's fixed per-feature window tuple that
    no experiment varies) stay declared on their Feature classes; this
    object carries the SHARED surface — the horizon, barrier, and the
    window families that experiments sweep.
    """

    # --- label / horizon -----------------------------------------------------
    m: int = M
    eta: float = ETA
    c: float = C

    # --- window families -------------------------------------------------------
    windows_f: tuple[int, ...] = _t(WINDOWS_F)
    windows_h: tuple[int, ...] = _t(WINDOWS_H)
    lags_f: tuple[int, ...] = _t(LAGS_F)
    windows_b: tuple[int, ...] = _t(WINDOWS_B)
    windows_bplus: tuple[int, ...] = _t(WINDOWS_BPLUS)
    windows_barrier: tuple[int, ...] = _t(WINDOWS_BARRIER)
    windows_breakout: tuple[int, ...] = _t(WINDOWS_BREAKOUT)
    windows_candle_roll: tuple[int, ...] = _t(WINDOWS_CANDLE_ROLL)
    windows_corr: tuple[int, ...] = _t(WINDOWS_CORR)
    windows_eq: tuple[int, ...] = _t(WINDOWS_EQ)
    windows_eq_pairs: tuple[tuple[int, int], ...] = _pairs(WINDOWS_EQ_PAIRS)
    halflives_eq: tuple[int, ...] = _t(HALFLIVES_EQ)
    windows_excursion: tuple[int, ...] = _t(WINDOWS_EXCURSION)
    windows_extreme: tuple[int, ...] = _t(WINDOWS_EXTREME)
    windows_flow_csum: tuple[int, ...] = _t(WINDOWS_FLOW_CSUM)
    windows_flow_pressure: tuple[int, ...] = _t(WINDOWS_FLOW_PRESSURE)
    windows_liq_amihud: tuple[int, ...] = _t(WINDOWS_LIQ_AMIHUD)
    windows_liq_rpv: tuple[int, ...] = _t(WINDOWS_LIQ_RPV)
    windows_logp_z: tuple[int, ...] = _t(WINDOWS_LOGP_Z)
    windows_maxret: tuple[int, ...] = _t(WINDOWS_MAXRET)
    windows_ofi_impulse: tuple[int, ...] = _t(WINDOWS_OFI_IMPULSE)
    windows_pentropy: tuple[int, ...] = _t(WINDOWS_PENTROPY)
    windows_pivot: tuple[int, ...] = _t(WINDOWS_PIVOT)
    pivot_q_values: tuple[int, ...] = _t(PIVOT_Q_VALUES)
    windows_quad_trend: tuple[int, ...] = _t(WINDOWS_QUAD_TREND)
    windows_rsi: tuple[int, ...] = _t(WINDOWS_RSI)
    windows_vol_decomp: tuple[int, ...] = _t(WINDOWS_VOL_DECOMP)
    windows_vol_jump: tuple[int, ...] = _t(WINDOWS_VOL_JUMP)
    windows_vol_ohlc: tuple[int, ...] = _t(WINDOWS_VOL_OHLC)
    windows_vol_signed: tuple[int, ...] = _t(WINDOWS_VOL_SIGNED)
    vol_pairs: tuple[tuple[int, int], ...] = _pairs(VOL_PAIRS)
    hitrate_windows_h: tuple[int, ...] = _t(HITRATE_WINDOWS_H)
    # --- derivatives families ---------------------------------------------------
    windows_basis: tuple[int, ...] = _t(WINDOWS_BASIS)
    windows_funding: tuple[int, ...] = _t(WINDOWS_FUNDING)
    windows_oi_chg: tuple[int, ...] = _t(WINDOWS_OI_CHG)
    windows_oi_regime: tuple[int, ...] = _t(WINDOWS_OI_REGIME)
    windows_options: tuple[int, ...] = _t(WINDOWS_OPTIONS)
    windows_vol_idx: tuple[int, ...] = _t(WINDOWS_VOL_IDX)

    # --- derived (properties: can never drift from their sources) -----------

    @property
    def phi(self) -> float:
        """Barrier in log-return space: ``c + eta``."""
        return self.c + self.eta

    @property
    def n_warmup(self) -> int:
        """Deepest lookback of the bar-level engine, in bars."""
        return max(
            max(self.windows_f) - 1,
            max(self.lags_f),
            self.m * max(self.windows_h),
        )

    @property
    def k_warmup(self) -> int:
        """``n_warmup`` in boundary rows (ceil division by ``m``)."""
        return (self.n_warmup + self.m - 1) // self.m

    # --- validation -----------------------------------------------------------

    def __post_init__(self) -> None:
        if not (isinstance(self.m, int) and self.m > 0):
            raise ConfigError(f"FeatureConfig.m must be a positive int; got {self.m!r}")
        for name in ("eta", "c"):
            v = getattr(self, name)
            if not (isinstance(v, float) and math.isfinite(v) and v >= 0.0):
                raise ConfigError(
                    f"FeatureConfig.{name} must be finite and >= 0; got {v!r}"
                )
        if not self.phi > 0.0:
            raise ConfigError(
                f"FeatureConfig barrier phi = c + eta must be > 0; got {self.phi!r}"
            )
        for f in fields(self):
            if not f.name.startswith(("windows_", "lags_", "halflives_", "pivot_q")):
                continue
            value = getattr(self, f.name)
            if f.name.endswith("_pairs"):
                if any(not (0 < s < l) for s, l in value):
                    raise ConfigError(
                        f"FeatureConfig.{f.name} entries must be (short, long) "
                        f"with 0 < short < long; got {value!r}"
                    )
                continue
            if not value:
                raise ConfigError(f"FeatureConfig.{f.name} must be non-empty")
        # Cross-field consistency: the eq pair interactions read the tier-1
        # proxies ``eq__mu_mean``/``eq__sigma_r`` AT the pair windows — a
        # pair window absent from ``windows_eq`` fails deep inside polars
        # with a missing-column error. Fail here, at construction, instead.
        eq_set = set(self.windows_eq)
        for s, l in self.windows_eq_pairs:
            if s not in eq_set or l not in eq_set:
                raise ConfigError(
                    f"FeatureConfig.windows_eq_pairs entry ({s}, {l}) references "
                    f"windows missing from windows_eq={sorted(eq_set)} — pair "
                    "interactions read the tier-1 eq proxies at those windows"
                )
            # w == 0 is the established "instantaneous / no window" sentinel
            # (e.g. windows_basis = (0, 5, 60): basis level, then rolling
            # stats). Negative widths are always an error.
            if any((not isinstance(w, int)) or w < 0 for w in value):
                raise ConfigError(
                    f"FeatureConfig.{f.name} must hold non-negative ints; got {value!r}"
                )


#: The production configuration — identical to the legacy module constants.
DEFAULT_CONFIG = FeatureConfig()

# Bit-parity tripwire: the derived properties must reproduce the legacy
# derived constants exactly. If utils and this object ever disagree, fail
# at import — not deep inside a training run.
assert DEFAULT_CONFIG.phi == PHI, "FeatureConfig.phi diverged from utils.PHI"
assert DEFAULT_CONFIG.n_warmup == N_WARMUP, "n_warmup diverged from utils.N_WARMUP"
assert DEFAULT_CONFIG.k_warmup == K_WARMUP, "k_warmup diverged from utils.K_WARMUP"


__all__ = [
    "FeatureConfig",
    "DEFAULT_CONFIG",
    # Legacy constant re-exports (modules not yet threaded; retire in Phase 5)
    "C",
    "EPS",
    "ETA",
    "HALFLIVES_EQ",
    "HITRATE_WINDOWS_H",
    "K_WARMUP",
    "LAGS_F",
    "M",
    "N_WARMUP",
    "PHI",
    "PIVOT_Q_VALUES",
    "VOL_PAIRS",
    "WINDOWS_B",
    "WINDOWS_BARRIER",
    "WINDOWS_BASIS",
    "WINDOWS_BPLUS",
    "WINDOWS_BREAKOUT",
    "WINDOWS_CANDLE_ROLL",
    "WINDOWS_CORR",
    "WINDOWS_EQ",
    "WINDOWS_EQ_PAIRS",
    "WINDOWS_EXCURSION",
    "WINDOWS_EXTREME",
    "WINDOWS_F",
    "WINDOWS_FLOW_CSUM",
    "WINDOWS_FLOW_PRESSURE",
    "WINDOWS_FUNDING",
    "WINDOWS_H",
    "WINDOWS_LIQ_AMIHUD",
    "WINDOWS_LIQ_RPV",
    "WINDOWS_LOGP_Z",
    "WINDOWS_MAXRET",
    "WINDOWS_OFI_IMPULSE",
    "WINDOWS_OI_CHG",
    "WINDOWS_OI_REGIME",
    "WINDOWS_OPTIONS",
    "WINDOWS_PENTROPY",
    "WINDOWS_PIVOT",
    "WINDOWS_QUAD_TREND",
    "WINDOWS_RSI",
    "WINDOWS_VOL_DECOMP",
    "WINDOWS_VOL_IDX",
    "WINDOWS_VOL_JUMP",
    "WINDOWS_VOL_OHLC",
    "WINDOWS_VOL_SIGNED",
]
