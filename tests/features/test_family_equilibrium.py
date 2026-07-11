"""Equilibrium-family (Round 2) tests.

Coverage:

  - Family registration / column inventory.
  - Causal-naming audit (every eq column matches ``__f__…``).
  - Warmup contract (declared null count == actual null+NaN count).
  - Past-only causality: mutating ``p[N]`` does NOT change tier-1
    equilibrium proxies at row ``N``; it DOES change them at row ``N+1``
    (the spike has now entered the past window).
  - EWMA pre-update causality: same probe on ``eq__ewm_innov_hz``.
  - Known-answer probes:
      * constant log-price series → residuals identically 0 post-warmup.
      * linear rising series → ``eq__mu_trend`` extrapolates correctly,
        ``eq__trend_resid_z`` ≈ 0, ``eq__mean_resid_hz`` strictly positive.
      * constant volume → ``eq__mu_vwap == eq__mu_mean``.
  - Sign / range invariants on the synthetic random fixture:
      * ``eq__proxy_dispersion_hz ≥ 0``.
      * ``eq__pullback_rising_eq ≥ 0``.
      * ``eq__above_falling_eq ≥ 0``.
      * residual / barrier interaction columns are finite and Float64.
  - Post-trim cleanliness: zero null + zero NaN + zero inf in every
    eq column when the engine is asked to trim by the declared warmup.

Engine-only — the pipeline-level warmup (K_WARMUP) requires a fixture of
~20k+ bars; covered by integration once the full pipeline is exercised
in the legacy parity tests elsewhere.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src import utils
from src.analytics.audits import causal_feature_audit
from src.features import FeatureEngine, get_registry
from src.features.config import (
    EPS,
    HALFLIVES_EQ,
    M,
    PHI,
    WINDOWS_EQ,
    WINDOWS_EQ_PAIRS,
)


# ---------------------------------------------------------------------------
# Synthetic bar fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eq_bars() -> pl.DataFrame:
    rng = np.random.default_rng(2026_05_12)
    n = 4_000
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    spread = np.abs(rng.normal(0.0, 0.5, n))
    high = close + spread
    low = np.maximum(close - spread, 0.001)
    open_ = np.clip(close + rng.normal(0.0, 0.2, n), low, high)
    volume = np.abs(rng.normal(100.0, 30.0, n)) + 1.0
    qv = volume * close
    nt = (np.abs(rng.normal(50.0, 20.0, n)).astype(np.int64)) + 1
    tbb = volume * rng.uniform(0.3, 0.7, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "quote_volume": qv,
            "num_trades": nt, "taker_buy_base": tbb,
        },
        index=ts,
    )
    df = utils.compute_base_series(df)
    return pl.from_pandas(df.reset_index(names="ts"))


@pytest.fixture(scope="module")
def eq_engine_result(eq_bars) -> pl.DataFrame:
    """Untrimmed engine output, so warmup contracts are visible."""
    engine = FeatureEngine(tiers=(1, 2), families=("eq",))
    return engine.transform(eq_bars, trim=False).data


@pytest.fixture(scope="module")
def eq_engine_specs():
    """Concrete spec instances + their (window, column_name) pairs."""
    out = []
    for cls in get_registry(families=("eq",)):
        inst = cls()
        for w, name in inst.expanded():
            out.append((inst, w, name))
    return out


# ---------------------------------------------------------------------------
# Registration / inventory
# ---------------------------------------------------------------------------


def test_eq_family_registers_expected_class_count():
    """22 Feature classes (10 tier-1 + 12 tier-2).

    The six import-time-generated pair classes became ONE config-driven
    ``EqPairInteractions`` (2026-07-11 FeatureConfig migration) — the pair
    grid now comes from ``cfg.windows_eq_pairs`` instead of a module
    constant. Column count and ORDER are unchanged (see
    ``test_eq_family_column_count`` and the contract-order note on the
    class); only the class structure changed.
    """
    classes = get_registry(families=("eq",))
    assert len(classes) == 22, [c.__name__ for c in classes]


def test_eq_family_column_count(eq_engine_specs):
    """Tier-1: 8 window features × 6 + 2 halflife × 3 = 54.
    Tier-2: 5 simple + 1 EWMA + 4 UER/BVE + 1 dispersion × 6 windows
            + 6 pair concrete classes (2 features × 3 pairs).
    Total exactly 123."""
    column_names = {name for _, _, name in eq_engine_specs}
    assert len(column_names) == 123


def test_eq_family_tier_split():
    classes = get_registry(families=("eq",))
    tier_counts = {1: 0, 2: 0}
    for cls in classes:
        tier_counts[cls.tier] += 1
    # 2 -> 12: the six generated pair classes are one class since the
    # FeatureConfig migration (same columns; see class-count test above).
    assert tier_counts == {1: 10, 2: 12}


# ---------------------------------------------------------------------------
# Causal naming audit
# ---------------------------------------------------------------------------


def test_eq_columns_pass_causal_audit(eq_engine_specs):
    """Every eq column must end in ``__f__…`` so the causal audit
    treats it as a frozen-up-to-now feature."""
    names = sorted({name for _, _, name in eq_engine_specs})
    result = causal_feature_audit(names)
    assert result.passed, (
        f"eq family failed causal audit: "
        f"suspect={result.suspect[:5]}, unmatched={result.unmatched[:5]}"
    )


# ---------------------------------------------------------------------------
# Warmup contract
# ---------------------------------------------------------------------------


def test_eq_declared_warmup_matches_actual_nulls(eq_engine_result, eq_engine_specs):
    """For every eq column, declared warmup must equal actual missing
    count. Missing = null + NaN (numpy-kernel features can emit either
    but we coerce to null at the eq feature layer)."""
    out = eq_engine_result
    mismatches = []
    for spec, w, name in eq_engine_specs:
        declared = spec.warmup_for(w)
        actual_null = int(out[name].null_count())
        actual_nan = int(out[name].is_nan().sum() or 0)
        actual = actual_null + actual_nan
        if actual != declared:
            mismatches.append((name, declared, actual))
    assert not mismatches, f"Warmup mismatches: {mismatches[:5]}"


def test_eq_post_trim_no_missing(eq_bars):
    """After the engine trims by max declared warmup, every eq column
    must be free of null, NaN, and inf. This is the contract the
    pipeline's impute layer relies on."""
    engine = FeatureEngine(tiers=(1, 2), families=("eq",))
    trimmed = engine.transform(eq_bars, trim=True).data
    eq_cols = [c for c in trimmed.columns if c.startswith("eq__")]
    issues = []
    for c in eq_cols:
        s = trimmed[c]
        n_null = int(s.null_count())
        n_nan = int(s.is_nan().sum() or 0)
        n_inf = int(s.is_infinite().sum() or 0)
        if n_null or n_nan or n_inf:
            issues.append((c, n_null, n_nan, n_inf))
    assert not issues, f"Post-trim issues: {issues[:5]}"


# ---------------------------------------------------------------------------
# Past-only causality — the central correctness property
# ---------------------------------------------------------------------------


def _engine_run(bars: pl.DataFrame) -> pl.DataFrame:
    return FeatureEngine(tiers=(1, 2), families=("eq",)).transform(
        bars, trim=False
    ).data


def test_tier1_proxies_are_past_only(eq_bars):
    """Mutate the close (and therefore ``p``) at a single row ``N``.
    Tier-1 mu / scale columns at row ``N`` must NOT change — their
    windows are strictly ``[N-W, N-1]``, excluding ``N``.

    Tier-1 columns at row ``N+1`` MAY change (the spike has entered
    the past window at that row); we don't check those here.
    """
    bars = eq_bars
    out_baseline = _engine_run(bars)

    # Inject a spike at row N. Choose N large enough that the longest
    # window has fully warmed up around it.
    N = 1_500
    # Build a perturbed close: multiply by exp(0.05) (5% log shock).
    close = bars["close"].to_numpy().copy()
    close[N] = close[N] * math.exp(0.05)
    bars_pert = bars.with_columns(pl.Series("close", close))
    # Recompute base series for the perturbed frame.
    df_pd = bars_pert.to_pandas().set_index("ts")
    # Only recompute the columns the eq family reads — that's ``p`` and ``r``.
    # close is the raw; p = log(close); r = p.diff().
    df_pd["p"] = np.log(df_pd["close"])
    df_pd["r"] = df_pd["p"].diff()
    bars_pert = pl.from_pandas(df_pd.reset_index(names="ts"))
    out_pert = _engine_run(bars_pert)

    # Tier-1 window-indexed mu columns that depend on p (or close/volume
    # for VWAP) at row N: must be unchanged.
    tier1_window_cols = [
        f"eq__mu_mean__f__w{w}" for w in WINDOWS_EQ
    ] + [
        f"eq__mu_median__f__w{w}" for w in WINDOWS_EQ
    ] + [
        f"eq__mu_range__f__w{w}" for w in WINDOWS_EQ
    ] + [
        f"eq__mu_trend__f__w{w}" for w in WINDOWS_EQ
    ] + [
        f"eq__trend_sresid__f__w{w}" for w in WINDOWS_EQ
    ] + [
        f"eq__mad_p__f__w{w}" for w in WINDOWS_EQ
    ]
    # sigma_r is a function of returns; r[N] uses p[N], so r[N] changes
    # when p[N] changes. sigma_r at row N uses past-only window over r,
    # which is [r_{N-W}, ..., r_{N-1}] — that includes r[N-W] through
    # r[N-1] only. r[N-1] uses p[N-1] and p[N-2] — not p[N]. So sigma_r
    # at row N is also unchanged.
    tier1_window_cols += [f"eq__sigma_r__f__w{w}" for w in WINDOWS_EQ]

    # VWAP at row N uses close[N-W..N-1] and volume[N-W..N-1] — not row N.
    tier1_window_cols += [f"eq__mu_vwap__f__w{w}" for w in WINDOWS_EQ]

    for col in tier1_window_cols:
        base_val = out_baseline[col][N]
        pert_val = out_pert[col][N]
        if base_val is None and pert_val is None:
            continue
        assert base_val == pytest.approx(pert_val, rel=1e-10, abs=1e-12), (
            f"{col} at row {N} changed under p[{N}] perturbation: "
            f"baseline={base_val}, perturbed={pert_val}"
        )


def test_ewm_innovation_is_pre_update_at_row_n(eq_bars):
    """The EWMA innovation at row N uses the EWMA state at row N-1
    (pre-update). Mutating p[N] must NOT change ``eq__ewm_innov_hz`` at
    row N's *equilibrium* term; the residual numerator does change (it
    uses p[N] directly). But the EWMA state at row N+1 DOES depend on
    p[N], so the innovation at row N+1 changes.

    What we assert: mutating p[N] keeps ``eq__mu_ewm__f__h{H}`` at row
    N unchanged (the equilibrium is pre-update) and changes it at row
    N+1.
    """
    bars = eq_bars
    out_baseline = _engine_run(bars)

    N = 1_500
    close = bars["close"].to_numpy().copy()
    close[N] = close[N] * math.exp(0.05)
    bars_pert = bars.with_columns(pl.Series("close", close))
    df_pd = bars_pert.to_pandas().set_index("ts")
    df_pd["p"] = np.log(df_pd["close"])
    df_pd["r"] = df_pd["p"].diff()
    bars_pert = pl.from_pandas(df_pd.reset_index(names="ts"))
    out_pert = _engine_run(bars_pert)

    for H in HALFLIVES_EQ:
        col_mu = f"eq__mu_ewm__f__h{H}"
        # Row N: equilibrium is pre-update -> unchanged.
        assert out_baseline[col_mu][N] == pytest.approx(
            out_pert[col_mu][N], rel=1e-10, abs=1e-12
        ), f"{col_mu}[N] should be invariant to p[N] (pre-update); H={H}"
        # Row N+1: the shock at N has updated the EWMA state -> changed.
        base_next = out_baseline[col_mu][N + 1]
        pert_next = out_pert[col_mu][N + 1]
        assert base_next != pytest.approx(pert_next, rel=1e-10, abs=1e-12), (
            f"{col_mu}[N+1] should reflect the shock at row N; H={H}"
        )


# ---------------------------------------------------------------------------
# Known-answer probes
# ---------------------------------------------------------------------------


def _make_bars_from_p(p: np.ndarray, *, volume_const: float = 100.0) -> pl.DataFrame:
    """Helper: build a minimal bar frame from a chosen log-price series.

    OHLC pinned at exp(p) (zero spread) and a constant non-zero volume.
    Enough to drive the eq family — `compute_base_series` will populate
    the derived columns (p, r, logvol, etc.).
    """
    n = len(p)
    close = np.exp(p)
    df = pd.DataFrame(
        {
            "open": close, "high": close, "low": close, "close": close,
            "volume": np.full(n, volume_const),
            "quote_volume": np.full(n, volume_const) * close,
            "num_trades": np.full(n, 10, dtype=np.int64),
            "taker_buy_base": np.full(n, volume_const * 0.5),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
    )
    df = utils.compute_base_series(df)
    return pl.from_pandas(df.reset_index(names="ts"))


def test_constant_price_residuals_are_essentially_zero():
    """On a perfectly-flat log price series, every residual feature
    that has a meaningful zero point should be essentially zero.

    Numerical note: ``*_hz`` residuals normalize by ``sigma·√M + EPS``
    with ``EPS = 1e-10``. On a flat series sigma is exactly zero, so
    the denominator collapses to EPS and amplifies the ~1e-15 polars
    rolling-mean round-off in the numerator into a ~1e-5 residual. The
    EPS floor is by design (prevents inf on degenerate windows) — the
    test asserts ``small`` rather than ``exactly zero``. 1e-3 is still
    far below any threshold a model would split on.
    """
    n = 1_500
    p = np.full(n, math.log(100.0))
    bars = _make_bars_from_p(p)
    out = _engine_run(bars)

    # Sample a row past the longest window's warmup.
    N = 1_200
    residual_cols = (
        [f"eq__mean_resid_hz__f__w{w}" for w in WINDOWS_EQ]
        + [f"eq__median_resid_madz__f__w{w}" for w in WINDOWS_EQ]
        + [f"eq__vwap_resid_hz__f__w{w}" for w in WINDOWS_EQ]
        + [f"eq__range_mid_resid_hz__f__w{w}" for w in WINDOWS_EQ]
        + [f"eq__trend_resid_z__f__w{w}" for w in WINDOWS_EQ]
        + [f"eq__ewm_innov_hz__f__h{h}" for h in HALFLIVES_EQ]
    )
    for col in residual_cols:
        val = float(out[col][N])
        assert abs(val) < 1e-3, f"{col} = {val} on flat series (expected ~0)"


def test_linear_trend_recovery():
    """``eq__mu_trend`` extrapolated to the current bar position should
    equal ``a + b·n`` on a noiseless linear series, and the trend
    residual std should be (numerically) zero."""
    n = 1_200
    a, b = math.log(100.0), 0.0003  # gentle upward drift
    p = a + b * np.arange(n, dtype=float)
    bars = _make_bars_from_p(p)
    out = _engine_run(bars)

    # Sample a row well past warmup.
    N = 1_100
    expected_mu = a + b * N
    for w in WINDOWS_EQ:
        mu = float(out[f"eq__mu_trend__f__w{w}"][N])
        sres = float(out[f"eq__trend_sresid__f__w{w}"][N])
        assert mu == pytest.approx(expected_mu, abs=1e-9), (
            f"mu_trend[N] should be {expected_mu}, got {mu} at W={w}"
        )
        assert sres == pytest.approx(0.0, abs=1e-9), (
            f"trend_sresid should be ~0 on noiseless line, got {sres} at W={w}"
        )


def test_linear_trend_resid_z_near_zero_on_perfect_line():
    """On a perfectly linear series, ``eq__trend_resid_z`` is
    ``(p - mu_trend) / (sresid + EPS) ≈ 0 / EPS = 0``. The trend has no
    deviation to explain, so the z-score collapses to zero."""
    n = 1_200
    p = math.log(100.0) + 0.0003 * np.arange(n, dtype=float)
    bars = _make_bars_from_p(p)
    out = _engine_run(bars)
    N = 1_100
    for w in WINDOWS_EQ:
        z = float(out[f"eq__trend_resid_z__f__w{w}"][N])
        assert abs(z) < 1e-5, f"trend_resid_z={z} on perfect line (expected ~0), W={w}"


def test_mean_resid_hz_positive_on_uptrend():
    """On a linear uptrend, the current price is above the past-only
    mean (which is the average of older, lower prices), so
    ``eq__mean_resid_hz`` is strictly positive."""
    n = 1_500
    p = math.log(100.0) + 0.0005 * np.arange(n, dtype=float)
    bars = _make_bars_from_p(p)
    out = _engine_run(bars)
    N = 1_400
    for w in WINDOWS_EQ:
        val = float(out[f"eq__mean_resid_hz__f__w{w}"][N])
        # Sigma_r is tiny (close to 0) on a noiseless line, so the
        # residual normalized by sigma·sqrt(M) can be very large.
        # We only assert sign here, not magnitude.
        assert val > 0, f"mean_resid_hz={val} on uptrend (expected >0), W={w}"


def test_constant_volume_implies_vwap_equals_mean():
    """When volume is constant across the window, VWAP = arithmetic mean
    of close. In log space: log(mean(close)) ≠ mean(log(close)) in
    general (Jensen), but with a flat log-price series the close is
    also constant, so log(VWAP) = log(close) = mu_mean. Use the flat-
    price series for an exact match."""
    n = 1_500
    p = np.full(n, math.log(100.0))
    bars = _make_bars_from_p(p, volume_const=42.0)
    out = _engine_run(bars)
    N = 1_400
    for w in WINDOWS_EQ:
        mu_mean = float(out[f"eq__mu_mean__f__w{w}"][N])
        mu_vwap = float(out[f"eq__mu_vwap__f__w{w}"][N])
        assert mu_mean == pytest.approx(mu_vwap, abs=1e-10), (
            f"mu_vwap should equal mu_mean on flat-price + flat-volume "
            f"series at W={w}: mean={mu_mean}, vwap={mu_vwap}"
        )


# ---------------------------------------------------------------------------
# Sign / range invariants on the random fixture
# ---------------------------------------------------------------------------


def test_proxy_dispersion_non_negative(eq_engine_result):
    """Population std of proxies is non-negative; dispersion / horizon
    vol is also non-negative."""
    out = eq_engine_result
    for w in WINDOWS_EQ:
        col = f"eq__proxy_dispersion_hz__f__w{w}"
        vals = out[col].drop_nulls().to_numpy()
        vals = vals[np.isfinite(vals)]
        assert (vals >= -1e-12).all(), (
            f"{col} has negative values: min={vals.min()}"
        )


def test_pullback_rising_eq_non_negative(eq_engine_result):
    """Pullback feature is a product of two clipped-non-negative
    expressions, must be >= 0."""
    out = eq_engine_result
    for s, l in WINDOWS_EQ_PAIRS:
        col = f"eq__pullback_rising_eq__f__w{s}__l{l}"
        vals = out[col].drop_nulls().to_numpy()
        vals = vals[np.isfinite(vals)]
        assert (vals >= -1e-12).all(), f"{col} has negative values"


def test_above_falling_eq_non_negative(eq_engine_result):
    out = eq_engine_result
    for s, l in WINDOWS_EQ_PAIRS:
        col = f"eq__above_falling_eq__f__w{s}__l{l}"
        vals = out[col].drop_nulls().to_numpy()
        vals = vals[np.isfinite(vals)]
        assert (vals >= -1e-12).all(), f"{col} has negative values"


def test_sigma_r_non_negative(eq_engine_result):
    """Standard deviation is non-negative by construction."""
    out = eq_engine_result
    for w in WINDOWS_EQ:
        vals = out[f"eq__sigma_r__f__w{w}"].drop_nulls().to_numpy()
        vals = vals[np.isfinite(vals)]
        assert (vals >= -1e-12).all()


def test_mad_p_non_negative(eq_engine_result):
    out = eq_engine_result
    for w in WINDOWS_EQ:
        vals = out[f"eq__mad_p__f__w{w}"].drop_nulls().to_numpy()
        vals = vals[np.isfinite(vals)]
        assert (vals >= -1e-12).all()


def test_trend_sresid_non_negative(eq_engine_result):
    out = eq_engine_result
    for w in WINDOWS_EQ:
        vals = out[f"eq__trend_sresid__f__w{w}"].drop_nulls().to_numpy()
        vals = vals[np.isfinite(vals)]
        assert (vals >= -1e-12).all()


def test_dtypes_are_float64(eq_engine_result, eq_engine_specs):
    """Every eq column must land as Float64 — the model layer and the
    impute layer both assume float dtype."""
    for _, _, name in eq_engine_specs:
        assert eq_engine_result.schema[name] == pl.Float64, (
            f"{name} has dtype {eq_engine_result.schema[name]}, expected Float64"
        )


# ---------------------------------------------------------------------------
# Imputation registry: every eq column maps to a finite fill value
# ---------------------------------------------------------------------------


def test_eq_columns_have_imputation_value(eq_engine_specs):
    """The catch-all ``^eq__`` pattern in ``get_imputation_value`` must
    return a finite value (0.0) for every eq column. Without this the
    impute step raises."""
    for _, _, name in eq_engine_specs:
        val = utils.get_imputation_value(name, p_hit_prior=0.5, cap_h_blocks=144)
        assert math.isfinite(val), f"{name}: impute value {val} is non-finite"
        assert val == 0.0, f"{name}: expected 0.0 fill, got {val}"


# ---------------------------------------------------------------------------
# UER and BVE basic shape — sanity around PHI
# ---------------------------------------------------------------------------


def test_uer_is_real_valued_finite(eq_engine_result):
    """UER columns (both proxies) must be finite after warmup. They can
    be positive or negative depending on whether the proxy fair value
    is above or below current price."""
    out = eq_engine_result
    for proxy in ("trend", "vwap"):
        for w in WINDOWS_EQ:
            col = f"eq__upside_to_eq_over_phi__via_{proxy}__f__w{w}"
            vals = out[col].drop_nulls().to_numpy()
            assert np.isfinite(vals).all(), (
                f"{col} has non-finite values after warmup"
            )


def test_bve_uer_relationship_via_trend():
    """``BVE = UER - sigma·sqrt(M)/(sigma·sqrt(M)+EPS)`` is not a closed
    form, but algebraically BVE = (mu - p - PHI) / horizon_vol, and
    UER = (mu - p) / PHI. So
        BVE - UER · PHI / horizon_vol = -PHI / horizon_vol
    For a noiseless line where mu_trend, sigma_r are well-defined, the
    relationship holds row-by-row."""
    n = 1_500
    p = math.log(100.0) + 0.0003 * np.arange(n, dtype=float) + 0.0001 * np.sin(
        np.arange(n) / 50.0
    )
    bars = _make_bars_from_p(p)
    out = _engine_run(bars)
    N = 1_400
    for w in WINDOWS_EQ:
        uer = float(out[f"eq__upside_to_eq_over_phi__via_trend__f__w{w}"][N])
        bve = float(out[f"eq__barrier_vs_eq_hz__via_trend__f__w{w}"][N])
        sigma = float(out[f"eq__sigma_r__f__w{w}"][N])
        # BVE - UER·PHI/horizon_vol == -PHI/horizon_vol
        horizon_vol = sigma * math.sqrt(M) + EPS
        lhs = bve - uer * PHI / horizon_vol
        rhs = -PHI / horizon_vol
        assert lhs == pytest.approx(rhs, rel=1e-8, abs=1e-12), (
            f"BVE/UER relationship broken at W={w}: lhs={lhs}, rhs={rhs}"
        )
