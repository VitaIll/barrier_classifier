"""End-to-end feature-engineering pipeline.

Orchestrates the pieces built in steps 1-14 to produce the final dataset
that the legacy notebook produces. The legacy ``compute_base_series`` and
``compute_derivatives_base_series`` are still used as base-series
generators (out of scope for this refactor); everything downstream is
the new polars stack.

Stages (mirrors the notebook):
  1. ``compute_base_series`` — adds r, rho, clv, etc. (legacy)
  2. ``compute_derivatives_base_series`` — adds basis_abs, pcr_oi, etc. (legacy, optional)
  3. ``FeatureEngine.transform`` — Tier-1 + Tier-2 features (no trim)
  4. ``compute_data_quality_flags_pl`` — bad_ohlc + gap
  5. Boundary sample (every M-th row)
  6. ``construct_labels_pl``
  7. ``compute_past_target_features_pl``
  8. ``compute_barrier_aware_features_pl``
  9. ``compute_block_features_pl``
 10. Warmup trim (k >= K_WARMUP) + drop NaN-label rows
 11. ``create_undef_flags_and_impute_pl``

Returns the final boundary-aligned dataframe ready for training.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
import polars as pl

from src import utils as _legacy
from src.features import FeatureEngine
from src.features.boundary import (
    compute_barrier_aware_features_pl,
    compute_block_features_pl,
    compute_past_target_autocorrelation_pl,
    compute_past_target_features_pl,
    construct_labels_pl,
)
from src.features.config import DEFAULT_CONFIG, FeatureConfig
from src.features.quality import (
    compute_data_quality_flags_pl,
    create_undef_flags_and_impute_pl,
)


# Families included by default (tier 1 + tier 2 of the bar-level engine).
# Round 1/3 additions (``extreme``, ``pivot``, ``flow``) are tier-2 families
# that depend on tier-1 outputs already in this list (``vol``, ``rolling``).
# Round 2 ``equilibrium`` is split across tier-1 (mu/sigma proxies) and tier-2
# (residuals, dispersion, pullback/overextension interactions); all of its
# dependencies are inside the family so registration order does not matter.
_DEFAULT_FAMILIES_NO_DERIV: tuple[str, ...] = (
    "lag", "rolling", "quantile", "vol", "candle", "trend", "activity",
    "correlation", "entropy", "event", "seasonality",
    "excursion", "liquidity",
    "extreme", "pivot", "flow",
    "eq",
)
_DERIVATIVES_FAMILIES: tuple[str, ...] = (
    "deriv_basis", "deriv_flow", "deriv_oi", "deriv_funding",
    "deriv_options", "deriv_volidx",
)


# Columns that are NOT model features even though they sit on the boundary
# frame. Mirrors the notebook's NON_FEATURE_COLS contract — labels, weights,
# raw OHLCV, base series, and derivatives base series. ``m_dn`` and
# ``tau_dn`` carry the optional downside-barrier diagnostics emitted by
# ``construct_labels_pl(..., add_triple_barrier_aux=True)``; they sit next
# to the long-only label and are never model features.
_LABEL_AUX_COLS: tuple[str, ...] = (
    "k", "ts", "y", "m_k", "tau_k", "phi", "m_dn", "tau_dn",
)
_RAW_COLS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "quote_volume",
    "num_trades", "taker_buy_base", "taker_buy_quote",
)
_BASE_COLS: tuple[str, ...] = (
    "p", "r", "rho", "r_oc", "g", "logvol", "logtrades", "logquotevol",
    "b", "ofi", "clv", "bodyfrac", "wickup", "wickdn", "vwap", "vwapdev",
    "qpertrade",
)
_DERIV_BASE_COLS: tuple[str, ...] = (
    "close_fut", "volume_fut", "quote_volume_fut", "taker_buy_base_fut",
    "num_trades_fut", "funding_rate", "oi_usd", "opt_oi",
    "put_open_interest", "call_open_interest", "opt_volume", "put_volume",
    "call_volume", "bvol", "basis_abs", "basis_pct", "tb_ratio_fut",
    "net_vol_fut", "pcr_oi", "pcr_vol",
)


def _to_polars(df_pd: pd.DataFrame) -> pl.DataFrame:
    """pandas → polars with tz stripped (keeps Datetime dtype on Windows)."""
    df = df_pd.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return pl.from_pandas(df.reset_index(names="ts"))


def _coerce_expected_numeric(df_pl: pl.DataFrame) -> pl.DataFrame:
    """Force any expected-numeric column to ``Float64`` with null preservation.

    A left-join onto a date range outside the EOH coverage window (e.g.
    2025 bars joined against 2023-05..2023-10 EOH option data) produces
    an all-null column. ``pl.from_pandas`` of such a column can land as
    ``Object``/``str`` dtype, which then breaks downstream feature compute
    (``pl.col(...).is_infinite()`` rejects ``str``). Cast defensively here.

    String sentinels (e.g. ``"NA"``, ``"null"``) silently become NaN
    under ``strict=False``. Warn when a non-numeric column gains NaN
    rows during the cast so silent data loss surfaces to the caller.
    """
    expected_numeric = set(_RAW_COLS) | set(_BASE_COLS) | set(_DERIV_BASE_COLS)
    casts: list[pl.Expr] = []
    cast_diagnostics: list[tuple[str, int]] = []
    for col in df_pl.columns:
        if col in expected_numeric and df_pl.schema[col] != pl.Float64:
            n_null_before = int(df_pl[col].null_count())
            casts.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
            cast_diagnostics.append((col, n_null_before))
            # Track non-numeric-source columns so we can detect cast losses
            # after the with_columns pass below.
            cast_diagnostics[-1] = (col, n_null_before)
    if casts:
        df_pl = df_pl.with_columns(casts)
        for col, n_null_before in cast_diagnostics:
            # Count NaN introduced by the cast (string sentinel -> NaN with
            # strict=False). null_count is checked too because all-null
            # columns from a missing join are expected.
            s = df_pl[col]
            n_null_after = int(s.null_count())
            n_nan_after = int(s.is_nan().sum() or 0)
            introduced = (n_null_after - n_null_before) + n_nan_after
            if introduced > 0:
                warnings.warn(
                    f"_coerce_expected_numeric: cast of column {col!r} "
                    f"introduced {introduced} non-finite cells "
                    f"(string sentinel or non-castable value).",
                    UserWarning,
                    stacklevel=2,
                )
    return df_pl


_AUTOCORR_WINDOWS_DEFAULT_BOUNDARY: tuple[int, ...] = (12, 24, 72, 144)
_AUTOCORR_WINDOWS_DEFAULT_1MIN: tuple[int, ...] = (60, 240, 1440, 2880)
_AUTOCORR_LAGS_DEFAULT: tuple[int, ...] = (1, 2, 5, 10)


def run_pipeline(
    df_raw_pd: pd.DataFrame,
    *,
    with_derivatives: bool = False,
    p_hit_prior: float = 0.5,
    cap_h_blocks: int | None = None,
    label_cadence: str = "boundary",
    enable_autocorrelation: bool | None = None,
    autocorr_windows: tuple[int, ...] | None = None,
    autocorr_lags: tuple[int, ...] = _AUTOCORR_LAGS_DEFAULT,
    barrier_source: str | None = None,
    add_triple_barrier_aux: bool = False,
    config: FeatureConfig | None = None,
) -> pl.DataFrame:
    """Run the full feature pipeline end-to-end on a pandas bar frame.

    ``df_raw_pd`` must have OHLCV columns and a tz-aware DatetimeIndex.
    For ``with_derivatives=True`` the frame must also carry the
    derivatives raw columns (close_fut, funding_rate, oi_usd, etc.) —
    the legacy ``compute_derivatives_base_series`` runs first to derive
    basis_abs, pcr_oi, and friends.

    ``label_cadence`` is the toggle for the new target definition:

    - ``"boundary"`` (default): legacy mode. Sample every M-th bar; one
      label per boundary; M-bar non-overlapping prediction windows. The
      output is row-aligned to the legacy schema and preserves bit-equal
      parity with ``utils.compute_*`` for the no-derivatives path.
    - ``"1min"``: every 1-min bar gets its own label (look M bars forward
      from that bar). Adjacent labels share M−1 of their M future bars,
      so they are strongly autocorrelated by construction. The new
      ``target__autocorr_lag{L}__h__w{W}`` features surface that
      autocorrelation in a strictly causal way (label-maturity shift of
      ``M // bar_stride`` rows).

    ``barrier_source`` controls the upper-barrier crossing test. The
    default is cadence-dependent:

    - ``"close"`` at boundary cadence — preserves the legacy close-
      confirmed label and bit-equal parity with ``utils.construct_labels``.
    - ``"high"`` at 1-min cadence — aligns the label with a long TP-limit
      order that fills when an intrabar high crosses the barrier (matches
      the simulator's ``exit_tp_or_expiry`` for a long position). Mixing
      close-based labels with high-based exits trains one event and
      trades a different one, so the high-based default is the correct
      target/execution alignment for the production strategy.

    Passing ``barrier_source`` explicitly overrides the cadence-based
    default in either direction.

    Reverting from ``"1min"`` to ``"boundary"`` is a one-line flip of
    this argument; the resulting frame is the legacy schema.
    """
    plan = _resolve_plan(
        label_cadence=label_cadence,
        barrier_source=barrier_source,
        enable_autocorrelation=enable_autocorrelation,
        autocorr_windows=autocorr_windows,
        cap_h_blocks=cap_h_blocks,
        config=config if config is not None else DEFAULT_CONFIG,
    )

    df_pl, df_raw_pl, impute_map = _build_bar_level(
        df_raw_pd,
        with_derivatives=with_derivatives,
        label_cadence=label_cadence,
        config=plan.config,
    )
    df_boundaries = _run_boundary_stages(
        df_pl,
        df_raw_pl,
        plan=plan,
        autocorr_lags=autocorr_lags,
        add_triple_barrier_aux=add_triple_barrier_aux,
    )

    # --- Stage 10: warmup trim + drop NaN labels ---------------------------
    # Warmup is in the row units of the current cadence:
    #   boundary cadence -> K_WARMUP boundary rows = N_WARMUP / M
    #   1-min cadence    -> N_WARMUP 1-min rows
    df_boundaries = (
        df_boundaries
        .filter(pl.col("k") >= plan.warmup_rows)
        .filter(pl.col("y").is_not_null())
    )

    return _impute_stage(
        df_boundaries, p_hit_prior=p_hit_prior, plan=plan, impute_map=impute_map
    )


def run_inference_pipeline(
    df_raw_pd: pd.DataFrame,
    *,
    with_derivatives: bool = False,
    p_hit_prior: float = 0.5,
    cap_h_blocks: int | None = None,
    label_cadence: str = "1min",
    enable_autocorrelation: bool | None = None,
    autocorr_windows: tuple[int, ...] | None = None,
    autocorr_lags: tuple[int, ...] = _AUTOCORR_LAGS_DEFAULT,
    barrier_source: str | None = None,
    boundary_tail_rows: int | None = 6_000,
    config: FeatureConfig | None = None,
) -> pl.DataFrame:
    """Feature pipeline for *serving*: same stages, no label filtering.

    Identical to :func:`run_pipeline` except:

    - rows whose labels are not yet mature (the last ``M`` bars — exactly
      the rows a live engine predicts on) are **kept**; their ``y``/``m_k``/
      ``tau_k`` are null;
    - the warmup *trim* is skipped — the caller guarantees window depth
      (the engine's :class:`~src.engine.guards.WarmupGuard`); every row of
      the output is only as warm as the input window makes it;
    - optionally, the boundary stages (labels, past-target, barrier, block)
      run on only the trailing ``boundary_tail_rows`` rows. Every boundary-
      stage lookback at 1-min cadence is ≤ ~2,920 rows (hit-rate/autocorr
      windows 2,880 + label-maturity shift M), so the default 6,000 leaves
      a 2× margin while skipping >80% of the boundary-stage work on a
      typical live buffer. Bar-level (tier-1/2) features always see the
      full window — their lookbacks are the deep ones.

    The last output row corresponds to the last input bar. Callers select
    model features by the feature-list contract; label columns of tail
    rows are null by construction and must not be consumed.
    """
    plan = _resolve_plan(
        label_cadence=label_cadence,
        barrier_source=barrier_source,
        enable_autocorrelation=enable_autocorrelation,
        autocorr_windows=autocorr_windows,
        cap_h_blocks=cap_h_blocks,
        config=config if config is not None else DEFAULT_CONFIG,
    )
    df_pl, df_raw_pl, impute_map = _build_bar_level(
        df_raw_pd,
        with_derivatives=with_derivatives,
        label_cadence=label_cadence,
        config=plan.config,
    )
    if boundary_tail_rows is not None and boundary_tail_rows < len(df_pl):
        if label_cadence == "boundary":
            raise ValueError(
                "boundary_tail_rows slicing is only supported at 1-min cadence "
                "(boundary cadence resamples rows and is not slice-invariant)"
            )
        # Both frames are row-aligned at 1-min cadence; slice them together
        # so construct_labels' positional indexing stays consistent.
        df_pl = df_pl.tail(boundary_tail_rows)
        df_raw_pl = df_raw_pl.tail(boundary_tail_rows)
    df_boundaries = _run_boundary_stages(
        df_pl,
        df_raw_pl,
        plan=plan,
        autocorr_lags=autocorr_lags,
        add_triple_barrier_aux=False,
    )
    return _impute_stage(
        df_boundaries, p_hit_prior=p_hit_prior, plan=plan, impute_map=impute_map
    )


@dataclass(frozen=True)
class _PipelinePlan:
    """Resolved cadence-dependent knobs shared by both entry points."""

    config: FeatureConfig
    label_cadence: str
    bar_stride: int
    barrier_source: str
    enable_autocorrelation: bool
    autocorr_windows: tuple[int, ...]
    hitrate_windows: tuple[int, ...]
    block_windows: tuple[int, ...]
    warmup_rows: int
    cap_h_blocks: int


def _resolve_plan(
    *,
    label_cadence: str,
    barrier_source: str | None,
    enable_autocorrelation: bool | None,
    autocorr_windows: tuple[int, ...] | None,
    cap_h_blocks: int | None,
    config: FeatureConfig,
) -> _PipelinePlan:
    if label_cadence not in ("boundary", "1min"):
        raise ValueError(
            f"label_cadence must be 'boundary' or '1min', got {label_cadence!r}"
        )
    M = config.m
    bar_stride = int(M) if label_cadence == "boundary" else 1
    if barrier_source is None:
        barrier_source = "close" if label_cadence == "boundary" else "high"
    if barrier_source not in ("close", "high"):
        raise ValueError(
            f"barrier_source must be 'close' or 'high', got {barrier_source!r}"
        )
    # Autocorrelation columns default on at 1-min (where labels overlap by M-1
    # bars and the signal is informative) and off at boundary cadence (where
    # the signal is weak and adding the columns silently changes the legacy
    # schema). Explicit True/False overrides this.
    if enable_autocorrelation is None:
        enable_autocorrelation = label_cadence == "1min"
    # Calendar-time semantics: at boundary cadence, "_h" windows are counted
    # in boundary rows (1 row = M bars). At 1-min cadence, they're counted in
    # 1-min rows. Scale by M so the *calendar* time of each window is preserved
    # across cadences — w=72 at boundary cadence (24h) becomes w=1440 at 1-min
    # cadence (also 24h). Column names follow the value used, so a model
    # trained at one cadence has a distinct feature vector from a model
    # trained at the other.
    scale = 1 if label_cadence == "boundary" else int(M)
    hitrate_windows = tuple(int(w) * scale for w in config.hitrate_windows_h)
    block_windows = tuple(int(w) * scale for w in config.windows_h)
    warmup_rows = (
        int(config.k_warmup) if label_cadence == "boundary" else int(config.n_warmup)
    )
    if cap_h_blocks is None:
        # cap_h_blocks is used by the impute step for hit__since at the
        # right calendar scale; max of the scaled hitrate windows is the
        # natural ceiling.
        cap_h_blocks = max(hitrate_windows)
    if autocorr_windows is None:
        autocorr_windows = (
            _AUTOCORR_WINDOWS_DEFAULT_BOUNDARY
            if label_cadence == "boundary"
            else _AUTOCORR_WINDOWS_DEFAULT_1MIN
        )
    return _PipelinePlan(
        config=config,
        label_cadence=label_cadence,
        bar_stride=bar_stride,
        barrier_source=barrier_source,
        enable_autocorrelation=bool(enable_autocorrelation),
        autocorr_windows=tuple(autocorr_windows),
        hitrate_windows=hitrate_windows,
        block_windows=block_windows,
        warmup_rows=warmup_rows,
        cap_h_blocks=int(cap_h_blocks),
    )


def _build_bar_level(
    df_raw_pd: pd.DataFrame,
    *,
    with_derivatives: bool,
    label_cadence: str,
    config: FeatureConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float]]:
    """Stages 1-4: base series, engine features, quality flags (+ impute map)."""
    # --- Stages 1-2: base series (legacy) ----------------------------------
    df_pd = _legacy.compute_base_series(df_raw_pd)
    if with_derivatives:
        df_pd = _legacy.compute_derivatives_base_series(df_pd)

    df_pl = _coerce_expected_numeric(_to_polars(df_pd))
    df_raw_pl = _coerce_expected_numeric(_to_polars(df_raw_pd))

    # --- Stage 3: bar-level features (Tier 1 + Tier 2) ---------------------
    families = list(_DEFAULT_FAMILIES_NO_DERIV)
    if with_derivatives:
        families.extend(_DERIVATIVES_FAMILIES)
    engine = FeatureEngine(tiers=(1, 2), families=tuple(families), config=config)
    impute_map = engine.imputation_map()
    df_pl = engine.transform(df_pl, trim=False).data

    # At 1-min cadence the boundary-sparse excursion drawup/drawdown columns
    # are NaN at every non-multiple-of-M row by construction, which would
    # leak a phase artifact (modulo-M missingness) through the imputation
    # step and create a synthetic "regime label" of zero information. The
    # every-row trailing variants ``excursion__roll_max_drawup__f__w*`` and
    # ``excursion__roll_max_drawdown__f__w*`` cover the same statistic at
    # every row, so the sparse pair is redundant at 1-min cadence — drop it.
    if label_cadence == "1min":
        sparse_excursion = [
            c for c in df_pl.columns
            if c.startswith("excursion__max_drawup__f__")
            or c.startswith("excursion__max_drawdown__f__")
        ]
        if sparse_excursion:
            df_pl = df_pl.drop(sparse_excursion)

    # --- Stage 4: data quality flags --------------------------------------
    df_pl = compute_data_quality_flags_pl(df_pl)
    return df_pl, df_raw_pl, impute_map


def _run_boundary_stages(
    df_pl: pl.DataFrame,
    df_raw_pl: pl.DataFrame,
    *,
    plan: _PipelinePlan,
    autocorr_lags: tuple[int, ...],
    add_triple_barrier_aux: bool,
) -> pl.DataFrame:
    """Stages 5-9: decision-row sampling, labels, boundary features."""
    cfg = plan.config
    # --- Stage 5: sample decision rows -------------------------------------
    # Boundary cadence keeps every M-th row; 1-min cadence keeps every row.
    if plan.label_cadence == "boundary":
        df_boundaries = df_pl.gather_every(cfg.m).with_columns(
            pl.int_range(pl.len()).alias("k")
        )
    else:
        df_boundaries = df_pl.with_columns(pl.int_range(pl.len()).alias("k"))

    # --- Stages 6-9: boundary stages ---------------------------------------
    df_boundaries = construct_labels_pl(
        df_boundaries,
        df_raw_pl,
        cfg.m,
        cfg.eta,
        cfg.c,
        bar_stride=plan.bar_stride,
        barrier_source=plan.barrier_source,
        add_triple_barrier_aux=add_triple_barrier_aux,
    )
    df_boundaries = compute_past_target_features_pl(
        df_boundaries,
        list(plan.block_windows),
        list(plan.hitrate_windows),
        bar_stride=plan.bar_stride,
        M=int(cfg.m),
    )
    if plan.enable_autocorrelation:
        df_boundaries = compute_past_target_autocorrelation_pl(
            df_boundaries,
            plan.autocorr_windows,
            bar_stride=plan.bar_stride,
            M=int(cfg.m),
            lags=tuple(autocorr_lags),
        )
    df_boundaries = compute_barrier_aware_features_pl(
        df_boundaries,
        cfg.windows_barrier,
        cfg.phi,
        cfg.m,
        cfg.vol_pairs,
        c=float(cfg.c),
    )
    df_boundaries = compute_block_features_pl(
        df_boundaries,
        df_raw_pl,
        cfg.m,
        list(plan.block_windows),
        bar_stride=plan.bar_stride,
    )
    return df_boundaries


def _impute_stage(
    df_boundaries: pl.DataFrame,
    *,
    p_hit_prior: float,
    plan: _PipelinePlan,
    impute_map: dict[str, float],
) -> pl.DataFrame:
    """Stage 11: undef flags + deterministic imputation."""
    # POSITIVE feature membership: a column is a feature because something
    # declared it — the engine emitted it (impute_map keys = the plan) or
    # a boundary-stage prefix declares it. The role tuples below classify
    # the known NON-features (labels, raw bars, base series); any column
    # in NEITHER set is contract drift and fails loudly. This replaces the
    # historical everything-except-exclusion-lists subtraction, under
    # which a new stray column silently became a model feature.
    from src.core.errors import ContractError
    from src.features.boundary import is_boundary_feature_column

    non_feature = set(_LABEL_AUX_COLS + _RAW_COLS + _BASE_COLS + _DERIV_BASE_COLS)
    feature_cols: list[str] = []
    unknown: list[str] = []
    for c in df_boundaries.columns:
        if c in impute_map or is_boundary_feature_column(c):
            feature_cols.append(c)
        elif c not in non_feature:
            unknown.append(c)
    if unknown:
        raise ContractError(
            f"pipeline produced {len(unknown)} column(s) with no declared "
            f"role: {unknown[:8]} — declare a Feature class / boundary "
            "prefix (feature) or add to the role tuples in pipeline.py "
            "(label/raw/base)."
        )
    df_final, _ = create_undef_flags_and_impute_pl(
        df_boundaries,
        feature_cols,
        p_hit_prior=p_hit_prior,
        cap_h_blocks=plan.cap_h_blocks,
        impute_map=impute_map,
    )
    return df_final
