"""Augment the research-prediction cache with the two columns the strategy
backtest needs but the offline-study cache doesn't store by default:

- ``r_realized`` — end-of-horizon realized log return: ``ln(close_{n_k+M} / close_{n_k})``.
  Computed from the raw 1-min bars (path data already on disk). Required for
  honest EV: ``y=0`` doesn't mean ``-φ``, it means the upper barrier wasn't
  hit; the realized loss could be anywhere from ``-∞`` to ``+φ``.
- ``mean_p_ve`` + ``knowledge_unc`` — virtual-ensemble mean and epistemic
  uncertainty per row. Requires the underlying CatBoost model trained with
  ``posterior_sampling=True`` (which the research model already is). Optional;
  the strategy degrades gracefully when these columns are absent.

Both augmentations are idempotent: passing a frame that already has the
target columns returns it unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# r_realized (end-of-horizon return)
# ---------------------------------------------------------------------------


def augment_cache_with_r_realized(
    cache: pd.DataFrame,
    raw_bars: pd.DataFrame,
    *,
    M: int = 20,
    skip_if_present: bool = True,
) -> pd.DataFrame:
    """Append ``r_realized`` to the cache.

    ``r_realized[k] = ln(close[n_k + M] / close[n_k])``. Falls back to NaN
    for the last boundary (no future horizon).

    ``raw_bars`` must be a DatetimeIndex-ed OHLC frame; the function joins on
    the boundary timestamp by ``asof`` to find ``close[n_k]`` and then jumps
    M bars forward.
    """
    if skip_if_present and "r_realized" in cache.columns:
        return cache.copy()
    if "ts" not in cache.columns:
        raise ValueError("cache must carry a 'ts' column")
    if "close" not in raw_bars.columns:
        raise ValueError("raw_bars must carry a 'close' column")

    raw = raw_bars.copy()
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw.sort_index()
    raw_close = raw["close"].to_numpy(dtype=float)

    out = cache.copy()
    # Guard against caches that carry ``ts`` as object/string — silent
    # ``.dt`` accessor failures here were a known confounder.
    if not pd.api.types.is_datetime64_any_dtype(out["ts"]):
        raise TypeError(
            f"cache['ts'] must be datetime-typed; got dtype={out['ts'].dtype}. "
            "Cast with pd.to_datetime upstream."
        )
    boundary_ts = pd.to_datetime(out["ts"]).dt.tz_localize(None) if out["ts"].dt.tz is not None else pd.to_datetime(out["ts"])
    # For each boundary, find the matching 1-min bar by exact ts (every boundary
    # ts MUST exist in raw — sampled from raw with iloc[::M]). If not, fall back
    # to asof.
    positions = np.full(len(out), -1, dtype=np.int64)
    raw_ts = pd.Series(np.arange(len(raw)), index=raw.index)
    for i, ts in enumerate(boundary_ts):
        try:
            positions[i] = int(raw_ts.loc[ts])
        except KeyError:
            # asof lookup for ts <= raw timestamp
            asof = raw_ts.index.searchsorted(ts, side="right") - 1
            if asof >= 0:
                positions[i] = int(asof)

    r_realized = np.full(len(out), np.nan, dtype=float)
    for i, n_k in enumerate(positions):
        if n_k < 0 or n_k + M >= len(raw_close):
            continue
        p0 = raw_close[n_k]
        p1 = raw_close[n_k + M]
        if p0 > 0 and p1 > 0:
            r_realized[i] = math.log(p1 / p0)
    out["r_realized"] = r_realized
    return out


# ---------------------------------------------------------------------------
# Boundary OHLC (from raw bars onto the cache by ts join)
# ---------------------------------------------------------------------------


def augment_cache_with_boundary_ohlc(
    cache: pd.DataFrame,
    raw_bars: pd.DataFrame,
    *,
    skip_if_present: bool = True,
) -> pd.DataFrame:
    """Append ``open / high / low / close`` to the cache by joining on ``ts``.

    The prediction cache from ``fast_train.compute_predictions`` carries the
    boundary timestamp but not the boundary-bar's OHLC. The simulator needs
    the close to fill entries / mark MTM, so we join from raw bars upfront.
    Idempotent — if all four columns are already present, returns the cache
    unchanged.
    """
    needed = ["open", "high", "low", "close"]
    if skip_if_present and all(c in cache.columns for c in needed):
        return cache.copy()

    raw = raw_bars.copy()
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw.sort_index()

    if not pd.api.types.is_datetime64_any_dtype(cache["ts"]):
        raise TypeError(
            f"cache['ts'] must be datetime-typed; got dtype={cache['ts'].dtype}. "
            "Cast with pd.to_datetime upstream."
        )
    boundary_ts = pd.to_datetime(cache["ts"])
    if boundary_ts.dt.tz is not None:
        boundary_ts = boundary_ts.dt.tz_localize(None)

    raw_sub = raw.loc[raw.index.isin(boundary_ts), needed]
    # Reindex to cache order
    raw_sub = raw_sub.reindex(boundary_ts.values)
    out = cache.copy()
    for col in needed:
        out[col] = raw_sub[col].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Virtual-ensemble quantities
# ---------------------------------------------------------------------------


def augment_cache_with_ve(
    cache: pd.DataFrame,
    model,
    feature_matrix_by_split: dict,
    feature_list: Sequence[str],
    *,
    virtual_ensembles_count: int = 20,
    skip_if_present: bool = True,
    return_samples: bool = False,
):
    """Append ``mean_p_ve`` and ``knowledge_unc`` to the cache.

    ``feature_matrix_by_split`` maps the cache's ``split`` value (``"val"`` /
    ``"test"``) to the feature matrix (DataFrame or ndarray) used for
    prediction. Row order within each split must match the cache rows for
    that split.

    When ``return_samples=True``, also returns the per-row, per-ensemble
    probabilities aligned to the cache (numpy array of shape ``(N, K_ve)``)
    so the strategy's Bayesian-Kelly sizer has the full posterior.
    """
    if skip_if_present and "mean_p_ve" in cache.columns and "knowledge_unc" in cache.columns:
        if return_samples:
            return cache.copy(), None
        return cache.copy()

    from src.analytics.uncertainty import predictive_uncertainty, virtual_ensemble_predictions

    out = cache.copy()
    n = len(out)
    mean_p_ve = np.full(n, np.nan, dtype=float)
    knowledge_unc = np.full(n, np.nan, dtype=float)
    samples_full = (
        np.full((n, virtual_ensembles_count), np.nan, dtype=float)
        if return_samples else None
    )

    for split_name, X in feature_matrix_by_split.items():
        mask = (out["split"] == split_name).to_numpy()
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        p_ve = virtual_ensemble_predictions(
            model, X, virtual_ensembles_count=virtual_ensembles_count,
            feature_list=feature_list,
        )
        # Length sanity
        if len(p_ve) != len(idx):
            raise ValueError(
                f"feature matrix for split {split_name!r} has {len(p_ve)} rows, "
                f"cache has {len(idx)} — must match"
            )
        decomp = predictive_uncertainty(p_ve)
        mean_p_ve[idx] = decomp["mean_p"]
        knowledge_unc[idx] = decomp["knowledge_uncertainty"]
        if return_samples:
            samples_full[idx, :] = p_ve

    out["mean_p_ve"] = mean_p_ve
    out["knowledge_unc"] = knowledge_unc
    if return_samples:
        return out, samples_full
    return out


def select_p_ve_samples(
    samples_full: Optional[np.ndarray],
    cache: pd.DataFrame,
    *,
    split: str,
) -> Optional[np.ndarray]:
    """Slice the VE-samples array to the rows of one split.

    Used by the simulator: ``p_ve_samples`` passed to ``simulate`` must be
    row-aligned to the cache subset that's being simulated.
    """
    if samples_full is None:
        return None
    mask = (cache["split"] == split).to_numpy()
    return samples_full[mask]
