"""Shared mathematical primitives used by feature families.

Every primitive returns a pl.Expr so the engine composes them inside a
single with_columns() pass. All primitives are causal-by-construction —
they never reference future rows.

Implemented in waves, each landing with adversarial + pandas-oracle tests:
- Step 2: safe_log_ratio, log_return, eps_safe_div, log1p_vol, clip_pos
- Step 3: rolling_mean, rolling_std_pop, rolling_sum, rolling_min, rolling_max
- Step 4: ewm_mean, rs_variance, z_score_rolling
- Step 5: population_corr, wilder_smooth
- Step 6 (this module): rolling_quantile, rolling_mad, perm_entropy_m3,
                        signed_run_dir, signed_run_length, signed_run_cumret
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from src.features.config import EPS


def safe_log_ratio(
    num: pl.Expr,
    den: pl.Expr,
    *,
    when: pl.Expr | None = None,
) -> pl.Expr:
    """log(num / den) where the guard is true; null otherwise.

    Default guard: ``num > 0 AND den > 0``. Matches the most common pattern
    in compute_base_series, e.g.::

        g = log(open / prev_close) when (open > 0) & (prev_close > 0)

    Use ``when=`` to override for non-default guards, e.g.::

        rho = safe_log_ratio(high, low, when=pl.col("high") > pl.col("low"))

    NaN inputs are excluded by ``is_finite()`` in the default guard, and
    null inputs propagate naturally because ``null > 0`` is null. Both
    fall through to the otherwise branch and produce null.

    When ``when=`` is supplied, the caller is responsible for excluding
    NaN/Inf if desired.

    Pandas equivalent::

        np.where(guard, np.log(num / den), np.nan)
    """
    if when is None:
        guard = num.is_finite() & den.is_finite() & (num > 0) & (den > 0)
    else:
        guard = when
    return pl.when(guard).then((num / den).log()).otherwise(None)


def log_return(x: pl.Expr) -> pl.Expr:
    """Log return: ``log(x_t) - log(x_{t-1})``. Row 0 is null.

    No guard — non-positive ``x`` produces ``-inf`` or ``NaN`` and propagates,
    matching the unguarded ``r = np.log(close).diff()`` in
    ``compute_base_series`` (utils.py:1517). Use :func:`safe_log_ratio` when
    a guard is required.

    Pandas equivalent::

        np.log(x).diff()
    """
    return x.log().diff()


def eps_safe_div(
    num: pl.Expr,
    den: pl.Expr,
    *,
    eps: float = EPS,
) -> pl.Expr:
    """``num / (den + eps)`` — bounded ratio for legitimately-zero denominators.

    Differs from :func:`safe_log_ratio` by NOT masking: it lets the ratio
    grow large but finite. Used pervasively in liquidity, barrier-distance,
    and derivatives ratio features (see utils.py:2114, 2125, 2148, 2521,
    2582, 2677 for examples).

    The default ``eps`` matches ``utils.EPS = 1e-10``; pass an explicit
    ``eps=`` only when reproducing legacy code with a different constant.

    Pandas equivalent::

        num / (den + EPS)
    """
    return num / (den + eps)


def log1p_vol(x: pl.Expr) -> pl.Expr:
    """``log(1 + x)``. Used for volume-like quantities (always non-negative).

    Implementation: ``(x + 1).log()``. Numerically equivalent to ``np.log1p``
    for ``x >> 1e-8`` (which covers all realistic OHLCV-derived inputs:
    volume, quote_volume, num_trades). For tiny ``x`` near float epsilon
    the implementations diverge by ~1 ulp; not a concern at our scale.

    Pandas equivalent::

        np.log1p(x)
    """
    return (x + 1).log()


def clip_pos(x: pl.Expr) -> pl.Expr:
    """``max(0, x)``. Clamp negatives to zero.

    Used to clamp variance estimators (Garman-Klass, Rogers-Satchell)
    before sqrt — RS variance can go negative on degenerate OHLC ticks.

    Pandas equivalent::

        np.maximum(0.0, x)
    """
    return x.clip(lower_bound=0.0)


# ---------------------------------------------------------------------------
# Step 3: rolling primitives
#
# All assume the engine has pre-processed inputs via ``fill_nan(None)`` so
# that float NaN is converted to polars null. With strict ``min_samples=w``
# (the default), any null in a window propagates to a null output, matching
# pandas ``rolling(w, min_periods=w).f()`` semantics.
# ---------------------------------------------------------------------------


def rolling_mean(
    x: pl.Expr,
    w: int,
    *,
    min_samples: int | None = None,
) -> pl.Expr:
    """Rolling arithmetic mean over ``w`` rows. Strict windowing by default.

    First ``w-1`` rows are null. ``min_samples`` defaults to ``w`` (refuse
    partial windows); pass ``min_samples=1`` for permissive partial means.

    Pandas equivalent::

        x.rolling(w, min_periods=w).mean()
    """
    return x.rolling_mean(window_size=w, min_samples=min_samples or w)


def rolling_std_pop(
    x: pl.Expr,
    w: int,
    *,
    min_samples: int | None = None,
) -> pl.Expr:
    """Rolling population standard deviation (``ddof=0``).

    The current pandas codebase uses ``ddof=0`` throughout (see utils.py
    lines 1621, 1628, 1629, 1776, 2388, 2588, 2592, 2650). Polars'
    ``rolling_std`` defaults to ``ddof=1`` — this primitive enforces
    ``ddof=0`` so callers cannot accidentally produce sample-std.

    Pandas equivalent::

        x.rolling(w, min_periods=w).std(ddof=0)
    """
    return x.rolling_std(window_size=w, min_samples=min_samples or w, ddof=0)


def rolling_sum(
    x: pl.Expr,
    w: int,
    *,
    min_samples: int | None = None,
) -> pl.Expr:
    """Rolling sum over ``w`` rows. Strict windowing by default.

    Pandas equivalent::

        x.rolling(w, min_periods=w).sum()
    """
    return x.rolling_sum(window_size=w, min_samples=min_samples or w)


def rolling_min(
    x: pl.Expr,
    w: int,
    *,
    min_samples: int | None = None,
) -> pl.Expr:
    """Rolling min over ``w`` rows. Strict windowing by default.

    Pandas equivalent::

        x.rolling(w, min_periods=w).min()
    """
    return x.rolling_min(window_size=w, min_samples=min_samples or w)


def rolling_max(
    x: pl.Expr,
    w: int,
    *,
    min_samples: int | None = None,
) -> pl.Expr:
    """Rolling max over ``w`` rows. Strict windowing by default.

    Pandas equivalent::

        x.rolling(w, min_periods=w).max()
    """
    return x.rolling_max(window_size=w, min_samples=min_samples or w)


# ---------------------------------------------------------------------------
# Step 4: composite primitives
# ---------------------------------------------------------------------------


def ewm_mean_halflife(x: pl.Expr, *, half_life: int) -> pl.Expr:
    """EWMA with half-life parametrization, ``adjust=False`` (causal recursive).

    ``alpha = 1 - 2^(-1/half_life)`` so the weight of the bar ``half_life``
    rows ago is exactly 0.5× the weight on the current bar:

        y_t = alpha · x_t + (1 - alpha) · y_{t-1},  y_0 = x_0

    Polars accepts ``half_life`` directly; ``adjust=False`` is critical
    because polars' default ``adjust=True`` uses normalized weights that
    do not match the spec's recursive update form.

    Wrapped here rather than called bare in feature classes so the
    ``adjust=False`` discipline is enforced in one place.
    """
    return x.ewm_mean(half_life=half_life, adjust=False)


def ewm_mean(x: pl.Expr, *, span: int, adjust: bool = False) -> pl.Expr:
    """Exponentially weighted mean with ``alpha = 2 / (span + 1)``.

    Defaults to ``adjust=False`` (causal recursive form
    ``y_t = α·x_t + (1-α)·y_{t-1}``) — this matches the existing pandas
    codebase (``compute_trend_momentum``'s ``ema(series, W)`` helper,
    utils.py:1862, and ``compute_funding_features`` at line 2703).

    Polars' native default is ``adjust=True`` which uses unequal weights
    near the start of the series; the parity hazard is silent.

    Pandas equivalent::

        x.ewm(span=span, adjust=False).mean()
    """
    return x.ewm_mean(span=span, adjust=adjust)


def rs_variance(o: pl.Expr, h: pl.Expr, l: pl.Expr, c: pl.Expr) -> pl.Expr:
    """Rogers-Satchell instantaneous variance for one OHLC bar.

    Formula:

        RS = log(h/o)·log(h/c) + log(l/o)·log(l/c)

    Per-bar value can be negative on degenerate ticks (h < o or h < c).
    Always apply ``clip_pos`` before ``sqrt`` when reducing to a volatility.

    The expression is unguarded — non-positive OHLC propagates as -inf/NaN,
    matching the existing pandas form at utils.py:1707-1710 and 1735-1738.

    Pandas equivalent::

        np.log(h/o) * np.log(h/c) + np.log(l/o) * np.log(l/c)
    """
    return (h / o).log() * (h / c).log() + (l / o).log() * (l / c).log()


def z_score_rolling(
    x: pl.Expr,
    w: int,
    *,
    min_samples: int | None = None,
) -> pl.Expr:
    """Rolling z-score: ``(x - rmean(x, w)) / rstd_pop(x, w)``.

    Returns null when the rolling std is zero (all-equal window) — matches
    the pandas pattern ``z.where(sigma != 0, np.nan)`` at utils.py:1856-1859
    and 1909-1912. Uses population std (``ddof=0``) consistent with the
    rest of the codebase.

    Pandas equivalent::

        mu = x.rolling(w, min_periods=w).mean()
        sigma = x.rolling(w, min_periods=w).std(ddof=0)
        ((x - mu) / sigma).where(sigma != 0, np.nan)
    """
    mu = rolling_mean(x, w, min_samples=min_samples)
    sigma = rolling_std_pop(x, w, min_samples=min_samples)
    return pl.when(sigma > 0).then((x - mu) / sigma).otherwise(None)


# ---------------------------------------------------------------------------
# Step 5: custom primitives (no polars native equivalent)
# ---------------------------------------------------------------------------


def population_corr(x: pl.Expr, y: pl.Expr, w: int) -> pl.Expr:
    """Rolling Pearson correlation with population variance (``ddof=0``).

    Computes ``cov(x, y) / (std_pop(x) * std_pop(y))`` via the algebraic
    decomposition:

        var_x  = E[x²] - E[x]²
        var_y  = E[y²] - E[y]²
        cov    = E[xy] - E[x]·E[y]
        corr   = cov / (sqrt(max(0, var_x)) * sqrt(max(0, var_y)))

    Result is null when either variance is exactly zero (constant window),
    matching the legacy ``where((var_x != 0) & (var_y != 0), np.nan)`` at
    utils.py:1939. The ``max(0, var)`` clamp guards against tiny-negative
    variances from floating-point cancellation; the strict ``!= 0`` guard
    is intentional and preserves legacy behavior on edge cases.

    Do NOT use polars' ``rolling_corr`` — it computes sample correlation
    (``ddof=1``) which differs from the codebase convention.

    Pandas equivalent (utils.py ``_rolling_corr_population``)::

        mean_x = x.rolling(W).mean()
        mean_y = y.rolling(W).mean()
        mean_x2 = (x**2).rolling(W).mean()
        mean_y2 = (y**2).rolling(W).mean()
        mean_xy = (x*y).rolling(W).mean()
        var_x = mean_x2 - mean_x**2
        var_y = mean_y2 - mean_y**2
        cov   = mean_xy - mean_x*mean_y
        std_x = np.sqrt(np.maximum(0, var_x))
        std_y = np.sqrt(np.maximum(0, var_y))
        (cov / (std_x*std_y)).where((var_x != 0) & (var_y != 0), np.nan)
    """
    mean_x = rolling_mean(x, w)
    mean_y = rolling_mean(y, w)
    mean_x2 = rolling_mean(x ** 2, w)
    mean_y2 = rolling_mean(y ** 2, w)
    mean_xy = rolling_mean(x * y, w)

    var_x = mean_x2 - mean_x ** 2
    var_y = mean_y2 - mean_y ** 2
    cov_xy = mean_xy - mean_x * mean_y

    std_x = clip_pos(var_x).sqrt()
    std_y = clip_pos(var_y).sqrt()

    return pl.when((var_x != 0) & (var_y != 0)).then(
        cov_xy / (std_x * std_y)
    ).otherwise(None)


def _wilder_smooth_np(x: np.ndarray, w: int) -> np.ndarray:
    """Standard Wilder smoothing on a numpy array.

    Seeds at index ``w-1`` with the simple mean of the first ``w`` values;
    subsequent values follow the Wilder recursion ``y[i] = ((w-1)·y[i-1] + x[i]) / w``
    (equivalent to ``alpha = 1/w`` exponential smoothing).

    Returns ``NaN`` for indices ``0..w-2`` and when ``len(x) < w``.
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n < w:
        return out
    out[w - 1] = float(np.mean(x[:w]))
    for i in range(w, n):
        out[i] = ((w - 1) * out[i - 1] + x[i]) / w
    return out


def wilder_smooth(x: pl.Expr, w: int) -> pl.Expr:
    """Wilder's smoothing (alpha=1/w), seeded with the SMA of the first w values.

    Output:
      - rows ``0..w-2``: null (warmup; seed not yet computed)
      - row ``w-1``: ``mean(x[0..w-1])`` (the seed)
      - row ``i >= w``: ``((w-1)·prev + x[i]) / w``

    No polars-native equivalent. Implemented via ``map_batches`` over a
    numpy kernel; fast enough at our scale (~1.5M rows × few windows).

    Pandas/numpy equivalent::

        out = np.full(n, np.nan)
        out[w-1] = np.mean(x[:w])
        for i in range(w, n):
            out[i] = ((w-1) * out[i-1] + x[i]) / w

    **RSI offset note**: the legacy ``_wilder_rsi`` (utils.py:1823-1844)
    seeds at index ``w`` using ``mean(x[1..w])`` instead of standard ``w-1``
    using ``mean(x[0..w-1])``. This is intentional — it skips ``r[0]`` which
    is NaN from ``log_return``. To match legacy RSI behavior with this
    primitive, apply ``wilder_smooth(x.shift(-1), w).shift(1)`` so the
    primitive's seed lands on ``mean(x[1..w])`` and is shifted back to
    index ``w``. The RSI feature class will encapsulate this.
    """
    return x.map_batches(
        lambda s: pl.Series(_wilder_smooth_np(s.to_numpy(), w)),
        return_dtype=pl.Float64,
    )


# ---------------------------------------------------------------------------
# Step 6: specialized primitives
# ---------------------------------------------------------------------------


def rolling_quantile(
    x: pl.Expr,
    w: int,
    q: float,
    *,
    min_samples: int | None = None,
) -> pl.Expr:
    """Rolling quantile with **linear** interpolation.

    Polars' default is ``interpolation='nearest'`` — a silent divergence
    from pandas / numpy default ``'linear'``. This primitive enforces
    ``'linear'`` so callers cannot accidentally produce different
    boundary values on even-length windows.

    Pandas/numpy equivalent (utils.py:1671-1673)::

        np.quantile(window, q, method='linear')
    """
    return x.rolling_quantile(
        quantile=q,
        interpolation="linear",
        window_size=w,
        min_samples=min_samples or w,
    )


def _rolling_mad_np(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling MAD over a numpy array: ``median(|x - median_w(x)|)``.

    Two-pass median per window — inner median over the window, outer
    median over the absolute deviations from that inner median. Computed
    in chunks to bound peak memory.

    Mirrors the math at utils.py:1675-1677, applied at every eligible
    row instead of only boundary indices.
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n < w:
        return out

    chunk_size = 2000 if w >= 720 else 5000
    valid_starts = np.arange(w - 1, n, dtype=np.int64)
    offsets = np.arange(w - 1, -1, -1, dtype=np.int64)

    for start in range(0, len(valid_starts), chunk_size):
        idx = valid_starts[start : start + chunk_size]
        rows = idx[:, None] - offsets[None, :]
        window_vals = x[rows]
        invalid = np.isnan(window_vals).any(axis=1)
        medians = np.quantile(window_vals, 0.5, axis=1, method="linear")
        abs_devs = np.abs(window_vals - medians[:, None])
        mads = np.quantile(abs_devs, 0.5, axis=1, method="linear")
        mads[invalid] = np.nan
        out[idx] = mads

    return out


def rolling_mad(x: pl.Expr, w: int) -> pl.Expr:
    """Rolling Median Absolute Deviation: ``median(|x - median_w(x)|)``.

    Two-pass median per window. NOT equivalent to
    ``rolling_median(|x - rolling_median(x)|)`` — the inner median is the
    SAME value for every position in the outer window.

    Pandas equivalent (utils.py:1675-1677)::

        med = window.quantile(0.5, method='linear')
        np.quantile(np.abs(window - med), 0.5, method='linear')

    The legacy is boundary-sparse (only at every M-th row); this primitive
    computes at every eligible row. Apply boundary masking at the feature
    layer for parity.
    """
    return x.map_batches(
        lambda s: pl.Series(_rolling_mad_np(s.to_numpy(), w)),
        return_dtype=pl.Float64,
    )


def _perm_entropy_m3_np(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling normalized permutation entropy at every row, m=3, tau=1.

    Replicates the cumulative-sum approach in
    compute_permutation_entropy (utils.py:2265-2299). Returns NaN where
    the window contains any non-finite value or the window is too small
    (``n_patterns < 5 * factorial(m) = 30``).
    """
    # Local import: utils owns the canonical _perm_codes_m3_tau1 with its
    # stable sorting-network tie-break. Will move to this module in step 15.
    from src.utils import _perm_codes_m3_tau1

    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    n_patterns = w - 2
    if n_patterns < 5 * math.factorial(3):
        return out

    codes = _perm_codes_m3_tau1(x)
    n_patterns_total = len(codes)
    if n_patterns_total < n_patterns:
        return out

    end_idx = np.arange(n_patterns - 1, n_patterns_total, dtype=np.int64)
    start_idx = end_idx - (n_patterns - 1)

    invalid = (codes == -1).astype(np.int32)
    invalid_cum = np.cumsum(invalid, dtype=np.int64)
    invalid_in_window = invalid_cum[end_idx] - np.where(
        start_idx > 0, invalid_cum[start_idx - 1], 0
    )
    valid_window = invalid_in_window == 0

    probs = []
    for code in range(6):
        ind = (codes == code).astype(np.int32)
        cum = np.cumsum(ind, dtype=np.int64)
        cnt = cum[end_idx] - np.where(start_idx > 0, cum[start_idx - 1], 0)
        probs.append(cnt.astype(float) / float(n_patterns))
    P = np.vstack(probs).T

    max_entropy = math.log(math.factorial(3))
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    H_norm = H / max_entropy
    H_norm[~valid_window] = np.nan

    bar_positions = end_idx + 2  # convert pattern index to bar index
    out[bar_positions] = H_norm
    return out


def perm_entropy_m3(x: pl.Expr, w: int) -> pl.Expr:
    """Rolling normalized permutation entropy, m=3, tau=1, in [0, 1].

    Output is null where the window contains any non-finite value or the
    window is too small for a stable estimate.

    Implemented via numpy ``map_batches`` because polars has no native
    permutation entropy. Pattern coding uses utils.``_perm_codes_m3_tau1``
    for guaranteed legacy parity.
    """
    return x.map_batches(
        lambda s: pl.Series(_perm_entropy_m3_np(s.to_numpy(), w)),
        return_dtype=pl.Float64,
    )


def _signed_run_np(
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed-streak state machine over a numpy array.

    Returns three arrays:
      - run_dir   (int8):  -1, 0, or +1 (0 means no active run)
      - run_len   (int32): bars in the current run (0 if no run)
      - run_cum   (float): cumulative sum within the current run

    Reset on non-finite or exactly-zero. Extend on same-sign. Flip on
    opposite-sign (length=1, cumret=ri).

    Mirrors utils.py:1975-1992. Pure-Python loop for clarity; ~4M iter/s
    on a typical machine, so ~1s on 1.5M rows × 3 calls.
    """
    n = len(r)
    run_dir = np.zeros(n, dtype=np.int8)
    run_len = np.zeros(n, dtype=np.int32)
    run_cum = np.zeros(n, dtype=float)
    direction = 0
    length = 0
    cumret = 0.0
    for i, ri in enumerate(r):
        if not np.isfinite(ri) or ri == 0.0:
            direction = 0
            length = 0
            cumret = 0.0
        else:
            sign = 1 if ri > 0 else -1
            if sign == direction:
                length += 1
                cumret += float(ri)
            else:
                direction = sign
                length = 1
                cumret = float(ri)
        run_dir[i] = direction
        run_len[i] = length
        run_cum[i] = cumret
    return run_dir, run_len, run_cum


def signed_run_dir(x: pl.Expr) -> pl.Expr:
    """Direction of the current signed run: -1, 0, or +1.

    See :func:`_signed_run_np` for state-machine semantics. Resets on
    NaN/inf/zero; same-sign extends; opposite-sign flips.

    Pandas equivalent: utils.compute_event_features ``run_dir`` channel.
    """
    return x.map_batches(
        lambda s: pl.Series(_signed_run_np(s.to_numpy())[0]),
        return_dtype=pl.Int8,
    )


def signed_run_length(x: pl.Expr) -> pl.Expr:
    """Length of the current signed run (0 when no active run)."""
    return x.map_batches(
        lambda s: pl.Series(_signed_run_np(s.to_numpy())[1]),
        return_dtype=pl.Int32,
    )


def signed_run_cumret(x: pl.Expr) -> pl.Expr:
    """Cumulative sum of values within the current signed run."""
    return x.map_batches(
        lambda s: pl.Series(_signed_run_np(s.to_numpy())[2]),
        return_dtype=pl.Float64,
    )


# ---------------------------------------------------------------------------
# Past-only linear OLS (equilibrium family helper)
#
# Used by ``eq__mu_trend`` and ``eq__trend_sresid``. Differs from
# ``_quad_trend_np`` in two ways:
#   1. Linear (degree 1) instead of quadratic.
#   2. Strictly past-only window: at row n the fit uses p[n-W..n-1], with
#      design coordinates x_i = i - n so x ranges from -W to -1. The
#      reported intercept is evaluated at x = 0 — i.e. extrapolated ONE
#      step beyond the past data, to the current-bar position. The
#      residual std is the population (ddof=0) standard deviation of the
#      fit residuals over the same past-only window.
# ---------------------------------------------------------------------------


def _past_only_linear_trend_np(
    p: np.ndarray, w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Past-only linear-OLS fit at each row.

    For each row ``n`` with ``n >= w``, fits ``p_i = a + b·x_i + e_i`` over
    the past-only window ``{n-w, ..., n-1}`` with coordinates
    ``x_i = i - n``. Returns:

      - ``mu_trend``: the intercept evaluated at ``x = 0`` (the current-bar
        extrapolation of the trend line).
      - ``s_resid``: ``sqrt(mean(e_i²))`` over the past-only window
        (population, ``ddof=0``).

    Rows ``0..w-1`` return ``NaN`` (warmup — the past window is incomplete).
    Windows containing any ``NaN`` return ``NaN`` for both outputs.

    Implementation note: residual variance is clamped at zero before sqrt
    to guard against tiny-negative cancellation residuals on perfect-line
    windows (mirrors the ``clip_pos`` pattern in ``VolGk`` / ``VolRs``).
    """
    n = len(p)
    mu_trend = np.full(n, np.nan, dtype=float)
    s_resid = np.full(n, np.nan, dtype=float)
    w_int = int(w)
    if n <= w_int:
        return mu_trend, s_resid

    # Fixed design moments (depend only on w).
    x = np.arange(-w_int, 0, dtype=float)
    x_mean = float(x.mean())  # = -(w+1)/2
    x_centered = x - x_mean
    sum_u2 = float((x_centered ** 2).sum())
    if sum_u2 <= 0.0:
        return mu_trend, s_resid

    # Eligible rows: first valid output is at row w (window covers rows
    # 0..w-1, strictly past). Process in chunks to bound peak memory.
    eligible = np.arange(w_int, n, dtype=np.int64)
    # offsets[j]: rows back from n. Window indices = n - offsets, so
    # column 0 is the oldest (n-w) and column w-1 is the newest (n-1).
    offsets = np.arange(w_int, 0, -1, dtype=np.int64)
    max_elements = 5_000_000
    chunk_size = int(min(20_000, max(1, max_elements // w_int)))

    for start in range(0, len(eligible), chunk_size):
        idx = eligible[start : start + chunk_size]
        rows = idx[:, None] - offsets[None, :]
        window_vals = p[rows]
        invalid = np.isnan(window_vals).any(axis=1)

        p_mean = window_vals.mean(axis=1)
        Sup = ((window_vals - p_mean[:, None]) * x_centered[None, :]).sum(axis=1)
        beta = Sup / sum_u2  # slope
        alpha = p_mean - beta * x_mean  # intercept at x = 0

        # Residuals against the past-only window.
        fitted = alpha[:, None] + beta[:, None] * x[None, :]
        resid = window_vals - fitted
        resid_var = (resid ** 2).mean(axis=1)
        resid_var = np.maximum(0.0, resid_var)
        sres = np.sqrt(resid_var)

        alpha[invalid] = np.nan
        sres[invalid] = np.nan
        mu_trend[idx] = alpha
        s_resid[idx] = sres

    return mu_trend, s_resid


def past_only_linear_trend_mu(x: pl.Expr, w: int) -> pl.Expr:
    """Past-only OLS fair value: the intercept at ``x=0`` of a linear fit
    over the trailing ``w`` rows ``{n-w, ..., n-1}``.

    See :func:`_past_only_linear_trend_np` for math details.
    """
    return x.map_batches(
        lambda s, ww=w: pl.Series(_past_only_linear_trend_np(s.to_numpy(), ww)[0]),
        return_dtype=pl.Float64,
    )


def past_only_linear_trend_sresid(x: pl.Expr, w: int) -> pl.Expr:
    """Past-only OLS residual std (population) over the trailing ``w`` rows."""
    return x.map_batches(
        lambda s, ww=w: pl.Series(_past_only_linear_trend_np(s.to_numpy(), ww)[1]),
        return_dtype=pl.Float64,
    )
