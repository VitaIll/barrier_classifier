"""Barrier labels: the label definition as a domain object + vectorized kernel.

The label (docs/MINIMAL_PROJECT_SPEC_v2 §6, re-homed here):

    y_k = 1[ max_{j=1..M} log(upper[n_k + j] / close[n_k]) >= phi ]

with ``m_k`` the max forward log-excursion, ``tau_k`` the first bar index
(1-based) at which the barrier is crossed, and — optionally — downside
diagnostics ``m_dn``/``tau_dn`` from the lows (triple-barrier family).

Three layers:

- :class:`BarrierSpec` — the *definition* as an immutable value object.
  Everything that must agree on the label (weights, purged splits, serving
  contracts) derives from this one object instead of loose ``M``/``phi``
  floats.
- :func:`barrier_label_arrays` — the vectorized numpy kernel. Bit-exact
  with the legacy per-row loop (same operation order: divide, log, max) and
  ~70x faster; processes in bounded-memory row chunks.
- :class:`BarrierLabeler` — the base-cadence (stride=1) block for the
  kernel pipeline: bar frame in, label columns out.

NaN/degenerate-price semantics (identical to the legacy loop, pinned by
parity tests):

- A window that ends past the data (``n_k + M >= n``) leaves ``y`` null —
  the label is *unknowable*, not negative.
- A non-finite forward max (NaN/inf from NaN or non-positive prices
  anywhere in the window, or a non-positive base) yields ``y = 0`` with
  ``m_k``/``tau_k`` null — "no observed hit", flagged by nullness.
- Downside diagnostics are computed only where the upside max was finite,
  and ``tau_dn`` only where the downside min was also finite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import polars as pl

from src.core.block import Block
from src.core.contracts import ColumnSpec, FrameSchema, require_columns
from src.core.errors import ConfigError, ContractError
from src.core.log import get_logger, timed

_log = get_logger("labels")

BarrierSource = Literal["close", "high"]

#: Bound on the temporary ``(rows, horizon)`` float blocks the kernel
#: materializes at once (~64 MB per block at float64).
_CHUNK_CELLS = 8_000_000


# ---------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarrierSpec:
    """Immutable definition of the barrier label.

    Parameters
    ----------
    horizon : int
        Forward window length in bars (the project's ``M``).
    upper : float
        Upper barrier in log-return space (the project's ``phi``).
    source : {"high", "close"}
        Which future series is tested against the barrier. ``"high"``
        matches a TP-limit fill on the intrabar high (production);
        ``"close"`` is the close-confirmed legacy definition.
    stride : int
        Bars between consecutive decision rows. 1 = a label at every bar
        (canonical 1-min cadence); ``horizon`` = non-overlapping legacy
        boundary cadence.
    downside : bool
        Also compute ``m_dn``/``tau_dn`` from the lows (diagnostics for the
        triple-barrier family; never model features).
    """

    horizon: int
    upper: float
    source: BarrierSource = "high"
    stride: int = 1
    downside: bool = False

    def __post_init__(self) -> None:
        if not (isinstance(self.horizon, int) and self.horizon > 0):
            raise ConfigError(f"BarrierSpec.horizon must be a positive int; got {self.horizon!r}")
        if not (isinstance(self.stride, int) and self.stride > 0):
            raise ConfigError(f"BarrierSpec.stride must be a positive int; got {self.stride!r}")
        if not (math.isfinite(self.upper) and self.upper > 0):
            raise ConfigError(f"BarrierSpec.upper must be finite and > 0; got {self.upper!r}")
        if self.source not in ("close", "high"):
            raise ConfigError(f"BarrierSpec.source must be 'close' or 'high'; got {self.source!r}")

    # -- derived quantities that must never be recomputed ad hoc -------------

    @property
    def maturity_shift(self) -> int:
        """Rows to shift so a referenced label's horizon has fully closed.

        At decision row ``t`` the label looks at bars
        ``[t*stride + 1, t*stride + horizon]``; it is *mature* at row
        ``t + horizon // stride``. Features reading past labels must shift
        by at least this much or they leak open horizons
        (López de Prado's overlapping-label problem).
        """
        return max(1, self.horizon // self.stride)

    def label_intervals(self, n_rows: int) -> np.ndarray:
        """``(n_rows, 2)`` array of ``[start, end)`` future-bar intervals.

        Row ``i`` uses raw bars ``[i*stride + 1, i*stride + horizon]``.
        Feed to ``src.analytics.sampling`` for concurrency / uniqueness
        weights and purged splits — same convention, one source of truth.
        """
        starts = np.arange(n_rows, dtype=np.int64) * self.stride + 1
        return np.column_stack([starts, starts + self.horizon])

    @property
    def source_column(self) -> str:
        return "high" if self.source == "high" else "close"

    # -- behaviors owned by the definition ------------------------------------

    def label(self, bars: pl.DataFrame, *, n_out: Optional[int] = None) -> pl.DataFrame:
        """Label a bar frame under THIS definition.

        The spec owns labeling the way a matrix owns its rank — callers
        ask the definition rather than threading (M, phi, source, stride)
        through a free function. Returns the decision-row label frame
        (see :func:`label_frame`).
        """
        return label_frame(bars, self, n_out=n_out)

    def uniqueness_weights(self, n_rows: int, *, normalize: bool = True) -> np.ndarray:
        """López-de-Prado uniqueness weights for THIS label definition.

        Adjacent overlapping labels (stride < horizon) describe the same
        economic event; these weights down-weight the redundancy. Derived
        from the same horizon/stride as :meth:`label_intervals`, so the
        weighting can never drift from the labels it corrects for.
        """
        from src.analytics.sampling import compute_uniqueness_weights

        return compute_uniqueness_weights(
            n_rows, self.horizon, bar_stride=self.stride, normalize=normalize
        )


# Label columns as a contract. ``y`` is null where the window is unknowable;
# ``m_k``/``tau_k`` are additionally null where the forward window was
# poisoned (see module docstring) — hence nullable, non-finite forbidden.
LABEL_SCHEMA = FrameSchema.of(
    "barrier_labels",
    ColumnSpec("y", pl.Float64, nullable=True, finite=True, bounds=(0.0, 1.0)),
    ColumnSpec("m_k", pl.Float64, nullable=True, finite=True),
    ColumnSpec("tau_k", pl.Float64, nullable=True, finite=True, bounds=(1.0, None)),
    ColumnSpec("phi", pl.Float64, nullable=False, finite=True, bounds=(0.0, None)),
)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelArrays:
    """Kernel output bundle (all ``float64``, NaN = null)."""

    y: np.ndarray
    m_k: np.ndarray
    tau_k: np.ndarray
    m_dn: Optional[np.ndarray] = None
    tau_dn: Optional[np.ndarray] = None

    @property
    def n_labeled(self) -> int:
        return int(np.isfinite(self.y).sum())

    @property
    def n_positive(self) -> int:
        return int(np.nansum(self.y))


def barrier_label_arrays(
    close: np.ndarray,
    upper: np.ndarray,
    *,
    horizon: int,
    phi: float,
    stride: int = 1,
    n_out: Optional[int] = None,
    low: Optional[np.ndarray] = None,
) -> LabelArrays:
    """Vectorized barrier labels over aligned price arrays.

    ``close`` is the entry-reference series; ``upper`` the barrier-crossing
    series (highs or closes); both length ``n`` and index-aligned.
    Decision row ``k`` references bar ``n_k = k * stride`` and looks at
    ``upper[n_k+1 : n_k+horizon+1]``. Rows whose window extends past the
    data stay NaN across all outputs.

    ``n_out`` sets the output length (defaults to ``ceil(n / stride)``,
    the number of decision rows the data supports). Passing ``low``
    enables the downside diagnostics.

    Bit-exact with the legacy per-row loop: same operation order
    (divide then log then max), same NaN propagation, same tie-breaking
    (first crossing wins). Memory-bounded: forward windows are
    materialized in row chunks of ≲64 MB.
    """
    if horizon <= 0:
        raise ConfigError(f"horizon must be > 0, got {horizon}")
    if stride <= 0:
        raise ConfigError(f"stride must be > 0, got {stride}")
    if not (math.isfinite(phi) and phi > 0):
        raise ConfigError(f"phi must be finite and > 0, got {phi!r}")

    close_a = np.asarray(close, dtype=np.float64)
    upper_a = np.asarray(upper, dtype=np.float64)
    if close_a.ndim != 1 or upper_a.ndim != 1:
        raise ContractError("close/upper must be 1-D arrays")
    n = close_a.shape[0]
    if upper_a.shape[0] != n:
        raise ContractError(
            f"close and upper must be aligned: len(close)={n}, "
            f"len(upper)={upper_a.shape[0]}"
        )
    low_a: Optional[np.ndarray] = None
    if low is not None:
        low_a = np.asarray(low, dtype=np.float64)
        if low_a.shape != (n,):
            raise ContractError(
                f"low must be aligned with close: len(low)={low_a.shape[0]}, n={n}"
            )

    K = int(n_out) if n_out is not None else -(-n // stride)  # ceil
    if K < 0:
        raise ConfigError(f"n_out must be >= 0, got {n_out}")

    y = np.full(K, np.nan)
    m_k = np.full(K, np.nan)
    tau_k = np.full(K, np.nan)
    m_dn = np.full(K, np.nan) if low_a is not None else None
    tau_dn = np.full(K, np.nan) if low_a is not None else None

    # Decision rows whose forward window is fully inside the data:
    # n_k + horizon < n  (strict, matching the legacy guard).
    ks = np.arange(K, dtype=np.int64)
    valid = ks[ks * stride + horizon < n]
    if valid.size == 0:
        return LabelArrays(y=y, m_k=m_k, tau_k=tau_k, m_dn=m_dn, tau_dn=tau_dn)

    # Forward windows of the barrier series: view row i = upper[i+1 : i+1+horizon].
    sw_up = np.lib.stride_tricks.sliding_window_view(upper_a[1:], horizon)
    sw_dn = (
        np.lib.stride_tricks.sliding_window_view(low_a[1:], horizon)
        if low_a is not None
        else None
    )

    chunk_rows = max(1, _CHUNK_CELLS // horizon)
    for start in range(0, valid.size, chunk_rows):
        ck = valid[start : start + chunk_rows]
        rows = ck * stride
        base = close_a[rows]

        # Same operation order as the legacy loop: divide, then log, then
        # max — keeps results bit-identical. errstate silenced because
        # non-finite outcomes are handled explicitly below.
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = sw_up[rows] / base[:, None]
            np.log(ret, out=ret)
        m = ret.max(axis=1)  # np.max semantics: any NaN in the window -> NaN

        finite = np.isfinite(m)
        y[ck] = np.where(finite, (m >= phi).astype(float), 0.0)
        m_k[ck] = np.where(finite, m, np.nan)

        hit = finite & (m >= phi)
        if hit.any():
            # First crossing, 1-based — argmax returns the first True.
            tau_k[ck[hit]] = (ret[hit] >= phi).argmax(axis=1) + 1.0

        if sw_dn is not None:
            # Legacy semantics: the downside block runs only where the
            # upside max was finite (the loop `continue`s past it).
            rows_f = rows[finite]
            ck_f = ck[finite]
            if rows_f.size:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ret_dn = sw_dn[rows_f] / close_a[rows_f][:, None]
                    np.log(ret_dn, out=ret_dn)
                m_dn_val = -ret_dn.min(axis=1)
                dn_finite = np.isfinite(m_dn_val)
                m_dn[ck_f[dn_finite]] = m_dn_val[dn_finite]  # type: ignore[index]
                dn_hit = (ret_dn <= -phi).any(axis=1) & dn_finite
                if dn_hit.any():
                    tau_dn[ck_f[dn_hit]] = (  # type: ignore[index]
                        (ret_dn[dn_hit] <= -phi).argmax(axis=1) + 1.0
                    )

    return LabelArrays(y=y, m_k=m_k, tau_k=tau_k, m_dn=m_dn, tau_dn=tau_dn)


def label_frame(
    bars: pl.DataFrame,
    spec: BarrierSpec,
    *,
    n_out: Optional[int] = None,
) -> pl.DataFrame:
    """Labels for a bar frame as a standalone decision-row frame.

    Returns ``ceil(len(bars) / spec.stride)`` rows (or ``n_out``) with
    columns ``y, m_k, tau_k, phi`` (+ ``m_dn, tau_dn`` when
    ``spec.downside``), NaN coerced to null. Row ``k`` refers to bar
    ``k * spec.stride``.
    """
    require_columns(bars, ("close", spec.source_column), context="label_frame")
    cols = {"close": bars["close"].to_numpy().astype(np.float64)}
    upper = (
        cols["close"]
        if spec.source == "close"
        else bars["high"].to_numpy().astype(np.float64)
    )
    low = None
    if spec.downside:
        require_columns(bars, ("low",), context="label_frame(downside)")
        low = bars["low"].to_numpy().astype(np.float64)

    with timed(_log, f"barrier labels over {bars.height:,} bars") as info:
        arrays = barrier_label_arrays(
            cols["close"],
            upper,
            horizon=spec.horizon,
            phi=spec.upper,
            stride=spec.stride,
            n_out=n_out,
            low=low,
        )
        info["labeled"] = f"{arrays.n_labeled:,}"
        info["positives"] = f"{arrays.n_positive:,}"
        info["spec"] = (
            f"M={spec.horizon},phi={spec.upper:g},src={spec.source},"
            f"stride={spec.stride}"
        )

    out = {
        "y": pl.Series("y", arrays.y).fill_nan(None),
        "m_k": pl.Series("m_k", arrays.m_k).fill_nan(None),
        "tau_k": pl.Series("tau_k", arrays.tau_k).fill_nan(None),
        "phi": pl.Series("phi", np.full(len(arrays.y), float(spec.upper))),
    }
    if spec.downside:
        assert arrays.m_dn is not None and arrays.tau_dn is not None
        out["m_dn"] = pl.Series("m_dn", arrays.m_dn).fill_nan(None)
        out["tau_dn"] = pl.Series("tau_dn", arrays.tau_dn).fill_nan(None)
    return pl.DataFrame(out)


# ---------------------------------------------------------------------------
# The block (base cadence: one decision row per bar)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarrierLabeler(Block):
    """Attach barrier-label columns to a base-cadence bar frame.

    Requires ``spec.stride == 1`` — at base cadence the decision rows ARE
    the bar rows, so labels attach 1:1. Sparse cadences (stride > 1) keep
    the two-frame orchestration in ``src/features`` until decision-row
    sampling becomes its own block (Phase 3).
    """

    spec: BarrierSpec

    log_component = "labels"

    def __post_init__(self) -> None:
        if self.spec.stride != 1:
            raise ConfigError(
                "BarrierLabeler requires spec.stride == 1 (base cadence); "
                f"got stride={self.spec.stride}. Use label_frame / "
                "barrier_label_arrays for sparse decision rows."
            )
        # requires depends on the spec — set per-instance (frozen dataclass:
        # use object.__setattr__, the sanctioned idiom for derived fields).
        needed = ["close"]
        if self.spec.source == "high":
            needed.append("high")
        if self.spec.downside:
            needed.append("low")
        object.__setattr__(self, "requires", tuple(dict.fromkeys(needed)))
        provides = ["y", "m_k", "tau_k", "phi"]
        if self.spec.downside:
            provides += ["m_dn", "tau_dn"]
        object.__setattr__(self, "provides", tuple(provides))

    def _apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        labels = label_frame(frame, self.spec, n_out=frame.height)
        return pl.concat([frame, labels], how="horizontal")
