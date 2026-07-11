"""Feature services: one research pipeline, two serving shapes.

- :class:`BatchFeatureService` — precompute the whole replay window in a
  single pipeline pass (bit-identical to the research dataset build by
  construction) and serve rows by timestamp. Fast path for replays and
  the parity harness.
- :class:`RollingFeatureService` — recompute on the trailing
  :class:`~src.engine.buffer.BarBuffer` each bar and take the last row.
  True-live path.

Both produce :class:`FeatureVector`s reconciled against the active model's
:class:`FeatureContract` — the ordered feature list *is* the interface,
and a row that cannot be reconciled never reaches the model.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

from src.engine.buffer import BarBuffer
from src.engine.domain import FeatureVector
from src.engine.errors import FeatureContractError
from src.features.pipeline import run_inference_pipeline


@dataclass(frozen=True)
class FeatureContract:
    """Everything the serving path must replicate from the training build.

    Travels with each model version (``contract.json``). The grid anchor
    is the *first raw bar* of the training frame — boundary-sparse kernels
    key off absolute row phase, so serving windows must align to it
    (see docs/ENGINE.md §4).
    """

    feature_list: tuple[str, ...]
    label_cadence: str = "1min"
    barrier_source: str = "high"
    with_derivatives: bool = False
    m: int = 20
    phi: float = 0.0025
    n_warmup: int = 20_159
    grid_anchor_ts: str = ""  # ISO-8601 UTC of training-frame bar 0
    p_hit_prior: float = 0.5
    cap_h_blocks: Optional[int] = None
    enable_autocorrelation: Optional[bool] = None
    autocorr_windows: Optional[tuple[int, ...]] = None
    autocorr_lags: tuple[int, ...] = (1, 2, 5, 10)

    def __post_init__(self) -> None:
        if not self.feature_list:
            raise FeatureContractError("FeatureContract requires a non-empty feature_list")
        if len(set(self.feature_list)) != len(self.feature_list):
            raise FeatureContractError("FeatureContract feature_list has duplicate names")
        if self.label_cadence not in ("boundary", "1min"):
            raise FeatureContractError(f"bad label_cadence {self.label_cadence!r}")
        if self.barrier_source not in ("high", "close"):
            raise FeatureContractError(f"bad barrier_source {self.barrier_source!r}")
        if not self.grid_anchor_ts:
            raise FeatureContractError("FeatureContract requires grid_anchor_ts")
        if not (isinstance(self.m, int) and self.m > 0):
            raise FeatureContractError(f"FeatureContract m must be a positive int, got {self.m!r}")
        if not (math.isfinite(self.phi) and self.phi > 0):
            raise FeatureContractError(f"FeatureContract phi must be finite and > 0, got {self.phi!r}")
        if not (isinstance(self.n_warmup, int) and self.n_warmup >= 0):
            raise FeatureContractError(
                f"FeatureContract n_warmup must be a non-negative int, got {self.n_warmup!r}"
            )

    @property
    def anchor(self) -> pd.Timestamp:
        ts = pd.Timestamp(self.grid_anchor_ts)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    @property
    def n_features(self) -> int:
        return len(self.feature_list)

    def pipeline_kwargs(self) -> dict:
        """Kwargs for ``run_inference_pipeline`` that reproduce training."""
        return dict(
            with_derivatives=self.with_derivatives,
            p_hit_prior=self.p_hit_prior,
            cap_h_blocks=self.cap_h_blocks,
            label_cadence=self.label_cadence,
            enable_autocorrelation=self.enable_autocorrelation,
            autocorr_windows=self.autocorr_windows,
            autocorr_lags=tuple(self.autocorr_lags),
            barrier_source=self.barrier_source,
        )

    # -- serialization ---------------------------------------------------- #

    def to_json(self, path: str | Path) -> None:
        payload = asdict(self)
        payload["feature_list"] = list(self.feature_list)
        payload["autocorr_lags"] = list(self.autocorr_lags)
        if self.autocorr_windows is not None:
            payload["autocorr_windows"] = list(self.autocorr_windows)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureContract":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["feature_list"] = tuple(payload["feature_list"])
        payload["autocorr_lags"] = tuple(payload.get("autocorr_lags", (1, 2, 5, 10)))
        if payload.get("autocorr_windows") is not None:
            payload["autocorr_windows"] = tuple(payload["autocorr_windows"])
        return cls(**payload)


def reconcile_row(
    frame: pl.DataFrame, contract: FeatureContract, *, row: int = -1
) -> np.ndarray:
    """Select the contract's features from a pipeline output row.

    Hard errors (never repairable at runtime): a contract feature missing
    from the frame, or a non-finite value *after* the pipeline's impute
    stage (which asserts finiteness — so this catches contract/pipeline
    version drift, not data problems).
    """
    missing = [c for c in contract.feature_list if c not in frame.columns]
    if missing:
        raise FeatureContractError(
            f"pipeline output is missing {len(missing)} contract feature(s), "
            f"e.g. {missing[:5]} — the model was trained against a different "
            "pipeline version"
        )
    vec = (
        frame.select(contract.feature_list)
        .row(row)
    )
    arr = np.asarray(vec, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        bad = [contract.feature_list[i] for i in np.where(~np.isfinite(arr))[0][:5]]
        raise FeatureContractError(
            f"non-finite feature value(s) after imputation at row {row}: {bad}"
        )
    return arr


def reconcile_matrix(frame: pl.DataFrame, contract: FeatureContract) -> np.ndarray:
    """Vectorized :func:`reconcile_row` over the full frame."""
    missing = [c for c in contract.feature_list if c not in frame.columns]
    if missing:
        raise FeatureContractError(
            f"pipeline output is missing {len(missing)} contract feature(s), "
            f"e.g. {missing[:5]}"
        )
    mat = frame.select(contract.feature_list).to_numpy().astype(np.float64, copy=False)
    if not np.all(np.isfinite(mat)):
        rows, cols = np.where(~np.isfinite(mat))
        raise FeatureContractError(
            f"non-finite feature values in batch frame: {len(rows)} cell(s), "
            f"first at row {rows[0]}, feature {contract.feature_list[cols[0]]!r}"
        )
    return mat


# Largest boundary-stage lookback at 1-min cadence (hit-rate/autocorr windows
# 2,880 + label-maturity shift M; see run_inference_pipeline). The rolling
# tail slice must comfortably exceed it or boundary features would be
# truncated relative to the batch/training path.
MAX_BOUNDARY_LOOKBACK_ROWS = 2_920


class RollingFeatureService:
    """Recompute the pipeline on the trailing buffer; serve the last row.

    ``boundary_tail_rows`` bounds the boundary-stage work. Every boundary
    lookback is ≤ ``MAX_BOUNDARY_LOOKBACK_ROWS`` at 1-min cadence, so the
    tail must exceed that or rolling features would silently diverge from
    batch; the constructor enforces the floor. ``None`` runs the full window.
    """

    def __init__(
        self,
        contract: FeatureContract,
        *,
        boundary_tail_rows: Optional[int] = 6_000,
    ) -> None:
        self.contract = contract
        if boundary_tail_rows is not None:
            tail = int(boundary_tail_rows)
            if tail <= MAX_BOUNDARY_LOOKBACK_ROWS:
                raise FeatureContractError(
                    f"boundary_tail_rows={tail} must exceed the max boundary "
                    f"lookback ({MAX_BOUNDARY_LOOKBACK_ROWS}) or rolling features "
                    "would diverge from the batch/training path"
                )
            self._tail_rows: Optional[int] = tail
        else:
            self._tail_rows = None
        self.last_feature_ms: float = float("nan")

    def latest(self, buffer: BarBuffer) -> FeatureVector:
        t0 = time.perf_counter()
        expect = buffer.last_ts
        if expect is None:
            raise FeatureContractError("RollingFeatureService.latest on an empty buffer")
        window = buffer.window_frame()
        try:
            out = run_inference_pipeline(
                window,
                boundary_tail_rows=self._tail_rows,
                **self.contract.pipeline_kwargs(),
            )
        except FeatureContractError:
            raise
        except Exception as exc:
            # A polars/pipeline failure on one window must degrade the bar
            # (typed error the engine catches), not crash the session.
            raise FeatureContractError(
                f"feature pipeline failed on the rolling window ending {expect} "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        if out.height == 0:
            raise FeatureContractError(
                f"feature pipeline returned no rows for the window ending {expect}"
            )
        arr = reconcile_row(out, self.contract, row=-1)
        got = pd.Timestamp(out["ts"][-1]).tz_localize("UTC")
        if got != expect.tz_convert("UTC"):
            raise FeatureContractError(
                f"pipeline last row ts {got} != buffer last ts {expect}"
            )
        self.last_feature_ms = (time.perf_counter() - t0) * 1e3
        return FeatureVector(ts=expect, values=arr)


class BatchFeatureService:
    """One pipeline pass over a full historical frame; O(1) row lookups.

    The frame passed to :meth:`precompute` should start at the contract's
    grid anchor (the training frame's first bar) for bit-parity with the
    research dataset; any suffix alignment on the M-grid is also accepted
    (phase-checked here).
    """

    def __init__(self, contract: FeatureContract) -> None:
        self.contract = contract
        self._matrix: Optional[np.ndarray] = None
        self._ts_index: dict[int, int] = {}
        self._ts: Optional[pd.DatetimeIndex] = None

    @property
    def is_ready(self) -> bool:
        return self._matrix is not None

    def precompute(self, df_raw: pd.DataFrame) -> int:
        """Run the pipeline once over ``df_raw`` (tz-aware 1-min frame).

        Returns the number of feature rows materialized.
        """
        if df_raw.index.tz is None:
            raise FeatureContractError("batch frame must have a tz-aware index")
        anchor = self.contract.anchor
        first = df_raw.index[0].tz_convert("UTC")
        offset_min = round((first - anchor).total_seconds() / 60.0)
        if offset_min % self.contract.m != 0:
            raise FeatureContractError(
                f"batch frame starts {offset_min} minutes after the grid anchor "
                f"({anchor}) — not a multiple of M={self.contract.m}; "
                "boundary-sparse features would phase-shift"
            )
        out = run_inference_pipeline(
            df_raw,
            boundary_tail_rows=None,  # full frame through every stage
            **self.contract.pipeline_kwargs(),
        )
        self._matrix = reconcile_matrix(out, self.contract)
        ts = out["ts"].to_numpy()
        idx = pd.DatetimeIndex(ts).tz_localize("UTC")
        self._ts = idx
        self._ts_index = {int(v): i for i, v in enumerate(idx.asi8)}
        return len(self._ts_index)

    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            raise FeatureContractError("BatchFeatureService.precompute has not run")
        return self._matrix

    def timestamps(self) -> pd.DatetimeIndex:
        if self._ts is None:
            raise FeatureContractError("BatchFeatureService.precompute has not run")
        return self._ts

    def row_index(self, ts: pd.Timestamp) -> int:
        key = int(pd.Timestamp(ts).value)
        try:
            return self._ts_index[key]
        except KeyError:
            raise FeatureContractError(
                f"no precomputed feature row for ts {ts} — batch frame does "
                "not cover the streamed bar"
            ) from None


def matured_label(
    buffer: BarBuffer, *, m: int, phi: float
) -> Optional[tuple[pd.Timestamp, int, float]]:
    """The label that matured at the buffer's newest bar, if any.

    At bar ``t`` the label of ``t − M`` is fully determined (its horizon
    ``[t−M+1, t]`` has closed). Returns ``(entry_ts, y, m_k)`` with
    high-source barrier semantics — the production label definition —
    or ``None`` while the buffer is shorter than ``M+1`` bars.
    """
    if len(buffer) < m + 1:
        return None
    closes = buffer.tail_closes(m + 1)
    highs = buffer.tail_highs(m + 1)
    base = closes[0]
    if not np.isfinite(base) or base <= 0:
        return None
    future_high = highs[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.log(future_high / base)
    m_k = float(np.nanmax(rets)) if np.isfinite(rets).any() else float("nan")
    y = int(np.isfinite(m_k) and m_k >= phi)
    last = buffer.last_ts
    assert last is not None
    entry_ts = last - pd.Timedelta(minutes=m)
    return entry_ts, y, m_k
