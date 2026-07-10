"""Scheduled retraining — the researched training procedure, automated.

A retrain run reproduces ``notebooks/02_build_features.ipynb`` +
``notebooks/03_train_model.ipynb`` end to end on a trailing data window:

1. ``run_pipeline`` (training mode) on the window;
2. asymmetric barrier-distance weights (``compute_training_weights``,
   ``use_dist=True, use_time=False``);
3. ``chronological_split_with_embargo`` (70/15/15, embargo 1,200 rows);
4. per-split López de Prado uniqueness weights (``normalize=False``),
   combined multiplicatively;
5. a single CatBoost with the research hyperparameters
   (``utils.CB_FIXED_PARAMS`` + legacy best params), time-aware Pools,
   ``early_stopping_rounds`` **on** — early firing is logged as a
   data-quality diagnostic, never suppressed and never fatal;
6. champion/challenger gate: both models are scored on the challenger's
   validation split; the challenger must not regress log-loss/PR-AUC
   beyond configured tolerances;
7. walk-forward threshold refresh: ``p_threshold`` re-derived from the
   *new* training slice (top-q, never val/test);
8. atomic publish to the :class:`~src.engine.model.ModelRegistry`.

Scheduling is **event-time** (the stream clock), so replayed history
triggers retrains deterministically at the same bars a live session
would.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from src import utils
from src.analytics.sampling import compute_uniqueness_weights
from src.analytics.thresholds import derive_top_q_threshold
from src.engine.errors import RetrainError
from src.engine.features import FeatureContract
from src.engine.model import ModelHandle, ModelRegistry, Thresholds
from src.features.pipeline import (
    _BASE_COLS,
    _DERIV_BASE_COLS,
    _LABEL_AUX_COLS,
    _RAW_COLS,
    run_pipeline,
)

_TRAIN_P_QUANTILE_LEVELS = (0.05, 0.10, 0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

# The exact production training params (nb03): CB_FIXED_PARAMS backbone +
# legacy best hyperparameters. ``iterations`` stays overridable — demo
# replays retrain with a small budget to exercise the plumbing.
_LEGACY_BEST_PARAMS = {
    "learning_rate": 0.01,
    "l2_leaf_reg": 0.1,
    "depth": 6,
    "rsm": 1.0,
    "subsample": 1.0,
    "mvs_reg": 3.0,
    "diffusion_temperature": 10000,
    "random_strength": 0.0,
}


def research_cb_params(*, iterations: int = 2000, verbose: int | bool = False) -> dict:
    params = {**utils.CB_FIXED_PARAMS, **_LEGACY_BEST_PARAMS,
              "iterations": int(iterations), "verbose": verbose}
    if not params.get("has_time", False):
        raise RetrainError(
            "CB_FIXED_PARAMS lost has_time=True; refusing to train without "
            "time-awareness on overlapping 1-min labels"
        )
    return params


@dataclass(frozen=True)
class RetrainPolicy:
    """When and how to retrain (all row counts are 1-min bars)."""

    enabled: bool = True
    every_bars: int = 7 * 1440              # event-time period between runs
    window_rows: Optional[int] = None        # None = everything the store has
    min_window_rows: int = 30 * 1440         # skip (recorded) if less history
    iterations: int = 2000
    top_q: float = 0.99
    embargo_rows: Optional[int] = None       # None = research default (60*M)
    train_frac: float = 0.70
    val_frac: float = 0.15
    # Champion/challenger tolerances (evaluated on the challenger's val split).
    max_logloss_regression: float = 0.005    # challenger may be at most this much worse
    min_pr_auc_ratio: float = 0.98           # challenger PR-AUC ≥ ratio × incumbent


@dataclass
class RetrainOutcome:
    status: str                  # "published" | "gate_rejected" | "skipped" | "failed"
    new_version: Optional[str] = None
    best_iter: Optional[int] = None
    n_rows: int = 0
    metrics: dict = field(default_factory=dict)
    notes: str = ""


def run_retrain_job(
    raw_frame: pd.DataFrame,
    *,
    contract: FeatureContract,
    incumbent: Optional[ModelHandle],
    registry: ModelRegistry,
    policy: RetrainPolicy,
) -> RetrainOutcome:
    """Train, gate, and (maybe) publish one challenger model.

    ``raw_frame`` is a tz-aware 1-minute spot frame; its head is trimmed
    to the first bar congruent with the contract's grid anchor (mod M) so
    the anchor lineage stays buffer-compatible across versions.
    """
    m = int(contract.m)
    if raw_frame.empty:
        return RetrainOutcome(status="skipped", notes="empty training window")

    anchor = contract.anchor
    first = raw_frame.index[0].tz_convert("UTC")
    off = int(round((first - anchor).total_seconds() / 60.0))
    pad = (-off) % m
    if pad:
        raw_frame = raw_frame.iloc[pad:]
    if len(raw_frame) < policy.min_window_rows:
        return RetrainOutcome(
            status="skipped",
            notes=f"window has {len(raw_frame):,} rows < min {policy.min_window_rows:,}",
        )

    # --- 1. Dataset build (training mode: warmup-trimmed, labeled) --------
    ds = run_pipeline(
        raw_frame,
        with_derivatives=contract.with_derivatives,
        p_hit_prior=contract.p_hit_prior,
        cap_h_blocks=contract.cap_h_blocks,
        label_cadence=contract.label_cadence,
        enable_autocorrelation=contract.enable_autocorrelation,
        autocorr_windows=contract.autocorr_windows,
        autocorr_lags=tuple(contract.autocorr_lags),
        barrier_source=contract.barrier_source,
    )
    if ds.is_empty():
        return RetrainOutcome(status="skipped", notes="pipeline produced no labeled rows")

    # --- 2. Asymmetric barrier-distance weights (nb02 §4) -----------------
    m_k = ds["m_k"].to_numpy()
    phi_arr = ds["phi"].to_numpy()
    phi_const = float(phi_arr[0])
    if not np.allclose(phi_arr, phi_const):
        raise RetrainError("phi is not constant across the dataset")
    w_combined, _, _, _ = utils.compute_training_weights(
        m_k=m_k, phi=phi_const, use_dist=True, use_time=False,
        w_max=utils.WEIGHT_DIST_W_MAX, q_tail=utils.WEIGHT_DIST_Q_TAIL,
        use_pos=False, normalize=False,
    )

    # --- 3. Feature contract check (pipeline drift is fatal) --------------
    non_feature = (
        set(_LABEL_AUX_COLS) | set(_RAW_COLS) | set(_BASE_COLS)
        | set(_DERIV_BASE_COLS) | {"weight"}
    )
    feature_cols = [
        c for c in ds.columns
        if c not in non_feature and not c.startswith("undef__")
    ]
    if tuple(feature_cols) != tuple(contract.feature_list):
        missing = set(contract.feature_list) - set(feature_cols)
        extra = set(feature_cols) - set(contract.feature_list)
        raise RetrainError(
            "retrain feature set diverged from the active contract "
            f"({len(missing)} missing, {len(extra)} extra; e.g. "
            f"missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}). "
            "Retraining must not silently change the feature contract."
        )

    # --- 4. Split + per-split uniqueness weights (nb03) -------------------
    embargo = (
        int(policy.embargo_rows) if policy.embargo_rows is not None
        else int(utils.recommended_embargo_for_cadence("1min", base_embargo=60, M=m))
    )
    # ``weight`` is not a pipeline column — nb02 appends it post-build; here
    # ``w_combined`` (computed above, row-aligned with ``ds``) plays that role.
    pdf = ds.select(["ts", "k", "y"] + feature_cols).to_pandas()
    del ds
    # Split-geometry preflight: with test_frac = 1 − train − val, the test
    # slice must survive the embargo (and so must val). The research
    # embargo (60·M = 1,200 rows) assumes research-scale windows; a small
    # window is a skip, not a crash.
    test_frac = 1.0 - policy.train_frac - policy.val_frac
    min_rows_for_split = int((embargo + 200) / max(test_frac, 1e-9))
    if len(pdf) < min_rows_for_split:
        return RetrainOutcome(
            status="skipped",
            n_rows=len(pdf),
            notes=(
                f"dataset has {len(pdf):,} labeled rows < {min_rows_for_split:,} "
                f"needed for a {policy.train_frac:.0%}/{policy.val_frac:.0%} split "
                f"with embargo={embargo} — widen the window or lower embargo_rows"
            ),
        )
    train_df, val_df, test_df = utils.chronological_split_with_embargo(
        pdf, train_frac=policy.train_frac, val_frac=policy.val_frac, embargo_k=embargo
    )
    if len(train_df) < 1_000 or len(val_df) < 200:
        return RetrainOutcome(
            status="skipped",
            notes=f"splits too small (train={len(train_df)}, val={len(val_df)})",
        )
    w_by_index = pd.Series(w_combined, index=pdf.index)
    u_train = compute_uniqueness_weights(
        n_rows=len(train_df), M=m, bar_stride=1, normalize=False
    )
    u_val = compute_uniqueness_weights(
        n_rows=len(val_df), M=m, bar_stride=1, normalize=False
    )
    train_weights = w_by_index.loc[train_df.index].to_numpy() * u_train
    val_weights = w_by_index.loc[val_df.index].to_numpy() * u_val
    for name, w in (("train_weights", train_weights), ("val_weights", val_weights)):
        if not np.isfinite(w).all() or (w <= 0).any():
            raise RetrainError(f"{name} has non-finite or non-positive values")

    # --- 5. Fit (early stopping ON — diagnostic, never suppressed) --------
    params = research_cb_params(iterations=policy.iterations)
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    train_pool = Pool(
        data=X_train, label=train_df["y"].astype(int).to_numpy(),
        timestamp=train_df["k"].to_numpy(dtype=np.uint32),
        weight=train_weights, feature_names=list(feature_cols),
    )
    val_pool = Pool(
        data=X_val, label=val_df["y"].astype(int).to_numpy(),
        timestamp=val_df["k"].to_numpy(dtype=np.uint32),
        weight=val_weights, feature_names=list(feature_cols),
    )
    model = CatBoostClassifier(**params)
    t0 = time.perf_counter()
    model.fit(train_pool, eval_set=val_pool)
    fit_seconds = time.perf_counter() - t0
    best_iter = int(model.get_best_iteration())
    es_note = ""
    if best_iter < int(0.30 * params["iterations"]):
        es_note = (
            f"early stopping fired suspiciously early (best_iter={best_iter} "
            f"< 30% of {params['iterations']}) — investigate data prep "
            "(labels, weights, split alignment, embargo, feature parity)"
        )

    # --- 6. Champion/challenger gate on the challenger's val split --------
    y_val = val_df["y"].astype(int).to_numpy()
    p_val_new = model.predict_proba(X_val)[:, 1]
    new_metrics = {
        "val_log_loss": float(log_loss(y_val, p_val_new, labels=[0, 1])),
        "val_pr_auc": float(average_precision_score(y_val, p_val_new)),
        "val_roc_auc": float(roc_auc_score(y_val, p_val_new)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "best_iter": best_iter,
        "fit_seconds": fit_seconds,
    }
    gate_passed = True
    gate_notes: list[str] = []
    if incumbent is not None:
        p_val_inc = incumbent.predict_matrix(X_val)
        inc_ll = float(log_loss(y_val, p_val_inc, labels=[0, 1]))
        inc_pr = float(average_precision_score(y_val, p_val_inc))
        new_metrics["incumbent_val_log_loss"] = inc_ll
        new_metrics["incumbent_val_pr_auc"] = inc_pr
        if new_metrics["val_log_loss"] > inc_ll + policy.max_logloss_regression:
            gate_passed = False
            gate_notes.append(
                f"log_loss {new_metrics['val_log_loss']:.5f} regresses vs "
                f"incumbent {inc_ll:.5f} beyond {policy.max_logloss_regression}"
            )
        if inc_pr > 0 and new_metrics["val_pr_auc"] < policy.min_pr_auc_ratio * inc_pr:
            gate_passed = False
            gate_notes.append(
                f"pr_auc {new_metrics['val_pr_auc']:.5f} < "
                f"{policy.min_pr_auc_ratio} × incumbent {inc_pr:.5f}"
            )
    if not gate_passed:
        return RetrainOutcome(
            status="gate_rejected", best_iter=best_iter, n_rows=len(pdf),
            metrics=new_metrics,
            notes="; ".join(gate_notes) + (f" | {es_note}" if es_note else ""),
        )

    # --- 7. Train-frozen threshold refresh --------------------------------
    p_train = model.predict_proba(X_train)[:, 1]
    thresholds = Thresholds(
        p_threshold=float(derive_top_q_threshold(p_train, q=policy.top_q)),
        top_q=policy.top_q,
        train_p_quantiles={
            str(q): float(np.quantile(p_train, q)) for q in _TRAIN_P_QUANTILE_LEVELS
        },
    )

    # --- 8. Publish --------------------------------------------------------
    new_contract = FeatureContract(
        feature_list=tuple(feature_cols),
        label_cadence=contract.label_cadence,
        barrier_source=contract.barrier_source,
        with_derivatives=contract.with_derivatives,
        m=m,
        phi=phi_const,
        n_warmup=contract.n_warmup,
        grid_anchor_ts=raw_frame.index[0].tz_convert("UTC").isoformat(),
        p_hit_prior=contract.p_hit_prior,
        cap_h_blocks=contract.cap_h_blocks,
        enable_autocorrelation=contract.enable_autocorrelation,
        autocorr_windows=contract.autocorr_windows,
        autocorr_lags=tuple(contract.autocorr_lags),
    )
    training_meta = {
        "source": "engine_retrain",
        "window_start": str(raw_frame.index[0]),
        "window_end": str(raw_frame.index[-1]),
        "n_raw_rows": int(len(raw_frame)),
        "n_dataset_rows": int(len(pdf)),
        "embargo_rows": embargo,
        "cb_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
        "weights": "asymmetric_barrier_distance * lopez_de_prado_avg_uniqueness",
        "early_stopping_note": es_note,
    }
    version = registry.publish(
        model=model, contract=new_contract, thresholds=thresholds,
        metrics=new_metrics, training_meta=training_meta, activate=False,
    )
    return RetrainOutcome(
        status="published", new_version=version, best_iter=best_iter,
        n_rows=len(pdf), metrics=new_metrics, notes=es_note,
    )


class Retrainer:
    """Event-time scheduler + worker-thread wrapper around the job.

    ``on_bar(ts)`` decides whether a run is due; ``poll()`` (called by the
    engine each bar) returns a finished :class:`RetrainOutcome` exactly
    once, at which point the engine activates + hot-swaps. ``threaded=False``
    runs synchronously inside ``on_bar`` — deterministic for tests and
    replays.
    """

    def __init__(
        self,
        *,
        policy: RetrainPolicy,
        registry: ModelRegistry,
        frame_provider: Callable[[Optional[int]], pd.DataFrame],
        incumbent_provider: Callable[[], Optional[ModelHandle]],
        threaded: bool = True,
    ) -> None:
        self.policy = policy
        self.registry = registry
        self._frames = frame_provider
        self._incumbent = incumbent_provider
        self._threaded = threaded
        self._first_bar_ts: Optional[pd.Timestamp] = None
        self._last_trigger_ts: Optional[pd.Timestamp] = None
        self._thread: Optional[threading.Thread] = None
        self._pending: Optional[tuple[pd.Timestamp, RetrainOutcome]] = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def due(self, ts: pd.Timestamp) -> bool:
        if not self.policy.enabled:
            return False
        if self.running:
            return False
        base = self._last_trigger_ts or self._first_bar_ts
        if base is None:
            return False
        return (ts - base) >= pd.Timedelta(minutes=self.policy.every_bars)

    def on_bar(self, ts: pd.Timestamp) -> None:
        if self._first_bar_ts is None:
            self._first_bar_ts = ts
        if not self.due(ts):
            return
        self._last_trigger_ts = ts
        if self._threaded:
            self._thread = threading.Thread(
                target=self._run, args=(ts,), name="engine-retrain", daemon=True
            )
            self._thread.start()
        else:
            self._run(ts)

    def _run(self, trigger_ts: pd.Timestamp) -> None:
        try:
            frame = self._frames(self.policy.window_rows)
            contract_owner = self._incumbent()
            if contract_owner is None:
                outcome = RetrainOutcome(status="skipped", notes="no incumbent contract")
            else:
                outcome = run_retrain_job(
                    frame,
                    contract=contract_owner.contract,
                    incumbent=contract_owner,
                    registry=self.registry,
                    policy=self.policy,
                )
        except Exception as exc:  # noqa: BLE001 — worker boundary
            outcome = RetrainOutcome(
                status="failed",
                notes=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
            )
        with self._lock:
            self._pending = (trigger_ts, outcome)

    def poll(self) -> Optional[tuple[pd.Timestamp, RetrainOutcome]]:
        with self._lock:
            pending, self._pending = self._pending, None
        return pending
