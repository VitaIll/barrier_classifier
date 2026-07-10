"""Threshold derivation from a frozen training-period reference distribution.

Two scalar thresholds the 1-min selective-entry strategy needs:

- ``derive_top_q_threshold(p_train, q)`` — the score gate. Returns the ``q``-th
  quantile of the training-period model scores. For "top 15%", call with
  ``q=0.85``.
- ``derive_conditional_unc_cap(p_train, unc_train, *, p_threshold, q)`` — the
  epistemic gate, computed conditional on the score gate being passed. Among
  training rows with ``p_train >= p_threshold``, returns the ``q``-th quantile
  of ``knowledge_unc``. For "low uncertainty compared to other predicted-positive
  rows", default ``q=0.5`` — i.e. the median — which halves the trade rate
  beyond what the score gate alone would admit.

Both helpers are pure functions of training-period numpy arrays; the notebook
supplies those arrays after scoring the saved model on the training slice.
The thresholds are *frozen* — no val/test data goes in.

Persisted-metadata loaders (``load_training_p_quantile``, ``load_training_p_max``)
read the same quantiles back from ``predictions_metadata_1min.json`` so a
notebook running after training does not need to re-score the train slice.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def derive_top_q_threshold(p_train: np.ndarray, *, q: float = 0.85) -> float:
    """Return the ``q``-th quantile of ``p_train``.

    ``q=0.85`` corresponds to "top 15%": only rows with score in the top
    15% of the training-period distribution pass this gate. ``p_train``
    must be the model's predicted probability over the *training* slice
    only — passing val/test scores is the leak this helper is designed to
    prevent.
    """
    arr = np.asarray(p_train, dtype=float)
    if arr.size == 0:
        raise ValueError("p_train is empty")
    if not (0.0 < float(q) < 1.0):
        raise ValueError(f"q must be in (0, 1); got {q}")
    return float(np.quantile(arr, float(q)))


def derive_conditional_unc_cap(
    p_train: np.ndarray,
    unc_train: np.ndarray,
    *,
    p_threshold: float,
    q: float = 0.5,
) -> float:
    """Return the ``q``-th quantile of ``unc_train`` conditional on ``p_train >= p_threshold``.

    The conditional cohort is the "predicted positive class" the user
    cares about: rows the score gate would already let through. The
    cap then says "among those, keep only the bottom ``q``-fraction by
    knowledge uncertainty". ``q=0.5`` halves the trade rate beyond gate 1;
    ``q=0.25`` quarters it; ``q=1.0`` is a no-op (returns the max).

    Raises if no training rows pass the score gate — would silently degrade
    to a vacuous cap otherwise.
    """
    p = np.asarray(p_train, dtype=float)
    u = np.asarray(unc_train, dtype=float)
    if p.shape != u.shape:
        raise ValueError(f"p_train shape {p.shape} != unc_train shape {u.shape}")
    if not (0.0 < float(q) <= 1.0):
        raise ValueError(f"q must be in (0, 1]; got {q}")
    mask = (p >= float(p_threshold)) & np.isfinite(u)
    if not mask.any():
        raise ValueError(
            f"no training rows pass p >= {p_threshold} with finite uncertainty; "
            f"cannot derive conditional cap"
        )
    return float(np.quantile(u[mask], float(q)))


def summarize_gate_overlap(
    p_train: np.ndarray,
    unc_train: np.ndarray,
    *,
    p_threshold: float,
    unc_cap: float,
) -> dict:
    """Diagnostic: how many training rows pass each gate and the combined gate.

    Returned dict has counts and pass-rates; used by the notebook to print
    a quick sanity table next to the frozen thresholds.
    """
    p = np.asarray(p_train, dtype=float)
    u = np.asarray(unc_train, dtype=float)
    n = int(p.size)
    score_pass = p >= float(p_threshold)
    unc_pass = u <= float(unc_cap)
    joint = score_pass & unc_pass & np.isfinite(u)
    return {
        "n_train": n,
        "n_score_pass": int(score_pass.sum()),
        "n_unc_pass": int(unc_pass.sum()),
        "n_joint_pass": int(joint.sum()),
        "score_pass_rate": float(score_pass.mean()),
        "unc_pass_rate": float(unc_pass.mean()),
        "joint_pass_rate": float(joint.mean()),
        "unc_pass_rate_given_score": (
            float((joint.sum()) / max(score_pass.sum(), 1))
        ),
        "p_threshold": float(p_threshold),
        "unc_cap": float(unc_cap),
    }


def load_training_p_quantile(
    metadata_path: str | os.PathLike,
    q: float,
    *,
    consumer_name: str = "this consumer",
) -> float:
    """Read the training-period probability quantile at level ``q`` from
    the training-metadata JSON.

    The convention (set in the training notebook) is a top-level key
    ``"train_p_quantiles": {"0.05": ..., "0.10": ..., ...}`` keyed by
    string-formatted quantile. Accepts ``"0.90"`` or ``"0.9"`` style keys.

    Raises a clear ``RuntimeError`` if the file is missing or the
    requested quantile isn't recorded.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise RuntimeError(
            f"{consumer_name} needs the training-period probability quantile "
            f"at q={q}, but {path} does not exist. "
            f"Run the training notebook (03_train_model.ipynb) first to "
            "produce predictions_metadata_1min.json with 'train_p_quantiles'."
        )
    with open(path) as f:
        meta = json.load(f)
    q_table = meta.get("train_p_quantiles")
    if not isinstance(q_table, dict):
        raise RuntimeError(
            f"{consumer_name} expected key 'train_p_quantiles' in {path}, "
            f"but it is missing or not a mapping. Re-run the training "
            "notebook to refresh it."
        )
    candidates = [f"{q:.2f}", f"{q:g}", f"{q}"]
    for k in candidates:
        if k in q_table:
            return float(q_table[k])
    raise RuntimeError(
        f"{consumer_name} requested training-period p-quantile q={q}, but "
        f"only {sorted(q_table.keys())} are recorded in {path}. "
        "Re-run the training notebook to refresh the quantile table."
    )


def load_training_p_max(
    metadata_path: str | os.PathLike,
    *,
    consumer_name: str = "this consumer",
) -> float:
    """Return the training-period ``p.max()`` from metadata.

    Used by reporting notebooks to size an edge-threshold grid upper
    bound without peeking at val/test ``p.max()``.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise RuntimeError(
            f"{consumer_name} needs the training-period p_max recorded by "
            f"the training notebook, but {path} does not exist."
        )
    with open(path) as f:
        meta = json.load(f)
    if "train_p_max" not in meta:
        raise RuntimeError(
            f"{consumer_name} expected key 'train_p_max' in {path}; not found. "
            "Re-run the training notebook to refresh metadata."
        )
    return float(meta["train_p_max"])
