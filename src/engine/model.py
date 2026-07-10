"""Model service: versioned registry + prediction handles.

Layout (under ``EngineConfig.model_dir``)::

    models/
      ACTIVE                # single line: the active version name
      v0001/
        model.cbm           # CatBoost artifact
        contract.json       # FeatureContract (incl. grid anchor)
        thresholds.json     # {"p_threshold": …, "top_q": …, "train_p_quantiles": {…}}
        metrics.json        # validation metrics recorded at training time
        training_meta.json  # provenance (rows, window, params, best_iter, …)

Publishing is atomic: artifacts are written to a temp directory, renamed
into place, and only then does ``ACTIVE`` repoint. A reader can never see
a half-written version.

``import_research_artifacts`` packages the existing research outputs
(``catboost_model_1min.cbm`` + metadata) as ``v0001`` so the engine's
first live model is *exactly* the researched one.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.engine.errors import ModelArtifactError
from src.engine.features import FeatureContract


@dataclass(frozen=True)
class Thresholds:
    """Train-frozen decision thresholds (never derived from val/test)."""

    p_threshold: float
    top_q: float
    train_p_quantiles: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "p_threshold": self.p_threshold,
            "top_q": self.top_q,
            "train_p_quantiles": self.train_p_quantiles,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Thresholds":
        return cls(
            p_threshold=float(d["p_threshold"]),
            top_q=float(d.get("top_q", float("nan"))),
            train_p_quantiles={str(k): float(v) for k, v in d.get("train_p_quantiles", {}).items()},
        )


class ModelHandle:
    """A loaded model version: CatBoost artifact + contract + thresholds."""

    def __init__(
        self,
        version: str,
        model: CatBoostClassifier,
        contract: FeatureContract,
        thresholds: Thresholds,
        metrics: dict,
    ) -> None:
        self.version = version
        self.model = model
        self.contract = contract
        self.thresholds = thresholds
        self.metrics = metrics
        self.last_predict_ms: float = float("nan")

    def predict_p(self, values: np.ndarray) -> float:
        """P(y=1) for one feature vector (contract order)."""
        t0 = time.perf_counter()
        p = float(self.model.predict_proba(values.reshape(1, -1))[0, 1])
        self.last_predict_ms = (time.perf_counter() - t0) * 1e3
        return p

    def predict_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Vectorized P(y=1) over an (N, n_features) matrix."""
        return self.model.predict_proba(matrix)[:, 1].astype(np.float64)


class ModelRegistry:
    """Versioned model store with an atomic ACTIVE pointer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def versions(self) -> list[str]:
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith("v") and (p / "model.cbm").exists()
        )

    def active_version(self) -> Optional[str]:
        pointer = self.root / "ACTIVE"
        if not pointer.exists():
            return None
        name = pointer.read_text(encoding="utf-8").strip()
        return name or None

    def has_active(self) -> bool:
        v = self.active_version()
        return v is not None and (self.root / v / "model.cbm").exists()

    # ------------------------------------------------------------------ #
    # Load / publish
    # ------------------------------------------------------------------ #

    def load(self, version: str) -> ModelHandle:
        vdir = self.root / version
        cbm = vdir / "model.cbm"
        if not cbm.exists():
            raise ModelArtifactError(f"model version {version!r} not found under {self.root}")
        model = CatBoostClassifier()
        model.load_model(str(cbm))
        contract = FeatureContract.from_json(vdir / "contract.json")
        thresholds = Thresholds.from_dict(
            json.loads((vdir / "thresholds.json").read_text(encoding="utf-8"))
        )
        metrics_path = vdir / "metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists() else {}
        )
        n_model_feats = model.feature_names_ is not None and len(model.feature_names_)
        if n_model_feats and n_model_feats != contract.n_features:
            raise ModelArtifactError(
                f"{version}: model expects {n_model_feats} features but the "
                f"contract lists {contract.n_features}"
            )
        return ModelHandle(version, model, contract, thresholds, metrics)

    def active(self) -> ModelHandle:
        v = self.active_version()
        if v is None:
            raise ModelArtifactError(
                f"no ACTIVE model under {self.root} — publish or import one first"
            )
        return self.load(v)

    def next_version_name(self) -> str:
        existing = self.versions()
        if not existing:
            return "v0001"
        return f"v{int(existing[-1][1:]) + 1:04d}"

    def publish(
        self,
        *,
        model: CatBoostClassifier,
        contract: FeatureContract,
        thresholds: Thresholds,
        metrics: dict,
        training_meta: dict,
        activate: bool = True,
    ) -> str:
        version = self.next_version_name()
        tmp = self.root / f".tmp_{version}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            model.save_model(str(tmp / "model.cbm"))
            contract.to_json(tmp / "contract.json")
            (tmp / "thresholds.json").write_text(
                json.dumps(thresholds.to_dict(), indent=2), encoding="utf-8"
            )
            (tmp / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            (tmp / "training_meta.json").write_text(
                json.dumps(training_meta, indent=2, default=str), encoding="utf-8"
            )
            tmp.rename(self.root / version)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if activate:
            self.activate(version)
        return version

    def activate(self, version: str) -> None:
        if not (self.root / version / "model.cbm").exists():
            raise ModelArtifactError(f"cannot activate missing version {version!r}")
        pointer = self.root / "ACTIVE"
        tmp = self.root / "ACTIVE.tmp"
        tmp.write_text(version, encoding="utf-8")
        tmp.replace(pointer)

    # ------------------------------------------------------------------ #
    # Research-artifact import
    # ------------------------------------------------------------------ #

    def import_research_artifacts(
        self,
        dataset_dir: str | Path,
        *,
        top_q: float = 0.99,
        activate: bool = True,
    ) -> str:
        """Package the notebook-produced research artifacts as a version.

        Reads ``catboost_model_1min.cbm``, ``feature_list_1min.json``,
        ``dataset_metadata_1min.json``, ``predictions_metadata_1min.json``
        and the train-score artifact, derives the top-q entry threshold on
        the *training* score distribution (never val/test), and publishes.
        """
        from src.analytics.thresholds import derive_top_q_threshold

        dataset_dir = Path(dataset_dir)
        cbm = dataset_dir / "catboost_model_1min.cbm"
        flist = dataset_dir / "feature_list_1min.json"
        meta_p = dataset_dir / "dataset_metadata_1min.json"
        if not (cbm.exists() and flist.exists() and meta_p.exists()):
            raise ModelArtifactError(
                f"research artifacts incomplete under {dataset_dir} — need "
                "catboost_model_1min.cbm, feature_list_1min.json, dataset_metadata_1min.json"
            )
        model = CatBoostClassifier()
        model.load_model(str(cbm))
        feature_list = tuple(json.loads(flist.read_text(encoding="utf-8")))
        meta = json.loads(meta_p.read_text(encoding="utf-8"))

        # Grid anchor = training-frame bar 0 = trimmed first row − N_WARMUP min.
        ts_start = pd.Timestamp(meta["ts_start"])
        if ts_start.tzinfo is None:
            ts_start = ts_start.tz_localize("UTC")
        anchor = ts_start - pd.Timedelta(minutes=int(meta["N_WARMUP"]))
        contract = FeatureContract(
            feature_list=feature_list,
            label_cadence=str(meta["label_cadence"]),
            barrier_source=str(meta["barrier_source"]),
            with_derivatives=False,
            m=int(meta["M"]),
            phi=float(meta["PHI"]),
            n_warmup=int(meta["N_WARMUP"]),
            grid_anchor_ts=anchor.isoformat(),
        )

        train_scores_p = dataset_dir / "analytics" / "train_scores_unc_1min.parquet"
        if not train_scores_p.exists():
            raise ModelArtifactError(
                f"missing {train_scores_p} — cannot derive the train-frozen "
                "entry threshold"
            )
        p_train = pd.read_parquet(train_scores_p, columns=["p_train"])["p_train"].to_numpy()
        p_threshold = float(derive_top_q_threshold(p_train, q=top_q))

        pred_meta_p = dataset_dir / "predictions_metadata_1min.json"
        train_p_quantiles: dict[str, float] = {}
        if pred_meta_p.exists():
            pred_meta = json.loads(pred_meta_p.read_text(encoding="utf-8"))
            train_p_quantiles = {
                str(k): float(v)
                for k, v in pred_meta.get("train_p_quantiles", {}).items()
            }
        thresholds = Thresholds(
            p_threshold=p_threshold, top_q=top_q, train_p_quantiles=train_p_quantiles
        )

        metrics_p = dataset_dir / "analytics" / "research_metrics_1min.json"
        metrics = (
            json.loads(metrics_p.read_text(encoding="utf-8"))
            if metrics_p.exists() else {}
        )
        training_meta = {
            "source": "research_import",
            "dataset_metadata": meta,
        }
        return self.publish(
            model=model, contract=contract, thresholds=thresholds,
            metrics=metrics, training_meta=training_meta, activate=activate,
        )
