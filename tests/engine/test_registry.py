"""Model registry: publish/activate/load, contract round-trip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from src.engine.errors import FeatureContractError, ModelArtifactError
from src.engine.features import FeatureContract, reconcile_matrix, reconcile_row
from src.engine.model import ModelRegistry, Thresholds

pytestmark = pytest.mark.engine

ANCHOR = "2025-01-01T00:01:00+00:00"


def tiny_model(n_features: int = 4, seed: int = 0) -> CatBoostClassifier:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, n_features))
    y = (X[:, 0] + rng.normal(0, 0.5, 400) > 0).astype(int)
    model = CatBoostClassifier(iterations=10, depth=2, verbose=False,
                               allow_writing_files=False)
    model.fit(X, y)
    return model


def tiny_contract(n_features: int = 4) -> FeatureContract:
    return FeatureContract(
        feature_list=tuple(f"f{i}__f__w0" for i in range(n_features)),
        grid_anchor_ts=ANCHOR,
    )


def test_contract_json_roundtrip(tmp_path):
    c = tiny_contract()
    path = tmp_path / "contract.json"
    c.to_json(path)
    back = FeatureContract.from_json(path)
    assert back == c
    assert back.anchor == pd.Timestamp(ANCHOR)


def test_contract_requires_anchor_and_features():
    with pytest.raises(FeatureContractError):
        FeatureContract(feature_list=(), grid_anchor_ts=ANCHOR)
    with pytest.raises(FeatureContractError):
        FeatureContract(feature_list=("a",), grid_anchor_ts="")


def test_publish_activate_load_roundtrip(tmp_path):
    reg = ModelRegistry(tmp_path / "models")
    assert reg.versions() == []
    assert not reg.has_active()
    with pytest.raises(ModelArtifactError):
        reg.active()

    contract = tiny_contract()
    v1 = reg.publish(
        model=tiny_model(), contract=contract,
        thresholds=Thresholds(p_threshold=0.6, top_q=0.99, train_p_quantiles={"0.5": 0.3}),
        metrics={"val_roc_auc": 0.7}, training_meta={"note": "test"},
    )
    assert v1 == "v0001"
    assert reg.active_version() == "v0001"

    handle = reg.active()
    assert handle.version == "v0001"
    assert handle.contract == contract
    assert handle.thresholds.p_threshold == 0.6
    assert handle.metrics["val_roc_auc"] == 0.7

    x = np.zeros(4)
    p = handle.predict_p(x)
    assert 0.0 <= p <= 1.0

    v2 = reg.publish(
        model=tiny_model(seed=1), contract=contract,
        thresholds=Thresholds(p_threshold=0.5, top_q=0.99, train_p_quantiles={}),
        metrics={}, training_meta={}, activate=False,
    )
    assert v2 == "v0002"
    assert reg.active_version() == "v0001"  # activate=False left the pointer
    reg.activate(v2)
    assert reg.active_version() == "v0002"
    assert reg.versions() == ["v0001", "v0002"]


def test_activate_missing_version_fails(tmp_path):
    reg = ModelRegistry(tmp_path / "models")
    with pytest.raises(ModelArtifactError):
        reg.activate("v0042")


def test_feature_count_mismatch_detected(tmp_path):
    reg = ModelRegistry(tmp_path / "models")
    reg.publish(
        model=tiny_model(n_features=4), contract=tiny_contract(n_features=3),
        thresholds=Thresholds(p_threshold=0.5, top_q=0.99, train_p_quantiles={}),
        metrics={}, training_meta={},
    )
    with pytest.raises(ModelArtifactError):
        reg.active()


# ---------------------------------------------------------------------------
# Contract reconciliation
# ---------------------------------------------------------------------------


def test_reconcile_row_and_matrix(tmp_path):
    import polars as pl

    contract = tiny_contract(3)
    frame = pl.DataFrame({
        "ts": [1, 2], "f0__f__w0": [1.0, 2.0], "f1__f__w0": [3.0, 4.0],
        "f2__f__w0": [5.0, 6.0], "extra": [0.0, 0.0],
    })
    row = reconcile_row(frame, contract, row=-1)
    np.testing.assert_allclose(row, [2.0, 4.0, 6.0])
    mat = reconcile_matrix(frame, contract)
    assert mat.shape == (2, 3)

    with pytest.raises(FeatureContractError):
        reconcile_row(frame.drop("f1__f__w0"), contract)
    bad = frame.with_columns(pl.lit(float("nan")).alias("f0__f__w0"))
    with pytest.raises(FeatureContractError):
        reconcile_row(bad, contract)
