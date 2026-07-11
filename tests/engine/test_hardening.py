"""Production-hardening unit tests.

Fast, hermetic checks for the engine's defensive layer: model-artifact
parity guards, feature-contract validation, rolling-service anti-skew
guards, crash-safe store flush, open-position snapshots, and config
validation. Heavier behaviours (risk kill-switch, resume roundtrip,
degraded bars, hot-swap guard, CLI) run end-to-end in
``test_engine_e2e.py`` where a real model/registry/stream exists.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, Pool

from src.engine.buffer import BarBuffer
from src.engine.engine import EngineConfig
from src.engine.errors import (
    ConfigError,
    FeatureContractError,
    ModelArtifactError,
    StoreError,
)
from src.engine.execution import PaperBroker
from src.engine.features import (
    MAX_BOUNDARY_LOOKBACK_ROWS,
    FeatureContract,
    RollingFeatureService,
)
from src.engine.model import ModelRegistry, Thresholds
from src.engine.retrain import RetrainPolicy
from src.engine.store import SQLiteStore

ANCHOR = "2025-01-01T00:00:00+00:00"


def _named_model(feature_names: list[str], seed: int = 0) -> CatBoostClassifier:
    """A tiny CatBoost carrying explicit feature names."""
    rng = np.random.default_rng(seed)
    n = len(feature_names)
    X = rng.random((60, n))
    y = (rng.random(60) > 0.5).astype(int)
    model = CatBoostClassifier(
        iterations=8, depth=2, learning_rate=0.2, verbose=False,
        allow_writing_files=False, random_seed=seed,
    )
    model.fit(Pool(data=X, label=y, feature_names=list(feature_names)))
    return model


def _contract(features: tuple[str, ...]) -> FeatureContract:
    return FeatureContract(feature_list=features, grid_anchor_ts=ANCHOR)


def _publish(reg: ModelRegistry, model, contract, *, p_threshold=0.5):
    return reg.publish(
        model=model, contract=contract,
        thresholds=Thresholds(p_threshold=p_threshold, top_q=0.99,
                              train_p_quantiles={"0.5": 0.3}),
        metrics={}, training_meta={},
    )


# --------------------------------------------------------------------------- #
# Model artifact guards
# --------------------------------------------------------------------------- #

def test_model_name_order_mismatch_is_rejected(tmp_path):
    reg = ModelRegistry(tmp_path / "m")
    model = _named_model(["a", "b", "c", "d"])
    # Contract lists the SAME names in a DIFFERENT order → must not load.
    _publish(reg, model, _contract(("d", "c", "b", "a")))
    with pytest.raises(ModelArtifactError, match="feature names do not match"):
        reg.active()


def test_model_name_match_loads(tmp_path):
    reg = ModelRegistry(tmp_path / "m")
    names = ["a", "b", "c", "d"]
    _publish(reg, _named_model(names), _contract(tuple(names)))
    handle = reg.active()
    assert tuple(handle.model.feature_names_) == tuple(handle.contract.feature_list)


def test_corrupt_cbm_raises_typed_error(tmp_path):
    reg = ModelRegistry(tmp_path / "m")
    v = _publish(reg, _named_model(["a", "b"]), _contract(("a", "b")))
    (reg.root / v / "model.cbm").write_bytes(b"not a real catboost artifact")
    with pytest.raises(ModelArtifactError, match="failed to load"):
        reg.load(v)


def test_malformed_thresholds_json_raises_typed_error(tmp_path):
    reg = ModelRegistry(tmp_path / "m")
    v = _publish(reg, _named_model(["a", "b"]), _contract(("a", "b")))
    (reg.root / v / "thresholds.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="malformed"):
        reg.load(v)


@pytest.mark.parametrize("bad_threshold", [float("nan"), 0.0, 1.0, -0.2, 1.7])
def test_out_of_range_threshold_rejected(tmp_path, bad_threshold):
    reg = ModelRegistry(tmp_path / "m")
    v = _publish(reg, _named_model(["a", "b"]), _contract(("a", "b")))
    (reg.root / v / "thresholds.json").write_text(
        json.dumps({"p_threshold": bad_threshold, "top_q": 0.99,
                    "train_p_quantiles": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ModelArtifactError, match="p_threshold"):
        reg.load(v)


def test_predict_p_rejects_wrong_length_vector(tmp_path):
    reg = ModelRegistry(tmp_path / "m")
    _publish(reg, _named_model(["a", "b", "c", "d"]), _contract(("a", "b", "c", "d")))
    handle = reg.active()
    assert 0.0 <= handle.predict_p(np.zeros(4)) <= 1.0  # correct length is fine
    with pytest.raises(ModelArtifactError, match="misaligned"):
        handle.predict_p(np.zeros(3))


# --------------------------------------------------------------------------- #
# FeatureContract validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kwargs, match", [
    (dict(feature_list=("a", "a")), "duplicate"),
    (dict(feature_list=("a",), m=0), "m must be"),
    (dict(feature_list=("a",), phi=0.0), "phi must be"),
    (dict(feature_list=("a",), phi=float("nan")), "phi must be"),
    (dict(feature_list=("a",), n_warmup=-1), "n_warmup must be"),
    (dict(feature_list=("a",), barrier_source="bogus"), "barrier_source"),
])
def test_feature_contract_rejects_bad_params(kwargs, match):
    with pytest.raises(FeatureContractError, match=match):
        FeatureContract(grid_anchor_ts=ANCHOR, **kwargs)


# --------------------------------------------------------------------------- #
# RollingFeatureService anti-skew guards
# --------------------------------------------------------------------------- #

def test_rolling_service_rejects_tail_below_lookback():
    c = _contract(("a",))
    with pytest.raises(FeatureContractError, match="max boundary lookback"):
        RollingFeatureService(c, boundary_tail_rows=MAX_BOUNDARY_LOOKBACK_ROWS)
    # A tail above the lookback, or None (full window), is fine.
    RollingFeatureService(c, boundary_tail_rows=MAX_BOUNDARY_LOOKBACK_ROWS + 1)
    RollingFeatureService(c, boundary_tail_rows=None)


def test_rolling_service_empty_buffer_errors():
    c = _contract(("a",))
    buf = BarBuffer(40, m=20, anchor_ts=pd.Timestamp(ANCHOR))
    with pytest.raises(FeatureContractError, match="empty buffer"):
        RollingFeatureService(c).latest(buf)


# --------------------------------------------------------------------------- #
# Store crash-safety + resume state
# --------------------------------------------------------------------------- #

def test_flush_is_atomic_and_retains_rows_on_failure():
    store = SQLiteStore(":memory:")
    # A valid bars row queued alongside a malformed predictions row (wrong arity).
    store._pending["bars"].append((60_000, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0))
    store._pending["predictions"].append((1, 2, 3))  # 3 values, needs 5
    with pytest.raises(StoreError, match="retained for retry"):
        store.flush()
    # Buffered rows survive for a retry; the transaction rolled back.
    assert store._pending["bars"], "buffered rows must survive a failed flush"
    n = store._conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    assert n == 0, "a failed flush must not partially commit"
    store.close()


def test_flush_commits_then_clears_on_success():
    store = SQLiteStore(":memory:")
    store._pending["bars"].append((60_000, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0))
    store.flush()
    assert not store._pending["bars"]
    assert store._conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 1
    store.close()


def test_busy_timeout_pragma_set(tmp_path):
    store = SQLiteStore(tmp_path / "s.db")
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5_000
    store.close()


def test_open_position_snapshot_roundtrip_and_resume_helpers():
    store = SQLiteStore(":memory:")
    assert store.load_open_positions() == []
    assert store.last_realized_cum() is None
    assert store.max_total_equity() is None
    assert store.max_trade_id() == 0
    assert store.max_order_id() == 0
    assert store.last_bar_ts() is None

    rows = [
        (100, 6_000_000, 1, 0.02, 50_000.0, 50_125.0, None, 1_000_100, 0.61, "v0001"),
        (140, 8_400_000, 1, 0.02, 50_500.0, 50_626.0, 49_000.0, 1_000_140, 0.72, "v0002"),
    ]
    store.snapshot_open_positions(rows)
    assert store.load_open_positions() == rows
    # A newer snapshot fully replaces the old one (it is a snapshot, not a log).
    store.snapshot_open_positions(rows[1:])
    assert store.load_open_positions() == rows[1:]
    store.snapshot_open_positions([])
    assert store.load_open_positions() == []

    ts = pd.Timestamp("2025-01-01 00:01:00", tz="UTC")
    store.record_equity(ts, realized_cum=0.010, unrealized=0.005, n_open=1, gross_size=0.02)
    store.record_equity(ts + pd.Timedelta(minutes=1), realized_cum=0.012,
                        unrealized=-0.001, n_open=1, gross_size=0.02)
    store.flush()
    assert store.last_realized_cum() == pytest.approx(0.012)
    assert store.max_total_equity() == pytest.approx(0.015)
    store.close()


# --------------------------------------------------------------------------- #
# PaperBroker order-id continuity
# --------------------------------------------------------------------------- #

def test_paper_broker_order_id_seed():
    assert PaperBroker()._next_order_id == 1
    assert PaperBroker(next_order_id=42)._next_order_id == 42
    with pytest.raises(ValueError):
        PaperBroker(next_order_id=0)


# --------------------------------------------------------------------------- #
# EngineConfig / RetrainPolicy validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kwargs", [
    dict(feature_mode="weird"),
    dict(exit_variant="weird"),
    dict(buffer_rows=0),
    dict(boundary_tail_rows=0),
    dict(min_ready_rows=0),
    dict(lot_size=0.0),
    dict(max_concurrent=0),
    dict(cost_per_trade=-0.001),
    dict(p_threshold_override=0.0),
    dict(p_threshold_override=1.5),
    dict(retention_days=0),
    dict(max_repair_gap=0),
    dict(halt_after_feature_errors=0),
    dict(log_every_bars=0),
    dict(max_drawdown=-0.1),
    dict(max_cumulative_loss=0.1),
])
def test_engine_config_rejects_bad_values(kwargs):
    with pytest.raises(ConfigError):
        EngineConfig(**kwargs).validate()


def test_engine_config_accepts_risk_knobs():
    EngineConfig(max_drawdown=0.2, max_cumulative_loss=-0.5, resume=True).validate()


def test_engine_config_rejects_retention_starving_retrain():
    cfg = EngineConfig(
        retention_days=7.0,
        retrain=RetrainPolicy(enabled=True, min_window_rows=30 * 1440),
    )
    with pytest.raises(ConfigError, match="starve"):
        cfg.validate()


@pytest.mark.parametrize("kwargs", [
    dict(every_bars=0),
    dict(iterations=0),
    dict(min_window_rows=0),
    dict(window_rows=0),
    dict(embargo_rows=-1),
    dict(top_q=1.0),
    dict(train_frac=0.0),
    dict(val_frac=1.0),
    dict(train_frac=0.9, val_frac=0.2),
    dict(min_pr_auc_ratio=0.0),
])
def test_retrain_policy_rejects_bad_values(kwargs):
    with pytest.raises(ConfigError):
        RetrainPolicy(**kwargs)


def test_from_toml_rejects_bad_retrain_table(tmp_path):
    toml = tmp_path / "e.toml"
    toml.write_text(
        'feature_mode = "batch"\n\n[retrain]\nnot_a_real_key = 3\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"\[retrain\]"):
        EngineConfig.from_toml(toml)


def test_from_toml_rejects_bad_field_type(tmp_path):
    toml = tmp_path / "e.toml"
    toml.write_text("buffer_rows = -5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="buffer_rows"):
        EngineConfig.from_toml(toml)
