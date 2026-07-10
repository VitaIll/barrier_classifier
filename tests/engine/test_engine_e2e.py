"""Synthetic end-to-end: the whole engine loop on a simulated stream.

Builds a self-contained world from scratch — synthetic parquet, a genesis
model trained through the real pipeline, a registry — then runs the
engine against a ReplaySource and asserts the full loop functioned:
guards, features, predictions, decisions, fills, persistence, an
event-time retrain with hot-swap, and batch/rolling serving parity.

No repository data artifacts are required; everything lives in tmp_path.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, Pool

from src.engine.engine import Engine, EngineConfig
from src.engine.features import FeatureContract
from src.engine.model import ModelRegistry, Thresholds
from src.engine.retrain import RetrainPolicy
from src.engine.sources import ReplaySource
from src.engine.store import SQLiteStore
from src.features.config import N_WARMUP
from src.features.pipeline import (
    _BASE_COLS,
    _DERIV_BASE_COLS,
    _LABEL_AUX_COLS,
    _RAW_COLS,
    run_pipeline,
)
from tests.engine.test_feature_inference import synthetic_raw

pytestmark = [pytest.mark.engine_slow]

STREAM_BARS = 700


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Synthetic parquet + genesis model registry, built once."""
    root = tmp_path_factory.mktemp("engine_world")
    n_total = N_WARMUP + 1_200 + STREAM_BARS
    raw = synthetic_raw(n_total, seed=123)
    spot_path = root / "klines_1m.parquet"
    raw.to_parquet(spot_path)

    genesis_rows = N_WARMUP + 1_200  # stream starts after these
    genesis_frame = raw.iloc[:genesis_rows]

    ds = run_pipeline(genesis_frame, label_cadence="1min")
    non_feature = (
        set(_LABEL_AUX_COLS) | set(_RAW_COLS) | set(_BASE_COLS)
        | set(_DERIV_BASE_COLS) | {"weight"}
    )
    feature_cols = [
        c for c in ds.columns
        if c not in non_feature and not c.startswith("undef__")
    ]
    X = ds.select(feature_cols).to_numpy().astype(float)
    y = ds["y"].to_numpy().astype(int)
    k = ds["k"].to_numpy().astype(np.uint32)
    model = CatBoostClassifier(
        iterations=30, depth=3, learning_rate=0.1, verbose=False,
        allow_writing_files=False, random_seed=0,
    )
    model.fit(Pool(data=X, label=y, timestamp=k, feature_names=list(feature_cols)))
    p_train = model.predict_proba(X)[:, 1]

    contract = FeatureContract(
        feature_list=tuple(feature_cols),
        label_cadence="1min",
        barrier_source="high",
        with_derivatives=False,
        n_warmup=N_WARMUP,
        grid_anchor_ts=raw.index[0].isoformat(),
    )
    registry = ModelRegistry(root / "models")
    registry.publish(
        model=model, contract=contract,
        thresholds=Thresholds(
            p_threshold=float(np.quantile(p_train, 0.90)), top_q=0.90,
            train_p_quantiles={"0.5": float(np.median(p_train))},
        ),
        metrics={"genesis": True}, training_meta={"rows": int(len(ds))},
    )
    stream_start = raw.index[genesis_rows]
    return {
        "root": root, "spot_path": spot_path, "raw": raw,
        "stream_start": stream_start, "registry_dir": root / "models",
    }


def _config(world, tmp_path, **overrides) -> EngineConfig:
    n_total = len(world["raw"])
    buffer_rows = n_total + (20 - n_total % 20) % 20
    cfg = EngineConfig(
        model_dir=world["registry_dir"],
        store_path=tmp_path / "engine.db",
        feature_mode="batch",
        buffer_rows=buffer_rows,
        min_ready_rows=N_WARMUP + 1,
        log_every_bars=10_000,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.validate()
    return cfg


def test_batch_replay_full_loop_with_retrain_and_hot_swap(world, tmp_path):
    cfg = _config(
        world, tmp_path,
        retrain=RetrainPolicy(
            enabled=True,
            every_bars=300,               # event-time: fires twice in 700 bars
            min_window_rows=N_WARMUP + 600,
            iterations=10,
            top_q=0.90,
            embargo_rows=60,              # 3×M — research-scale 1,200 needs research-scale windows
        ),
        retrain_threaded=False,           # deterministic inside the replay
        p_threshold_override=None,
    )
    source = ReplaySource(world["spot_path"], start=world["stream_start"])
    engine = Engine(cfg, source=source)
    events = {"swaps": [], "retrains": [], "trades": []}
    from src.engine.domain import EventType
    engine.on(EventType.MODEL_SWAPPED, events["swaps"].append)
    engine.on(EventType.RETRAIN_COMPLETED, events["retrains"].append)
    engine.on(EventType.TRADE_CLOSED, events["trades"].append)

    report = engine.run()

    # --- Loop mechanics -----------------------------------------------------
    assert report.n_bars == STREAM_BARS
    assert report.n_predictions == STREAM_BARS  # batch rows embed full history
    assert not report.halted
    assert report.feature_errors == 0
    assert report.guard_repairs == 0

    # --- Persistence ----------------------------------------------------------
    counts = engine.store.counts()
    assert counts["predictions"] == STREAM_BARS
    assert counts["equity"] == STREAM_BARS
    assert counts["bars"] >= STREAM_BARS  # bootstrap + streamed
    assert counts["labels"] > 0           # matured labels recorded

    # --- Retraining + hot swap -------------------------------------------------
    assert report.retrain_runs >= 1
    assert len(events["retrains"]) == report.retrain_runs
    statuses = {o.status for o in events["retrains"]}
    assert statuses <= {"published", "gate_rejected", "skipped"}
    if "published" in statuses:
        assert len(report.model_versions_used) >= 2
        assert events["swaps"], "published retrain must hot-swap"
        reg = ModelRegistry(world["registry_dir"])
        assert reg.active_version() == report.model_versions_used[-1]

    # --- Ledger consistency ------------------------------------------------------
    assert len(events["trades"]) == report.n_trades
    if report.n_trades:
        t = report.trades
        assert np.isfinite(t["weighted_net_log_return"].to_numpy()).all()
        realized = float(t["weighted_net_log_return"].sum())
        assert realized == pytest.approx(report.realized_cum_log_return, abs=1e-9)
    engine.store.close()


def test_rolling_matches_batch_on_shared_history(world, tmp_path):
    """The anti-skew guard: the true-live rolling path must reproduce the
    batch path's predictions exactly when both see the same history."""
    cfg_batch = _config(world, tmp_path, store_path=tmp_path / "b.db")
    src_b = ReplaySource(world["spot_path"], start=world["stream_start"])
    eng_b = Engine(cfg_batch, source=src_b)
    rep_b = eng_b.run(max_bars=2)
    p_batch = pd.read_sql_query(
        "SELECT ts_ms, p FROM predictions ORDER BY ts_ms",
        eng_b.store.read_connection(),
    )
    eng_b.store.close()

    cfg_roll = _config(world, tmp_path, store_path=tmp_path / "r.db",
                       feature_mode="rolling")
    src_r = ReplaySource(world["spot_path"], start=world["stream_start"])
    eng_r = Engine(cfg_roll, source=src_r)
    rep_r = eng_r.run(max_bars=2)
    p_roll = pd.read_sql_query(
        "SELECT ts_ms, p FROM predictions ORDER BY ts_ms",
        eng_r.store.read_connection(),
    )
    eng_r.store.close()

    assert rep_b.n_predictions == rep_r.n_predictions == 2
    pd.testing.assert_series_equal(p_batch["ts_ms"], p_roll["ts_ms"])
    np.testing.assert_allclose(
        p_batch["p"].to_numpy(), p_roll["p"].to_numpy(), rtol=0, atol=1e-12,
    )


def test_config_toml_roundtrip(tmp_path):
    toml = tmp_path / "engine.toml"
    toml.write_text(
        """
feature_mode = "batch"
lot_size = 0.05
max_concurrent = 10

[retrain]
enabled = true
every_bars = 1440
iterations = 50
""",
        encoding="utf-8",
    )
    cfg = EngineConfig.from_toml(toml)
    assert cfg.lot_size == 0.05
    assert cfg.retrain.enabled and cfg.retrain.every_bars == 1440

    bad = tmp_path / "bad.toml"
    bad.write_text('nonsense_key = 1\n', encoding="utf-8")
    from src.engine.errors import ConfigError
    with pytest.raises(ConfigError):
        EngineConfig.from_toml(bad)
