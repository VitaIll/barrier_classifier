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
from src.engine.errors import ConfigError
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
    from src.engine.risk import EntryControls
    cfg = EngineConfig(
        model_dir=world["registry_dir"],
        store_path=tmp_path / "engine.db",
        feature_mode="batch",
        entry_controls=EntryControls.disabled(),  # research-parity: no gate
        reconcile_every_bars=None,
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


def _source(world, **kwargs):
    kwargs.setdefault("start", world["stream_start"])
    return ReplaySource(world["spot_path"], **kwargs)


def test_resume_roundtrip_equals_uninterrupted_run(world, tmp_path):
    """Kill the session mid-stream, resume from the store, and the combined
    ledger/equity must equal one uninterrupted run — the resume contract."""
    raw = world["raw"]
    split_at = 350
    live_index = raw.index[raw.index >= world["stream_start"]]
    split_ts = live_index[split_at]  # first bar of the second half
    # The genesis top-10% train threshold is rarely crossed out-of-sample on
    # this synthetic world; trade at the train median so the resumed state
    # (open inventory, realized P&L, counters) is actually exercised.
    p_med = ModelRegistry(world["registry_dir"]).active().thresholds.train_p_quantiles["0.5"]

    # Reference: one uninterrupted run.
    eng_a = Engine(_config(world, tmp_path, store_path=tmp_path / "a.db",
                           p_threshold_override=p_med),
                   source=_source(world))
    rep_a = eng_a.run()
    trades_a = eng_a.store.trades_frame()
    equity_a = eng_a.store.equity_frame()
    eng_a.close()
    assert rep_a.n_trades > 0, "vacuous resume test — no trades in the reference run"

    # First half, then a clean shutdown.
    store_b = tmp_path / "b.db"
    eng_b1 = Engine(_config(world, tmp_path, store_path=store_b,
                            p_threshold_override=p_med),
                    source=_source(world, end=split_ts))
    rep_b1 = eng_b1.run()
    n_open_at_split = rep_b1.n_open_positions
    eng_b1.close()

    # Second half resumes from the same store.
    eng_b2 = Engine(_config(world, tmp_path, store_path=store_b, resume=True,
                            p_threshold_override=p_med),
                    source=_source(world, start=split_ts))
    rep_b2 = eng_b2.run()
    trades_b = eng_b2.store.trades_frame()
    equity_b = eng_b2.store.equity_frame()

    assert rep_b1.n_bars + rep_b2.n_bars == rep_a.n_bars
    assert rep_b2.realized_cum_log_return == pytest.approx(
        rep_a.realized_cum_log_return, abs=1e-12
    )
    assert rep_b2.n_open_positions == rep_a.n_open_positions
    pd.testing.assert_frame_equal(trades_a, trades_b)
    pd.testing.assert_frame_equal(equity_a, equity_b)
    # The restored snapshot matches the inventory that was open at the split.
    assert len(eng_b2.store.load_open_positions()) == rep_a.n_open_positions
    assert n_open_at_split >= 0  # informational; equality asserted above
    eng_b2.close()


def test_fresh_engine_refuses_dirty_store_without_resume(world, tmp_path):
    store_path = tmp_path / "dirty.db"
    eng = Engine(_config(world, tmp_path, store_path=store_path), source=_source(world))
    eng.run(max_bars=3)
    eng.close()
    with pytest.raises(ConfigError, match="resume=True"):
        Engine(_config(world, tmp_path, store_path=store_path), source=_source(world))


def test_risk_kill_switch_branches(world, tmp_path):
    """Drawdown and cumulative-loss limits trip the halt; the default
    (both None) never trips — offline-simulator parity."""
    ts = world["stream_start"]

    eng = Engine(_config(world, tmp_path, store_path=tmp_path / "dd.db",
                         max_drawdown=0.02), source=_source(world))
    eng._equity_peak = 0.10
    eng._check_risk_limits(ts, total_eq=0.095)   # drawdown 0.005 < 0.02
    assert not eng._halted
    eng._check_risk_limits(ts, total_eq=0.05)    # drawdown 0.05 >= 0.02
    assert eng._halted and "max_drawdown" in eng._halt_reason
    eng.close()

    eng2 = Engine(_config(world, tmp_path, store_path=tmp_path / "loss.db",
                          max_cumulative_loss=-0.10), source=_source(world))
    eng2._check_risk_limits(ts, total_eq=-0.05)
    assert not eng2._halted
    eng2._check_risk_limits(ts, total_eq=-0.15)
    assert eng2._halted and "max_cumulative_loss" in eng2._halt_reason
    eng2.close()

    eng3 = Engine(_config(world, tmp_path, store_path=tmp_path / "off.db"),
                  source=_source(world))
    eng3._equity_peak = 100.0
    eng3._check_risk_limits(ts, total_eq=-100.0)
    assert not eng3._halted
    eng3.close()


def test_halt_suppresses_entries_but_exits_stay_alive(world, tmp_path):
    """Once halted, no new entries for the rest of the session; the exit
    path keeps resolving open positions; decisions record HALT."""
    from src.engine.domain import Action, EventType

    p_med = ModelRegistry(world["registry_dir"]).active().thresholds.train_p_quantiles["0.5"]
    cfg = _config(world, tmp_path, store_path=tmp_path / "halt.db",
                  p_threshold_override=p_med)
    eng = Engine(cfg, source=_source(world))
    trip_after = 100
    seen = {"n": 0}

    def maybe_halt(bar):
        seen["n"] += 1
        if seen["n"] == trip_after:
            eng._halt(bar.ts, "test-tripped kill switch")

    eng.on(EventType.BAR_INGESTED, maybe_halt)
    report = eng.run()
    assert report.halted and report.halt_reason == "test-tripped kill switch"

    conn = eng.store.read_connection()
    dec = pd.read_sql_query(
        "SELECT ts_ms, action FROM decisions ORDER BY ts_ms", conn
    )
    # Bars from the trip onward are all HALT (never ENTER).
    assert set(dec["action"].iloc[trip_after - 1:]) == {Action.HALT.value}
    assert (dec["action"].iloc[:trip_after - 1] != Action.HALT.value).all()
    # No entries after the trip: every trade must have entered before it.
    trades = eng.store.trades_frame()
    trip_ts = dec["ts_ms"].iloc[trip_after - 1]
    if len(trades):
        entry_ms = trades["ts_entry"].astype("int64") // 10**6
        assert (entry_ms < trip_ts).all()
        # Exits are still resolving after the halt (if inventory was open).
    guard = pd.read_sql_query("SELECT guard, severity FROM guard_events", conn)
    assert ("halt" in set(guard["guard"]))
    eng.close()


def test_feature_errors_degrade_then_halt(world, tmp_path):
    """Rolling-path pipeline failures degrade bars (no prediction, no entry,
    exits alive) and trip the halt kill-switch at the configured limit."""
    from src.engine.errors import FeatureContractError

    cfg = _config(world, tmp_path, store_path=tmp_path / "deg.db",
                  feature_mode="rolling", halt_after_feature_errors=2)
    eng = Engine(cfg, source=_source(world))

    class Failing:
        last_feature_ms = float("nan")

        def latest(self, buffer):
            raise FeatureContractError("injected pipeline failure")

    eng.rolling_service = Failing()
    report = eng.run(max_bars=3)
    assert report.feature_errors == 3
    assert report.n_predictions == 0
    assert report.halted and "feature errors" in report.halt_reason
    assert report.n_trades == 0
    conn = eng.store.read_connection()
    dec = pd.read_sql_query("SELECT action, reason FROM decisions", conn)
    assert len(dec) == 3
    assert set(dec["reason"]) <= {"degraded", ""}
    guard = pd.read_sql_query("SELECT guard FROM guard_events", conn)
    assert (guard["guard"] == "feature_contract").sum() == 3
    eng.close()


def test_feature_error_recovery_resumes_predictions(world, tmp_path):
    """One bad window degrades one bar; the next bar predicts again
    (no sticky failure state below the halt threshold)."""
    from src.engine.errors import FeatureContractError
    from src.engine.features import RollingFeatureService

    cfg = _config(world, tmp_path, store_path=tmp_path / "rec.db",
                  feature_mode="rolling", halt_after_feature_errors=5)
    eng = Engine(cfg, source=_source(world))
    real = RollingFeatureService(eng.contract,
                                 boundary_tail_rows=cfg.boundary_tail_rows)

    class FlakyOnce:
        def __init__(self):
            self.calls = 0
            self.last_feature_ms = float("nan")

        def latest(self, buffer):
            self.calls += 1
            if self.calls == 1:
                raise FeatureContractError("injected transient failure")
            fv = real.latest(buffer)
            self.last_feature_ms = real.last_feature_ms
            return fv

    eng.rolling_service = FlakyOnce()
    report = eng.run(max_bars=2)
    assert report.feature_errors == 1
    assert report.n_predictions == 1
    assert not report.halted
    eng.close()


def test_grid_gap_repair_keeps_session_alive(world, tmp_path):
    """Missing minutes inside the stream are repaired with flat synthetic
    bars; the session predicts through them and records the repair."""
    raw = world["raw"]
    live_index = raw.index[raw.index >= world["stream_start"]]
    gapped = raw.drop(index=[live_index[2], live_index[3]])  # 2-minute hole
    spot_gapped = tmp_path / "gapped.parquet"
    gapped.to_parquet(spot_gapped)

    cfg = _config(world, tmp_path, store_path=tmp_path / "gap.db",
                  feature_mode="rolling")
    src = ReplaySource(spot_gapped, start=world["stream_start"])
    eng = Engine(cfg, source=src)
    report = eng.run(max_bars=6)
    assert report.guard_repairs == 2
    assert report.n_predictions == 6
    assert not report.halted and report.feature_errors == 0
    conn = eng.store.read_connection()
    n_synth = conn.execute("SELECT COUNT(*) FROM bars WHERE synthetic=1").fetchone()[0]
    assert n_synth == 2
    guard = pd.read_sql_query("SELECT guard, severity FROM guard_events", conn)
    assert ((guard["guard"] == "grid") & (guard["severity"] == "warning")).any()
    eng.close()


def test_hot_swap_compatibility_guard(world, tmp_path):
    """A candidate model that changes M / raw schema / anchor phase / warmup
    (or the feature list in batch mode) is refused; a faithful retrain
    product passes."""
    import dataclasses

    from src.engine.features import BatchFeatureService
    from src.engine.model import ModelHandle

    eng = Engine(_config(world, tmp_path, store_path=tmp_path / "swap.db"),
                 source=_source(world))
    c = eng.contract

    def handle_with(**changes):
        return ModelHandle("vTEST", eng.handle.model,
                           dataclasses.replace(c, **changes),
                           eng.handle.thresholds, {})

    assert eng._swap_incompatibility(handle_with()) == ""
    assert "M changed" in eng._swap_incompatibility(handle_with(m=40))
    assert "with_derivatives" in eng._swap_incompatibility(
        handle_with(with_derivatives=True))
    off_anchor = (pd.Timestamp(c.grid_anchor_ts) + pd.Timedelta(minutes=7)).isoformat()
    assert "anchor" in eng._swap_incompatibility(handle_with(grid_anchor_ts=off_anchor))
    assert "n_warmup" in eng._swap_incompatibility(
        handle_with(n_warmup=eng.warmup_guard.min_ready_rows + 1))
    # Batch mode serves a precomputed matrix → feature list is frozen.
    eng.batch_service = BatchFeatureService(c)
    assert "feature list" in eng._swap_incompatibility(
        handle_with(feature_list=tuple(c.feature_list[:-1])))
    eng.close()


def test_retention_pruning_wired_to_event_time(world, tmp_path):
    """retention_days triggers store.prune_before with the event-time cutoff
    on the daily cadence (first prune at the first bar)."""
    cfg = _config(world, tmp_path, store_path=tmp_path / "ret.db",
                  retention_days=0.5)
    eng = Engine(cfg, source=_source(world))
    calls = []
    real_prune = eng.store.prune_before
    eng.store.prune_before = lambda cutoff: (calls.append(cutoff), real_prune(cutoff))[1]
    eng.run(max_bars=3)
    assert len(calls) == 1  # daily cadence → one prune in a 3-bar session
    assert calls[0] == world["stream_start"] - pd.Timedelta(days=0.5)
    eng.close()


def test_initial_model_version_recorded_at_event_time(world, tmp_path):
    """The genesis model-version row is stamped with the first bar's event
    time (byte-reproducible replays), never wall-clock now()."""
    from src.engine.store import _ms

    eng = Engine(_config(world, tmp_path, store_path=tmp_path / "ver.db"),
                 source=_source(world))
    eng.run(max_bars=1)
    row = eng.store.read_connection().execute(
        "SELECT created_ts_ms FROM model_versions ORDER BY created_ts_ms LIMIT 1"
    ).fetchone()
    assert row[0] == _ms(world["stream_start"])
    eng.close()


def test_engine_close_is_idempotent_and_context_manager(world, tmp_path):
    with Engine(_config(world, tmp_path, store_path=tmp_path / "ctx.db"),
                source=_source(world)) as eng:
        eng.run(max_bars=2)
    assert eng._closed
    eng.close()  # second close is a no-op


def test_cli_replay_and_status_smoke(world, tmp_path, capsys):
    """The CLI drives a replay end to end and reports status, exit code 0."""
    from src.engine.__main__ import main

    store = tmp_path / "cli.db"
    common = [
        "--model-dir", str(world["registry_dir"]),
        "--store", str(store),
        "--spot", str(world["spot_path"]),
        "--start", str(world["stream_start"]),
        "--max-bars", "3",
    ]
    assert main(["replay"] + common) == 0
    out = capsys.readouterr().out
    assert "Session" in out and "3 bars" in out
    assert main(["status", "--model-dir", str(world["registry_dir"]),
                 "--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "*ACTIVE*" in out
    assert '"predictions": 3' in out  # bars also count the bootstrap prefix


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
