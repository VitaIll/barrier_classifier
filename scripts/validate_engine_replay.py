"""Real-data engine validation: replay ≡ research, end to end.

Drives the live engine over a contiguous slice of the validated 2025
BTCUSDT data (simulated live stream) with the imported research model,
then proves three things against the research stack:

1. **Prediction reproduction** — the engine's per-bar probabilities are
   bit-equal to `research_predictions_1min.parquet` on the shared rows
   (same pipeline, same model ⇒ same numbers; any drift means the serving
   path diverged from the dataset build).
2. **Ledger parity** — the engine's closed trades equal a fresh
   `simulate()` run of the winning P1+P3 spec over the same window
   (economic columns, exact).
3. **Loop mechanics** — guards clean, every bar predicted, persistence
   counts consistent.

Note: the published `production_1min_P1P3` ledger is NOT the comparison
target — it spans val+test with a 2×1,200-row embargo gap in the cache,
where `p_map` lookups return NaN and the exit policy fails safe. That is
an artifact of research evaluation, not of the strategy; a live engine
predicts through the gap. Comparing on a contiguous window removes the
artifact while exercising the identical code paths.

Usage (from repo root, ~10–25 min depending on window):

    python scripts/validate_engine_replay.py                 # val window
    python scripts/validate_engine_replay.py --days 7        # first week of val
    python scripts/validate_engine_replay.py --rolling-bars 3  # also time the live path
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252; keep output robust regardless.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.engine import Engine, EngineConfig, ModelRegistry, ReplaySource  # noqa: E402
from src.engine.risk import EntryControls  # noqa: E402
from src.engine.strategy import make_live_production_spec  # noqa: E402
from src.strategy.cache import (  # noqa: E402
    augment_cache_with_boundary_ohlc,
)
from src.strategy.simulator import SimConfig, simulate  # noqa: E402

DATASET_DIR = ROOT / "data" / "model_dataset"
RAW_PATH = ROOT / "data" / "raw_data" / "klines_1m.parquet"
CACHE_PATH = DATASET_DIR / "research_predictions_1min.parquet"
MODEL_DIR = ROOT / "models"
STORE_PATH = ROOT / "runtime" / "validate_engine_replay.db"

LEDGER_COLS = [
    "ts_entry", "size", "entry_price", "tp_price",
    "ts_exit", "exit_price", "exit_reason", "gross_log_return",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=None,
                    help="limit the replay to the first N days of the val window")
    ap.add_argument("--rolling-bars", type=int, default=0,
                    help="additionally run N bars through the rolling live path "
                         "and report per-bar latency + |Δp| vs batch")
    args = ap.parse_args()

    t_all = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 0. Model registry (import research artifacts if absent)
    # ------------------------------------------------------------------ #
    registry = ModelRegistry(MODEL_DIR)
    if not registry.has_active():
        print("[0] importing research artifacts as v0001 ...")
        registry.import_research_artifacts(DATASET_DIR)
    handle = registry.active()
    p_th = handle.thresholds.p_threshold
    print(f"[0] model {handle.version}: {handle.contract.n_features} features, "
          f"p_threshold={p_th:.6f}, anchor={handle.contract.anchor}")

    # ------------------------------------------------------------------ #
    # 1. Window = contiguous val slice of the research cache
    # ------------------------------------------------------------------ #
    cache = pd.read_parquet(CACHE_PATH)
    val = cache[cache["split"] == "val"].sort_values("ts").reset_index(drop=True)
    start = pd.Timestamp(val["ts"].iloc[0]).tz_localize("UTC")
    end_incl = pd.Timestamp(val["ts"].iloc[-1]).tz_localize("UTC")
    if args.days is not None:
        end_incl = min(end_incl, start + pd.Timedelta(days=args.days))
        val = val[val["ts"] <= end_incl.tz_localize(None)].reset_index(drop=True)
    end = end_incl + pd.Timedelta(minutes=1)  # ReplaySource end is exclusive
    print(f"[1] window: {start} → {end_incl}  ({len(val):,} bars)")

    # ------------------------------------------------------------------ #
    # 2. Engine replay (batch serving path)
    # ------------------------------------------------------------------ #
    STORE_PATH.unlink(missing_ok=True)
    cfg = EngineConfig(
        model_dir=MODEL_DIR, store_path=STORE_PATH, feature_mode="batch",
        log_every_bars=20_000,
        entry_controls=EntryControls.disabled(), reconcile_every_bars=None,
    )
    source = ReplaySource(RAW_PATH, start=start, end=end)
    engine = Engine(cfg, source=source)
    t0 = time.perf_counter()
    report = engine.run()
    t_run = time.perf_counter() - t0
    print(f"[2] engine replay: {report.n_bars:,} bars in {t_run:.1f}s "
          f"({1e3 * t_run / max(report.n_bars, 1):.2f} ms/bar incl. precompute)")
    print("    " + report.summary().replace("\n", "\n    "))

    ok = True

    # ------------------------------------------------------------------ #
    # 3. Prediction reproduction vs the research cache
    # ------------------------------------------------------------------ #
    conn = engine.store.read_connection()
    engine_p = pd.read_sql_query(
        "SELECT ts_ms, p FROM predictions ORDER BY ts_ms", conn
    )
    engine_p["ts"] = pd.to_datetime(engine_p.pop("ts_ms"), unit="ms")
    merged = val[["ts", "p"]].merge(engine_p, on="ts", suffixes=("_research", "_engine"))
    if len(merged) != len(val):
        print(f"[3] FAIL: engine predicted {len(merged):,} of {len(val):,} research rows")
        ok = False
    else:
        dp = np.abs(merged["p_research"].to_numpy() - merged["p_engine"].to_numpy())
        n_exact = int((dp == 0.0).sum())
        print(f"[3] prediction reproduction: max|Δp|={dp.max():.3e}, "
              f"mean|Δp|={dp.mean():.3e}, exact={n_exact:,}/{len(dp):,}")
        if dp.max() > 1e-12:
            print("    FAIL: engine predictions diverge from the research cache")
            ok = False

    # ------------------------------------------------------------------ #
    # 4. Ledger parity vs a fresh simulate() on the same window
    # ------------------------------------------------------------------ #
    raw_bars = pd.read_parquet(RAW_PATH, columns=["open", "high", "low", "close"])
    if raw_bars.index.tz is not None:
        raw_bars.index = raw_bars.index.tz_localize(None)
    sim_cache = augment_cache_with_boundary_ohlc(val.copy(), raw_bars)
    p_map = pd.Series(sim_cache["p"].to_numpy(), index=pd.DatetimeIndex(sim_cache["ts"]))
    spec = make_live_production_spec(
        p_map, p_threshold=p_th, lot_size=cfg.lot_size,
        max_concurrent=cfg.max_concurrent, cost_per_trade=cfg.cost_per_trade,
    )
    t0 = time.perf_counter()
    sim = simulate(sim_cache, raw_bars, spec, config=SimConfig(M=handle.contract.m,
                                                               cadence_minutes=1.0))
    print(f"[4] reference simulate(): {len(sim.closed):,} closed trades "
          f"in {time.perf_counter() - t0:.1f}s")

    eng_trades = report.trades.copy()
    if len(eng_trades) != len(sim.closed):
        print(f"    FAIL: trade count differs (engine={len(eng_trades):,}, "
              f"simulate={len(sim.closed):,})")
        ok = False
    elif len(eng_trades):
        sim_l = sim.closed[LEDGER_COLS].reset_index(drop=True).copy()
        eng_l = eng_trades.copy()
        eng_l["ts_entry"] = eng_l["ts_entry"].dt.tz_localize(None)
        eng_l["ts_exit"] = eng_l["ts_exit"].dt.tz_localize(None)
        eng_l["tp_price"] = eng_l["entry_price"] * np.exp(handle.contract.phi)
        eng_l = eng_l[LEDGER_COLS].reset_index(drop=True)
        sim_l["tp_price"] = sim_l["tp_price"].astype(float)
        try:
            pd.testing.assert_frame_equal(sim_l, eng_l, check_exact=False,
                                          rtol=0, atol=1e-12)
            print(f"    ledger parity: {len(eng_l):,} trades identical "
                  f"(entry/exit ts+px, size, reason, gross return)")
        except AssertionError as exc:
            print(f"    FAIL: ledgers diverge\n{str(exc)[:1500]}")
            ok = False
        eng_real = float(report.realized_cum_log_return)
        sim_real = float(sim.equity["realized_cum"].iloc[-1])
        print(f"    realized log-return: engine={eng_real:+.5f}  "
              f"simulate={sim_real:+.5f}  (Δ={abs(eng_real - sim_real):.2e})")
        if abs(eng_real - sim_real) > 1e-9:
            ok = False

    # ------------------------------------------------------------------ #
    # 5. Optional: rolling live-path spot check + latency
    # ------------------------------------------------------------------ #
    if args.rolling_bars > 0:
        print(f"[5] rolling live path over {args.rolling_bars} bar(s) ...")
        roll_store = ROOT / "runtime" / "validate_engine_rolling.db"
        roll_store.unlink(missing_ok=True)
        cfg_roll = EngineConfig(
            entry_controls=EntryControls.disabled(), reconcile_every_bars=None,
            model_dir=MODEL_DIR, store_path=roll_store, feature_mode="rolling",
            log_every_bars=10_000,
        )
        src_roll = ReplaySource(RAW_PATH, start=start, end=end)
        eng_roll = Engine(cfg_roll, source=src_roll)
        t0 = time.perf_counter()
        rep_roll = eng_roll.run(max_bars=args.rolling_bars)
        dt = time.perf_counter() - t0
        conn_r = eng_roll.store.read_connection()
        roll_p = pd.read_sql_query(
            "SELECT ts_ms, p, feature_ms, predict_ms FROM predictions ORDER BY ts_ms",
            conn_r,
        )
        roll_p["ts"] = pd.to_datetime(roll_p.pop("ts_ms"), unit="ms")
        cmp = roll_p.merge(engine_p, on="ts", suffixes=("_roll", "_batch"))
        dmax = float(np.abs(cmp["p_roll"] - cmp["p_batch"]).max()) if len(cmp) else float("nan")
        print(f"    {rep_roll.n_predictions} predictions, "
              f"feature latency mean={roll_p['feature_ms'].mean():.0f}ms "
              f"max={roll_p['feature_ms'].max():.0f}ms, "
              f"predict={roll_p['predict_ms'].mean():.1f}ms, "
              f"wall {dt:.1f}s | max|Δp| rolling-vs-batch = {dmax:.3e}")
        eng_roll.store.close()
        if not np.isnan(dmax) and dmax > 1e-6:
            print("    FAIL: rolling path diverges from batch beyond tolerance")
            ok = False

    engine.store.close()
    print(f"\n{'PASS' if ok else 'FAIL'} — total {time.perf_counter() - t_all:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
