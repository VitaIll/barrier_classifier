"""End-to-end engine backtest — a profitability *measurement*, not a target.

Drives the live engine (batch serving path — numerically identical to the
rolling live loop, enforced by parity tests) over a historical window via
``ReplaySource``, then reports economic metrics from the resulting trade
ledger and equity curve. No parameter is chosen here; the same code path a
live deployment runs produces the numbers.

Prediction parity against the research cache is asserted first, so any
divergence between the served model and the researched dataset fails loudly
before any P&L is read.

Split semantics (research cache ``research_predictions_1min.parquet``):

- ``test``  — true out-of-sample: never trained on, never used for early
  stopping, threshold, or model selection. The headline window.
- ``val``   — used for early stopping and model selection; in-sample-adjacent.

Usage::

    python scripts/engine_backtest.py                 # test split, 5 bp cost
    python scripts/engine_backtest.py --split val
    python scripts/engine_backtest.py --slippage-bp 5 # extra per-round-trip drag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252; keep output robust regardless.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.engine import Engine, EngineConfig, ModelRegistry, ReplaySource  # noqa: E402

DATASET_DIR = ROOT / "data" / "model_dataset"
RAW_PATH = ROOT / "data" / "raw_data" / "klines_1m.parquet"
CACHE_PATH = DATASET_DIR / "research_predictions_1min.parquet"
MODEL_DIR = ROOT / "models"

MINUTES_PER_YEAR = 365.25 * 24 * 60


def _max_drawdown_log(eq_log: np.ndarray) -> float:
    """Max peak-to-trough drop of a log-equity curve (log-return units)."""
    if len(eq_log) == 0:
        return 0.0
    return float((np.maximum.accumulate(eq_log) - eq_log).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--days", type=float, default=None,
                    help="limit to the first N days of the split")
    ap.add_argument("--cost-per-trade", type=float, default=0.0005,
                    help="round-trip cost fraction (research default 5 bp)")
    ap.add_argument("--slippage-bp", type=float, default=0.0,
                    help="extra round-trip slippage in basis points, added "
                         "to --cost-per-trade")
    args = ap.parse_args()

    cost = args.cost_per_trade + args.slippage_bp * 1e-4

    registry = ModelRegistry(MODEL_DIR)
    if not registry.has_active():
        print("[0] importing research artifacts as v0001 ...")
        registry.import_research_artifacts(DATASET_DIR)
    handle = registry.active()

    cache = pd.read_parquet(CACHE_PATH)
    split = cache[cache["split"] == args.split].sort_values("ts").reset_index(drop=True)
    start = pd.Timestamp(split["ts"].iloc[0]).tz_localize("UTC")
    end_incl = pd.Timestamp(split["ts"].iloc[-1]).tz_localize("UTC")
    if args.days is not None:
        end_incl = min(end_incl, start + pd.Timedelta(days=args.days))
        split = split[split["ts"] <= end_incl.tz_localize(None)].reset_index(drop=True)
    end = end_incl + pd.Timedelta(minutes=1)  # ReplaySource end is exclusive
    span_min = (end_incl - start).total_seconds() / 60.0 + 1
    span_days = span_min / (24 * 60)

    print(f"=== ENGINE BACKTEST — split={args.split!r} ===")
    print(f"model {handle.version}: p_threshold={handle.thresholds.p_threshold:.6f}, "
          f"cost={cost * 1e4:.1f} bp round-trip "
          f"({args.cost_per_trade * 1e4:.1f} research + {args.slippage_bp:.1f} slippage)")
    print(f"window: {start} -> {end_incl}  ({len(split):,} bars, {span_days:.1f} days)")
    if args.split == "test":
        print("test split is out-of-sample: not trained on, not used for early "
              "stopping, threshold derivation, or model selection")
    else:
        print("NOTE: val split was used for early stopping and model selection "
              "— treat these numbers as in-sample-adjacent")

    store_path = ROOT / "runtime" / f"backtest_{args.split}.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(store_path) + suffix).unlink(missing_ok=True)
    cfg = EngineConfig(
        model_dir=MODEL_DIR, store_path=store_path, feature_mode="batch",
        cost_per_trade=cost, log_every_bars=20_000,
    )
    engine = Engine(cfg, source=ReplaySource(RAW_PATH, start=start, end=end))
    report = engine.run()
    print()
    print(report.summary())

    # --- Prediction parity vs the research cache (fail loud) ----------------
    conn = engine.store.read_connection()
    eng_p = pd.read_sql_query("SELECT ts_ms, p FROM predictions ORDER BY ts_ms", conn)
    eng_p["ts"] = pd.to_datetime(eng_p.pop("ts_ms"), unit="ms")
    merged = split[["ts", "p"]].merge(eng_p, on="ts", suffixes=("_research", "_engine"))
    dp = np.abs(merged["p_research"].to_numpy() - merged["p_engine"].to_numpy())
    parity_ok = len(merged) == len(split) and float(dp.max()) <= 1e-12
    print(f"\n[parity] engine vs research cache: {len(merged):,}/{len(split):,} rows, "
          f"max|Δp|={dp.max():.2e} -> {'OK (bit-exact)' if parity_ok else 'DIVERGENCE'}")
    if not parity_ok:
        print("ABORT: served model diverges from the researched dataset — "
              "P&L below would not measure the researched strategy.")
        engine.close()
        return 1

    # --- Economic metrics ----------------------------------------------------
    trades = report.trades.copy()
    equity = report.equity.copy()
    n_trades = len(trades)

    realized_log = float(report.realized_cum_log_return)
    open_mtm = float(report.unrealized_log_return)
    total_log = realized_log + open_mtm
    equity_mult = float(np.exp(total_log))
    ann_log = total_log * (MINUTES_PER_YEAR / span_min)

    eq = equity.set_index("ts")["equity"].sort_index()   # realized + unrealized, log units
    daily_pnl = eq.resample("1D").last().ffill().diff().dropna()
    sharpe = (
        float(daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(365.25))
        if len(daily_pnl) > 2 and daily_pnl.std(ddof=1) > 0 else float("nan")
    )
    mdd_log = _max_drawdown_log(eq.to_numpy())

    if n_trades:
        net = trades["weighted_net_log_return"].to_numpy()
        gross_sum = float((trades["gross_log_return"] * trades["size"]).sum())
        cost_drag = gross_sum - float(net.sum())
        hit_rate = float((net > 0).mean()) * 100.0
        avg_hold_min = float(
            (trades["ts_exit"] - trades["ts_entry"]).dt.total_seconds().mean() / 60.0
        )
        sized_legs_per_day = float(2.0 * trades["size"].sum() / span_days)
    else:
        gross_sum = cost_drag = 0.0
        hit_rate = avg_hold_min = sized_legs_per_day = float("nan")

    avg_n_open = float(equity["n_open"].mean())
    avg_gross = float(equity["gross_size"].mean())

    print("\n--- Performance ---------------------------------------------------")
    print(f"  closed trades          : {n_trades:,}  ({n_trades / span_days:.1f}/day, "
          f"avg hold {avg_hold_min:.1f} min)  +{report.n_open_positions} still open")
    print(f"  realized log-return    : {realized_log:+.5f}   "
          f"(open positions MTM {open_mtm:+.5f})")
    print(f"  total return           : {(equity_mult - 1) * 100:+.3f}%  "
          f"(equity multiple x{equity_mult:.4f}, incl. open MTM)")
    print(f"  annualized (linear log): {(np.exp(ann_log) - 1) * 100:+.2f}%")
    print(f"  Sharpe (daily, rf=0)   : {sharpe:.2f}")
    print(f"  max drawdown           : {mdd_log:.5f} log "
          f"({(1 - np.exp(-mdd_log)) * 100:.3f}%)")
    print(f"  hit rate               : {hit_rate:.1f}%")
    print(f"  avg open positions     : {avg_n_open:.1f}  "
          f"(avg gross size {avg_gross:.3f} of max 1.0)")
    print(f"  turnover               : {sized_legs_per_day:.3f} sized legs/day")
    print(f"  cost drag              : {cost_drag:.5f} log at {cost * 1e4:.1f} bp "
          f"({cost_drag / gross_sum * 100 if gross_sum else float('nan'):.0f}% of gross)"
          if n_trades else "  cost drag              : n/a")

    # --- Cost / slippage sensitivity (linear in sized round-trips) -----------
    if n_trades:
        size_sum = float(trades["size"].sum())
        print("\n--- Cost sensitivity (round-trip, realized trades only) -----------")
        for bp in (0.0, 2.5, 5.0, 7.5, 10.0, 20.0):
            net_at = gross_sum - size_sum * bp * 1e-4
            print(f"  {bp:5.1f} bp : net {net_at:+.5f} log  "
                  f"({(np.exp(net_at) - 1) * 100:+.3f}%)")
        breakeven_bp = gross_sum / size_sum * 1e4 if size_sum else float("nan")
        print(f"  breakeven cost: {breakeven_bp:.1f} bp round-trip")

    # --- Per-month breakdown --------------------------------------------------
    if n_trades:
        print("\n--- Per-month breakdown (by exit) ---------------------------------")
        tm = trades.copy()
        tm["month"] = tm["ts_exit"].dt.to_period("M")
        g = tm.groupby("month")["weighted_net_log_return"].agg(["count", "sum"])
        wins = tm.groupby("month")["weighted_net_log_return"].apply(lambda s: (s > 0).mean())
        for month, row in g.iterrows():
            print(f"  {month}: {int(row['count']):>4} trades, "
                  f"net {row['sum']:+.5f} log ({(np.exp(row['sum']) - 1) * 100:+.3f}%), "
                  f"hit {wins.loc[month] * 100:.0f}%")

    # --- Benchmark -------------------------------------------------------------
    raw = pd.read_parquet(RAW_PATH, columns=["close"])
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    win = raw[(raw.index >= start) & (raw.index <= end_incl)]
    bh_log = float(np.log(win["close"].iloc[-1] / win["close"].iloc[0]))
    print("\n--- Benchmark ------------------------------------------------------")
    print(f"  BTC buy-and-hold (1.0 size): {bh_log:+.5f} log "
          f"({(np.exp(bh_log) - 1) * 100:+.2f}%)")
    print(f"  strategy avg gross exposure: {avg_gross:.3f} — compare per unit of "
          f"exposure, not headline vs headline")

    engine.close()
    print("\n(This backtest is a measurement, not a target — reported as-is.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
