"""Run strategy variants for the three improvements (P1, P3) on the current
model, save per-variant artifacts under data/model_dataset/strategy/variants/,
and emit a single comparison summary.

Variants:
  baseline                 : TOP_Q=0.95, exit_tp_or_expiry (matches nb05 current production)
  P1_top1pct               : TOP_Q=0.99, exit_tp_or_expiry
  P3_letrun                : TOP_Q=0.95, let-winners-run (HOLD=P_95, no SL)
  P3_letrun_sl2phi         : TOP_Q=0.95, let-winners-run (HOLD=P_95, SL=2*phi)
  P1_P3_combined           : TOP_Q=0.99, let-winners-run (HOLD=P_99, no SL)
  P1_P3_combined_sl2phi    : TOP_Q=0.99, let-winners-run (HOLD=P_99, SL=2*phi)

Each variant reuses the same simulator + cache + raw bars, so the only
moving parts are the score gate threshold and the exit policy.

P2 (longer feature windows) is a separate workflow — requires retraining,
handled by a dedicated script.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.thresholds import (  # noqa: E402
    derive_top_q_threshold,
)
from src.strategy.baseline import COST_PER_TRADE  # noqa: E402
from src.features.config import M as M_CFG, PHI as PHI_CFG  # noqa: E402
from src.strategy.cache import (  # noqa: E402
    augment_cache_with_boundary_ohlc,
    augment_cache_with_r_realized,
)
from src.strategy.policy import (  # noqa: E402
    RiskConfig,
    StrategySpec,
    exit_tp_or_expiry,
    gate_score_above,
    make_exit_let_winners_run,
    make_exit_let_winners_run_monotonic,
    score_raw_p,
    size_clip,
    size_constant,
)
from src.strategy.simulator import SimConfig, simulate  # noqa: E402

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DATASET_DIR = ROOT / "data" / "model_dataset"
CACHE_PATH = DATASET_DIR / "research_predictions_1min.parquet"
RAW_PATH = ROOT / "data" / "raw_data" / "klines_1m.parquet"
TRAIN_ART_PATH = DATASET_DIR / "analytics" / "train_scores_unc_1min.parquet"
VAL_TEST_ART_PATH = DATASET_DIR / "analytics" / "val_test_ve_unc_1min.parquet"
OUT_ROOT = DATASET_DIR / "strategy" / "variants"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Constants (must match nb05)
# -----------------------------------------------------------------------------
M = int(M_CFG)
PHI = float(PHI_CFG)
LOT_SIZE = 0.02
MAX_CONCURRENT = 50
COST = COST_PER_TRADE
MAX_HORIZON_BOUNDARIES = 1_000_000  # effectively no time-based expiry


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and augment the live cache + raw + training arts."""
    print("[load] cache ...", end="", flush=True)
    t0 = time.perf_counter()
    cache = pd.read_parquet(CACHE_PATH)
    ve = pd.read_parquet(VAL_TEST_ART_PATH)
    cache = cache.merge(ve, on=["k", "split"], how="left")
    assert cache["knowledge_unc"].notna().all(), "VE merge left NaNs"
    raw_bars = pd.read_parquet(RAW_PATH, columns=["open", "high", "low", "close"])
    if raw_bars.index.tz is not None:
        raw_bars.index = raw_bars.index.tz_localize(None)
    cache = augment_cache_with_boundary_ohlc(cache, raw_bars)
    cache = augment_cache_with_r_realized(cache, raw_bars, M=M)
    print(f" {time.perf_counter()-t0:.1f}s  ({len(cache):,} rows)")
    print("[load] train scores ...", end="", flush=True)
    t0 = time.perf_counter()
    train_art = pd.read_parquet(TRAIN_ART_PATH)
    print(f" {time.perf_counter()-t0:.1f}s  ({len(train_art):,} rows)")
    sim_cache = (
        pd.concat([cache[cache["split"] == "val"], cache[cache["split"] == "test"]])
        .sort_values("ts")
        .reset_index(drop=True)
    )
    return sim_cache, raw_bars, train_art, cache


def _build_spec(
    name: str,
    p_threshold: float,
    exit_policy,
    description: str,
) -> StrategySpec:
    """Builds a 1-min selective-entry spec with the given exit policy."""
    return StrategySpec(
        name=name,
        requires=("ve_diag",),
        score_fn=score_raw_p,
        entry_gates=(lambda s, t=p_threshold: gate_score_above(s, t),),
        sizer=lambda s, sz=LOT_SIZE: size_clip(size_constant(s, default=sz), max_size=1.0),
        exit_policy=exit_policy,
        bulk_close=lambda s: None,
        risk=RiskConfig(
            cost_per_trade=COST,
            max_open_positions=MAX_CONCURRENT,
            max_gross_size=MAX_CONCURRENT * LOT_SIZE + 1e-6,
            max_horizon_boundaries=MAX_HORIZON_BOUNDARIES,
            position_mtm_floor_log_return=None,
        ),
        description=description,
    )


def _build_p_map(sim_cache: pd.DataFrame) -> "pd.Series[float]":
    """ts -> p mapping for let-winners-run lookups."""
    return pd.Series(sim_cache["p"].to_numpy(), index=pd.DatetimeIndex(sim_cache["ts"]))


def _summarize_run(result, raw_bars: pd.DataFrame, params: dict, name: str) -> dict:
    eq = result.equity.copy()
    eq["ts"] = pd.to_datetime(eq["ts"])
    span_days = (eq["ts"].max() - eq["ts"].min()).total_seconds() / 86400.0
    daily = eq.set_index("ts")["realized_cum"].resample("1D").last().ffill()
    daily_ret = daily.diff().fillna(daily.iloc[0])
    annualized_realized = float(daily.iloc[-1]) * (365.0 / span_days)
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(365.0))
        if daily_ret.std() > 1e-18 else float("nan")
    )
    total_eq = (eq["realized_cum"] + eq["unrealized"]).to_numpy()
    peaks = np.maximum.accumulate(total_eq)
    dd_series = total_eq - peaks
    max_paper_dd = float(-dd_series.min())
    worst_dd_idx = int(np.argmin(dd_series))
    GROSS_CAP = MAX_CONCURRENT * LOT_SIZE
    avg_util = float((eq["gross_size"] / GROSS_CAP).mean())
    peak_util = float((eq["gross_size"] / GROSS_CAP).max())

    # BTC b&h over same span
    btc_slice = raw_bars.loc[
        (raw_bars.index >= eq["ts"].min()) & (raw_bars.index <= eq["ts"].max()), "close"
    ]
    btc_log_return = float(np.log(btc_slice.iloc[-1] / btc_slice.iloc[0]))
    btc_annualized = btc_log_return * (365.0 / span_days)

    # Exit reason distribution
    exit_reasons = (
        result.closed["exit_reason"].value_counts().to_dict() if len(result.closed) else {}
    )

    # Per-trade economics
    if len(result.closed) > 0:
        gross = result.closed["gross_log_return"].astype(float)
        net = gross - COST
        # Aggregate hold_minutes
        ts_entry = pd.to_datetime(result.closed["ts_entry"])
        ts_exit = pd.to_datetime(result.closed["ts_exit"])
        hold_min = (ts_exit - ts_entry).dt.total_seconds() / 60.0
        per_trade = {
            "n_closed": int(len(result.closed)),
            "gross_log_return_mean": float(gross.mean()),
            "gross_log_return_median": float(gross.median()),
            "gross_log_return_max": float(gross.max()),
            "gross_log_return_min": float(gross.min()),
            "net_log_return_sum": float(net.sum()),
            "win_count_net": int((net > 0).sum()),
            "loss_count_net": int((net <= 0).sum()),
            "hold_min_median": float(hold_min.median()),
            "hold_min_mean": float(hold_min.mean()),
            "hold_min_p90": float(hold_min.quantile(0.90)),
            "hold_min_max": float(hold_min.max()),
        }
    else:
        per_trade = {"n_closed": 0}

    summary = {
        "variant": name,
        "span_days": span_days,
        "n_signals": int(eq["opened_this_step"].sum()),
        "n_closed": int(len(result.closed)),
        "n_open_at_end": int(eq["n_open"].iloc[-1]),
        "exit_reasons": exit_reasons,
        "realized_log_return": float(daily.iloc[-1]),
        "realized_annualized": annualized_realized,
        "unrealized_at_end": float(eq["unrealized"].iloc[-1]),
        "total_log_return": float(daily.iloc[-1]) + float(eq["unrealized"].iloc[-1]),
        "sharpe_realized_daily": sharpe,
        "max_paper_dd": max_paper_dd,
        "worst_dd_ts": str(eq["ts"].iloc[worst_dd_idx]),
        "avg_utilization": avg_util,
        "peak_utilization": peak_util,
        "btc_log_return": btc_log_return,
        "btc_annualized": btc_annualized,
        "alpha_over_btc": (float(daily.iloc[-1]) + float(eq["unrealized"].iloc[-1])) - btc_log_return,
        "per_trade": per_trade,
        "parameters": params,
    }
    return summary


def _print_headline(s: dict) -> None:
    p = s["parameters"]
    print()
    print("=" * 76)
    print(f"VARIANT: {s['variant']}")
    print("=" * 76)
    print(f"  TOP_Q={p.get('TOP_Q')}  P_th={p.get('P_threshold'):.4f}  "
          f"exit={p.get('exit_policy_name')}  "
          f"hold_th={p.get('hold_threshold')}  sl_log={p.get('sl_log_return')}")
    print(f"  signals={s['n_signals']:,}  closed={s['n_closed']:,}  open_at_end={s['n_open_at_end']}")
    print(f"  exit reasons: {s['exit_reasons']}")
    print(f"  realized {s['realized_log_return']*100:+.3f}%  "
          f"unrealized {s['unrealized_at_end']*100:+.3f}%  "
          f"TOTAL {s['total_log_return']*100:+.3f}%")
    print(f"  realized annualized {s['realized_annualized']*100:+.2f}%  "
          f"sharpe daily {s['sharpe_realized_daily']:.2f}  "
          f"max DD {s['max_paper_dd']*100:.2f}%")
    print(f"  vs BTC {s['btc_log_return']*100:+.2f}% over span  → alpha {s['alpha_over_btc']*100:+.2f}%")
    pt = s["per_trade"]
    if pt.get("n_closed", 0) > 0:
        print(f"  per-trade: gross median {pt['gross_log_return_median']*1e4:+.2f}bp  "
              f"max {pt['gross_log_return_max']*1e4:+.2f}bp  "
              f"min {pt['gross_log_return_min']*1e4:+.2f}bp  "
              f"hold median {pt['hold_min_median']:.0f}min  "
              f"p90 {pt['hold_min_p90']:.0f}min")
    print(f"  util avg {s['avg_utilization']*100:.1f}%  peak {s['peak_utilization']*100:.1f}%")


def main() -> None:
    sim_cache, raw_bars, train_art, _ = _load_data()
    p_train = train_art["p_train"].to_numpy()

    P_95 = float(derive_top_q_threshold(p_train, q=0.95))
    P_99 = float(derive_top_q_threshold(p_train, q=0.99))
    print(f"\nThresholds  P_95={P_95:.4f}  P_99={P_99:.4f}  PHI={PHI:.4f}  COST={COST:.4f}")

    # Pre-build the ts -> p map for let-winners-run lookups.
    p_map = _build_p_map(sim_cache)

    cfg = SimConfig(M=M)

    # Define variants.
    # No SL anywhere: user constraint "do not release into loss; if we hold
    # and run into loss we wait as we do with other positions." The
    # let-winners-run policies only ever exit when the position is in the
    # TP zone and conviction has dropped.
    variants: list[tuple[str, float, callable, dict]] = [
        (
            "baseline",
            P_95,
            exit_tp_or_expiry,
            {"TOP_Q": 0.95, "P_threshold": P_95, "exit_policy_name": "exit_tp_or_expiry",
             "hold_threshold": None, "sl_log_return": None},
        ),
        (
            "P1_top1pct",
            P_99,
            exit_tp_or_expiry,
            {"TOP_Q": 0.99, "P_threshold": P_99, "exit_policy_name": "exit_tp_or_expiry",
             "hold_threshold": None, "sl_log_return": None},
        ),
        (
            "P3_letrun_thr_p95",
            P_95,
            make_exit_let_winners_run(p_map, hold_threshold=P_95, sl_log_return=None),
            {"TOP_Q": 0.95, "P_threshold": P_95,
             "exit_policy_name": "make_exit_let_winners_run (threshold)",
             "hold_threshold": P_95, "sl_log_return": None},
        ),
        (
            "P1_P3_thr_p99",
            P_99,
            make_exit_let_winners_run(p_map, hold_threshold=P_99, sl_log_return=None),
            {"TOP_Q": 0.99, "P_threshold": P_99,
             "exit_policy_name": "make_exit_let_winners_run (threshold)",
             "hold_threshold": P_99, "sl_log_return": None},
        ),
        (
            "P1_P3_monotonic",
            P_99,
            make_exit_let_winners_run_monotonic(p_map),
            {"TOP_Q": 0.99, "P_threshold": P_99,
             "exit_policy_name": "make_exit_let_winners_run_monotonic",
             "hold_threshold": "p_now > p_max_so_far (strict monotonic growth)",
             "sl_log_return": None},
        ),
    ]

    summaries: list[dict] = []
    for name, p_thr, exit_fn, params in variants:
        print(f"\n>>> Running variant: {name}")
        spec = _build_spec(
            name=name, p_threshold=p_thr, exit_policy=exit_fn,
            description=f"variant={name}",
        )
        t0 = time.perf_counter()
        result = simulate(sim_cache, raw_bars, spec, config=cfg)
        dt = time.perf_counter() - t0
        print(f"  sim {dt:.1f}s")
        s = _summarize_run(result, raw_bars, params, name)
        summaries.append(s)
        _print_headline(s)

        # Save per-variant artifacts.
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(s, f, indent=2, default=str)
        if len(result.closed) > 0:
            closed = result.closed.copy()
            # add hold_minutes for downstream parity with nb05
            ts_entry = pd.to_datetime(closed["ts_entry"])
            ts_exit = pd.to_datetime(closed["ts_exit"])
            closed["hold_minutes"] = (ts_exit - ts_entry).dt.total_seconds() / 60.0
            closed.to_parquet(out_dir / "closed_trades.parquet", index=False)

    # Write cross-variant comparison.
    print()
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"{'variant':<30}  {'TOP_Q':>5}  {'n_sig':>6}  {'n_cls':>6}  {'real%':>8}  "
          f"{'unr%':>8}  {'tot%':>8}  {'a-BTC%':>8}  {'DD%':>6}")
    for s in summaries:
        p = s["parameters"]
        print(
            f"{s['variant']:<30}  "
            f"{p.get('TOP_Q', '-'):>5}  "
            f"{s['n_signals']:>6,}  "
            f"{s['n_closed']:>6,}  "
            f"{s['realized_log_return']*100:>+8.3f}  "
            f"{s['unrealized_at_end']*100:>+8.3f}  "
            f"{s['total_log_return']*100:>+8.3f}  "
            f"{s['alpha_over_btc']*100:>+8.2f}  "
            f"{s['max_paper_dd']*100:>6.2f}"
        )
    with open(OUT_ROOT / "comparison_summary.json", "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nSaved: {OUT_ROOT / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
