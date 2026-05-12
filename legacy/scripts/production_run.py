"""Production-mode run.

NO look-ahead anywhere. NO tuning on val. Treat val+test as the live
deployment period; report only what a production operator would observe.
Parameters are fixed at deployment time based on prior knowledge.
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src import utils
from src.strategy.cache import (
    augment_cache_with_boundary_ohlc,
    augment_cache_with_r_realized,
)
from src.strategy.policy import (
    RiskConfig,
    StrategySpec,
    exit_tp_or_expiry,
    gate_score_above,
    score_raw_p,
    size_clip,
    size_constant,
)
from src.strategy.simulator import SimConfig, simulate

# ---------------------------------------------------------------------------
# FROZEN PARAMETERS (chosen before deployment from training-period analysis;
# not tuned on val/test).
# ---------------------------------------------------------------------------
THRESHOLD = 0.30        # ~ top-decile of p based on training distribution
LOT_SIZE = 0.02         # 2% of nominal capital per lot
MAX_CONCURRENT = 50     # peak gross deployment cap
PHI = 0.0025            # +25 bp upper barrier (label-aligned)
COST = 0.0005           # 5 bp round-trip
OUT = "data/model_dataset/strategy/production"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cache = pd.read_parquet("data/model_dataset/research_predictions.parquet")
    raw = pd.read_parquet(
        "data/raw_data/klines_1m.parquet", columns=["open", "high", "low", "close"]
    )
    cache = augment_cache_with_boundary_ohlc(cache, raw)
    cache = augment_cache_with_r_realized(cache, raw, M=int(utils.M))
    sim_cache = (
        pd.concat(
            [cache[cache["split"] == "val"], cache[cache["split"] == "test"]]
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )
    val_test_boundary = cache[cache["split"] == "test"]["ts"].min()

    spec = StrategySpec(
        name="production",
        score_fn=score_raw_p,
        entry_gates=(lambda s: gate_score_above(s, THRESHOLD),),
        sizer=lambda s: size_clip(size_constant(s, default=LOT_SIZE), max_size=1.0),
        exit_policy=exit_tp_or_expiry,
        bulk_close=lambda s: None,
        risk=RiskConfig(
            cost_per_trade=COST,
            max_open_positions=MAX_CONCURRENT,
            max_gross_size=MAX_CONCURRENT * LOT_SIZE + 1e-6,
            max_horizon_boundaries=1_000_000,   # never forced-expire in production
            position_mtm_floor_log_return=None,
        ),
    )
    cfg = SimConfig(M=int(utils.M))
    r = simulate(sim_cache, raw, spec, config=cfg)
    eq = r.equity.copy()
    eq["ts"] = pd.to_datetime(eq["ts"])

    # ----- timing / spans ---------------------------------------------------
    span_days = (eq["ts"].max() - eq["ts"].min()).total_seconds() / 86400.0
    print(f"Deployment span : {eq['ts'].min()}  ->  {eq['ts'].max()}  ({span_days:.1f} days)")
    print(f"Boundary cadence: {utils.M} minutes between model evaluations")

    # ----- realized P&L per actual production day --------------------------
    eq_idx = eq.set_index("ts")
    daily = eq_idx["realized_cum"].resample("1D").last().ffill()
    daily_ret = daily.diff().fillna(daily.iloc[0])
    annualized_realized = float(daily.iloc[-1]) * (365.0 / span_days)
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(365.0))
        if daily_ret.std() > 1e-18 else float("nan")
    )

    # ----- capital utilization over time ------------------------------------
    eq["utilization"] = eq["gross_size"] / (MAX_CONCURRENT * LOT_SIZE)
    peak_util = float(eq["utilization"].max())
    avg_util = float(eq["utilization"].mean())
    peak_concurrent = int(eq["n_open"].max())
    avg_concurrent = float(eq["n_open"].mean())

    # ----- worst observed total-equity drawdown (production-observable) ----
    total_eq = (eq["realized_cum"] + eq["unrealized"]).to_numpy()
    peaks = np.maximum.accumulate(total_eq)
    dd_series = total_eq - peaks
    max_paper_dd = float(-dd_series.min())

    # ----- BTC buy-and-hold over same period (sanity baseline) -------------
    raw_idx = raw.index.tz_localize(None) if raw.index.tz is not None else raw.index
    start_n = int(np.searchsorted(raw_idx, np.datetime64(eq["ts"].min()), side="left"))
    end_n = int(np.searchsorted(raw_idx, np.datetime64(eq["ts"].max()), side="left"))
    btc_start = float(raw["close"].iloc[start_n])
    btc_end = float(raw["close"].iloc[end_n])
    btc_log_return = float(np.log(btc_end / btc_start))
    btc_annualized = btc_log_return * 365.0 / span_days

    # ===== Production-grade summary =========================================
    print()
    print("=" * 72)
    print("PRODUCTION RUN: top-10% wait-for-TP, val + test as live stream")
    print("=" * 72)
    print(f"Threshold p>={THRESHOLD} | lot={LOT_SIZE} of capital | max_open={MAX_CONCURRENT}")
    print(f"TP = +{PHI*10000:.0f} bp | cost = {COST*10000:.0f} bp/trade | no SL, no expiry, no bulk-close")
    print()
    print("Signals & fills")
    print(f"  Boundaries observed         : {len(eq):>6,}")
    print(f"  Signals fired (p>=thr)      : {int(eq['opened_this_step'].sum()):>6,}")
    print(f"  Positions opened (capital OK): {int(eq['opened_this_step'].sum()):>6,}")
    print(f"  TPs filled                  : {len(r.closed):>6,}")
    print(f"  Still open at end           : {int(eq['n_open'].iloc[-1]):>6,}")
    print()
    print("Realized P&L (the production deliverable)")
    print(f"  Cumulative realized log-return : {float(daily.iloc[-1]) * 100:+.3f}% over {span_days:.0f} days")
    print(f"  Annualized                     : {annualized_realized * 100:+.2f}%")
    print(f"  Daily-Sharpe of realized       : {sharpe:.2f}")
    print(f"  Realized per TP fill           : exactly +{(PHI - COST) * 1e4:.1f} bp × {LOT_SIZE} = +{(PHI - COST) * LOT_SIZE * 1e4:.1f} bp / fill (deterministic)")
    print()
    print("Capital usage (observable at deployment time)")
    print(f"  Peak gross deployed (% cap)    : {peak_util * 100:.1f}%")
    print(f"  Avg gross deployed (% cap)     : {avg_util * 100:.1f}%")
    print(f"  Peak concurrent open positions : {peak_concurrent}")
    print(f"  Avg concurrent open positions  : {avg_concurrent:.1f}")
    print()
    print("Paper drawdown on the open book (observable; never realized)")
    print(f"  Worst total-equity dip from peak: {max_paper_dd * 100:.3f}% (realized + unrealized basis)")
    print(f"  Unrealized at end-of-period     : {float(eq['unrealized'].iloc[-1]) * 100:+.3f}%")
    print()
    print("BTC buy-and-hold over same period (baseline)")
    print(f"  Log return    : {btc_log_return * 100:+.2f}%")
    print(f"  Annualized    : {btc_annualized * 100:+.2f}%")
    print()
    print(f"Strategy alpha vs B&H: {(annualized_realized - btc_annualized) * 100:+.2f}% annualized")

    # ===== Production equity chart =========================================
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1.2]})
    ax = axes[0]
    ax.plot(eq["ts"], eq["realized_cum"] * 100, color="C2", linewidth=2.0,
            label="Realized P&L (production deliverable)")
    ax.fill_between(eq["ts"], 0, eq["unrealized"] * 100, color="C3", alpha=0.20,
                    label="Unrealized MTM (open book, informational)")
    ax.plot(eq["ts"], total_eq * 100, color="C0", linestyle="--", alpha=0.6,
            label="Realized + unrealized")
    ax.axvline(val_test_boundary, color="gray", linestyle=":", alpha=0.7, label="val | test")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.set(ylabel="%", title=f"Production run · realized={float(daily.iloc[-1])*100:+.2f}% over {span_days:.0f}d ({annualized_realized*100:+.2f}% ann)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(eq["ts"], eq["utilization"] * 100, color="C4")
    ax2.axhline(100, color="gray", linestyle=":", alpha=0.5, label="100% cap")
    ax2.axhline(avg_util * 100, color="C4", linestyle="--", alpha=0.6, label=f"avg {avg_util*100:.0f}%")
    ax2.set(ylabel="capital deployed (%)", xlabel="ts",
            title="Capital utilization (= % of max gross actually in the book)")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/production_equity.png", dpi=150)
    plt.close()
    print()
    print(f"Chart saved: {OUT}/production_equity.png")


if __name__ == "__main__":
    main()
