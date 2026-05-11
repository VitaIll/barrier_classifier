"""Threshold + sizing sweep with the wait-for-TP discipline.

Six quantile thresholds (top 30% -> top 5%) x two sizing schemes
(flat vs tiered-boost). Same TP-only exit policy, same capital cap, same
costs. Reports realized P&L, capital efficiency, paper drawdown.

Tiered boost: lot multiplier 1x / 2x / 4x for p in
    [thr, thr+0.10), [thr+0.10, thr+0.20), [thr+0.20, +inf).
Base lot is set so that the FLAT version uses exactly the same nominal
capital ceiling (max_gross = 1.0); the BOOST version concentrates that
same capital on higher-confidence signals.
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
)
from src.strategy.simulator import SimConfig, simulate


OUT = "data/model_dataset/strategy/threshold_sweep"
BASE_LOT = 0.02         # tier-1 lot under both schemes
MAX_GROSS = 1.0         # 100% capital cap, both schemes
MAX_OPEN = 300          # not the binding constraint; gross is
COST = 0.0005
PHI = 0.0025


def make_flat_sizer(base_lot: float, threshold: float):
    def sizer(state) -> float:
        p = state.p_calibrated
        if not np.isfinite(p) or p < threshold:
            return 0.0
        return base_lot
    return sizer


def make_tiered_sizer(base_lot: float, threshold: float):
    """1x at threshold, 2x once p exceeds threshold+0.10, 4x past threshold+0.20."""
    def sizer(state) -> float:
        p = state.p_calibrated
        if not np.isfinite(p) or p < threshold:
            return 0.0
        if p >= threshold + 0.20:
            return base_lot * 4.0
        if p >= threshold + 0.10:
            return base_lot * 2.0
        return base_lot
    return sizer


def make_spec(name: str, threshold: float, sizer) -> StrategySpec:
    return StrategySpec(
        name=name,
        score_fn=score_raw_p,
        entry_gates=(lambda s: gate_score_above(s, threshold),),
        sizer=sizer,
        exit_policy=exit_tp_or_expiry,
        bulk_close=lambda s: None,
        risk=RiskConfig(
            cost_per_trade=COST,
            max_open_positions=MAX_OPEN,
            max_gross_size=MAX_GROSS,
            max_horizon_boundaries=1_000_000,
            position_mtm_floor_log_return=None,
        ),
    )


def run_one(spec, sim_cache, raw_bars, cfg):
    r = simulate(sim_cache, raw_bars, spec, config=cfg)
    eq = r.equity
    span_days = (
        pd.Timestamp(eq["ts"].max()) - pd.Timestamp(eq["ts"].min())
    ).total_seconds() / 86400.0
    realized = float(eq["realized_cum"].iloc[-1])
    unrealized = float(eq["unrealized"].iloc[-1])
    total = (eq["realized_cum"] + eq["unrealized"]).to_numpy()
    peaks = np.maximum.accumulate(total)
    max_dd = float(-(total - peaks).min())
    n_signals_in_data = int(eq["opened_this_step"].sum())
    n_filled = int((r.closed["exit_reason"] == "tp").sum()) if len(r.closed) else 0
    n_open_at_end = int(eq["n_open"].iloc[-1])
    return {
        "n_filled": n_filled,
        "n_signals_taken": n_signals_in_data,
        "n_open_at_end": n_open_at_end,
        "realized_pct": realized * 100,
        "annualized_pct": realized * 100 * 365.0 / span_days,
        "unrealized_at_end_pct": unrealized * 100,
        "max_paper_dd_pct": max_dd * 100,
        "avg_util_pct": float(eq["gross_size"].mean()) / MAX_GROSS * 100,
        "peak_util_pct": float(eq["gross_size"].max()) / MAX_GROSS * 100,
        "peak_concurrent": int(eq["n_open"].max()),
        "calmar": realized * (365.0 / span_days) / max_dd if max_dd > 0 else float("inf"),
        "equity_df": eq[["ts", "realized_cum", "unrealized"]].copy(),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cache = pd.read_parquet("data/model_dataset/research_predictions.parquet")
    raw = pd.read_parquet(
        "data/raw_data/klines_1m.parquet", columns=["open", "high", "low", "close"]
    )
    cache = augment_cache_with_boundary_ohlc(cache, raw)
    cache = augment_cache_with_r_realized(cache, raw, M=int(utils.M))
    sim_cache = (
        pd.concat([cache[cache["split"] == "val"], cache[cache["split"] == "test"]])
        .sort_values("ts")
        .reset_index(drop=True)
    )

    cfg = SimConfig(M=int(utils.M))

    quantiles = [30, 25, 20, 15, 10, 5]
    thresholds = {q: float(np.quantile(sim_cache["p"], 1.0 - q / 100.0)) for q in quantiles}
    print("Threshold table:")
    for q in quantiles:
        n_above = int((sim_cache["p"] >= thresholds[q]).sum())
        print(f"  top {q:>2}% -> p >= {thresholds[q]:.4f}  ({n_above} signals in val+test)")
    print()

    rows = []
    equities_flat = {}
    equities_boost = {}
    for q in quantiles:
        thr = thresholds[q]
        # Distribution of p above threshold across tiers
        p_above = sim_cache.loc[sim_cache["p"] >= thr, "p"]
        n_t1 = int(((p_above >= thr) & (p_above < thr + 0.10)).sum())
        n_t2 = int(((p_above >= thr + 0.10) & (p_above < thr + 0.20)).sum())
        n_t3 = int((p_above >= thr + 0.20).sum())
        # FLAT
        sp_flat = make_spec(f"top{q}_flat", thr, make_flat_sizer(BASE_LOT, thr))
        r_flat = run_one(sp_flat, sim_cache, raw, cfg)
        equities_flat[q] = r_flat["equity_df"]
        # BOOST
        sp_boost = make_spec(f"top{q}_boost", thr, make_tiered_sizer(BASE_LOT, thr))
        r_boost = run_one(sp_boost, sim_cache, raw, cfg)
        equities_boost[q] = r_boost["equity_df"]
        rows.append(
            {"variant": "flat",  "top_q": q, "thr": thr, "n_t1": n_t1, "n_t2": n_t2, "n_t3": n_t3, **{k: v for k, v in r_flat.items()  if k != "equity_df"}}
        )
        rows.append(
            {"variant": "boost", "top_q": q, "thr": thr, "n_t1": n_t1, "n_t2": n_t2, "n_t3": n_t3, **{k: v for k, v in r_boost.items() if k != "equity_df"}}
        )
        print(
            f"top{q:>2}% (thr={thr:.4f}, tiers {n_t1}/{n_t2}/{n_t3}) | "
            f"FLAT  : taken={r_flat['n_signals_taken']:>4} filled={r_flat['n_filled']:>4} "
            f"realized={r_flat['realized_pct']:+5.2f}% paper_dd={r_flat['max_paper_dd_pct']:.2f}% calmar={r_flat['calmar']:.2f}"
        )
        print(
            f"top{q:>2}% (thr={thr:.4f}, tiers {n_t1}/{n_t2}/{n_t3}) | "
            f"BOOST : taken={r_boost['n_signals_taken']:>4} filled={r_boost['n_filled']:>4} "
            f"realized={r_boost['realized_pct']:+5.2f}% paper_dd={r_boost['max_paper_dd_pct']:.2f}% calmar={r_boost['calmar']:.2f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/threshold_sizing_sweep.csv", index=False)
    print()
    print(f"Saved: {OUT}/threshold_sizing_sweep.csv")

    # Compact comparison view: flat vs boost across quantiles
    pivot_real = df.pivot(index="top_q", columns="variant", values="realized_pct")
    pivot_real["boost_vs_flat_pp"] = pivot_real["boost"] - pivot_real["flat"]
    pivot_real["boost_vs_flat_ratio"] = pivot_real["boost"] / pivot_real["flat"]
    print()
    print("Realized P&L (% over 104 days) — flat vs boost:")
    print(pivot_real.round(2).to_string())
    print()
    pivot_dd = df.pivot(index="top_q", columns="variant", values="max_paper_dd_pct")
    pivot_calmar = df.pivot(index="top_q", columns="variant", values="calmar")
    print("Max paper drawdown (%):")
    print(pivot_dd.round(2).to_string())
    print()
    print("Calmar (annualized realized / max paper DD):")
    print(pivot_calmar.round(2).to_string())

    # Equity overlay
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    palette = plt.get_cmap("viridis")
    for i, q in enumerate(quantiles):
        c = palette(i / max(len(quantiles) - 1, 1))
        axes[0].plot(equities_flat[q]["ts"], equities_flat[q]["realized_cum"] * 100,
                     color=c, label=f"top{q}%", linewidth=1.6)
        axes[1].plot(equities_boost[q]["ts"], equities_boost[q]["realized_cum"] * 100,
                     color=c, label=f"top{q}%", linewidth=1.6)
    axes[0].set(ylabel="realized cum (%)", title="Flat sizing (base_lot = 0.02)")
    axes[1].set(ylabel="realized cum (%)", xlabel="ts",
                title="Boost sizing (1x/2x/4x by p-tier, same max_gross = 1.0)")
    for ax in axes:
        ax.legend(loc="upper left", ncol=3, fontsize=9)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/equity_overlay.png", dpi=150)
    plt.close()
    print()
    print(f"Saved: {OUT}/equity_overlay.png")


if __name__ == "__main__":
    main()
