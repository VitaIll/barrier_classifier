"""Analytics for the pure wait-for-TP strategy.

Four panels on a single representative run (top-10% of p, lot=0.02, max_open=50):
1. Realized vs unrealized equity over time (realized is monotone; unrealized fluctuates)
2. Time-to-TP distribution for closed positions
3. Worst MTM observed per closed position (how deep underwater before recovery)
4. Capacity / capital tradeoff sweep
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

OUT = "data/model_dataset/strategy/wait_for_tp"
os.makedirs(OUT, exist_ok=True)


def wait_for_tp(thr: float, lot: float, max_c: int) -> StrategySpec:
    return StrategySpec(
        name=f"wait_tp_max{max_c}",
        score_fn=score_raw_p,
        entry_gates=(lambda s, t=thr: gate_score_above(s, t),),
        sizer=lambda s, sz=lot: size_clip(size_constant(s, default=sz), max_size=1.0),
        exit_policy=exit_tp_or_expiry,
        bulk_close=lambda s: None,
        risk=RiskConfig(
            cost_per_trade=0.0005,
            max_open_positions=max_c,
            max_gross_size=max_c * lot + 0.001,
            max_horizon_boundaries=1_000_000,
            position_mtm_floor_log_return=None,
        ),
    )


def main() -> None:
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
    p_top10 = float(np.quantile(sim_cache["p"], 0.90))
    val_test_boundary = cache[cache["split"] == "test"]["ts"].min()
    cfg = SimConfig(M=int(utils.M), quantile_window=500, quantile_min_warmup=80)

    # ===== Representative run: max_open=50, lot=0.02 =====
    s = wait_for_tp(p_top10, lot=0.02, max_c=50)
    r = simulate(sim_cache, raw, s, config=cfg)
    print(f"Representative run: top-10% (thr={p_top10:.4f}), lot=0.02, max_open=50")
    print(
        f"  {len(r.closed)} TP fills, {r.equity['n_open'].iloc[-1]} open at end, "
        f"realized={r.equity['realized_cum'].iloc[-1] * 10000:+.1f}bp, "
        f"unrealized={r.equity['unrealized'].iloc[-1] * 10000:+.1f}bp"
    )

    # === ANALYTIC 1: realized vs unrealized equity over time ===
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(
        r.equity["ts"], r.equity["realized_cum"] * 10000,
        color="C2", linewidth=2.0, label="Realized (TP fills only) — monotone",
    )
    ax.fill_between(
        r.equity["ts"], 0, r.equity["unrealized"] * 10000,
        color="C3", alpha=0.25, label="Unrealized MTM of open book",
    )
    ax.plot(
        r.equity["ts"],
        (r.equity["realized_cum"] + r.equity["unrealized"]) * 10000,
        color="C0", linestyle="--", alpha=0.7, label="Total = realized + unrealized",
    )
    ax.axvline(val_test_boundary, color="gray", linestyle=":", alpha=0.7, label="val | test")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.set(
        xlabel="ts", ylabel="log-return (bp)",
        title="Wait-for-TP: realized P&L is monotone; unrealized fluctuates but never gets realized",
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/01_realized_vs_unrealized.png", dpi=150)
    plt.close()

    real_diff = np.diff(r.equity["realized_cum"].to_numpy())
    print(
        f"  monotonicity check: min step in realized = {real_diff.min() * 10000:+.4f}bp "
        f"(>= 0 confirms no realized loss ever)"
    )

    # === ANALYTIC 2: time-to-TP distribution ===
    closed = r.closed.copy()
    closed["ts_entry"] = pd.to_datetime(closed["ts_entry"].astype(str)).map(
        lambda x: x.tz_localize(None) if x.tzinfo is not None else x
    )
    closed["ts_exit"] = pd.to_datetime(closed["ts_exit"].astype(str)).map(
        lambda x: x.tz_localize(None) if x.tzinfo is not None else x
    )
    closed["hold_minutes"] = (
        closed["ts_exit"] - closed["ts_entry"]
    ).dt.total_seconds() / 60.0

    print()
    print("Time-to-TP for closed positions:")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        v = float(closed["hold_minutes"].quantile(q))
        if v < 60:
            label = f"{v:.0f} min"
        elif v < 1440:
            label = f"{v:.0f} min ({v / 60:.1f} h)"
        else:
            label = f"{v:.0f} min ({v / 1440:.1f} d)"
        print(f"  p{int(q * 100):>2}: {label}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    clip = float(closed["hold_minutes"].quantile(0.99))
    axes[0].hist(
        closed["hold_minutes"].clip(upper=clip), bins=50, color="C0", alpha=0.7
    )
    med = float(closed["hold_minutes"].median())
    axes[0].axvline(med, color="C1", linestyle="--", label=f"median = {med:.0f}min")
    axes[0].set(
        xlabel="hold time (minutes, clipped at p99)",
        ylabel="# closed positions",
        title="Time-to-TP distribution (linear)",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    sorted_h = np.sort(closed["hold_minutes"].to_numpy())
    ecdf = np.arange(1, len(sorted_h) + 1) / len(sorted_h)
    axes[1].semilogx(sorted_h, ecdf, color="C0")
    for q, lab in [(0.50, "p50"), (0.90, "p90"), (0.99, "p99")]:
        v = float(closed["hold_minutes"].quantile(q))
        axes[1].axvline(v, color="gray", linestyle=":", alpha=0.5)
        axes[1].text(
            v, q, f"  {lab}={v:.0f}min",
            verticalalignment="bottom", fontsize=9,
        )
    axes[1].set(
        xlabel="hold time (minutes, log scale)", ylabel="ECDF",
        title="Time-to-TP ECDF (log-x)",
    )
    axes[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(f"{OUT}/02_time_to_tp.png", dpi=150)
    plt.close()

    # === ANALYTIC 3: worst MTM observed per closed position ===
    print()
    print("Worst-MTM per closed position (deepest underwater before TP eventually fired):")
    raw_idx = (
        raw.index.tz_localize(None) if raw.index.tz is not None else raw.index
    )
    raw_low = raw["low"].to_numpy(dtype=float)
    worst_list: list[float] = []
    for _, row in closed.iterrows():
        entry_p = float(row["entry_price"])
        ts_in = pd.Timestamp(row["ts_entry"])
        ts_out = pd.Timestamp(row["ts_exit"])
        # Index slice between (ts_in, ts_out]
        lo = int(np.searchsorted(raw_idx, np.datetime64(ts_in), side="right"))
        hi = int(np.searchsorted(raw_idx, np.datetime64(ts_out), side="right"))
        if hi <= lo:
            worst_list.append(0.0)
            continue
        worst_low = float(raw_low[lo:hi].min())
        worst_list.append(float(np.log(worst_low / entry_p)))
    closed["worst_mtm_bp"] = np.asarray(worst_list, dtype=float) * 10000.0
    print(
        f"  median worst MTM: {closed['worst_mtm_bp'].median():+.0f} bp     "
        f"(p25/p75: {closed['worst_mtm_bp'].quantile(0.25):+.0f} / {closed['worst_mtm_bp'].quantile(0.75):+.0f})"
    )
    for cutoff in [-50, -100, -200, -500, -1000]:
        frac = float((closed["worst_mtm_bp"] < cutoff).mean())
        print(f"  fraction of positions whose MTM dipped below {cutoff:+5d} bp: {frac:.2%}")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(
        closed["worst_mtm_bp"].clip(lower=-1500, upper=0), bins=60, color="C3", alpha=0.7
    )
    for q in [0.05, 0.25, 0.50, 0.75]:
        v = float(closed["worst_mtm_bp"].quantile(q))
        ax.axvline(v, color="gray", linestyle=":", alpha=0.6)
        ax.text(
            v, ax.get_ylim()[1] * 0.95, f"p{int(q * 100)}={v:.0f}",
            rotation=90, verticalalignment="top", fontsize=8,
        )
    ax.set(
        xlabel="worst MTM observed during position life (bp, clipped at -1500)",
        ylabel="# closed positions",
        title="How deep did each position go before TP eventually fired? (closed positions only)",
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_worst_mtm_per_position.png", dpi=150)
    plt.close()

    # === ANALYTIC 4: capacity / capital tradeoff ===
    print()
    print("Capacity sweep: how many signals get filled vs how deep the open book runs?")
    print(
        f"{'max_open':>8} {'lot':>6} {'taken':>6} {'TP':>5} {'open_end':>8} "
        f"{'realized_bp':>11} {'max_concurrent':>14} {'min_unreal_bp':>14}"
    )
    cap_rows = []
    for max_c, lot in [
        (5, 0.2), (10, 0.1), (20, 0.05), (50, 0.02),
        (100, 0.01), (200, 0.005), (500, 0.002),
    ]:
        sp = wait_for_tp(p_top10, lot, max_c)
        rr = simulate(sim_cache, raw, sp, config=cfg)
        n_signals = int(rr.equity["opened_this_step"].sum())
        n_tp = int((rr.closed["exit_reason"] == "tp").sum()) if len(rr.closed) else 0
        n_open = int(rr.equity["n_open"].iloc[-1])
        realized_bp = float(rr.equity["realized_cum"].iloc[-1] * 10000)
        max_open_book = int(rr.equity["n_open"].max())
        min_unreal_bp = float(rr.equity["unrealized"].min() * 10000)
        cap_rows.append(
            {
                "max_open": max_c, "lot": lot,
                "taken": n_signals, "TP": n_tp, "open_end": n_open,
                "realized_bp": realized_bp,
                "max_concurrent": max_open_book,
                "min_unreal_bp": min_unreal_bp,
            }
        )
        print(
            f"{max_c:>8} {lot:>6.3f} {n_signals:>6} {n_tp:>5} {n_open:>8} "
            f"{realized_bp:>+9.1f}   {max_open_book:>14} {min_unreal_bp:>+12.1f}"
        )
    pd.DataFrame(cap_rows).to_csv(f"{OUT}/04_capacity_sweep.csv", index=False)

    # Capacity plot
    cap_df = pd.DataFrame(cap_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(
        cap_df["max_open"], cap_df["taken"] / cap_df["taken"].max(),
        marker="o", color="C2", label="signals taken / max",
    )
    axes[0].set_xscale("log")
    axes[0].set(
        xlabel="max_open_positions (log)", ylabel="fraction of signals taken",
        title="Coverage: how many signals fill as we grow capacity",
    )
    axes[0].axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()

    axes[1].plot(
        cap_df["max_open"], cap_df["min_unreal_bp"], marker="o", color="C3",
        label="worst observed unrealized book (bp)",
    )
    axes[1].plot(
        cap_df["max_open"], cap_df["realized_bp"], marker="s", color="C2",
        label="final realized (bp)",
    )
    axes[1].set_xscale("log")
    axes[1].set(
        xlabel="max_open_positions (log)", ylabel="log-return (bp)",
        title="Realized gain vs worst observed unrealized drawdown",
    )
    axes[1].axhline(0, color="gray", linestyle=":", alpha=0.5)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_capacity_tradeoff.png", dpi=150)
    plt.close()

    print()
    print(f"All artifacts saved under {OUT}/")
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
