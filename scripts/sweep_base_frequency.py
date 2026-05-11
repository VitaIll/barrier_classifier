"""Base-frequency entries: open at each 1-min bar inside a still-active
boundary signal window. The model's p_k is valid for the M=20 future
bars after boundary k, so at each of those 20 1-min bars we re-evaluate
the entry gate using p_k forward-filled and the *current 1-min bar's
close* as the entry price.

This expands the cache from boundary cadence (every 20 min) to 1-min
cadence: ~20x more potential entries per boundary signal. At top-5%
threshold, where boundary signals are rare but clustered, this should
dramatically densify entries inside the active window.

Compares boundary-cadence and base-frequency variants across the same
threshold/sizing sweep as scripts/sweep_thresholds_and_sizing.py.
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


OUT = "data/model_dataset/strategy/base_frequency"
BASE_LOT = 0.02
MAX_GROSS = 1.0
MAX_OPEN = 300
COST = 0.0005
PHI = 0.0025
M = 20  # 1-min bars per boundary


def expand_cache_to_1min(
    cache: pd.DataFrame, raw_bars: pd.DataFrame
) -> pd.DataFrame:
    """Forward-fill the boundary signal across each 20-min window.

    Returns a frame indexed at 1-min cadence with the latest active
    boundary's ``p``, ``regime``, ``phi`` carried forward, plus the
    1-min bar OHLC. Rows BEFORE any boundary signal are dropped, and
    rows where the carried signal has aged past M bars are also dropped
    (so each boundary's p is used for exactly M follow-on bars at most).
    """
    cache_sorted = cache.sort_values("ts").reset_index(drop=True)
    cache_ts = pd.to_datetime(cache_sorted["ts"])
    # Strip tz to align with raw bars convention
    if hasattr(cache_ts, "dt") and cache_ts.dt.tz is not None:
        cache_ts = cache_ts.dt.tz_localize(None)

    raw_idx = raw_bars.index.tz_localize(None) if raw_bars.index.tz is not None else raw_bars.index
    raw_clipped = raw_bars.copy()
    raw_clipped.index = raw_idx
    raw_clipped = raw_clipped.sort_index()
    # Keep only raw bars in the deployment span
    ts_min, ts_max = cache_ts.min(), cache_ts.max()
    mask = (raw_clipped.index >= ts_min) & (raw_clipped.index <= ts_max + pd.Timedelta(minutes=M))
    raw_sub = raw_clipped.loc[mask].copy()

    # Build a per-1-min-bar lookup of (active_boundary_ts, p, regime).
    # The active boundary for ts is the most recent boundary whose ts <= bar_ts.
    boundary_lookup = pd.DataFrame(
        {
            "boundary_ts": cache_ts.values,
            "p": cache_sorted["p"].values,
            "regime": cache_sorted["regime"].values,
            "phi": cache_sorted["phi"].values,
        }
    ).sort_values("boundary_ts")

    # Use merge_asof for forward-fill: for each raw bar's ts, find the latest
    # boundary_ts <= bar_ts (i.e., direction='backward').
    raw_with_ts = raw_sub.reset_index().rename(columns={"index": "ts"})
    joined = pd.merge_asof(
        raw_with_ts.sort_values("ts"),
        boundary_lookup,
        left_on="ts",
        right_on="boundary_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    # Drop bars before any boundary
    joined = joined.dropna(subset=["p"]).reset_index(drop=True)
    # Drop bars more than M minutes after their carried boundary (signal expired)
    age_min = (joined["ts"] - joined["boundary_ts"]).dt.total_seconds() / 60.0
    joined = joined[age_min <= float(M)].reset_index(drop=True)
    # Assign unique k per 1-min row (the simulator uses k for position identity)
    joined["k"] = np.arange(len(joined), dtype=np.int64)
    joined["y"] = np.nan  # unused at base frequency
    joined["split"] = "live"
    return joined[
        ["k", "ts", "y", "p", "regime", "phi", "open", "high", "low", "close"]
    ]


def make_flat_sizer(base_lot: float, threshold: float):
    def sizer(state) -> float:
        p = state.p_calibrated
        if not np.isfinite(p) or p < threshold:
            return 0.0
        return base_lot
    return sizer


def make_tiered_sizer(base_lot: float, threshold: float):
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
            max_horizon_boundaries=1_000_000_000,
            position_mtm_floor_log_return=None,
        ),
    )


def run_one(spec, sim_cache, raw_bars, cfg) -> dict:
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
    return {
        "n_filled": int((r.closed["exit_reason"] == "tp").sum()) if len(r.closed) else 0,
        "n_signals_taken": int(eq["opened_this_step"].sum()),
        "n_open_at_end": int(eq["n_open"].iloc[-1]),
        "realized_pct": realized * 100,
        "annualized_pct": realized * 100 * 365.0 / span_days,
        "unrealized_at_end_pct": unrealized * 100,
        "max_paper_dd_pct": max_dd * 100,
        "calmar": (realized * 365.0 / span_days) / max_dd if max_dd > 0 else float("inf"),
        "avg_util_pct": float(eq["gross_size"].mean()) / MAX_GROSS * 100,
        "peak_util_pct": float(eq["gross_size"].max()) / MAX_GROSS * 100,
        "peak_concurrent": int(eq["n_open"].max()),
        "equity": eq[["ts", "realized_cum", "unrealized"]].copy(),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cache = pd.read_parquet("data/model_dataset/research_predictions.parquet")
    raw = pd.read_parquet(
        "data/raw_data/klines_1m.parquet", columns=["open", "high", "low", "close"]
    )
    cache = augment_cache_with_boundary_ohlc(cache, raw)
    cache = augment_cache_with_r_realized(cache, raw, M=int(utils.M))
    val_test_cache = (
        pd.concat([cache[cache["split"] == "val"], cache[cache["split"] == "test"]])
        .sort_values("ts").reset_index(drop=True)
    )

    print("Expanding boundary cache -> 1-min cache (forward-fill p over each M-bar window)...")
    expanded = expand_cache_to_1min(val_test_cache, raw)
    print(f"  Boundary cache : {len(val_test_cache):,} rows ({val_test_cache['ts'].min()} -> {val_test_cache['ts'].max()})")
    print(f"  Expanded cache : {len(expanded):,} rows ({expanded['ts'].min()} -> {expanded['ts'].max()})")
    print(f"  Expansion ratio: {len(expanded)/len(val_test_cache):.2f}x")

    cfg = SimConfig(M=int(utils.M))
    quantiles = [30, 20, 10, 5]
    thresholds = {q: float(np.quantile(val_test_cache["p"], 1.0 - q / 100.0)) for q in quantiles}

    print()
    print("Threshold table (computed on boundary cache; applied uniformly to both runs):")
    for q in quantiles:
        n_boundary = int((val_test_cache["p"] >= thresholds[q]).sum())
        n_basefreq = int((expanded["p"] >= thresholds[q]).sum())
        print(f"  top {q:>2}% -> p >= {thresholds[q]:.4f}  (boundary: {n_boundary} signals; base-freq: {n_basefreq} 1-min bars)")
    print()

    rows = []
    equities = {}
    for q in quantiles:
        thr = thresholds[q]
        for variant_label, sizer_factory in [("flat", make_flat_sizer), ("boost", make_tiered_sizer)]:
            sizer = sizer_factory(BASE_LOT, thr)
            for cadence_label, sim_cache in [("boundary", val_test_cache), ("basefreq", expanded)]:
                name = f"top{q}_{variant_label}_{cadence_label}"
                spec = make_spec(name, thr, sizer)
                r = run_one(spec, sim_cache, raw, cfg)
                equities[name] = r.pop("equity")
                rows.append(
                    {"top_q": q, "variant": variant_label, "cadence": cadence_label, "thr": thr, **r}
                )
                print(
                    f"top{q:>2}% {variant_label:>5} {cadence_label:>9} | "
                    f"taken={r['n_signals_taken']:>5} filled={r['n_filled']:>5} "
                    f"realized={r['realized_pct']:+5.2f}% paper_dd={r['max_paper_dd_pct']:.2f}% "
                    f"calmar={r['calmar']:.2f} util_avg={r['avg_util_pct']:.0f}% peak_concurrent={r['peak_concurrent']}"
                )
        print()

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/sweep.csv", index=False)
    print(f"Saved: {OUT}/sweep.csv")

    # Side-by-side comparison
    print()
    print("Realized P&L (%) — boundary vs base-frequency:")
    pivot = df.pivot_table(
        index=["top_q", "variant"],
        columns="cadence",
        values="realized_pct",
    )
    pivot["uplift_basefreq_x"] = pivot["basefreq"] / pivot["boundary"]
    print(pivot.round(2).to_string())

    print()
    print("Calmar — boundary vs base-frequency:")
    pivot_c = df.pivot_table(
        index=["top_q", "variant"],
        columns="cadence",
        values="calmar",
    )
    print(pivot_c.round(2).to_string())

    print()
    print("Max paper DD (%) — boundary vs base-frequency:")
    pivot_dd = df.pivot_table(
        index=["top_q", "variant"],
        columns="cadence",
        values="max_paper_dd_pct",
    )
    print(pivot_dd.round(2).to_string())

    # Equity overlay: boundary vs basefreq at each (q, variant)
    fig, axes = plt.subplots(len(quantiles), 2, figsize=(14, 3.0 * len(quantiles)), sharex=True)
    for i, q in enumerate(quantiles):
        for j, variant in enumerate(["flat", "boost"]):
            ax = axes[i, j]
            for cad in ["boundary", "basefreq"]:
                key = f"top{q}_{variant}_{cad}"
                e = equities[key]
                ax.plot(e["ts"], e["realized_cum"] * 100, label=f"{cad}", linewidth=1.5)
            ax.set(title=f"top {q}% — {variant}", ylabel="realized cum (%)")
            ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/equity_overlay.png", dpi=150)
    plt.close()
    print()
    print(f"Saved: {OUT}/equity_overlay.png")


if __name__ == "__main__":
    main()
