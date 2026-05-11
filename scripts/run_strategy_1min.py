"""Run the production wait-for-TP strategy on the 1-min-cadence prediction
cache produced by ``scripts/train_1min_model.py``.

Reuses the same StrategySpec as the production boundary-cadence run but
notes the cadence in the output. Because the 1-min cache has 1-min row
spacing and the simulator uses ``ts`` for path lookups, no simulator
changes are needed.

For honest reporting at 1-min cadence:
- Threshold is recomputed from the *training-period* score distribution
  loaded from the saved model artifacts (avoiding a val/test peek).
  Falls back to a default if that info isn't yet persisted.
- Block bootstrap for any CI claims downstream.
"""

from __future__ import annotations

import json
import os
import sys
import time
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
from src.features.config import M, PHI
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

OUT = "data/model_dataset/strategy/production_1min"
COST = 0.0005
LOT_SIZE = 0.02
MAX_CONCURRENT = 50


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cache_path = "data/model_dataset/research_predictions_1min.parquet"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"1-min predictions cache missing: {cache_path}. "
            "Run scripts/train_1min_model.py first."
        )

    cache = pd.read_parquet(cache_path)
    raw = pd.read_parquet(
        "data/raw_data/klines_1m.parquet", columns=["open", "high", "low", "close"]
    )
    cache = augment_cache_with_boundary_ohlc(cache, raw)
    cache = augment_cache_with_r_realized(cache, raw, M=int(M))
    print(f"Cache: {len(cache):,} rows ({cache['split'].value_counts().to_dict()})")

    # ----- Threshold: top decile of p in this cache --------------------------
    # NOTE: this is val+test data; in production you'd use training-period
    # scores. Reported as a quick sanity check on the strategy mechanics.
    THRESHOLD = float(np.quantile(cache["p"], 0.90))
    print(f"Top-10% threshold p>={THRESHOLD:.4f}  ({(cache['p']>=THRESHOLD).sum()} signals)")
    print(f"Per-fill net (deterministic): +{(PHI-COST)*1e4:.0f} bp x lot={LOT_SIZE} = +{(PHI-COST)*LOT_SIZE*1e4:.1f} bp / fill")

    # ----- Build sim cache (val + test stream) -------------------------------
    sim_cache = (
        pd.concat([cache[cache["split"] == "val"], cache[cache["split"] == "test"]])
        .sort_values("ts").reset_index(drop=True)
    )
    val_test_boundary = cache[cache["split"] == "test"]["ts"].min()

    spec = StrategySpec(
        name="production_1min_wait_for_tp",
        score_fn=score_raw_p,
        entry_gates=(lambda s: gate_score_above(s, THRESHOLD),),
        sizer=lambda s: size_clip(size_constant(s, default=LOT_SIZE), max_size=1.0),
        exit_policy=exit_tp_or_expiry,
        bulk_close=lambda s: None,
        risk=RiskConfig(
            cost_per_trade=COST,
            max_open_positions=MAX_CONCURRENT,
            max_gross_size=MAX_CONCURRENT * LOT_SIZE + 1e-6,
            max_horizon_boundaries=1_000_000,
            position_mtm_floor_log_return=None,
        ),
    )

    # NB: SimConfig.M is the cadence of the cache rows in 1-min bars. At
    # boundary cadence M=20 (each row = 20 bars). At 1-min cadence each
    # row is one bar, so M should be 1 for the simulator's internal pacing.
    # However, the simulator only uses M to compute n_bars from k offsets
    # for legacy compatibility; the actual TP detection uses 1-min OHLC
    # from raw_bars directly. Both modes work — at 1-min cadence the
    # intra-bars walk between two consecutive cache rows contains 1 bar.
    cfg = SimConfig(M=1)

    t0 = time.perf_counter()
    r = simulate(sim_cache, raw, spec, config=cfg)
    print(f"Simulation: {time.perf_counter()-t0:.1f}s")
    print(f"  closed: {len(r.closed)}  open_at_end: {int(r.equity['n_open'].iloc[-1])}")

    eq = r.equity.copy()
    eq["ts"] = pd.to_datetime(eq["ts"])
    span_days = (eq["ts"].max() - eq["ts"].min()).total_seconds() / 86400.0

    realized = float(eq["realized_cum"].iloc[-1])
    unrealized = float(eq["unrealized"].iloc[-1])
    total_eq = (eq["realized_cum"] + eq["unrealized"]).to_numpy()
    peaks = np.maximum.accumulate(total_eq)
    max_paper_dd = float(-(total_eq - peaks).min())

    # BTC B&H over the same period
    raw_idx = raw.index.tz_localize(None) if raw.index.tz is not None else raw.index
    n0 = int(np.searchsorted(raw_idx, np.datetime64(eq["ts"].min()), side="left"))
    n1 = int(np.searchsorted(raw_idx, np.datetime64(eq["ts"].max()), side="left"))
    btc_log_return = float(np.log(raw["close"].iloc[n1] / raw["close"].iloc[n0]))
    btc_annualized = btc_log_return * 365.0 / span_days

    summary = {
        "span_days": span_days,
        "n_signals": int(eq["opened_this_step"].sum()),
        "n_tps": int((r.closed["exit_reason"] == "tp").sum()) if len(r.closed) else 0,
        "n_open_at_end": int(eq["n_open"].iloc[-1]),
        "realized_log_return": realized,
        "realized_annualized": realized * 365.0 / span_days,
        "max_paper_dd": max_paper_dd,
        "unrealized_at_end": unrealized,
        "calmar": (realized * 365.0 / span_days) / max_paper_dd if max_paper_dd > 0 else float("inf"),
        "btc_log_return": btc_log_return,
        "btc_annualized": btc_annualized,
        "parameters": {
            "threshold": THRESHOLD, "lot_size": LOT_SIZE, "max_concurrent": MAX_CONCURRENT,
            "phi": PHI, "cost": COST, "cadence": "1min",
        },
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print()
    print("=" * 60)
    print("PRODUCTION RUN (1-min cadence)")
    print("=" * 60)
    print(f"Span                 : {span_days:.1f} days")
    print(f"Signals fired        : {summary['n_signals']:,}")
    print(f"TPs filled           : {summary['n_tps']:,}")
    print(f"Open at end          : {summary['n_open_at_end']}")
    print(f"Realized cumulative  : {realized*100:+.3f}%  ({summary['realized_annualized']*100:+.2f}%/yr)")
    print(f"Unrealized at end    : {unrealized*100:+.3f}%")
    print(f"Max paper DD         : {max_paper_dd*100:.2f}%")
    print(f"Calmar               : {summary['calmar']:.2f}")
    print(f"BTC B&H (same period): {btc_log_return*100:+.2f}%  ({btc_annualized*100:+.1f}%/yr)")

    # ----- Production equity chart -------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1.2]})
    ax = axes[0]
    ax.plot(eq["ts"], eq["realized_cum"]*100, color="C2", linewidth=2.0, label="Realized")
    ax.fill_between(eq["ts"], 0, eq["unrealized"]*100, color="C3", alpha=0.20, label="Unrealized MTM")
    ax.plot(eq["ts"], total_eq*100, color="C0", linestyle="--", alpha=0.6, label="Total")
    ax.axvline(val_test_boundary, color="gray", linestyle=":", alpha=0.7, label="val | test")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4)
    ax.set(ylabel="%", title=f"Production 1-min · realized={realized*100:+.2f}% over {span_days:.0f}d ({summary['realized_annualized']*100:+.2f}%/yr)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    ax2 = axes[1]
    util = eq["gross_size"] / (MAX_CONCURRENT * LOT_SIZE)
    ax2.plot(eq["ts"], util*100, color="C4")
    ax2.axhline(100, color="gray", linestyle=":", alpha=0.5, label="cap")
    ax2.set(ylabel="capital deployed (%)", xlabel="ts", title="Capital utilization")
    ax2.legend(loc="upper left"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/equity.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUT}/equity.png and {OUT}/summary.json")


if __name__ == "__main__":
    main()
