"""Run the winning P1+P3 strategy spec on the current model and produce the
full nb05 dashboard set so the result can be inspected visually.

Winning spec (from the strategy-variants sweep):
  - TOP_Q = 0.99 (top 1% of training scores)
  - exit: make_exit_let_winners_run (threshold-based, hold_threshold = P_99,
    no SL — "don't release into loss")
  - lot_size = 0.02 (2% per lot)
  - max_concurrent = 50
  - cost = 5 bp round-trip

Outputs (under data/model_dataset/strategy/production_1min_P1P3/):
  - production_summary.json
  - dashboard.png
  - trade_zoom_grid.png
  - local_min_rank.png
  - hold_and_worst_mtm.png
  - closed_trades.parquet
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.thresholds import derive_top_q_threshold  # noqa: E402
from src.strategy.baseline import COST_PER_TRADE, compute_btc_buy_and_hold  # noqa: E402
from src.features.config import M as M_CFG, PHI as PHI_CFG  # noqa: E402
from src.strategy.cache import (  # noqa: E402
    augment_cache_with_boundary_ohlc,
    augment_cache_with_r_realized,
)
from src.strategy.policy import (  # noqa: E402
    RiskConfig,
    StrategySpec,
    gate_score_above,
    make_exit_let_winners_run,
    score_raw_p,
    size_clip,
    size_constant,
)
from src.strategy.simulator import SimConfig, simulate  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
TOP_Q = 0.99
LOT_SIZE = 0.02
MAX_CONCURRENT = 50
COST = COST_PER_TRADE
MAX_HORIZON_BOUNDARIES = 1_000_000

M = int(M_CFG)
PHI = float(PHI_CFG)
DATASET_DIR = ROOT / "data" / "model_dataset"
CACHE_PATH = DATASET_DIR / "research_predictions_1min.parquet"
RAW_PATH = ROOT / "data" / "raw_data" / "klines_1m.parquet"
TRAIN_ART_PATH = DATASET_DIR / "analytics" / "train_scores_unc_1min.parquet"
VAL_TEST_ART_PATH = DATASET_DIR / "analytics" / "val_test_ve_unc_1min.parquet"
OUT_DIR = DATASET_DIR / "strategy" / "production_1min_P1P3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"ROOT     : {ROOT}")
print(f"M, PHI   : {M}, {PHI}")
print(f"OUT_DIR  : {OUT_DIR}")


# -----------------------------------------------------------------------------
# 1. Load + augment
# -----------------------------------------------------------------------------
train_art = pd.read_parquet(TRAIN_ART_PATH)
p_train = train_art["p_train"].to_numpy()
P_99 = float(derive_top_q_threshold(p_train, q=TOP_Q))
print(f"P_99 (entry + hold threshold) = {P_99:.4f}")

cache = pd.read_parquet(CACHE_PATH)
ve = pd.read_parquet(VAL_TEST_ART_PATH)
cache = cache.merge(ve, on=["k", "split"], how="left")
assert cache["knowledge_unc"].notna().all()

raw_bars = pd.read_parquet(RAW_PATH, columns=["open", "high", "low", "close"])
if raw_bars.index.tz is not None:
    raw_bars.index = raw_bars.index.tz_localize(None)

cache = augment_cache_with_boundary_ohlc(cache, raw_bars)
cache = augment_cache_with_r_realized(cache, raw_bars, M=M)

sim_cache = (
    pd.concat([cache[cache["split"] == "val"], cache[cache["split"] == "test"]])
    .sort_values("ts").reset_index(drop=True)
)
val_test_boundary = pd.Timestamp(cache[cache["split"] == "test"]["ts"].min())
print(f"Live stream : {sim_cache['ts'].min()} -> {sim_cache['ts'].max()}  ({len(sim_cache):,} rows)")
print(f"val|test ts : {val_test_boundary}")


# -----------------------------------------------------------------------------
# 2. Build spec + simulate
# -----------------------------------------------------------------------------
p_map = pd.Series(sim_cache["p"].to_numpy(), index=pd.DatetimeIndex(sim_cache["ts"]))
exit_policy = make_exit_let_winners_run(p_map, hold_threshold=P_99, sl_log_return=None)

spec = StrategySpec(
    name="winning_P1_P3_letrun_p99",
    requires=("ve_diag",),
    score_fn=score_raw_p,
    entry_gates=(lambda s, t=P_99: gate_score_above(s, t),),
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
    description=(
        "P1: top-1% selective entry (p>=P_99); "
        "P3: let-winners-run when model conviction still in top-1% at TP signal; "
        "no SL (don't release into loss)"
    ),
)

cfg = SimConfig(M=M)
t0 = time.perf_counter()
result = simulate(sim_cache, raw_bars, spec, config=cfg)
print(f"\n[simulate] {time.perf_counter()-t0:.1f}s  closed={len(result.closed):,}  "
      f"open at end={int(result.equity['n_open'].iloc[-1])}")

# -----------------------------------------------------------------------------
# 3. Headline metrics
# -----------------------------------------------------------------------------
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
GROSS_CAP = MAX_CONCURRENT * LOT_SIZE
peak_util = float((eq["gross_size"] / GROSS_CAP).max())
avg_util = float((eq["gross_size"] / GROSS_CAP).mean())
peak_concurrent = int(eq["n_open"].max())
total_eq = (eq["realized_cum"] + eq["unrealized"]).to_numpy()
peaks = np.maximum.accumulate(total_eq)
dd_series = total_eq - peaks
max_paper_dd = float(-dd_series.min())
worst_dd_idx = int(np.argmin(dd_series))
worst_dd_ts = eq["ts"].iloc[worst_dd_idx]
btc_log_return, btc_annualized = compute_btc_buy_and_hold(
    raw_bars, eq["ts"].min(), eq["ts"].max(), span_days=span_days,
)
calmar = annualized_realized / max_paper_dd if max_paper_dd > 0 else float("inf")
total_log_return = float(daily.iloc[-1]) + float(eq["unrealized"].iloc[-1])

prod_summary = {
    "spec_name": spec.name,
    "spec_description": spec.description,
    "span_days": span_days,
    "n_signals": int(eq["opened_this_step"].sum()),
    "n_closed": int(len(result.closed)),
    "n_open_at_end": int(eq["n_open"].iloc[-1]),
    "exit_reasons": result.closed["exit_reason"].value_counts().to_dict() if len(result.closed) else {},
    "realized_log_return": float(daily.iloc[-1]),
    "realized_annualized": annualized_realized,
    "unrealized_at_end": float(eq["unrealized"].iloc[-1]),
    "total_log_return": total_log_return,
    "sharpe_realized_daily": sharpe,
    "calmar": calmar,
    "max_paper_dd": max_paper_dd,
    "worst_dd_ts": str(worst_dd_ts),
    "avg_utilization": avg_util,
    "peak_utilization": peak_util,
    "peak_concurrent": peak_concurrent,
    "btc_log_return": btc_log_return,
    "btc_annualized": btc_annualized,
    "alpha_over_btc": total_log_return - btc_log_return,
    "parameters": {
        "TOP_Q": TOP_Q, "P_99": float(P_99), "lot_size": LOT_SIZE,
        "max_concurrent": MAX_CONCURRENT, "phi": PHI, "cost": COST,
        "hold_threshold": float(P_99), "sl_log_return": None,
    },
}
with open(OUT_DIR / "production_summary.json", "w") as f:
    json.dump(prod_summary, f, indent=2, default=str)
print(f"[save] {OUT_DIR / 'production_summary.json'}")
print(f"  total={total_log_return*100:+.2f}%  alpha={prod_summary['alpha_over_btc']*100:+.2f}%  "
      f"DD={max_paper_dd*100:.2f}%  closed={int(len(result.closed))}")

# -----------------------------------------------------------------------------
# 4. Charts
# -----------------------------------------------------------------------------
deploy_start, deploy_end = eq["ts"].min(), eq["ts"].max()
btc_slice = raw_bars.loc[(raw_bars.index >= deploy_start) & (raw_bars.index <= deploy_end), "close"]
btc_idx = btc_slice.index

# Composition bins by current MTM in log-return space.
# Letting winners run means MTM is NOT capped at +phi any more — extended bins.
BIN_EDGES = [-np.inf, -8*PHI, -4*PHI, -2*PHI, -PHI, 0.0, PHI, 4*PHI, np.inf]
BIN_LABELS = [
    f"< -{8*PHI*1e4:.0f}bp (stressed)",
    f"-{8*PHI*1e4:.0f} to -{4*PHI*1e4:.0f}bp",
    f"-{4*PHI*1e4:.0f} to -{2*PHI*1e4:.0f}bp",
    f"-{2*PHI*1e4:.0f} to -{PHI*1e4:.0f}bp",
    f"-{PHI*1e4:.0f} to 0bp",
    f"0 to +{PHI*1e4:.0f}bp (below TP)",
    f"+{PHI*1e4:.0f} to +{4*PHI*1e4:.0f}bp (running)",
    f"> +{4*PHI*1e4:.0f}bp (deep run)",
]
BIN_COLORS = ["#7a0d0d", "#c0392b", "#e67e22", "#f1c40f", "#bdc3c7", "#a9dfbf", "#2ecc71", "#16a085"]

closed = result.closed.copy()
if len(closed):
    closed["ts_entry"] = pd.to_datetime(closed["ts_entry"]).map(
        lambda x: x.tz_localize(None) if getattr(x, "tz", None) is not None else x
    )
    closed["ts_exit"] = pd.to_datetime(closed["ts_exit"]).map(
        lambda x: x.tz_localize(None) if getattr(x, "tz", None) is not None else x
    )


def reconstruct_composition(eq_df, closed_df, raw_bars_df, bin_edges, n_bins):
    eq_ts = pd.to_datetime(eq_df["ts"].values)
    n = len(eq_ts)
    counts = np.zeros((n, n_bins), dtype=int)
    if len(closed_df) == 0:
        return counts
    close_series = raw_bars_df["close"].reindex(pd.DatetimeIndex(eq_ts), method="ffill")
    close_arr = close_series.to_numpy(dtype=float)
    eq_idx_local = pd.DatetimeIndex(eq_ts)
    edges = np.asarray(bin_edges)
    for _, tr in closed_df.iterrows():
        i0 = int(eq_idx_local.searchsorted(tr["ts_entry"], side="right") - 1)
        i1 = int(eq_idx_local.searchsorted(tr["ts_exit"], side="right") - 1)
        i0 = max(0, i0)
        i1 = min(n - 1, i1)
        if i1 < i0:
            continue
        ep = float(tr["entry_price"])
        if ep <= 0:
            continue
        mtm = np.log(close_arr[i0:i1+1] / ep)
        bin_ix = np.clip(np.digitize(mtm, edges) - 1, 0, n_bins - 1)
        rng = np.arange(i0, i1 + 1)
        for k_idx, b in zip(rng, bin_ix):
            counts[k_idx, b] += 1
    return counts


t0 = time.perf_counter()
counts = reconstruct_composition(eq, closed, raw_bars, BIN_EDGES, len(BIN_LABELS))
print(f"[composition] {time.perf_counter()-t0:.1f}s")

# ---- DASHBOARD: 7-panel time-aligned stack -----------------------------
fig = plt.figure(figsize=(15, 22))
gs = gridspec.GridSpec(
    nrows=7, ncols=1, figure=fig,
    height_ratios=[1.2, 2.6, 1.0, 2.0, 1.6, 1.6, 2.0],
    hspace=0.18, top=0.97, bottom=0.04, left=0.06, right=0.97,
)
ax_btc = fig.add_subplot(gs[0])
ax_eq = fig.add_subplot(gs[1], sharex=ax_btc)
ax_dd = fig.add_subplot(gs[2], sharex=ax_btc)
ax_comp = fig.add_subplot(gs[3], sharex=ax_btc)
ax_score = fig.add_subplot(gs[4], sharex=ax_btc)
ax_unc = fig.add_subplot(gs[5], sharex=ax_btc)
ax_swim = fig.add_subplot(gs[6], sharex=ax_btc)
all_axes = [ax_btc, ax_eq, ax_dd, ax_comp, ax_score, ax_unc, ax_swim]
for a in all_axes:
    a.axvline(val_test_boundary, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    a.axvline(worst_dd_ts, color="black", linestyle=(0, (3, 1, 1, 1)), linewidth=1, alpha=0.5)

ax_btc.plot(btc_idx, btc_slice.values, color="#34495e", linewidth=0.6)
ax_btc.set_ylabel("BTC close ($)")
ax_btc.set_title(f"BTC spot (1-min)  ·  dotted: val|test  ·  dot-dash: worst-DD ts  ·  span {span_days:.1f}d")

realized = eq["realized_cum"].to_numpy() * 100
unreal = eq["unrealized"].to_numpy() * 100
total = realized + unreal
ax_eq.plot(eq["ts"], realized, color="#27ae60", linewidth=1.5, label="realized P&L (cum)")
ax_eq.fill_between(eq["ts"], 0, unreal, color="#e74c3c", alpha=0.18, label="unrealized MTM (open book)")
ax_eq.plot(eq["ts"], total, color="#2c3e50", linestyle="--", linewidth=1.0, alpha=0.7, label="total")
ax_eq.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
ax_eq.set_ylabel("% (log-returns)")
ax_eq.set_title(
    f"Equity  ·  realized {realized[-1]:+.2f}%  ·  total {total[-1]:+.2f}%  "
    f"·  ann. realized {annualized_realized*100:+.1f}%  ·  alpha vs BTC {prod_summary['alpha_over_btc']*100:+.2f}%"
)
ax_eq.legend(loc="upper left", ncol=3, frameon=False)

dd_pct = dd_series * 100
ax_dd.fill_between(eq["ts"], 0, dd_pct, color="#7f8c8d", alpha=0.4)
ax_dd.plot(eq["ts"], dd_pct, color="#7f8c8d", linewidth=0.5)
ax_dd.axhline(-max_paper_dd * 100, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
ax_dd.set_ylabel("DD (%)")
ax_dd.set_title(f"Drawdown on total equity  ·  max {-max_paper_dd*100:.2f}% at {worst_dd_ts.strftime('%Y-%m-%d %H:%M')}")

ts_arr = eq["ts"].values
bottom = np.zeros(len(eq))
for b in range(len(BIN_LABELS)):
    ax_comp.fill_between(
        ts_arr, bottom, bottom + counts[:, b],
        color=BIN_COLORS[b], alpha=0.85, label=BIN_LABELS[b], step="post",
    )
    bottom = bottom + counts[:, b]
ax_comp.set_ylabel("# open lots")
ax_comp.set_title(
    f"Portfolio composition  ·  let-winners-run extends ITM bands  ·  peak {peak_concurrent}/{MAX_CONCURRENT} lots"
)
ax_comp.legend(loc="upper left", ncol=4, frameon=False, fontsize=7)

ax_score.plot(eq["ts"], eq["p"], color="#95a5a6", linewidth=0.4, alpha=0.7, label="p (raw model prob)")
ax_score.axhline(P_99, color="#e67e22", linestyle="-", linewidth=1.0, label=f"P_99 = {P_99:.4f}")
fired = eq[eq["opened_this_step"].astype(bool)]
ax_score.scatter(
    fired["ts"], fired["p"], s=10, color="#27ae60", edgecolor="black",
    linewidth=0.2, zorder=3, label=f"fires (n={len(fired)})",
)
ax_score.set_ylabel("p")
ax_score.set_title("Score `p` with top-1% entry gate  ·  green dots = entries")
ax_score.legend(loc="upper left", ncol=3, frameon=False, fontsize=7)

ax_unc.plot(eq["ts"], eq["knowledge_unc"], color="#3498db", linewidth=0.4, alpha=0.7, label="knowledge_unc")
ax_unc.set_ylabel("MI")
ax_unc.set_title("Epistemic uncertainty (VE knowledge / MI)  ·  unused in P1+P3 (no unc gate)")
ax_unc.legend(loc="upper left", ncol=2, frameon=False, fontsize=7)

if len(closed) > 0:
    closed_sorted = closed.sort_values("ts_entry").reset_index(drop=True)
    for i, tr in closed_sorted.iterrows():
        # color by exit reason: tp_market green, tp blue, sl red, expiry grey
        if tr["exit_reason"] == "tp_market":
            color = "#27ae60"
        elif tr["exit_reason"] == "tp":
            color = "#3498db"
        elif tr["exit_reason"] == "sl":
            color = "#e74c3c"
        else:
            color = "#bdc3c7"
        ax_swim.hlines(i, tr["ts_entry"], tr["ts_exit"], color=color, alpha=0.65, linewidth=0.6)
ax_swim.set_ylabel("trade idx (entry order)")
n_tp_market = int((result.closed["exit_reason"] == "tp_market").sum()) if len(result.closed) else 0
n_tp = int((result.closed["exit_reason"] == "tp").sum()) if len(result.closed) else 0
ax_swim.set_title(
    f"Position swimlane  ·  green=tp_market ({n_tp_market})  ·  blue=tp ({n_tp})  ·  hold-time = bar length"
)
ax_swim.set_xlabel("ts")

for a in all_axes[:-1]:
    plt.setp(a.get_xticklabels(), visible=False)
ax_swim.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
ax_swim.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

fig.savefig(OUT_DIR / "dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[save] {OUT_DIR / 'dashboard.png'}")

# ---- Trade-window zoom-ins -------------------------------------------------
if len(closed) > 0:
    closed["hold_min"] = (closed["ts_exit"] - closed["ts_entry"]).dt.total_seconds() / 60.0
    raw_idx_local = raw_bars.index
    raw_low_arr = raw_bars["low"].to_numpy()
    raw_close_arr = raw_bars["close"].to_numpy()

    worst = np.zeros(len(closed))
    for i, (_, tr) in enumerate(closed.iterrows()):
        lo = int(np.searchsorted(raw_idx_local, np.datetime64(tr["ts_entry"]), side="right"))
        hi = int(np.searchsorted(raw_idx_local, np.datetime64(tr["ts_exit"]), side="right"))
        if hi > lo:
            worst[i] = float(np.log(raw_low_arr[lo:hi].min() / tr["entry_price"]))
    closed["worst_mtm"] = worst

    # Local-min rank
    ranks = np.full(len(closed), np.nan)
    for i, (_, tr) in enumerate(closed.iterrows()):
        ts_entry = pd.Timestamp(tr["ts_entry"])
        i0 = int(np.searchsorted(raw_idx_local, np.datetime64(ts_entry - pd.Timedelta(minutes=60)), side="left"))
        i1 = int(np.searchsorted(raw_idx_local, np.datetime64(ts_entry + pd.Timedelta(minutes=60)), side="right"))
        if i1 <= i0 + 1:
            continue
        window = raw_close_arr[i0:i1]
        ep = float(tr["entry_price"])
        ranks[i] = float((window < ep).sum()) / float(len(window) - 1)
    closed["local_min_rank"] = ranks

    # Choose interesting trades for zoom-in
    n_per = 4
    rng = np.random.default_rng(0)
    fastest = closed.nsmallest(n_per, "hold_min")
    slowest = closed.nlargest(n_per, "hold_min")
    biggest = closed.nlargest(n_per, "gross_log_return")  # biggest run-ups
    rand_idx = rng.choice(closed.index, size=min(n_per, len(closed)), replace=False)
    randoms = closed.loc[rand_idx]
    sample = pd.concat([fastest, slowest, biggest, randoms]).drop_duplicates(subset=["ts_entry"]).reset_index(drop=True)
    sample["group"] = (
        ["fastest"] * len(fastest) + ["slowest"] * len(slowest)
        + ["biggest"] * len(biggest) + ["random"] * len(randoms)
    )[:len(sample)]

    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), sharey=False)
    for ax, (_, tr) in zip(axes.flat, sample.iterrows()):
        ts_in = pd.Timestamp(tr["ts_entry"])
        ts_out = pd.Timestamp(tr["ts_exit"])
        t0 = ts_in - pd.Timedelta(minutes=60)
        t1 = ts_out + pd.Timedelta(minutes=30)
        i0 = int(np.searchsorted(raw_idx_local, np.datetime64(t0), side="left"))
        i1 = int(np.searchsorted(raw_idx_local, np.datetime64(t1), side="right"))
        if i1 <= i0:
            ax.axis("off")
            continue
        win_idx = raw_idx_local[i0:i1]
        win_close = raw_close_arr[i0:i1]
        ax.plot(win_idx, win_close, color="#34495e", linewidth=0.7)
        ax.axvline(ts_in, color="#27ae60", linewidth=1.0, alpha=0.9)
        ax.axvline(ts_out, color="#e67e22", linewidth=1.0, alpha=0.9)
        ax.axhline(float(tr["entry_price"]), color="#27ae60", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.axhline(float(tr["tp_price"]), color="#e67e22", linestyle=":", linewidth=0.7, alpha=0.6)
        hold_min = (ts_out - ts_in).total_seconds() / 60.0
        gross_bp = tr["gross_log_return"] * 1e4
        ax.set_title(
            f"{tr.get('group', '?')}  ·  hold {hold_min:.0f}min  ·  +{gross_bp:.0f}bp",
            fontsize=8,
        )
        ax.tick_params(axis="x", rotation=0, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.suptitle(
        "Trade-window zoom  ·  ±60min around entry  ·  green=entry, orange=exit  ·  "
        "biggest = largest realized gain (let-winners-run effect)",
        y=1.005, fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "trade_zoom_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {OUT_DIR / 'trade_zoom_grid.png'}")

    # ---- Local-min rank distribution ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ranks_clean = closed["local_min_rank"].dropna().to_numpy()
    axes[0].hist(ranks_clean, bins=40, color="#3498db", alpha=0.7, edgecolor="black", linewidth=0.4)
    axes[0].axvline(
        np.median(ranks_clean), color="#e67e22", linestyle="--", linewidth=1.5,
        label=f"median = {np.median(ranks_clean):.2f}",
    )
    axes[0].axvline(0.5, color="gray", linestyle=":", linewidth=1.0, label="uniform reference")
    axes[0].set_xlabel("entry rank within ±60-min window  (0=local min, 1=local max)")
    axes[0].set_ylabel("# trades")
    axes[0].set_title(f"Local-min-rank distribution  ·  n={len(ranks_clean)}")
    axes[0].legend()
    axes[0].set_xlim(0, 1)
    sorted_r = np.sort(ranks_clean)
    ecdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    axes[1].plot(sorted_r, ecdf, color="#3498db")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle=":", alpha=0.6, label="uniform reference")
    axes[1].set_xlabel("local-min rank")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title("ECDF — above diagonal means entries cluster near local minima")
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "local_min_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {OUT_DIR / 'local_min_rank.png'}")

    # ---- Hold and worst MTM -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sorted_h = np.sort(closed["hold_min"].to_numpy())
    ecdf = np.arange(1, len(sorted_h) + 1) / len(sorted_h)
    axes[0].semilogx(sorted_h, ecdf, color="#3498db")
    for q, lab in [(0.50, "p50"), (0.90, "p90"), (0.99, "p99")]:
        v = float(np.quantile(closed["hold_min"], q))
        axes[0].axvline(v, color="gray", linestyle=":", alpha=0.5)
        axes[0].text(v, q, f"  {lab}={v:.0f}min", verticalalignment="bottom", fontsize=8)
    axes[0].set_xlabel("hold time (log min)")
    axes[0].set_ylabel("ECDF")
    axes[0].set_title("Hold-time ECDF (let-winners-run can extend holds significantly)")

    worst_bp = closed["worst_mtm"].to_numpy() * 1e4
    axes[1].hist(np.clip(worst_bp, -1500, 0), bins=60, color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.2)
    for q in [0.05, 0.25, 0.50, 0.75]:
        v = float(np.quantile(worst_bp, q))
        axes[1].axvline(v, color="gray", linestyle=":", alpha=0.5)
        axes[1].text(
            v, axes[1].get_ylim()[1] * 0.95, f"p{int(q*100)}={v:.0f}",
            rotation=90, va="top", fontsize=8,
        )
    axes[1].set_xlabel("worst MTM observed (bp, clipped at -1500)")
    axes[1].set_ylabel("# positions")
    axes[1].set_title("Worst-MTM per position (path-peeking)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "hold_and_worst_mtm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {OUT_DIR / 'hold_and_worst_mtm.png'}")

    # ---- Gross-PnL distribution (NEW — specific to let-winners-run) --------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    gross_bp = closed["gross_log_return"].to_numpy() * 1e4
    axes[0].hist(gross_bp, bins=80, color="#27ae60", alpha=0.7, edgecolor="black", linewidth=0.2)
    axes[0].axvline(PHI * 1e4, color="#e67e22", linestyle="--", linewidth=1.2,
                    label=f"static TP = +{PHI*1e4:.0f}bp")
    axes[0].axvline(0, color="gray", linestyle="-", linewidth=0.5)
    for q in [0.05, 0.50, 0.90, 0.99]:
        v = float(np.quantile(gross_bp, q))
        axes[0].axvline(v, color="black", linestyle=":", alpha=0.4)
        axes[0].text(v, axes[0].get_ylim()[1] * 0.9, f"p{int(q*100)}={v:.0f}",
                     rotation=90, va="top", fontsize=7)
    axes[0].set_xlabel("gross log return (bp)")
    axes[0].set_ylabel("# trades")
    axes[0].set_title(
        f"Per-trade gross PnL distribution  ·  median {np.median(gross_bp):.1f}bp  "
        f"·  vs static TP +{PHI*1e4:.0f}bp"
    )
    axes[0].legend()

    # Cum PnL by trade order
    sorted_by_exit = closed.sort_values("ts_exit")
    cum_pnl = (sorted_by_exit["gross_log_return"] - COST).cumsum() * LOT_SIZE
    axes[1].plot(pd.to_datetime(sorted_by_exit["ts_exit"]), cum_pnl * 100, color="#27ae60", linewidth=1.0)
    axes[1].set_xlabel("exit ts")
    axes[1].set_ylabel("cumulative realized PnL (%)")
    axes[1].set_title(f"Cumulative realized PnL over time  ·  final {cum_pnl.iloc[-1]*100:+.2f}%")
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "gross_pnl_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {OUT_DIR / 'gross_pnl_distribution.png'}")

    closed.to_parquet(OUT_DIR / "closed_trades.parquet", index=False)
    print(f"[save] {OUT_DIR / 'closed_trades.parquet'}  ({len(closed):,} rows)")

print()
print("=" * 70)
print("HEADLINE  ·  P1+P3 winning spec")
print("=" * 70)
print(f"  Signals fired         : {prod_summary['n_signals']:,}")
print(f"  Trades closed         : {prod_summary['n_closed']:,}  (open at end {prod_summary['n_open_at_end']})")
print(f"  Realized              : {prod_summary['realized_log_return']*100:+.3f}%")
print(f"  Unrealized at end     : {prod_summary['unrealized_at_end']*100:+.3f}%")
print(f"  TOTAL                 : {prod_summary['total_log_return']*100:+.3f}%")
print(f"  Realized annualized   : {prod_summary['realized_annualized']*100:+.2f}%")
print(f"  Sharpe (daily ann.)   : {prod_summary['sharpe_realized_daily']:.2f}")
print(f"  Max paper DD          : {prod_summary['max_paper_dd']*100:.2f}%")
print(f"  Calmar                : {prod_summary['calmar']:.2f}")
print(f"  vs BTC                : {prod_summary['btc_log_return']*100:+.2f}%  → alpha {prod_summary['alpha_over_btc']*100:+.2f}%")
print(f"  Per-trade gross median: {np.median(closed['gross_log_return']) * 1e4:+.2f}bp" if len(closed) else "")
print(f"  Per-trade gross max   : {closed['gross_log_return'].max() * 1e4:+.2f}bp" if len(closed) else "")
