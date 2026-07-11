"""Reporting helpers: headline table, equity ladder, drawdown, regime attribution.

Consumes ``SimResult`` objects produced by the simulator and turns them
into pandas DataFrames suitable for direct display in the notebook. Plot
helpers are lazy on ``matplotlib``.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-spec headline table
# ---------------------------------------------------------------------------


def _annualization_factor_from_cadence_minutes(cadence_minutes: float) -> float:
    """Bars-per-year given a per-boundary cadence in minutes (24/7 markets)."""
    if cadence_minutes <= 0:
        return 1.0
    return (365.0 * 24.0 * 60.0) / float(cadence_minutes)


def _sharpe(per_step_returns: np.ndarray, *, annualization: float) -> float:
    """Annualized Sharpe assuming zero-rate; NaN if std == 0."""
    r = np.asarray(per_step_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std(ddof=1) < 1e-18:
        return float("nan")
    return float(np.sqrt(annualization) * r.mean() / r.std(ddof=1))


def _max_drawdown_log(equity_log: np.ndarray) -> float:
    """Max drawdown on cumulative log-equity (log-return units, magnitude)."""
    e = np.asarray(equity_log, dtype=float)
    if len(e) == 0:
        return 0.0
    peaks = np.maximum.accumulate(e)
    dd = e - peaks
    return float(-dd.min())


def headline_row(
    spec_name: str,
    closed: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    cost_per_trade: float,
    cadence_minutes: float,
) -> dict:
    """Compute the headline metrics row for one SimResult.

    ``cadence_minutes`` is REQUIRED — at 1-min cadence (M=20) the simulator
    walks one row per minute (cadence_minutes=1), whereas the legacy 20-min-
    boundary mode uses cadence_minutes=20. A default of 20 used to under-
    annualize 1-min Sharpe by sqrt(20). Pass ``r.config.get('cadence_minutes', 20.0)``
    or wire the value explicitly from upstream.

    Returns a dict ready to be appended to a DataFrame.
    """
    n_trades = int(len(closed))
    if n_trades == 0:
        return {
            "spec": spec_name, "n_trades": 0, "hit_rate": float("nan"),
            "mean_net_log_pnl": 0.0, "std_net_log_pnl": float("nan"),
            "total_log_pnl": 0.0,
            "ann_sharpe": float("nan"),
            "sharpe_realized": float("nan"),
            "sharpe_equity_step": float("nan"),
            "max_drawdown_log": 0.0,
            "trades_per_day": 0.0,
            "tp_rate": float("nan"), "expiry_rate": float("nan"),
            "bulk_close_rate": float("nan"),
        }
    net_per_trade = (closed["gross_log_return"] - cost_per_trade) * closed["size"]
    is_hit = closed["exit_reason"].eq("tp").astype(int)
    annualization = _annualization_factor_from_cadence_minutes(cadence_minutes)

    # Per-step returns from the equity panel (mixes realized + unrealized
    # MTM) — annualized on the per-bar cadence.
    eq = equity["equity"].to_numpy() if "equity" in equity.columns else np.zeros(0)
    step_returns = np.diff(eq, prepend=0.0) if len(eq) else np.zeros(0)
    sharpe_equity_step = _sharpe(step_returns, annualization=annualization)

    # Per-closed-trade realized Sharpe: weighted_net_log_return per trade,
    # annualized on a *trade-frequency* basis (n_trades / span). Avoids the
    # MTM contribution that contaminates the equity-step series.
    span_days = (
        (equity["ts"].max() - equity["ts"].min()).total_seconds() / 86400.0
        if "ts" in equity.columns and len(equity) >= 2 else 0.0
    )
    if span_days > 0 and n_trades > 1:
        trades_per_year = n_trades * (365.0 / span_days)
        sharpe_realized = _sharpe(
            net_per_trade.to_numpy(dtype=float), annualization=trades_per_year
        )
    else:
        sharpe_realized = float("nan")

    tp_rate = float((closed["exit_reason"] == "tp").mean())
    expiry_rate = float((closed["exit_reason"] == "expiry").mean())
    bulk_rate = float(
        closed["exit_reason"].astype(str).str.startswith("bulk_").mean()
    )

    return {
        "spec": spec_name,
        "n_trades": n_trades,
        "hit_rate": float(is_hit.mean()),
        "mean_net_log_pnl": float(net_per_trade.mean()),
        "std_net_log_pnl": float(net_per_trade.std(ddof=1)) if n_trades > 1 else float("nan"),
        "total_log_pnl": float(eq[-1]) if len(eq) else float(net_per_trade.sum()),
        # ``ann_sharpe`` retained as alias of ``sharpe_equity_step`` for
        # back-compat with notebooks / sort logic in headline_table.
        "ann_sharpe": sharpe_equity_step,
        "sharpe_equity_step": sharpe_equity_step,
        "sharpe_realized": sharpe_realized,
        "max_drawdown_log": _max_drawdown_log(eq),
        "trades_per_day": float(n_trades / span_days) if span_days > 0 else float("nan"),
        "tp_rate": tp_rate,
        "expiry_rate": expiry_rate,
        "bulk_close_rate": bulk_rate,
    }


def headline_table(
    results: Iterable, *, cadence_minutes: float | None = None
) -> pd.DataFrame:
    """One row per ``SimResult`` with the headline metrics. Sorted by Sharpe.

    ``cadence_minutes`` is the per-row cadence the simulator was driven at —
    20.0 for the legacy 20-min-boundary mode, 1.0 for the 1-min canonical
    spec. If ``None``, the function reads ``r.config['cadence_minutes']``
    from each SimResult and falls back to 20.0 only if absent (preserves
    the old default for legacy callers but logs no warning).
    """
    rows = []
    for r in results:
        cost = r.config.get("cost_per_trade_override")
        if cost is None:
            # Fall back to phi-default if not in config
            cost = 0.0005
        cad = (
            float(cadence_minutes)
            if cadence_minutes is not None
            else float(r.config.get("cadence_minutes", 20.0))
        )
        rows.append(
            headline_row(
                r.spec_name,
                r.closed,
                r.equity,
                cost_per_trade=float(cost),
                cadence_minutes=cad,
            )
        )
    df = pd.DataFrame(rows)
    if "ann_sharpe" in df.columns:
        df = df.sort_values("ann_sharpe", ascending=False, na_position="last").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Deflated Sharpe (Bailey & López de Prado)
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    sharpe: float, n_obs: int, *, n_trials: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """Probability that the observed Sharpe exceeds a *threshold* Sharpe,
    given the multiple-testing context implied by ``n_trials``.

    Reference: Bailey & López de Prado (2014). For ``n_obs`` per-step return
    observations (the equity diff length, NOT n_trades), and ``n_trials``
    strategy variants screened, returns DSR in [0, 1]. Higher = more robust.
    Returns NaN when the underlying Sharpe distribution is degenerate.
    """
    from scipy.stats import norm

    if not np.isfinite(sharpe) or n_obs < 2 or n_trials < 1:
        return float("nan")
    # Variance of estimated SR under non-normal returns (Lo 2002)
    sigma_sr2 = (
        (1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe * sharpe) / (n_obs - 1.0)
    )
    if sigma_sr2 <= 0:
        return float("nan")
    sigma_sr = math.sqrt(sigma_sr2)
    # Expected max Sharpe under H0 over n_trials independent trials
    e_max = (
        (1.0 - np.euler_gamma) * norm.ppf(1.0 - 1.0 / max(n_trials, 1))
        + np.euler_gamma * norm.ppf(1.0 - 1.0 / (max(n_trials, 1) * math.e))
    )
    sr_threshold = e_max * sigma_sr
    # P(observed SR > threshold | true SR = 0), penalized for multiple testing
    z = (sharpe - sr_threshold) / sigma_sr
    return float(norm.cdf(z))


# ---------------------------------------------------------------------------
# Per-regime attribution
# ---------------------------------------------------------------------------


def regime_attribution(
    closed: pd.DataFrame, *, n_terciles: int = 3, cost_per_trade: float
) -> pd.DataFrame:
    """Per-regime-tercile breakdown of trades and P&L.

    Uses the ``regime_quantile_at_entry`` column on the closed-trade ledger
    (stamped by the simulator at entry time).
    """
    if closed.empty or "regime_quantile_at_entry" not in closed.columns:
        return pd.DataFrame()
    df = closed.copy()
    df["net_log_pnl"] = (df["gross_log_return"] - cost_per_trade) * df["size"]
    # Tercile rank within entries (alternative: by raw quantile thresholds).
    # Use ``labels=False`` then map to names — ``pd.qcut`` with explicit
    # ``labels=[...]`` crashes when ``duplicates="drop"`` shrinks the bucket
    # count below the label-list length.
    name_map = {0: "low", 1: "med", 2: "high"}
    tercile_idx = pd.qcut(
        df["regime_quantile_at_entry"], n_terciles,
        labels=False, duplicates="drop",
    )
    df["regime_tercile"] = tercile_idx.map(name_map).astype(
        pd.CategoricalDtype(categories=["low", "med", "high"], ordered=True)
    )
    grouped = df.groupby("regime_tercile", observed=False).agg(
        n_trades=("net_log_pnl", "size"),
        hit_rate=("exit_reason", lambda s: float((s == "tp").mean())),
        mean_net_log_pnl=("net_log_pnl", "mean"),
        total_net_log_pnl=("net_log_pnl", "sum"),
    )
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------------


def time_to_tp_analysis(
    closed: pd.DataFrame,
    raw_bars: pd.DataFrame,
    *,
    phi: float,
    look_ahead_bars: int = 14400,  # 14400 1-min bars = 10 days
) -> pd.DataFrame:
    """For each closed position, find when TP actually fired (if any) and when
    it *would* have fired had we held longer.

    Directly tests the "patient wait-for-level" thesis: for non-TP exits, does
    the path eventually cross +φ above entry? If yes, the M-bar horizon was
    the bug; if no for many cases, mean-reversion / regime decay is dominant.

    Returns one row per closed position with:
    - ``time_to_exit_bars`` — 1-min bars from entry to actual exit
    - ``would_have_hit_tp_bars`` — bars from entry to when high ≥ entry·exp(+φ),
      searching forward up to ``look_ahead_bars``; NaN if level never reached
    - ``exit_reason`` — passthrough from the ledger
    - ``gross_log_return`` — passthrough; meaningful for ledger context

    Causality note: this is a *post-hoc* analysis (uses future bars after the
    actual exit). It tells you whether the strategy's design left money on
    the table, not what the strategy itself observed at decision time.
    """
    if closed.empty:
        return pd.DataFrame()
    raw = raw_bars.copy()
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw.sort_index()
    raw_high = raw["high"].to_numpy(dtype=float)
    raw_idx = raw.index.values  # tz-naive datetime64

    def _to_naive(ts) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return t

    rows = []
    for _, row in closed.iterrows():
        entry_price = float(row["entry_price"])
        tp_target = entry_price * math.exp(float(phi))
        ts_entry_n = np.datetime64(_to_naive(row["ts_entry"]))
        ts_exit_n = np.datetime64(_to_naive(row["ts_exit"]))
        n_entry = int(np.searchsorted(raw_idx, ts_entry_n, side="left"))
        # If exact match, that's the index; otherwise searchsorted gives the
        # insertion point — use it as a safe asof.
        if n_entry >= len(raw_idx) or raw_idx[n_entry] != ts_entry_n:
            n_entry = max(0, n_entry - 1)
        n_exit = int(np.searchsorted(raw_idx, ts_exit_n, side="left"))
        if n_exit >= len(raw_idx) or raw_idx[n_exit] != ts_exit_n:
            n_exit = max(n_entry, n_exit - 1)

        # Look forward from entry up to look_ahead_bars for first high >= tp_target
        end = min(len(raw_high), n_entry + 1 + int(look_ahead_bars))
        forward = raw_high[n_entry + 1 : end] if n_entry + 1 < end else np.array([])
        hit_idx = np.argmax(forward >= tp_target) if len(forward) else -1
        would_have_hit = (
            int(hit_idx + 1) if (len(forward) and forward[hit_idx] >= tp_target) else -1
        )
        rows.append(
            {
                "k_entry": int(row["k_entry"]),
                "ts_entry": row["ts_entry"],
                "exit_reason": row["exit_reason"],
                "time_to_exit_bars": int(max(0, n_exit - n_entry)),
                "would_have_hit_tp_bars": int(would_have_hit) if would_have_hit > 0 else None,
                "gross_log_return": float(row["gross_log_return"]),
                "size": float(row["size"]),
            }
        )
    return pd.DataFrame(rows)


def time_to_tp_summary(ttp_df: pd.DataFrame) -> dict:
    """Top-line numbers for the time-to-TP analysis: what fraction of
    non-TP exits would have eventually hit, and at what typical lag."""
    if ttp_df.empty:
        return {}
    n = len(ttp_df)
    tp_now = (ttp_df["exit_reason"] == "tp").sum()
    non_tp = ttp_df[ttp_df["exit_reason"] != "tp"]
    n_non_tp = len(non_tp)
    if n_non_tp == 0:
        return {
            "n_closed": int(n), "n_tp": int(tp_now),
            "n_non_tp": 0, "frac_non_tp_eventually_hit": 0.0,
        }
    hit_eventually = non_tp["would_have_hit_tp_bars"].notna()
    n_eventual = int(hit_eventually.sum())
    eventual_bars = non_tp.loc[hit_eventually, "would_have_hit_tp_bars"]
    return {
        "n_closed": int(n),
        "n_tp": int(tp_now),
        "n_non_tp": int(n_non_tp),
        "n_non_tp_eventually_hit": n_eventual,
        "frac_non_tp_eventually_hit": float(n_eventual / n_non_tp),
        "median_eventual_bars": float(eventual_bars.median()) if n_eventual else float("nan"),
        "p25_eventual_bars": float(eventual_bars.quantile(0.25)) if n_eventual else float("nan"),
        "p75_eventual_bars": float(eventual_bars.quantile(0.75)) if n_eventual else float("nan"),
        "p95_eventual_bars": float(eventual_bars.quantile(0.95)) if n_eventual else float("nan"),
    }


def cluster_summary(cluster_log: pd.DataFrame) -> dict:
    """Aggregate cluster-log statistics into a compact summary dict."""
    if cluster_log.empty:
        return {"n_clusters": 0}
    return {
        "n_clusters": int(len(cluster_log)),
        "mean_duration_boundaries": float(cluster_log["duration_boundaries"].mean()),
        "median_duration_boundaries": float(cluster_log["duration_boundaries"].median()),
        "mean_n_entries": float(cluster_log["n_entries"].mean()),
        "mean_cluster_pnl": float(cluster_log["cluster_pnl"].mean()),
        "max_cluster_pnl": float(cluster_log["cluster_pnl"].max()),
        "min_cluster_pnl": float(cluster_log["cluster_pnl"].min()),
        "n_ending_bulk_regime": int((cluster_log["end_reason"] == "bulk_regime").sum()),
        "n_ending_bulk_unc": int((cluster_log["end_reason"] == "bulk_unc").sum()),
        "n_ending_bulk_cluster_loss": int(
            (cluster_log["end_reason"] == "bulk_cluster_loss").sum()
        ),
        "n_ending_expiry_flat": int((cluster_log["end_reason"] == "expiry_flat").sum()),
        "n_ending_tp": int((cluster_log["end_reason"] == "tp").sum()),
    }


# ---------------------------------------------------------------------------
# Plot helpers (lazy matplotlib import)
# ---------------------------------------------------------------------------


def plot_equity_ladder(
    results: Iterable,
    *,
    ax=None,
    split_marker_ts: Optional[pd.Timestamp] = None,
):
    """One line per SimResult, equity over time. Optional vertical line at
    the val-to-test boundary."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 5.4))
    palette = plt.get_cmap("tab10")
    for i, r in enumerate(results):
        if r.equity.empty:
            continue
        ax.plot(
            r.equity["ts"], r.equity["equity"],
            color=palette(i % 10), label=r.spec_name, alpha=0.9,
        )
    if split_marker_ts is not None:
        ax.axvline(
            split_marker_ts, linestyle="--", color="gray", alpha=0.6,
            label="val → test",
        )
    ax.axhline(0.0, linestyle=":", color="gray", alpha=0.5)
    ax.set(
        xlabel="ts", ylabel="cumulative log-return (net)",
        title="Equity ladder (net cumulative log-return per strategy spec)",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_drawdown(result, *, ax=None):
    """Drawdown of one spec over time."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
    if result.equity.empty:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        return ax
    e = result.equity["equity"].to_numpy()
    peaks = np.maximum.accumulate(e)
    dd = e - peaks
    ax.fill_between(result.equity["ts"], dd, 0.0, color="C3", alpha=0.4)
    ax.set(xlabel="ts", ylabel="drawdown (log-return)",
           title=f"Drawdown: {result.spec_name}  (max = {-dd.min():.4f})")
    ax.grid(alpha=0.3)
    return ax


def plot_regime_attribution(closed: pd.DataFrame, *, cost_per_trade: float, ax=None):
    """Bar chart of total net log-PnL per regime tercile."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    df = regime_attribution(closed, cost_per_trade=cost_per_trade)
    if df.empty:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        return ax
    ax.bar(df["regime_tercile"].astype(str), df["total_net_log_pnl"], color="C0", alpha=0.85)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set(
        xlabel="regime tercile (low/med/high vol)",
        ylabel="total net log-PnL",
        title="P&L attribution by entry-time vol regime",
    )
    for i, row in df.iterrows():
        ax.text(
            i, row["total_net_log_pnl"],
            f"n={int(row['n_trades'])}\nhit={row['hit_rate']:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Trade forensics — kernels behind SimResult's owner-attached methods.
# Ported from notebook 05's inline helpers (2026-07-11): the result object
# owns its trade analysis; callers inject the market context (raw bars).
# ---------------------------------------------------------------------------


def trade_composition(
    equity: pd.DataFrame,
    closed: pd.DataFrame,
    raw_bars: pd.DataFrame,
    *,
    bin_edges: np.ndarray,
) -> dict:
    """Open-book composition per equity step, bucketed by current MTM.

    For each equity row, every open trade is marked to the last known
    close and assigned to an MTM bucket (``bin_edges``, log-return units,
    ``len(bin_edges) - 1`` buckets). Returns a dict of aligned arrays:

    - ``counts``     (n_steps, n_bins) — open lots per MTM bucket
    - ``sum_mtm``    (n_steps,)        — summed unweighted open MTM
    - ``n_open``     (n_steps,)        — open-lot count
    - ``worst_mtm``  (n_steps,)        — deepest open-lot MTM (NaN if flat)
    """
    edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(edges) - 1
    if n_bins < 1:
        raise ValueError("bin_edges must define at least one bucket")
    eq_idx = pd.DatetimeIndex(pd.to_datetime(equity["ts"].values))
    n = len(eq_idx)
    counts = np.zeros((n, n_bins), dtype=int)
    sum_mtm = np.zeros(n, dtype=float)
    n_open = np.zeros(n, dtype=int)
    worst_mtm = np.full(n, np.nan, dtype=float)
    if len(closed) == 0 or n == 0:
        return {
            "counts": counts, "sum_mtm": sum_mtm,
            "n_open": n_open, "worst_mtm": worst_mtm,
        }
    close_arr = (
        raw_bars["close"].reindex(eq_idx, method="ffill").to_numpy(dtype=float)
    )
    entry_ts = closed["ts_entry"].to_numpy()
    exit_ts = closed["ts_exit"].to_numpy()
    entry_px = closed["entry_price"].to_numpy(dtype=float)
    i0s = np.maximum(eq_idx.searchsorted(entry_ts, side="right") - 1, 0)
    i1s = np.minimum(eq_idx.searchsorted(exit_ts, side="right") - 1, n - 1)
    for i0, i1, ep in zip(i0s, i1s, entry_px):
        if i1 < i0 or ep <= 0:
            continue
        mtm = np.log(close_arr[i0 : i1 + 1] / ep)
        bin_ix = np.clip(np.digitize(mtm, edges) - 1, 0, n_bins - 1)
        rows = np.arange(i0, i1 + 1)
        np.add.at(counts, (rows, bin_ix), 1)
        sum_mtm[rows] += mtm
        n_open[rows] += 1
        worst_mtm[rows] = np.where(
            np.isnan(worst_mtm[rows]), mtm, np.fmin(worst_mtm[rows], mtm)
        )
    return {
        "counts": counts, "sum_mtm": sum_mtm,
        "n_open": n_open, "worst_mtm": worst_mtm,
    }


def entry_local_min_rank(
    closed: pd.DataFrame,
    raw_bars: pd.DataFrame,
    *,
    window_minutes: int = 60,
) -> np.ndarray:
    """How close each entry sat to a local price minimum, in [0, 1].

    Rank = fraction of closes within ±``window_minutes`` of the entry that
    are BELOW the entry price. 0 means the entry was the local minimum;
    1 means it was the local maximum. NaN when the window is degenerate.
    """
    ranks = np.full(len(closed), np.nan)
    if len(closed) == 0:
        return ranks
    raw_idx = raw_bars.index
    raw_close = raw_bars["close"].to_numpy(dtype=float)
    half = pd.Timedelta(minutes=int(window_minutes))
    for i, (ts_entry, entry_price) in enumerate(
        zip(closed["ts_entry"], closed["entry_price"])
    ):
        ts_entry = pd.Timestamp(ts_entry)
        i0 = int(raw_idx.searchsorted(ts_entry - half, side="left"))
        i1 = int(raw_idx.searchsorted(ts_entry + half, side="right"))
        if i1 <= i0 + 1:
            continue
        window = raw_close[i0:i1]
        ranks[i] = float((window < float(entry_price)).sum()) / float(
            len(window) - 1
        )
    return ranks


def select_trade_sample(
    closed: pd.DataFrame,
    raw_bars: pd.DataFrame,
    *,
    n_per_group: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Representative trades for zoom-in inspection.

    Four groups: fastest / slowest holds, deepest intra-trade worst MTM,
    and a seeded random draw — deduplicated by entry timestamp. Adds
    ``hold_min``, ``worst_mtm``, and ``group`` columns; the input frame is
    not mutated.
    """
    if len(closed) == 0:
        return closed.copy()
    df = closed.copy().reset_index(drop=True)
    df["hold_min"] = (
        (df["ts_exit"] - df["ts_entry"]).dt.total_seconds() / 60.0
    )
    raw_idx = raw_bars.index
    raw_low = raw_bars["low"].to_numpy(dtype=float)
    worst = np.zeros(len(df))
    for i, (ts_entry, ts_exit, entry_price) in enumerate(
        zip(df["ts_entry"], df["ts_exit"], df["entry_price"])
    ):
        lo = int(raw_idx.searchsorted(np.datetime64(ts_entry), side="right"))
        hi = int(raw_idx.searchsorted(np.datetime64(ts_exit), side="right"))
        if hi > lo:
            worst[i] = float(np.log(raw_low[lo:hi].min() / float(entry_price)))
    df["worst_mtm"] = worst
    fastest = df.nsmallest(n_per_group, "hold_min")
    slowest = df.nlargest(n_per_group, "hold_min")
    deepest = df.nsmallest(n_per_group, "worst_mtm")
    rng = np.random.default_rng(seed)
    random_idx = rng.choice(df.index, size=min(n_per_group, len(df)), replace=False)
    randoms = df.loc[random_idx]
    selected = (
        pd.concat([fastest, slowest, deepest, randoms])
        .drop_duplicates(subset=["ts_entry"])
        .reset_index(drop=True)
    )
    selected["group"] = (
        ["fastest"] * len(fastest)
        + ["slowest"] * len(slowest)
        + ["worst-MTM"] * len(deepest)
        + ["random"] * len(randoms)
    )[: len(selected)]
    return selected
