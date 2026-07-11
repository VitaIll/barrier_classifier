# Production Operations Runbook

How to take the engine from research artifacts to live Binance trading,
operate it, and recover it. Companion to [ENGINE.md](ENGINE.md) (the
architectural contract). **Read the residual-risk register at the bottom
before arming live execution — this document does not claim the absence
of risk; it enumerates and bounds it.**

## 1. Safety ladder

Live trading is gated three times; each gate is crossed explicitly:

| Gate | Default | Crossed by |
|---|---|---|
| Network | Binance **testnet** (`testnet.binance.vision`) | `--mainnet` |
| Execution | **dry-run** (orders built + validated against real exchange filters, not sent) | `--execute` |
| Credentials | none | `BINANCE_API_KEY` / `BINANCE_API_SECRET` env vars |

Run each rung for long enough to trust it: dry-run on testnet → execute
on testnet (play money, full order path incl. fills/duplicates/rate
limits) → dry-run on mainnet (real market data, no orders) → execute on
mainnet. **Never skip the testnet-execute rung** — it is the only place
the full order path is exercised without capital at risk.

## 2. Topology — two processes

Data acquisition and trading are **separate processes** that share only a
durable feed store (SQLite/WAL):

```
[ feed process ]  Binance --> BinanceKlineSource --> FeedWriter --> feed.db
                                                                      |
[ engine process ] feed.db --> FeedSource --> Engine --> BinanceBroker --> orders
```

The engine holds no market-data connection; it consumes bars from the
store. The feed process holds no credentials (public data) and no engine
state. Either can restart independently — bars keep landing while the
engine is down; resume tails the feed from where it stopped.

## 3. Bring-up

```bash
# 0. Once: research artifacts -> model registry (v0001 + ACTIVE pointer)
python -m src.engine import-model --dataset-dir data/model_dataset

# 1. Verify on real history: research-cache parity + feed-topology parity
python scripts/validate_engine_replay.py
python scripts/validate_feed_chain.py

# 2. Start the DATA FEED (its own process/terminal; public data, no keys)
python -m src.engine feed --feed runtime/feed.db

# 3. Trade — testnet, dry-run (default): consume the feed, no orders sent
python -m src.engine live --feed runtime/feed.db --store runtime/live.db \
    --alert-webhook https://hooks.slack.com/services/...

# 4. Testnet, executing:
BINANCE_API_KEY=... BINANCE_API_SECRET=... \
python -m src.engine live --feed runtime/feed.db --execute \
    --trade-capital 10000 --store runtime/live.db

# 5. Mainnet (after the ladder): add --mainnet to BOTH feed and live;
#    start SMALL trade-capital.
```

Startup mechanics: the FEED process REST-backfills history into the store
(~30 paginated kline requests to fill the buffer window) then polls for
closed candles every `--poll-seconds`. The ENGINE bootstraps its buffer
from the store and tails new bars. First prediction fires when the warmup
guard is satisfied. Per-bar compute is ~18s features + ~3ms predict
against a 60s bar interval — ~40s headroom; if the engine falls behind,
`FeedSource` catches up from the store (bars are durable, never lost).

`--trade-capital` maps strategy size fractions to money: the production
spec's 0.02 lots × 50 max concurrent = 100% of trade-capital deployed at
saturation. Size it as the amount you are prepared to have fully invested
(the researched strategy holds through drawdowns; see the backtest
caveats in ENGINE.md §10).

Set the account-level kill-switch unless you deliberately want
research-faithful unlimited drawdown: in the TOML config,
`max_drawdown = 0.05` (log-return units) and/or `max_cumulative_loss`.

## 4. What runs in parallel

- **Trading loop** (main thread): bar → features → predict → decide →
  execute → persist. Crash-safe: every bar's state is flushed to the
  SQLite WAL store; `--resume` restores portfolio/P&L/counters exactly
  (pinned by test).
- **Retraining** (worker thread, `retrain_threaded = true`): event-time
  scheduled (`retrain.every_bars`), rebuilds the dataset from the store's
  own bars (gap-repair synthetic bars excluded from training rows),
  trains the challenger with the frozen research procedure, and only
  publishes if the champion/challenger gate passes on the challenger's
  validation split. Publication hot-swaps at a bar boundary after a
  serving-compatibility check (feature list/anchor/warmup). CatBoost and
  polars release the GIL for the heavy work; the trading loop's ~40s/bar
  headroom absorbs contention. A failed or rejected retrain NEVER touches
  the serving model — you get an alert and the incumbent keeps trading.

## 5. Monitoring

- **Alerts** (`--alert-webhook`, Slack-compatible JSON): `halt`
  (kill-switch or feature-error streak), `execution_failure`
  (ledger/exchange divergence — page-worthy), `reconcile` mismatch on
  resume, `retrain` outcomes. Alert delivery failures never take the
  loop down (they degrade to log lines).
- **Status**: `python -m src.engine status` — registry versions + store
  row counts. The store is the audit trail: `bars`, `predictions`,
  `decisions`, `orders`, `fills`, `trades`, `equity`, `guard_events`,
  `model_versions`, `retrain_runs`.
- **Logs**: one INFO line per `log_every_bars` with equity/positions;
  guards log every repair. Ship stderr to your collector of choice.
- Watch for: `guard_events` severity=error, degraded-bar streaks
  (feature failures), rising `synthetic` bar counts (feed quality), and
  retrain `gate_rejected` streaks (regime drift — consider widening the
  training window or investigating features).

## 6. Failure modes and recovery

**Process crash / host reboot.** Restart with `--resume` and the same
`--store`. The engine restores open positions, realized P&L, and
counters, then — with a live broker — runs **reconciliation**: ledger
exposure vs actual exchange balance. A mismatch alerts and logs but does
not trade it away automatically; resolve manually (below), then restart.

**Order execution failure** (rejection after bounded retries, sustained
5xx/429): the engine records a guard event, alerts, and **halts new
entries** (exits keep resolving). The ledger and exchange may have
diverged. Recovery:
1. Check the exchange's order history for the failing `bc-<tag>-<n>`
   client order id (idempotency: a duplicate-id rejection is auto-fetched
   and treated as success, so double-sends cannot double-fill).
2. Compare `python -m src.engine status` trades/fills vs exchange trade
   history; flatten any unmatched exchange position manually.
3. Restart with `--resume`; reconciliation confirms alignment.

**Feed stall / network outage.** The source back-fills missed closed
candles on reconnection (in order, no gaps) up to `max_backfill_bars`
(default 10,000 ≈ 7 days); beyond that it refuses silent catch-up —
restart so the bootstrap path rebuilds the buffer instead. Grid guard
fabricates flat bars only for upstream DATA gaps (missing candles in
history), never for connectivity stalls.

**Kill-switch trip.** Entries stop for the session; exits continue. The
alert carries the reason. Investigate before restarting — a drawdown
halt on this strategy usually means the no-stop-loss saturation risk
materialized (ENGINE.md §10 caveats).

**Bad model publish.** Cannot happen silently: the gate rejects
regressions, the compatibility check rejects serving drift, and
activation is an atomic pointer swap. To roll back manually: stop,
repoint `models/ACTIVE` to the previous version, restart with `--resume`.

## 7. Upgrades

Stop the process (SIGINT — graceful: store drained), deploy code, run the
full test suite (`pytest --all`), restart with `--resume`. The hot-swap
compatibility check also protects across restarts: a model incompatible
with the running configuration refuses to serve.

## 8. Residual-risk register (read before arming)

Explicitly NOT eliminated by software, in decreasing order of teeth:

1. **Strategy risk.** The researched P1+P3 spec has no stop-loss and
   saturates gross exposure in selloffs; validation-period performance
   concentrated in one regime month. The kill-switch bounds session loss
   only if you configure it.
2. **Fill quality.** Research assumed TP touch-fills and entry at close
   with 5bp round-trip cost; live market orders pay spread + impact.
   Slippage is measured (assumed vs actual fill in the store) — watch it
   and re-cost the strategy if it erodes the researched edge (test-split
   breakeven was ~33bp).
3. **Exchange risk.** Outages, cancel-only windows, rate-limit storms,
   filter changes. Bounded by retries, the halt path, and reconciliation
   — not eliminated.
4. **Single-process, single-venue.** No redundancy layer; a dead host
   stops trading (fails safe: positions rest on the exchange, resume +
   reconcile recovers state).
5. **Serving-latency margin.** ~18s/bar features against a 60s cadence is
   a 3× margin, not 30×; do not add features without re-measuring
   (Phase 3b streaming serving is the designed fix).
6. **Clock skew.** Signed requests tolerate `recvWindow` (5s); keep NTP
   on the host.

## 9. Full-suite gate

Everything above sits on 1,100+ tests: live≡offline ledger parity by
construction, serve≡train feature parity, resume≡uninterrupted equality,
golden ledger corpus, kill-switch/degrade/recover e2e, and the hermetic
Binance adapter suite (signing, retries, pagination, closed-candle
discipline, filters, idempotency, reconciliation). CI runs all of it on
every push.
