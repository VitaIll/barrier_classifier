---
name: live-ops
description: Operating the live trading system — safety ladder, two-process bring-up, engine configuration, production monitoring, alerting, and failure recovery. Use this whenever starting/stopping/resuming the feed or engine, going to testnet or mainnet, configuring risk controls or the kill-switch, interpreting alerts or guard events, recovering from a crash/halt/reconcile mismatch, or answering "is the system healthy / what is it doing / how do I run it". docs/PRODUCTION.md is the full runbook; this is the operational index to it.
---

# Live operations

Full runbook: `docs/PRODUCTION.md` (read §9 residual-risk register before
arming anything). This skill is the fast path for an operator or LLM.

## Safety ladder — never skip a rung

| Gate | Default | Crossed by |
|---|---|---|
| Network | Binance **testnet** | `--mainnet` |
| Execution | **dry-run** (orders built + filter-validated, not sent) | `--execute` |
| Credentials | none | `BINANCE_API_KEY` / `BINANCE_API_SECRET` env vars |

Ladder: dry-run testnet → **execute testnet (mandatory rung — the only
full order path without capital)** → dry-run mainnet → execute mainnet
small. Two more arming guards: a drifted numeric stack refuses `--execute`
(`--allow-stack-drift` is the deliberate override — see `environment`
skill) and `--no-risk-controls` is research-parity only, NOT production.

## Topology + bring-up (two processes, share only feed.db)

```bash
# 0. once: research artifacts -> registry v0001 + ACTIVE
python -m src.engine import-model --dataset-dir data/model_dataset
# 1. verify on real history (dev machine with data/)
python scripts/validate_engine_replay.py && python scripts/validate_feed_chain.py
# 2. DATA FEED process (public data, no creds)
python -m src.engine feed --feed runtime/feed.db
# 3. ENGINE process (consumes the feed store; broker only for orders)
python -m src.engine live --feed runtime/feed.db --store runtime/live.db \
    --alert-webhook <slack-compatible-url>
# 4. arm on testnet:  ... live --execute --trade-capital 10000
# 5. mainnet: add --mainnet to BOTH processes; start SMALL
```

The engine holds zero market-data code; if it falls behind or dies, bars
keep landing in the store and `--resume` catches up. Per-bar budget:
~18 s features + ~3 ms predict against 60 s bars (3× margin — do not add
features without re-measuring).

## Configuration that matters (TOML via `--config`, fields of `EngineConfig`)

- Strategy (research-frozen): `lot_size=0.02`, `max_concurrent=50`,
  `cost_per_trade=0.0005`, threshold from the model version.
- **Kill-switch is OFF by default** (research parity): set `max_drawdown`
  (e.g. 0.05, log-return units) and/or `max_cumulative_loss` for real money.
- Pre-trade `entry_controls` ON by default: 300 entries/day, 0.05 bar-move
  collar, 0.10 daily-loss stop, 0.10 per-entry capital fraction.
- `reconcile_every_bars=1440` (daily ledger-vs-exchange check),
  `halt_after_feature_errors=25`, `log_every_bars=1440`,
  `[retrain]` table for scheduled retraining (champion/challenger gated).
- `--trade-capital` = amount you accept being FULLY invested (0.02 × 50
  saturates at 100%; the strategy holds through drawdowns, no stop-loss).

## Monitoring

- `python -m src.engine status` — numeric-stack table, registry versions
  (+ACTIVE), store row counts.
- Alerts (Slack-shape webhook): `halt` (kill-switch/feature streak),
  `execution_failure` (**page-worthy**: ledger/exchange divergence),
  `reconcile` mismatch, `retrain` outcomes. Alert delivery failure never
  kills the loop.
- Watch in the store: `guard_events` severity=error; degraded-bar streaks;
  rising synthetic-bar counts (feed quality); `retrain_runs` gate_rejected
  streaks (regime drift). One INFO log line per `log_every_bars` with
  equity/positions.
- Slippage: the store records assumed vs actual fills — re-cost the
  strategy if live costs approach the researched breakeven (~33 bp test).

## Failure recovery (details: PRODUCTION.md §6)

- **Crash/reboot** → restart with `--resume` + same `--store`; broker
  sessions auto-reconcile; a mismatch alerts and WAITS for the operator.
- **ExecutionError halt** (entries stopped, exits alive) → check exchange
  order history for the `bc-<tag>-<n>` client id (idempotent: duplicate-id
  rejection auto-fetches the landed order) → compare `status` trades vs
  exchange history → flatten any unmatched position manually — directly on
  the exchange (UI or plain API order), NOT through the engine, which
  deliberately never auto-flattens an unexplained delta → `--resume`.
- **Feed stall** → source back-fills in order up to `max_backfill_bars`
  (10,000 ≈ 7 d); beyond that restart so bootstrap rebuilds the buffer.
- **Kill-switch trip** → investigate BEFORE restarting (usually the no-SL
  saturation risk materializing).
- **Bad model publish** — cannot happen silently (gate + compatibility
  check + atomic pointer); manual rollback: stop, repoint `models/ACTIVE`
  to the previous version, restart `--resume`.
- **Host migration** (moving to a new machine): first verify the machine
  per the `environment` skill (new OS = revalidation event). Then move
  state: the engine store (`--store` file, e.g. `runtime/live.db`) MUST
  travel — it is the ledger `--resume` restores; copy the `models/`
  registry directory as-is (versions are self-contained, `ACTIVE` is a
  pointer; re-running `import-model` would need the research dataset).
  `feed.db` may be copied or rebuilt — the feed process backfills on
  restart. Start feed + engine `--resume`; reconciliation confirms
  alignment against the exchange.

## Update triggers — edit THIS skill when

- CLI commands/flags change in `src/engine/__main__.py`
  (`tests/test_skills.py` verifies every command and flag named here).
- `EngineConfig` defaults change (risk controls, reconcile cadence, kill-
  switch semantics) or new controls land in `src/engine/risk.py`.
- The recovery procedures in `docs/PRODUCTION.md` §6 change.
- Alert kinds change in `src/engine/alerts.py`.
