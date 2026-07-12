"""Engine CLI: ``python -m src.engine <command>``.

Commands
--------
import-model   Package the research artifacts (data/model_dataset) as the
               registry's first version.
replay         Run the engine against a simulated historical stream
               (the definition-of-done path).
run            Alias of replay with feature_mode='rolling' — exercises the
               true per-bar live path (slower, higher fidelity to production).
status         Show registry versions and store counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

from src.engine.engine import Engine, EngineConfig
from src.engine.environment import enforce_stack, stack_report
from src.engine.errors import EngineError
from src.engine.model import ModelRegistry
from src.engine.retrain import RetrainPolicy
from src.engine.sources import FeedSource, ReplaySource
from src.engine.store import SQLiteStore

logger = logging.getLogger("src.engine.cli")


def _install_graceful_stop() -> None:
    """Translate SIGTERM into KeyboardInterrupt so orchestrator termination
    (not just Ctrl-C) drains the store through the same clean path."""
    if not hasattr(signal, "SIGTERM"):
        return

    def _handler(signum, frame):  # noqa: ANN001
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):
        pass  # not the main thread, or unsupported on this platform


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None,
                   help="TOML config (fields of EngineConfig; [retrain] table)")
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--store", type=Path, default=Path("runtime/engine.db"))
    p.add_argument("--spot", type=Path,
                   default=Path("data/raw_data/klines_1m.parquet"))
    p.add_argument("--start", type=str, default=None, help="stream start (UTC)")
    p.add_argument("--end", type=str, default=None, help="stream end (UTC, exclusive)")
    p.add_argument("--max-bars", type=int, default=None)
    p.add_argument("--retrain-every-days", type=float, default=None,
                   help="enable event-time retraining at this cadence")
    p.add_argument("--retrain-iterations", type=int, default=200,
                   help="CatBoost iterations for scheduled retrains")
    p.add_argument("--speed", type=float, default=None,
                   help="replay pacing (1.0 = real time; default: as fast as possible)")
    p.add_argument("--resume", action="store_true",
                   help="continue a prior session in --store (restore open "
                        "positions, equity, and counters)")
    p.add_argument("-v", "--verbose", action="store_true")


def _build_config(args: argparse.Namespace, feature_mode: str) -> EngineConfig:
    if args.config is not None:
        cfg = EngineConfig.from_toml(args.config)
    else:
        cfg = EngineConfig()
    cfg.model_dir = args.model_dir
    cfg.store_path = args.store
    cfg.feature_mode = feature_mode
    if args.resume:
        cfg.resume = True
    if args.retrain_every_days is not None:
        cfg.retrain = RetrainPolicy(
            enabled=True,
            every_bars=int(args.retrain_every_days * 1440),
            iterations=args.retrain_iterations,
        )
        cfg.retrain_threaded = False  # deterministic inside replays
    cfg.validate()
    return cfg


def _cmd_import_model(args: argparse.Namespace) -> int:
    registry = ModelRegistry(args.model_dir)
    version = registry.import_research_artifacts(args.dataset_dir, top_q=args.top_q)
    handle = registry.load(version)
    print(f"imported research artifacts as {version}")
    print(f"  features    : {handle.contract.n_features}")
    print(f"  grid anchor : {handle.contract.anchor}")
    print(f"  p_threshold : {handle.thresholds.p_threshold:.6f} (top_q={args.top_q})")
    return 0


def _cmd_replay(args: argparse.Namespace, feature_mode: str) -> int:
    for drift in enforce_stack(strict=False, context="replay"):
        logger.warning("STACK DRIFT (parity not guaranteed): %s", drift)
    cfg = _build_config(args, feature_mode)
    source = ReplaySource(
        args.spot, start=args.start, end=args.end, speed=args.speed,
    )
    _install_graceful_stop()
    with Engine(cfg, source=source) as engine:
        try:
            report = engine.run(max_bars=args.max_bars)
        except KeyboardInterrupt:
            logger.warning("interrupted — store drained; restart with --resume to continue")
            return 130
    print()
    print(report.summary())
    if not report.trades.empty:
        by_reason = report.trades.groupby("exit_reason")["weighted_net_log_return"].agg(
            ["count", "sum"]
        )
        print("\nClosed trades by exit reason:")
        print(by_reason.to_string())
    return 0


def _cmd_feed(args: argparse.Namespace) -> int:
    """Run the DATA FEED service: pull closed bars from Binance into a
    durable feed store. Runs as its own process; the engine consumes the
    store. No credentials needed (public market data)."""
    from src.data.binance import BinanceClient, BinanceKlineSource
    from src.data.feed import FeedStore, FeedWriter, install_graceful_stop

    client = BinanceClient(testnet=not args.mainnet)
    source = BinanceKlineSource(
        client, symbol=args.symbol, poll_seconds=args.poll_seconds
    )
    store = FeedStore(args.feed)
    store.set_meta("symbol", args.symbol)
    store.set_meta("network", "mainnet" if args.mainnet else "testnet")
    writer = FeedWriter(source, store, backfill_rows=args.backfill_rows)
    install_graceful_stop(writer)
    logger.warning(
        "feed service: %s %s -> %s",
        "MAINNET" if args.mainnet else "TESTNET", args.symbol, args.feed,
    )
    try:
        n = writer.run(max_bars=args.max_bars)
    finally:
        store.close()
    print(f"feed writer appended {n} streamed bar(s); store now holds {FeedStore(args.feed, read_only=True).count()} bars")
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    """Live trading: CONSUME an external feed store, EXECUTE on Binance.

    The engine reads bars from the feed store (fed by a separate
    ``feed`` process) — it holds no market-data connection. The broker
    independently connects to Binance to place orders. Safety gates on
    execution are unchanged: ``--mainnet`` + ``--execute`` + env creds."""
    from src.data.feed import FeedStore
    from src.engine.binance import BinanceBroker, BinanceClient

    # Environment gate first: an armed session on a drifted numeric stack
    # serves the model inputs it was never validated on. Strict when
    # executing; dry-run sessions get a loud warning instead.
    strict = args.execute and not args.allow_stack_drift
    for drift in enforce_stack(strict=strict, context="live"):
        logger.warning("STACK DRIFT (parity not guaranteed): %s", drift)

    cfg = _build_config(args, feature_mode="rolling")
    cfg.alert_webhook_url = args.alert_webhook or cfg.alert_webhook_url
    if args.no_risk_controls:
        from src.engine.risk import EntryControls
        cfg.entry_controls = EntryControls.disabled()
        logger.warning("PRE-TRADE RISK CONTROLS DISABLED (--no-risk-controls)")
    cfg.validate()
    feed_store = FeedStore(args.feed, read_only=True)
    source = FeedSource(
        feed_store, poll_seconds=args.poll_seconds,
        idle_timeout_seconds=args.idle_timeout,
    )
    client = BinanceClient(testnet=not args.mainnet)
    broker = BinanceBroker(
        client=client, symbol=args.symbol,
        trade_capital=args.trade_capital, dry_run=not args.execute,
    )
    mode = (
        f"{'MAINNET' if args.mainnet else 'TESTNET'}/"
        f"{'EXECUTE' if args.execute else 'DRY-RUN'}"
    )
    logger.warning(
        "live session: %s %s capital=%.2f  (data <- %s)",
        mode, args.symbol, args.trade_capital, args.feed,
    )
    _install_graceful_stop()
    try:
        with Engine(cfg, source=source, broker=broker) as engine:
            try:
                report = engine.run(max_bars=args.max_bars)
            except KeyboardInterrupt:
                source.close()
                logger.warning("interrupted — store drained; restart with --resume")
                return 130
    finally:
        feed_store.close()
    print()
    print(report.summary())
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    print(stack_report())
    registry = ModelRegistry(args.model_dir)
    print(f"registry: {registry.root}")
    active = registry.active_version()
    for v in registry.versions():
        marker = " *ACTIVE*" if v == active else ""
        print(f"  {v}{marker}")
    if Path(args.store).exists():
        store = SQLiteStore(args.store)
        try:
            print(f"store: {args.store}")
            print(json.dumps(store.counts(), indent=2))
        finally:
            store.close()
    else:
        print(f"store: {args.store} (not created yet)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.engine",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import-model", help="package research artifacts as v0001")
    p_import.add_argument("--model-dir", type=Path, default=Path("models"))
    p_import.add_argument("--dataset-dir", type=Path, default=Path("data/model_dataset"))
    p_import.add_argument("--top-q", type=float, default=0.99)

    p_replay = sub.add_parser("replay", help="simulated historical stream (batch features)")
    _add_common(p_replay)
    p_run = sub.add_parser("run", help="simulated stream through the rolling live path")
    _add_common(p_run)

    p_feed = sub.add_parser(
        "feed", help="DATA FEED service: Binance bars -> durable feed store"
    )
    p_feed.add_argument("--feed", type=Path, default=Path("runtime/feed.db"),
                        help="feed store path (the engine consumes this)")
    p_feed.add_argument("--symbol", default="BTCUSDT")
    p_feed.add_argument("--poll-seconds", type=float, default=2.0)
    p_feed.add_argument("--backfill-rows", type=int, default=50_000)
    p_feed.add_argument("--mainnet", action="store_true")
    p_feed.add_argument("--max-bars", type=int, default=None)
    p_feed.add_argument("-v", "--verbose", action="store_true")

    p_live = sub.add_parser(
        "live", help="live trading: consume a feed store, execute on Binance"
    )
    _add_common(p_live)
    p_live.add_argument("--feed", type=Path, default=Path("runtime/feed.db"),
                        help="feed store to consume (run `feed` to populate it)")
    p_live.add_argument("--symbol", default="BTCUSDT")
    p_live.add_argument("--trade-capital", type=float, default=10_000.0,
                        help="quote units mapped to the strategy's size fractions")
    p_live.add_argument("--poll-seconds", type=float, default=2.0)
    p_live.add_argument("--idle-timeout", type=float, default=None,
                        help="end the session if the feed is silent this long (s)")
    p_live.add_argument("--mainnet", action="store_true",
                        help="broker targets api.binance.com instead of testnet")
    p_live.add_argument("--execute", action="store_true",
                        help="disable dry-run: actually send orders")
    p_live.add_argument("--alert-webhook", default=None,
                        help="POST operational alerts to this URL")
    p_live.add_argument("--no-risk-controls", action="store_true",
                        help="disable the pre-trade entry controls "
                             "(research-faithful; NOT for production)")
    p_live.add_argument("--allow-stack-drift", action="store_true",
                        help="arm --execute even if the numeric stack differs "
                             "from the validated pins (parity NOT guaranteed)")

    p_status = sub.add_parser("status", help="registry + store overview")
    p_status.add_argument("--model-dir", type=Path, default=Path("models"))
    p_status.add_argument("--store", type=Path, default=Path("runtime/engine.db"))

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        if args.command == "import-model":
            return _cmd_import_model(args)
        if args.command == "replay":
            return _cmd_replay(args, feature_mode="batch")
        if args.command == "run":
            return _cmd_replay(args, feature_mode="rolling")
        if args.command == "feed":
            return _cmd_feed(args)
        if args.command == "live":
            return _cmd_live(args)
        if args.command == "status":
            return _cmd_status(args)
    except EngineError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
