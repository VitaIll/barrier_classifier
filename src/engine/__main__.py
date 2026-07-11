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
from src.engine.errors import EngineError
from src.engine.model import ModelRegistry
from src.engine.retrain import RetrainPolicy
from src.engine.sources import ReplaySource
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


def _cmd_status(args: argparse.Namespace) -> int:
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
        if args.command == "status":
            return _cmd_status(args)
    except EngineError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
