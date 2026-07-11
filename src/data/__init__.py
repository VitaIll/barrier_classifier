"""Market-data acquisition — the external feed service.

This package runs as a SEPARATE process from the trading engine
(docs/PRODUCTION.md). A :class:`~src.data.feed.FeedWriter` drives an
exchange adapter (:mod:`src.data.binance`) and appends closed bars to a
durable :class:`~src.data.feed.FeedStore`; the engine consumes that store
through ``src.engine.sources.FeedSource``. The only contract between the
two is rows in a WAL-mode SQLite file — no shared in-process state.
"""
