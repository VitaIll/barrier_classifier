"""Operational alerting — the engine's outbound "wake a human" channel.

Fired on the events an operator must act on: kill-switch halts, order
execution failures (ledger/exchange divergence), reconciliation
mismatches, and retrain outcomes. An :class:`AlertSink` is injectable;
:class:`WebhookAlerter` POSTs a JSON payload to a configured URL
(Slack/Discord/PagerDuty-style receivers all accept this shape or a thin
proxy of it); :class:`NullAlerter` is the default no-op.

Alert delivery must never take the trading loop down: failures are
logged and swallowed (the log line itself is the fallback alert).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger("src.engine.alerts")


@runtime_checkable
class AlertSink(Protocol):
    def send(self, *, level: str, event: str, message: str, **context: Any) -> None:
        ...


class NullAlerter:
    """Default sink: alerts appear in the log only."""

    def send(self, *, level: str, event: str, message: str, **context: Any) -> None:
        logger.log(
            logging.ERROR if level == "error" else logging.INFO,
            "ALERT[%s/%s]: %s %s",
            level, event, message, context or "",
        )


class WebhookAlerter:
    """POST alert payloads to a webhook URL.

    Payload: ``{"ts": <unix seconds>, "level", "event", "message",
    "context": {...}, "text": "<level upper> engine/<event>: <message>"}``
    — the ``text`` field makes it directly consumable by Slack-compatible
    receivers. Delivery is best-effort with one retry; failures log and
    return (never raise into the trading loop).
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 5.0,
        transport=None,
    ) -> None:
        if not url:
            raise ValueError("WebhookAlerter requires a non-empty url")
        self.url = url
        self.timeout = float(timeout)
        if transport is None:
            import requests

            def transport(u: str, payload: dict, timeout: float):  # noqa: ANN001
                return requests.post(u, json=payload, timeout=timeout).status_code

        self._transport = transport

    def send(self, *, level: str, event: str, message: str, **context: Any) -> None:
        payload = {
            "ts": time.time(),
            "level": level,
            "event": event,
            "message": message,
            "context": _jsonable(context),
            "text": f"{level.upper()} engine/{event}: {message}",
        }
        for attempt in (1, 2):
            try:
                status = self._transport(self.url, payload, self.timeout)
                if int(status) < 300:
                    return
                logger.warning(
                    "alert webhook returned HTTP %s (attempt %d)", status, attempt
                )
            except Exception as exc:  # delivery must never break the loop
                logger.warning(
                    "alert webhook delivery failed (attempt %d): %s", attempt, exc
                )
        logger.error(
            "ALERT DELIVERY FAILED [%s/%s]: %s %s", level, event, message, context
        )


def _jsonable(context: dict) -> dict:
    out = {}
    for k, v in context.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out
