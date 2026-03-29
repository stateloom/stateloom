"""Alerting subsystem — webhook notifications for key gateway events.

Extends beyond blast radius to cover budget, kill switch, rate limiting,
job failures, and compliance violations. All webhook calls are fire-and-forget
(daemon thread, fail-open).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("stateloom.observability.alerting")


class AlertManager:
    """Manages webhook-based alerts for StateLoom events.

    Supports multiple webhook URLs and event type filtering.
    All HTTP calls are fire-and-forget on daemon threads.
    """

    def __init__(self, webhook_url: str = "", webhook_urls: list[str] | None = None) -> None:
        self._urls: list[str] = []
        if webhook_url:
            self._urls.append(webhook_url)
        if webhook_urls:
            self._urls.extend(webhook_urls)

    def fire(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Send an alert to all configured webhook URLs.

        Args:
            event_type: The type of event (e.g., "budget_exceeded",
                "kill_switch_activated", "rate_limit_rejected").
            payload: Event-specific data dict.
        """
        if not self._urls:
            return

        full_payload = {
            "source": "stateloom",
            "event_type": event_type,
            **payload,
        }

        for url in self._urls:
            thread = threading.Thread(
                target=self._send,
                args=(url, full_payload),
                daemon=True,
            )
            thread.start()

    @staticmethod
    def _send(url: str, payload: dict[str, Any]) -> None:
        """Send a webhook (fire-and-forget, fail-open)."""
        try:
            import httpx

            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=payload)
                logger.debug("Alert webhook sent to %s (status=%d)", url, resp.status_code)
        except Exception:
            logger.debug("Alert webhook failed for %s", url, exc_info=True)
