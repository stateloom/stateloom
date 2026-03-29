"""Webhook delivery with HMAC-SHA256 signatures and exponential backoff."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from stateloom.core.job import Job

logger = logging.getLogger("stateloom.jobs.webhook")


class WebhookDelivery:
    """Delivers job results to webhook URLs with HMAC signing and retries."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        global_secret: str = "",
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._global_secret = global_secret

    def deliver(self, job: Job, store: Any) -> None:
        """Deliver webhook in a daemon thread (fire-and-forget)."""
        if not job.webhook_url:
            return
        thread = threading.Thread(
            target=self._deliver_sync,
            args=(job, store),
            daemon=True,
        )
        thread.start()

    def _deliver_sync(self, job: Job, store: Any) -> None:
        """Synchronous webhook delivery with retries."""
        payload = self._build_payload(job)
        body = json.dumps(payload, default=str)
        secret = job.webhook_secret or self._global_secret
        headers = self._build_headers(job, body, secret)

        job.webhook_status = "pending"
        store.save_job(job)

        last_error = ""
        for attempt in range(self._max_retries):
            job.webhook_attempts = attempt + 1
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(
                        job.webhook_url,
                        content=body,
                        headers=headers,
                    )
                    resp.raise_for_status()

                job.webhook_status = "delivered"
                job.webhook_last_error = ""
                store.save_job(job)
                logger.debug("Webhook delivered for job %s", job.id)
                return
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Webhook attempt %d/%d failed for job %s: %s",
                    attempt + 1,
                    self._max_retries,
                    job.id,
                    e,
                )
                if attempt < self._max_retries - 1:
                    backoff = 2**attempt
                    time.sleep(backoff)

        job.webhook_status = "failed"
        job.webhook_last_error = last_error
        store.save_job(job)
        logger.error(
            "Webhook delivery failed for job %s after %d attempts", job.id, self._max_retries
        )

    def _build_payload(self, job: Job) -> dict:
        """Build the webhook payload."""
        payload: dict[str, Any] = {
            "job_id": job.id,
            "status": job.status.value,
            "provider": job.provider,
            "model": job.model,
            "created_at": job.created_at.isoformat(),
        }
        if job.result is not None:
            payload["result"] = job.result
        if job.error:
            payload["error"] = job.error
            payload["error_code"] = job.error_code
        if job.completed_at:
            payload["completed_at"] = job.completed_at.isoformat()
        if job.session_id:
            payload["session_id"] = job.session_id
        return payload

    def _build_headers(self, job: Job, body: str, secret: str) -> dict[str, str]:
        """Build headers including HMAC signature if secret is available."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-StateLoom-Event": "job.completed",
            "X-StateLoom-Job-Id": job.id,
        }
        if secret:
            sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
            headers["X-StateLoom-Signature"] = f"sha256={sig}"
        return headers
