"""Tests for webhook delivery with HMAC-SHA256 signatures."""

import hashlib
import hmac
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.job import Job
from stateloom.core.types import JobStatus
from stateloom.jobs.webhook import WebhookDelivery


@pytest.fixture
def completed_job():
    return Job(
        status=JobStatus.COMPLETED,
        provider="openai",
        model="gpt-4",
        result={"choices": [{"message": {"content": "hello"}}]},
        webhook_url="https://example.com/webhook",
        webhook_secret="test-secret",
        session_id="sess-1",
    )


@pytest.fixture
def failed_job():
    return Job(
        status=JobStatus.FAILED,
        provider="openai",
        model="gpt-4",
        error="Budget exceeded",
        error_code="BUDGET_ERROR",
        webhook_url="https://example.com/webhook",
        completed_at=datetime.now(timezone.utc),
    )


class TestWebhookPayload:
    def test_completed_payload(self, completed_job):
        delivery = WebhookDelivery()
        payload = delivery._build_payload(completed_job)

        assert payload["job_id"] == completed_job.id
        assert payload["status"] == "completed"
        assert payload["provider"] == "openai"
        assert payload["model"] == "gpt-4"
        assert "result" in payload
        assert "created_at" in payload
        assert "session_id" in payload

    def test_failed_payload(self, failed_job):
        delivery = WebhookDelivery()
        payload = delivery._build_payload(failed_job)

        assert payload["status"] == "failed"
        assert payload["error"] == "Budget exceeded"
        assert payload["error_code"] == "BUDGET_ERROR"

    def test_payload_without_result(self):
        job = Job(status=JobStatus.COMPLETED, webhook_url="https://example.com")
        delivery = WebhookDelivery()
        payload = delivery._build_payload(job)
        assert "result" not in payload


class TestWebhookHeaders:
    def test_headers_with_secret(self, completed_job):
        delivery = WebhookDelivery()
        body = json.dumps({"test": "data"})
        headers = delivery._build_headers(completed_job, body, "test-secret")

        assert headers["Content-Type"] == "application/json"
        assert headers["X-StateLoom-Event"] == "job.completed"
        assert headers["X-StateLoom-Job-Id"] == completed_job.id
        assert "X-StateLoom-Signature" in headers

        # Verify HMAC
        expected_sig = hmac.new(b"test-secret", body.encode(), hashlib.sha256).hexdigest()
        assert headers["X-StateLoom-Signature"] == f"sha256={expected_sig}"

    def test_headers_without_secret(self, completed_job):
        delivery = WebhookDelivery()
        body = json.dumps({"test": "data"})
        headers = delivery._build_headers(completed_job, body, "")

        assert "X-StateLoom-Signature" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_global_secret_fallback(self):
        delivery = WebhookDelivery(global_secret="global-secret")
        job = Job(webhook_url="https://example.com", webhook_secret="")
        body = json.dumps({"test": "data"})

        # When job has no secret, _deliver_sync uses global_secret
        secret = job.webhook_secret or delivery._global_secret
        headers = delivery._build_headers(job, body, secret)
        assert "X-StateLoom-Signature" in headers


class TestWebhookDeliverySync:
    @patch("stateloom.jobs.webhook.httpx.Client")
    def test_successful_delivery(self, mock_client_cls, completed_job):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        store = MagicMock()
        delivery = WebhookDelivery(timeout=5.0, max_retries=1)
        delivery._deliver_sync(completed_job, store)

        assert completed_job.webhook_status == "delivered"
        mock_client.post.assert_called_once()

    @patch("stateloom.jobs.webhook.httpx.Client")
    def test_failed_delivery_after_retries(self, mock_client_cls, completed_job):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("connection refused")
        mock_client_cls.return_value = mock_client

        store = MagicMock()
        delivery = WebhookDelivery(timeout=1.0, max_retries=2)
        delivery._deliver_sync(completed_job, store)

        assert completed_job.webhook_status == "failed"
        assert completed_job.webhook_last_error == "connection refused"
        assert completed_job.webhook_attempts == 2

    @patch("stateloom.jobs.webhook.httpx.Client")
    def test_retry_succeeds_on_second_attempt(self, mock_client_cls, completed_job):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = [Exception("timeout"), mock_response]
        mock_client_cls.return_value = mock_client

        store = MagicMock()
        delivery = WebhookDelivery(timeout=1.0, max_retries=3)
        delivery._deliver_sync(completed_job, store)

        assert completed_job.webhook_status == "delivered"
        assert completed_job.webhook_attempts == 2

    def test_no_delivery_without_webhook_url(self):
        job = Job(webhook_url="")
        delivery = WebhookDelivery()
        # deliver() should return immediately without starting a thread
        delivery.deliver(job, MagicMock())
        # No assertion needed — just verify it doesn't crash


class TestWebhookHMAC:
    def test_hmac_verification(self, completed_job):
        delivery = WebhookDelivery()
        payload = delivery._build_payload(completed_job)
        body = json.dumps(payload, default=str)
        secret = "my-secret"

        headers = delivery._build_headers(completed_job, body, secret)
        sig = headers["X-StateLoom-Signature"]

        # Verify the signature manually
        expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        assert sig == f"sha256={expected}"

    def test_different_secrets_produce_different_signatures(self, completed_job):
        delivery = WebhookDelivery()
        body = json.dumps({"test": "data"})

        headers1 = delivery._build_headers(completed_job, body, "secret1")
        headers2 = delivery._build_headers(completed_job, body, "secret2")

        assert headers1["X-StateLoom-Signature"] != headers2["X-StateLoom-Signature"]
