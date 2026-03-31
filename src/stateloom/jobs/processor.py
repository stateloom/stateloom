"""Background job processor with ThreadPoolExecutor and polling."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

from stateloom.core.context import set_current_session
from stateloom.core.errors import StateLoomError
from stateloom.core.event import AsyncJobEvent
from stateloom.core.job import Job
from stateloom.core.types import JobStatus
from stateloom.jobs.queue import JobQueue

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.jobs.processor")


class JobProcessor:
    """Background worker pool that polls for pending jobs and processes them."""

    def __init__(self, gate: Gate, queue: JobQueue, max_workers: int = 4) -> None:
        self._gate = gate
        self._queue = queue
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._poll_thread: threading.Thread | None = None
        self._active_jobs: dict[str, Future[None]] = {}
        self._active_jobs_lock = threading.Lock()
        self._shutdown_event = threading.Event()

    def start(self) -> None:
        """Start the executor and polling thread."""
        if self._executor is not None:
            return

        # Crash recovery: reset stale RUNNING jobs to PENDING
        self._queue.recover_stale()

        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="stateloom-job",
        )
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="stateloom-job-poller"
        )
        self._poll_thread.start()
        logger.info("Job processor started with %d workers", self._max_workers)

    def _poll_loop(self) -> None:
        """Poll for pending jobs every second."""
        while not self._shutdown_event.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.debug("Poll loop error", exc_info=True)
            self._shutdown_event.wait(timeout=1.0)

    def _poll_once(self) -> None:
        """Check for pending jobs and submit them to the pool."""
        if self._executor is None:
            return

        with self._active_jobs_lock:
            active_count = len(self._active_jobs)
        if active_count >= self._max_workers:
            return

        pending = self._queue.dequeue(self._max_workers - active_count)
        for job in pending:
            if self._shutdown_event.is_set():
                break
            with self._active_jobs_lock:
                if job.id in self._active_jobs:
                    continue
                future = self._executor.submit(self._process_job, job)
                self._active_jobs[job.id] = future

    def _process_job(self, job: Job) -> None:
        """Process a single job through the middleware pipeline."""
        start_time = time.perf_counter()

        # Mark as RUNNING
        self._queue.mark_running(job)
        self._broadcast_job_update(job)

        try:
            # Create a session for this job
            session = self._gate.session_manager.create(
                session_id=job.session_id or None,
                name=f"job-{job.id}",
                org_id=job.org_id,
                team_id=job.team_id,
            )
            # Set agent session fields from job metadata
            if job.metadata and job.metadata.get("agent_id"):
                session.agent_id = job.metadata["agent_id"]
                session.agent_slug = job.metadata.get("agent_slug", "")
                session.agent_version_id = job.metadata.get("agent_version_id", "")
                session.agent_version_number = job.metadata.get("agent_version_number", 0)
                session.agent_name = job.metadata.get("agent_slug", "")
            set_current_session(session)
            self._gate.store.save_session(session)

            if not job.session_id:
                job.session_id = session.id

            # Build and execute the LLM call through the pipeline
            result = self._execute_llm_call(job, session)

            # Serialize result
            job.result = self._serialize_result(result)
            self._queue.mark_completed(job)

            # End session
            from stateloom.core.types import SessionStatus

            session.end(SessionStatus.COMPLETED)
            self._gate.store.save_session(session)

        except StateLoomError as e:
            # Terminal error — no retry
            job.error = str(e)
            job.error_code = getattr(e, "error_code", "STATELOOM_ERROR")
            self._queue.mark_failed(job)
        except Exception as e:
            job.error = str(e)
            # Check if this is a permanent error (no point retrying)
            err_msg = str(e).lower()
            is_permanent = any(
                hint in err_msg
                for hint in (
                    "api_key",
                    "api key",
                    "authentication",
                    "invalid key",
                    "permission",
                    "unauthorized",
                    "import",
                    "no module",
                )
            )
            if is_permanent or job.retry_count >= job.max_retries:
                job.error_code = "PERMANENT_ERROR" if is_permanent else "TRANSIENT_ERROR"
                self._queue.mark_failed(job)
                if is_permanent:
                    logger.error(
                        "Job %s failed permanently: %s",
                        job.id,
                        e,
                    )
            else:
                self._queue.requeue(job)
                logger.warning(
                    "Job %s failed (attempt %d/%d), re-queuing: %s",
                    job.id,
                    job.retry_count,
                    job.max_retries,
                    e,
                )
        finally:
            set_current_session(None)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Record event
            self._record_event(job, elapsed_ms)
            self._broadcast_job_update(job)

            # Deliver webhook if configured
            if job.webhook_url and job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                try:
                    from stateloom.jobs.webhook import WebhookDelivery

                    delivery = WebhookDelivery(
                        timeout=self._gate.config.async_jobs_webhook_timeout,
                        max_retries=self._gate.config.async_jobs_webhook_retries,
                        global_secret=self._gate.config.async_jobs_webhook_secret,
                    )
                    delivery.deliver(job, self._gate.store)
                except Exception:
                    logger.debug("Webhook delivery setup failed", exc_info=True)

            with self._active_jobs_lock:
                self._active_jobs.pop(job.id, None)

    def _execute_llm_call(self, job: Job, session: Any) -> Any:
        """Execute the LLM call through the middleware pipeline."""
        provider = job.provider
        model = job.model
        request_kwargs = dict(job.request_kwargs)

        # Merge messages into request_kwargs
        if job.messages and "messages" not in request_kwargs:
            request_kwargs["messages"] = job.messages
        if model and "model" not in request_kwargs:
            request_kwargs["model"] = model

        # Build a dummy LLM call that uses the provider SDK
        llm_call = self._build_llm_call(provider, request_kwargs)

        result = self._gate.pipeline.execute_sync(
            provider=provider,
            method="chat.completions.create",
            model=model,
            request_kwargs=request_kwargs,
            session=session,
            config=self._gate.config,
            llm_call=llm_call,
        )
        return result

    def _build_llm_call(self, provider: str, request_kwargs: dict[str, Any]) -> Any:
        """Build a callable that invokes the original (unpatched) provider SDK.

        The job processor already runs its own pipeline via execute_sync().
        Using the original SDK method avoids the interceptor firing a second
        pipeline pass (which would double-count events, cost, etc.).
        """
        from stateloom.intercept.unpatch import get_original

        def _call() -> Any:
            if provider in ("openai", ""):
                import openai

                oai_client = openai.OpenAI()
                original = get_original(type(oai_client.chat.completions), "create")
                if original:
                    return original(oai_client.chat.completions, **request_kwargs)
                return oai_client.chat.completions.create(**request_kwargs)
            elif provider == "anthropic":
                import anthropic

                anth_client = anthropic.Anthropic()
                original = get_original(type(anth_client.messages), "create")
                if original:
                    return original(anth_client.messages, **request_kwargs)
                return anth_client.messages.create(**request_kwargs)
            elif provider in ("google", "gemini"):
                from google.generativeai import GenerativeModel  # type: ignore[attr-defined]

                messages = request_kwargs.get("messages", [])
                model_name = request_kwargs.get("model", "gemini-2.5-flash")

                contents: list[dict[str, Any]] = []
                system_instruction = None
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        system_instruction = content
                        continue
                    gemini_role = "model" if role == "assistant" else "user"
                    contents.append({"role": gemini_role, "parts": [{"text": str(content)}]})

                gen_config: dict[str, Any] = {}
                for k in ("temperature", "top_p", "top_k"):
                    if k in request_kwargs:
                        gen_config[k] = request_kwargs[k]
                if "max_tokens" in request_kwargs:
                    gen_config["max_output_tokens"] = request_kwargs["max_tokens"]

                gen_model = GenerativeModel(
                    model_name,
                    system_instruction=system_instruction,
                )
                original = get_original(GenerativeModel, "generate_content")
                if original:
                    return original(
                        gen_model,
                        contents,
                        generation_config=gen_config or None,
                    )
                return gen_model.generate_content(
                    contents,  # type: ignore[arg-type]
                    generation_config=gen_config or None,  # type: ignore[arg-type]
                )
            else:
                import openai

                oai_client2 = openai.OpenAI()
                original = get_original(type(oai_client2.chat.completions), "create")
                if original:
                    return original(oai_client2.chat.completions, **request_kwargs)
                return oai_client2.chat.completions.create(**request_kwargs)

        return _call

    def _serialize_result(self, result: Any) -> dict[str, Any]:
        """Serialize an LLM response to a JSON-safe dict."""
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        # Try pydantic model_dump (OpenAI/Anthropic SDK objects)
        if hasattr(result, "model_dump"):
            return cast(dict[str, Any], result.model_dump())
        if hasattr(result, "to_dict"):
            return cast(dict[str, Any], result.to_dict())
        return {"raw": str(result)}

    def _record_event(self, job: Job, elapsed_ms: float) -> None:
        """Record an AsyncJobEvent."""
        try:
            event = AsyncJobEvent(
                session_id=job.session_id,
                job_id=job.id,
                job_status=job.status.value,
                provider=job.provider,
                model=job.model,
                webhook_url=job.webhook_url,
                webhook_status=job.webhook_status,
                error=job.error,
                processing_time_ms=elapsed_ms,
            )
            self._gate.store.save_event(event)
        except Exception:
            logger.debug("Failed to record job event", exc_info=True)

    def _broadcast_job_update(self, job: Job) -> None:
        """Broadcast job status change via WebSocket."""
        try:
            from stateloom.dashboard.ws import broadcast_sync

            broadcast_sync(
                {
                    "type": "job_update",
                    "data": {
                        "job_id": job.id,
                        "status": job.status.value,
                        "provider": job.provider,
                        "model": job.model,
                    },
                }
            )
        except Exception:
            pass

    def shutdown(self, drain_timeout: float = 5.0) -> None:
        """Graceful shutdown: stop polling, drain in-flight jobs."""
        self._shutdown_event.set()

        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

        with self._active_jobs_lock:
            self._active_jobs.clear()

        logger.info("Job processor shut down")
