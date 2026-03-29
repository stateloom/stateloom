"""Network blocker for strict replay mode.

Patches httpx, requests, and urllib3 to block outbound HTTP calls
that are not captured via @gate.tool() during replay.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from stateloom.core.errors import StateLoomSideEffectError
from stateloom.intercept.unpatch import register_patch

logger = logging.getLogger("stateloom.replay.network_blocker")


class NetworkBlocker:
    """Blocks outbound HTTP calls during strict replay mode.

    Patches httpx.Client.send, httpx.AsyncClient.send,
    requests.Session.send, and urllib3.HTTPConnectionPool.urlopen
    to raise StateLoomSideEffectError for any host not in allowed_hosts.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._allowed_hosts: set[str] = set()
        self._active = False
        self._originals: list[tuple[Any, str, Any]] = []
        self._step_counter = 0

    def activate(self, allowed_hosts: set[str] | None = None) -> None:
        """Activate the network blocker, patching HTTP libraries."""
        self._allowed_hosts = allowed_hosts or set()
        self._active = True
        self._patch_httpx()
        self._patch_requests()
        self._patch_urllib3()
        logger.info(
            f"[StateLoom] Network blocker active for session '{self._session_id}', "
            f"allowed hosts: {self._allowed_hosts or 'none'}"
        )

    def deactivate(self) -> None:
        """Deactivate the network blocker, restoring original methods."""
        self._active = False
        for target, method_name, original in self._originals:
            try:
                setattr(target, method_name, original)
            except Exception as e:
                logger.warning(f"[StateLoom] Failed to unpatch {target}.{method_name}: {e}")
        self._originals.clear()

    def _check_host(self, host: str) -> None:
        """Raise StateLoomSideEffectError if the host is not allowed."""
        if not self._active:
            return
        # Strip port from host
        clean_host = host.split(":")[0] if ":" in host else host
        if clean_host not in self._allowed_hosts:
            self._step_counter += 1
            raise StateLoomSideEffectError(
                host=clean_host,
                session_id=self._session_id,
                step=self._step_counter,
            )

    def _extract_host_from_url(self, url: Any) -> str:
        """Extract host from a URL string or httpx.URL object."""
        url_str = str(url)
        parsed = urlparse(url_str)
        return parsed.hostname or url_str

    def _patch_httpx(self) -> None:
        """Patch httpx.Client.send and httpx.AsyncClient.send."""
        try:
            import httpx

            # Patch sync client
            original_send = httpx.Client.send
            blocker = self

            def patched_send(self_client: Any, request: Any, **kwargs: Any) -> Any:
                host = blocker._extract_host_from_url(request.url)
                blocker._check_host(host)
                return original_send(self_client, request, **kwargs)

            httpx.Client.send = patched_send  # type: ignore[assignment]  # Monkey-patching for network blocking
            self._originals.append((httpx.Client, "send", original_send))
            register_patch(
                httpx.Client, "send", original_send, "httpx.Client.send (replay blocker)"
            )

            # Patch async client
            original_async_send = httpx.AsyncClient.send

            async def patched_async_send(self_client: Any, request: Any, **kwargs: Any) -> Any:
                host = blocker._extract_host_from_url(request.url)
                blocker._check_host(host)
                return await original_async_send(self_client, request, **kwargs)

            httpx.AsyncClient.send = patched_async_send  # type: ignore[assignment]  # Monkey-patching for network blocking
            self._originals.append((httpx.AsyncClient, "send", original_async_send))
            register_patch(
                httpx.AsyncClient,
                "send",
                original_async_send,
                "httpx.AsyncClient.send (replay blocker)",
            )
        except ImportError:
            logger.debug("[StateLoom] httpx not installed, skipping patch")

    def _patch_requests(self) -> None:
        """Patch requests.Session.send."""
        try:
            import requests

            original_send = requests.Session.send
            blocker = self

            def patched_send(self_session: Any, request: Any, **kwargs: Any) -> Any:
                host = blocker._extract_host_from_url(request.url)
                blocker._check_host(host)
                return original_send(self_session, request, **kwargs)

            requests.Session.send = patched_send  # type: ignore[assignment]  # Monkey-patching for network blocking
            self._originals.append((requests.Session, "send", original_send))
            register_patch(
                requests.Session,
                "send",
                original_send,
                "requests.Session.send (replay blocker)",
            )
        except ImportError:
            logger.debug("[StateLoom] requests not installed, skipping patch")

    def _patch_urllib3(self) -> None:
        """Patch urllib3.HTTPConnectionPool.urlopen."""
        try:
            import urllib3

            original_urlopen = urllib3.HTTPConnectionPool.urlopen
            blocker = self

            def patched_urlopen(self_pool: Any, method: str, url: str, **kwargs: Any) -> Any:
                host = self_pool.host
                blocker._check_host(host)
                return original_urlopen(self_pool, method, url, **kwargs)

            urllib3.HTTPConnectionPool.urlopen = patched_urlopen  # type: ignore[assignment]  # Monkey-patching for network blocking
            self._originals.append((urllib3.HTTPConnectionPool, "urlopen", original_urlopen))
            register_patch(
                urllib3.HTTPConnectionPool,
                "urlopen",
                original_urlopen,
                "urllib3.HTTPConnectionPool.urlopen (replay blocker)",
            )
        except ImportError:
            logger.debug("[StateLoom] urllib3 not installed, skipping patch")
