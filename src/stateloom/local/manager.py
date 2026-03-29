"""Managed Ollama lifecycle — download, start, stop, health-check."""

from __future__ import annotations

import atexit
import logging
import os
import platform
import signal
import stat
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("stateloom.local.manager")

DEFAULT_PORT = 11435
OLLAMA_DIR = Path.home() / ".stateloom" / "ollama"
BIN_PATH = OLLAMA_DIR / "bin" / "ollama"
PID_PATH = OLLAMA_DIR / "ollama.pid"
LOG_PATH = OLLAMA_DIR / "ollama.log"
VERSION_PATH = OLLAMA_DIR / "version"

_GITHUB_LATEST = "https://github.com/ollama/ollama/releases/latest"


def _get_download_url(tag: str) -> str:
    """Return the platform-specific archive download URL for the given release tag.

    Ollama releases are now distributed as archives:
    - macOS: ``ollama-darwin.tgz`` (gzip tar)
    - Linux: ``ollama-linux-{arch}.tar.zst`` (zstd tar)
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    base = f"https://github.com/ollama/ollama/releases/download/{tag}"

    if system == "darwin":
        return f"{base}/ollama-darwin.tgz"

    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return f"{base}/ollama-linux-amd64.tar.zst"
        if machine in ("aarch64", "arm64"):
            return f"{base}/ollama-linux-arm64.tar.zst"
        raise RuntimeError(f"Unsupported Linux architecture: {machine}")

    raise RuntimeError(f"Unsupported platform: {system}")


class OllamaManager:
    """Downloads, starts, stops, and health-checks a managed Ollama instance.

    Storage layout (~/.stateloom/ollama/):
        bin/ollama   — the binary
        version      — installed version string
        ollama.pid   — PID of managed process
        ollama.log   — stdout/stderr from managed process
    """

    def __init__(
        self,
        *,
        ollama_dir: Path | None = None,
    ) -> None:
        self._dir = ollama_dir or OLLAMA_DIR
        self._bin = self._dir / "bin" / "ollama"
        self._pid_path = self._dir / "ollama.pid"
        self._log_path = self._dir / "ollama.log"
        self._version_path = self._dir / "version"
        self._process: subprocess.Popen[Any] | None = None
        self._atexit_registered = False

    # --- Install ---

    def install(
        self,
        version: str = "latest",
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[Path, str]:
        """Download the Ollama binary for the current platform.

        Args:
            version: Release tag (e.g. "v0.9.0") or "latest".
            progress: Optional callback ``(bytes_downloaded, total_bytes)``.

        Returns:
            ``(binary_path, version_tag)`` tuple.
        """
        tag = self._resolve_version(version)
        url = _get_download_url(tag)

        self._dir.mkdir(parents=True, exist_ok=True)
        bin_dir = self._dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading Ollama %s from %s", tag, url)
        self._download_and_extract(url, bin_dir, progress=progress)

        # Set executable permission on the binary
        self._bin.chmod(self._bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

        # Write version file
        self._version_path.write_text(tag)

        logger.info("Installed Ollama %s at %s", tag, self._bin)
        return self._bin, tag

    def is_installed(self) -> bool:
        """Check if the managed Ollama binary exists."""
        return self._bin.exists()

    def installed_version(self) -> str | None:
        """Return the installed version tag, or None if not installed."""
        if not self._version_path.exists():
            return None
        return self._version_path.read_text().strip()

    # --- Start / Stop ---

    def start(self, port: int = DEFAULT_PORT) -> int:
        """Start a managed Ollama instance on the given port.

        Returns the PID of the started process.

        Raises:
            RuntimeError: If the binary is not installed.
            RuntimeError: If Ollama fails to start within the health-check timeout.
        """
        if self.is_running(port=port):
            pid = self._read_pid()
            logger.info("Managed Ollama already running (PID %s)", pid)
            if pid is not None:
                return pid

        if not self.is_installed():
            raise RuntimeError("Ollama binary not installed. Run: stateloom ollama install")

        self._dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(self._log_path, "a")

        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"

        proc = subprocess.Popen(
            [str(self._bin), "serve"],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self._process = proc
        self._pid_path.write_text(str(proc.pid))

        # Wait for health check
        if not self._wait_for_health(port, timeout=10.0):
            self.stop()
            raise RuntimeError(
                f"Ollama failed to become healthy on port {port} within 10s. "
                f"Check logs: {self._log_path}"
            )

        # Register atexit cleanup
        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True

        logger.info("Managed Ollama started (PID %d, port %d)", proc.pid, port)
        return proc.pid

    def stop(self) -> None:
        """Stop the managed Ollama process."""
        pid = self._read_pid()
        if pid is None:
            return

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            self._pid_path.unlink(missing_ok=True)
            return

        # Wait up to 5s for graceful shutdown
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.2)
        else:
            # Force kill if still alive
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

        self._pid_path.unlink(missing_ok=True)
        self._process = None
        logger.info("Managed Ollama stopped (PID %d)", pid)

    def is_running(self, *, port: int = DEFAULT_PORT) -> bool:
        """Check if the managed Ollama process is alive and responding."""
        pid = self._read_pid()
        if pid is None:
            return False

        # Check process is alive
        try:
            os.kill(pid, 0)
        except OSError:
            self._pid_path.unlink(missing_ok=True)
            return False

        # Check HTTP health
        return self._http_health_check(port)

    # --- Ensure helpers ---

    def ensure_running(self, port: int = DEFAULT_PORT) -> None:
        """Ensure a managed Ollama instance is running.

        Raises RuntimeError if the binary is not installed.
        """
        if self.is_running(port=port):
            return
        self.start(port=port)

    def ensure_model(
        self,
        model: str,
        port: int = DEFAULT_PORT,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Ensure a model is available, pulling it if necessary.

        Args:
            model: Model name (e.g. "llama3.2").
            port: Port the managed Ollama is running on.
            progress: Optional callback receiving progress dicts from the pull stream.
        """
        host = f"http://127.0.0.1:{port}"
        try:
            with httpx.Client(base_url=host, timeout=httpx.Timeout(10.0)) as client:
                resp = client.get("/api/tags")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                # Check by name (Ollama returns names like "llama3.2:latest")
                for m in models:
                    name = m.get("name", "")
                    if name == model or name.startswith(f"{model}:"):
                        return
        except Exception:
            logger.debug("Failed to check models, will attempt pull", exc_info=True)

        # Model not present — pull it
        logger.info("Pulling model '%s'...", model)
        self._pull_model(model, port=port, progress=progress)

    # --- Internal helpers ---

    def _resolve_version(self, version: str) -> str:
        """Resolve 'latest' to an actual release tag."""
        if version != "latest":
            return version

        try:
            with httpx.Client(follow_redirects=False, timeout=httpx.Timeout(15.0)) as client:
                resp = client.head(_GITHUB_LATEST)
                if resp.status_code in (301, 302):
                    location = resp.headers.get("location", "")
                    # e.g. https://github.com/ollama/ollama/releases/tag/v0.9.0
                    tag = location.rsplit("/", 1)[-1]
                    if tag:
                        return tag
                # Fallback: follow redirect and parse URL
                resp2 = client.get(_GITHUB_LATEST, follow_redirects=True)
                tag = str(resp2.url).rsplit("/", 1)[-1]
                if tag:
                    return tag
        except Exception as e:
            raise RuntimeError(f"Failed to resolve latest Ollama version: {e}") from e

        raise RuntimeError("Could not determine latest Ollama release tag")

    def _download_and_extract(
        self,
        url: str,
        dest_dir: Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Download an archive and extract it into *dest_dir*.

        Supports ``.tgz`` (gzip tar) and ``.tar.zst`` (zstd tar) archives.
        """
        import io
        import tarfile
        import tempfile

        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(300.0),
        ) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                buf = io.BytesIO()

                for chunk in resp.iter_bytes(chunk_size=65536):
                    buf.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, total)

        raw = buf.getvalue()

        if url.endswith(".tar.zst"):
            try:
                import zstandard
            except ImportError:
                raise RuntimeError(
                    "zstandard package required for Linux install. Run: pip install zstandard"
                )
            dctx = zstandard.ZstdDecompressor()
            raw = dctx.decompress(raw, max_output_size=len(raw) * 10)

        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
            for member in tf.getmembers():
                # Flatten: extract files directly into dest_dir regardless of
                # any directory nesting inside the archive.
                member.name = Path(member.name).name
                if not member.name:
                    continue
                tf.extract(member, path=dest_dir, filter="data")

    def _read_pid(self) -> int | None:
        """Read PID from the PID file, returning None if absent or stale."""
        if not self._pid_path.exists():
            return None
        try:
            pid = int(self._pid_path.read_text().strip())
        except (ValueError, OSError):
            return None
        # Check if process is alive
        try:
            os.kill(pid, 0)
        except OSError:
            self._pid_path.unlink(missing_ok=True)
            return None
        return pid

    def _http_health_check(self, port: int) -> bool:
        """Quick HTTP health check — GET /api/tags returns 200."""
        try:
            with httpx.Client(timeout=httpx.Timeout(3.0)) as client:
                resp = client.get(f"http://127.0.0.1:{port}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def _wait_for_health(self, port: int, timeout: float = 10.0) -> bool:
        """Poll until health check passes or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._http_health_check(port):
                return True
            time.sleep(0.3)
        return False

    def _pull_model(
        self,
        model: str,
        port: int = DEFAULT_PORT,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Pull a model via the Ollama HTTP API."""
        import json

        host = f"http://127.0.0.1:{port}"
        with httpx.Client(
            base_url=host,
            timeout=httpx.Timeout(600.0),
        ) as client:
            with client.stream("POST", "/api/pull", json={"model": model}) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if progress:
                        progress(data)

    def _atexit_cleanup(self) -> None:
        """Cleanup handler registered via atexit."""
        try:
            self.stop()
        except Exception:
            pass
