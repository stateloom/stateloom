"""WebSocket handler for live dashboard updates."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.dashboard.ws")

_LOG_WS_QUEUE_SIZE = 500

# Thread-safe set of connected WebSocket clients
_clients: set[WebSocket] = set()
_clients_lock = threading.Lock()


def create_websocket_route(gate: Gate):
    """Create the WebSocket endpoint handler."""

    async def websocket_handler(websocket: WebSocket):
        await websocket.accept()
        with _clients_lock:
            _clients.add(websocket)
        logger.debug("[StateLoom] WebSocket client connected (%d total)", len(_clients))

        try:
            while True:
                # Keep connection alive, handle any client messages
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                    # Handle ping/pong
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    # Send a heartbeat to detect dead connections
                    try:
                        await websocket.send_text(json.dumps({"type": "heartbeat"}))
                    except Exception:
                        break
                except Exception:
                    # Any receive error means the connection is dead
                    break
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("WebSocket handler error", exc_info=True)
        finally:
            with _clients_lock:
                _clients.discard(websocket)
            # Attempt graceful close (ignore errors on already-closed sockets)
            try:
                await websocket.close()
            except Exception:
                pass
            logger.debug("[StateLoom] WebSocket client disconnected (%d total)", len(_clients))

    return websocket_handler


async def broadcast_event(event_data: dict) -> None:
    """Broadcast an event to all connected WebSocket clients."""
    with _clients_lock:
        if not _clients:
            return
        snapshot = set(_clients)

    message = json.dumps({"type": "new_event", "data": event_data})
    disconnected = set()

    for client in snapshot:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)

    if disconnected:
        with _clients_lock:
            _clients.difference_update(disconnected)


async def broadcast_session_update(session_data: dict) -> None:
    """Broadcast a session update to all connected clients."""
    with _clients_lock:
        if not _clients:
            return
        snapshot = set(_clients)

    message = json.dumps({"type": "session_update", "data": session_data})
    disconnected = set()

    for client in snapshot:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)

    if disconnected:
        with _clients_lock:
            _clients.difference_update(disconnected)


def create_log_websocket_route(gate: Gate):
    """Create a WebSocket endpoint for streaming server logs (debug mode only)."""

    async def log_ws_handler(websocket: WebSocket):
        if not getattr(gate.config, "debug", False):
            await websocket.close(code=1008, reason="Debug mode not enabled")
            return

        from stateloom.dashboard.log_buffer import get_log_buffer

        buf = get_log_buffer()
        if buf is None:
            await websocket.close(code=1008, reason="Log buffer not available")
            return

        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=_LOG_WS_QUEUE_SIZE)
        buf.subscribe(queue)

        try:
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=30)
                    await websocket.send_text(json.dumps({"type": "log", "data": entry}))
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"type": "heartbeat"}))
                except Exception:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            buf.unsubscribe(queue)
            try:
                await websocket.close()
            except Exception:
                pass

    return log_ws_handler


def broadcast_sync(event_data: dict) -> None:
    """Fire-and-forget broadcast from a synchronous context."""
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(broadcast_event(event_data))
        loop.close()
    except Exception:
        pass
