"""E2E: Call auto-routed locally while shadow is also enabled.

Catches the race condition where shadow calls were counted for
locally-routed requests. The shadow should be cancelled when the
call is routed locally.

Expected: cloud_calls=0, local_calls=1, shadow total_calls=0.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from tests.test_e2e.helpers import invoke_pipeline, make_ollama_response


def test_local_with_shadow(e2e_gate, api_client):
    gate = e2e_gate(
        local_model="llama3.2",
        shadow=True,
        auto_route=True,
        cache=False,
    )
    client = api_client(gate)

    ollama_resp = make_ollama_response("Local answer", model="llama3.2")

    router = gate._auto_router
    shadow_mw = gate._shadow_middleware

    with (
        patch.object(router._client, "chat", return_value=ollama_resp),
        patch.object(router._client, "is_available", return_value=True),
        patch.object(shadow_mw._client, "chat", return_value=ollama_resp),
    ):
        router._ollama_available = True
        router._ollama_check_time = time.monotonic()

        request_kwargs = {"messages": [{"role": "user", "content": "Hi"}]}

        def should_not_be_called():
            raise AssertionError("Cloud LLM call should not be made")

        with gate.session(session_id="e2e-local-shadow-1") as session:
            invoke_pipeline(
                gate,
                session,
                request_kwargs,
                llm_call=should_not_be_called,
                model="gpt-3.5-turbo",
            )

    # Give shadow thread a brief moment to complete (if it ran at all)
    time.sleep(0.3)

    # --- Dashboard assertions ---
    stats = client.get("/stats").json()
    assert stats["cloud_calls"] == 0
    assert stats["local_calls"] == 1

    shadow_resp = client.get("/shadow/metrics").json()
    # Shadow should have been cancelled (locally-routed calls cancel shadow)
    # /shadow/metrics excludes cancelled events
    assert shadow_resp["total_calls"] == 0, (
        f"Expected shadow total_calls=0 for locally-routed call, got {shadow_resp['total_calls']}"
    )
