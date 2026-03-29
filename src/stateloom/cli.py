"""StateLoom CLI — replay, pull, and test commands.

Usage:
    stateloom replay <session-id> --mock-until-step N [--strict/--no-strict] [--allow-hosts HOST]
    stateloom pull <session-id>
    stateloom test --suite <name> [--mock-llm]
"""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="stateloom")
def main() -> None:
    """StateLoom — The first stateful AI gateway."""


@main.command()
@click.argument("session_id")
@click.option("--mock-until-step", "-m", required=True, type=int, help="Mock steps 1 through N.")
@click.option("--strict/--no-strict", default=True, help="Block outbound HTTP calls in replay.")
@click.option(
    "--allow-hosts", "-a", multiple=True, help="Hosts to allow through the network blocker."
)
def replay(
    session_id: str, mock_until_step: int, strict: bool, allow_hosts: tuple[str, ...]
) -> None:
    """Time-travel debugging: replay a session."""
    import stateloom

    gate = stateloom.init(
        auto_patch=True,
        dashboard=False,
        console_output=True,
    )
    try:
        gate.replay(
            session=session_id,
            mock_until_step=mock_until_step,
            strict=strict,
            allow_hosts=list(allow_hosts) if allow_hosts else None,
        )
    except stateloom.StateLoomReplayError as e:
        click.echo(f"Replay error: {e}", err=True)
        raise SystemExit(1)
    finally:
        stateloom.shutdown()


@main.command()
@click.argument("session_id")
def pull(session_id: str) -> None:
    """Pull a session from the control plane for local debugging."""
    click.echo(
        f"Control plane not configured. Cannot pull session '{session_id}'.\n"
        f"See https://docs.stateloom.io/control-plane for setup instructions."
    )


@main.command()
@click.option("--suite", "-s", required=True, help="Test suite name.")
@click.option(
    "--mock-llm/--no-mock-llm", default=False, help="Mock LLM calls with cached responses."
)
def test(suite: str, mock_llm: bool) -> None:
    """Run regression tests against pinned sessions."""
    click.echo(
        f"Test suite '{suite}' (mock-llm={mock_llm})\n"
        f"Regression testing not yet implemented.\n"
        f"See https://docs.stateloom.io/testing for details."
    )


if __name__ == "__main__":
    main()
