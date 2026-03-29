"""Entry point shim — the installed script imports ``stateloom.cli.main:cli``."""

from stateloom.cli import main as cli

__all__ = ["cli"]
