"""Harmless fixture pack module; never imported by manifest or catalog loading."""

from __future__ import annotations

from unigym.pack_api import PackContext


def register(context: PackContext) -> None:
    """Do nothing; the fixture exists only to prove context-aware code loading."""
    return None
