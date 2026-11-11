"""Harmless fixture pack module; never imported by manifest or catalog loading."""

from __future__ import annotations


def register() -> None:
    """Do nothing; the fixture exists only to prove code loading is lazy."""
    return None
