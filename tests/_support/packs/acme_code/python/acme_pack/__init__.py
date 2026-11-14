"""Harmless fixture pack module; never imported by manifest or catalog loading."""

from __future__ import annotations

from unigym.pack_api import PackContext
from unigym.scenes.scene_ir import Scene


def register(context: PackContext) -> None:
    """Do nothing; the fixture exists only to prove context-aware code loading."""
    return None


def table_top(params: dict, context: object) -> Scene:
    """Build the fixture tabletop scene from the validated parameter mapping."""
    return Scene(name="tabletop")
