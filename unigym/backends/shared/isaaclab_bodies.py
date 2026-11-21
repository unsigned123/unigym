"""Shared IsaacLab profile body-binding resolution.

IsaacLab observations and the runtime query both map a robot-profile USD body
name onto the live articulation body order.  The matching rule (exact name,
then a unique path-leaf match) owns here so the env and the query cannot drift.
Callers raise their own domain error when the candidate set is not exactly one.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["body_binding_candidates"]


def body_binding_candidates(body_names: Sequence[str], usd_name: str) -> tuple[int, ...]:
    """Return every live body index matching one profile USD body name.

    Args:
        body_names: The live articulation body order.
        usd_name: The profile binding name (a bare name or a USD prim path).

    Returns:
        The matching indexes in live body order.  An exact name match wins
        immediately; otherwise the last prim-path component is matched.
    """
    names = tuple(str(name) for name in body_names)
    if usd_name in names:
        return (names.index(usd_name),)
    leaf = usd_name.rstrip("/").rsplit("/", 1)[-1]
    return tuple(
        index
        for index, name in enumerate(names)
        if name == leaf or name.rstrip("/").rsplit("/", 1)[-1] == leaf
    )
