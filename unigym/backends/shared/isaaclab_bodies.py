"""Shared IsaacLab profile body-binding resolution.

IsaacLab observations and the runtime query both map a robot-profile USD body
name onto the live articulation body order.  The matching rule (exact name,
then a unique path-leaf match) owns here so the env and the query cannot drift.
Callers raise their own domain error when the candidate set is not exactly one.

The live articulation body order is always ``data.body_names``.  The env-site
resolver below is the single boundary that turns one robot-profile USD binding
into a live body index; it fails fast when the live data block exposes no body
names instead of falling back to a stale authored list.  This module stays
import-safe without the Kit so the env-site contract is unit-testable with
fakes.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "body_binding_candidates",
    "resolve_site_body_index",
    "site_body_index",
]


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


def site_body_index(body_names: Sequence[str], usd_name: str, site_name: str) -> int:
    """Resolve one profile USD body name against the live articulation order.

    Raises:
        ValueError: The binding matches zero or several live bodies.
    """
    names = tuple(str(name) for name in body_names)
    matches = body_binding_candidates(names, usd_name)
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"observations.site {site_name!r}: body binding {usd_name!r} does not "
        f"resolve to exactly one live body; candidates {list(matches)}; "
        f"live bodies: {list(names)}"
    )


def resolve_site_body_index(
    data: object,
    *,
    usd_name: str,
    site_name: str,
    owner: str,
    backend: str,
) -> int:
    """Resolve one env-site body index from the live ``data.body_names``.

    This is the environment-site resolution boundary shared by IsaacLab2 and
    IsaacLab3.  ``data.body_names`` is the single live articulation body source;
    an absent or empty block fails fast with a teaching error naming the site,
    owner and USD binding.  It never falls back to ``articulation.body_names``
    or ``asset.body_names``.

    Raises:
        ValueError: The live data block exposes no body names, or the binding
            does not match exactly one live body.
    """
    raw_names = getattr(data, "body_names", None)
    body_names = () if raw_names is None else tuple(str(name) for name in raw_names)
    if not body_names:
        raise ValueError(
            f"{backend}: observations.site {site_name!r} owner articulation "
            f"{owner!r} exposes no live data.body_names; cannot resolve the "
            f"site body {usd_name!r}"
        )
    return site_body_index(body_names, usd_name, site_name)
