"""Field names and prim paths for the IsaacLab asset-scene fields.

One owner for the naming rule: file-backed asset instances and the
primitive passthrough fields both reserve their own namespace under
``{ENV_REGEX_NS}`` so neither can shadow the other, the ``World.add``
static fallback, or the ``Prop_*`` contact targets.
"""

from __future__ import annotations

__all__ = [
    "asset_field",
    "asset_prim_path",
    "static_field",
    "static_prim_path",
    "static_shape_field",
    "static_shape_prim_path",
]


def _env_prefix_sentinel() -> str:
    """Import the shared contact-namespace sentinel at call time.

    ``unigym.contacts.base`` is the single owner of the token, but it pulls
    torch; the import stays lazy so this naming module is engine-safe.
    """
    from unigym.contacts.base import ENV_PREFIX_SENTINEL

    return ENV_PREFIX_SENTINEL


def asset_field(scene: str, instance_id: str) -> str:
    """Return the scene field of one file-backed asset instance."""
    return f"asset_{scene}_{instance_id}"


def asset_prim_path(scene: str, instance_id: str) -> str:
    """Return the prim path of one file-backed asset instance.

    The path is flat (``Scene_<scene>_<id>``) because IsaacLab spawners do not
    create intermediate parent prims: a nested ``Scene_<scene>/<id>`` leaf would
    fail with ``Unable to find source prim path``.
    """
    return f"{_env_prefix_sentinel()}/Scene_{scene}_{instance_id}"


def static_field(instance_id: str) -> str:
    """Return the single-shape primitive field (``World.add`` naming)."""
    return f"static_{instance_id}"


def static_prim_path(instance_id: str) -> str:
    """Return the single-shape primitive prim path (``World.add`` naming)."""
    return f"{_env_prefix_sentinel()}/Static_{instance_id}"


def static_shape_field(instance_id: str, shape_name: str) -> str:
    """Return one shape field of a multi-shape primitive payload."""
    return f"static_{instance_id}_{shape_name}"


def static_shape_prim_path(instance_id: str, shape_name: str) -> str:
    """Return one shape prim path of a multi-shape primitive payload."""
    return f"{_env_prefix_sentinel()}/Static_{instance_id}/{shape_name}"
