"""Field names and prim paths for the IsaacLab asset-scene fields.

One owner for the naming rule: file-backed asset instances and the
primitive passthrough fields both reserve their own namespace under
``{ENV_REGEX_NS}`` so neither can shadow the other, the ``World.add``
static fallback, or the ``Prop_*`` contact targets.
"""

from __future__ import annotations

from unigym.contacts.base import ENV_PREFIX_SENTINEL

__all__ = [
    "asset_field",
    "asset_prim_path",
    "static_field",
    "static_prim_path",
    "static_shape_field",
    "static_shape_prim_path",
]


def asset_field(scene: str, instance_id: str) -> str:
    """Return the scene field of one file-backed asset instance."""
    return f"asset_{scene}_{instance_id}"


def asset_prim_path(scene: str, instance_id: str) -> str:
    """Return the prim path of one file-backed asset instance."""
    return f"{ENV_PREFIX_SENTINEL}/Scene_{scene}/{instance_id}"


def static_field(instance_id: str) -> str:
    """Return the single-shape primitive field (``World.add`` naming)."""
    return f"static_{instance_id}"


def static_prim_path(instance_id: str) -> str:
    """Return the single-shape primitive prim path (``World.add`` naming)."""
    return f"{ENV_PREFIX_SENTINEL}/Static_{instance_id}"


def static_shape_field(instance_id: str, shape_name: str) -> str:
    """Return one shape field of a multi-shape primitive payload."""
    return f"static_{instance_id}_{shape_name}"


def static_shape_prim_path(instance_id: str, shape_name: str) -> str:
    """Return one shape prim path of a multi-shape primitive payload."""
    return f"{ENV_PREFIX_SENTINEL}/Static_{instance_id}/{shape_name}"
