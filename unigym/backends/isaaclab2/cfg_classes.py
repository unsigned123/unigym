"""The opaque external IsaacLab cfg-class table boundary.

IsaacLab ships no usable type stubs for its ``*Cfg`` dataclasses here, so the
table values stay ``Any``.  This module is the single IsaacLab class boundary:
every other module addresses the table by its explicit key set and never
re-declares the opaque values.  ``total=False`` because offline tests inject a
partial fake table; a missing required class is a builder teaching error, not a
typing error.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict

__all__ = ["CfgClassKey", "IsaacLabCfgClasses", "cfg_class"]

CfgClassKey: TypeAlias = Literal[
    "AssetBaseCfg",
    "RigidObjectCfg",
    "UsdFileCfg",
    "UrdfFileCfg",
    "CollisionPropertiesCfg",
    "RigidBodyPropertiesCfg",
    "MassPropertiesCfg",
    "ArticulationRootPropertiesCfg",
    "PreviewSurfaceCfg",
    "CuboidCfg",
    "SphereCfg",
    "CylinderCfg",
    "CapsuleCfg",
    "GroundPlaneCfg",
]


class IsaacLabCfgClasses(TypedDict, total=False):
    """The external IsaacLab cfg classes the asset/static seams resolve.

    Values are ``Any`` on purpose: they are opaque external classes without
    stubs.  Do not add ``Any`` to the builders; they consume this table.
    """

    AssetBaseCfg: Any
    RigidObjectCfg: Any
    UsdFileCfg: Any
    UrdfFileCfg: Any
    CollisionPropertiesCfg: Any
    RigidBodyPropertiesCfg: Any
    MassPropertiesCfg: Any
    ArticulationRootPropertiesCfg: Any
    PreviewSurfaceCfg: Any
    CuboidCfg: Any
    SphereCfg: Any
    CylinderCfg: Any
    CapsuleCfg: Any
    GroundPlaneCfg: Any


def cfg_class(classes: IsaacLabCfgClasses, key: CfgClassKey) -> Any:
    """Return one required opaque external class from a (possibly partial) table.

    Offline tests inject only the classes a field form needs, so the table is
    partial by design.  A builder always requires its own keys; an absent key
    is a version mismatch against the pinned ``isaaclab2`` lane.
    """
    value = classes.get(key)
    if value is None:
        raise ValueError(
            f"isaaclab cfg class table is missing {key!r}; the pinned "
            f"isaaclab2 lane must expose it -- check the isaaclab2 environment"
        )
    return value
