"""Compatibility facade over the IsaacLab 2 authored robot body inventory.

The isaaclab contact-sensor objects must exist BEFORE scene cloning, while the
post-spawn ``robot.data.body_names`` does not exist yet.  The USD traversal is
owned by ``unigym.backends.isaaclab2.body_inventory``; this module keeps the
historical ``robot_rigid_bodies`` import path as a thin wrapper over that
authored inventory.

The wrapper reports authored USD bodies only.  It is not the live PhysX body
universe: B2 measures ``robot.data.body_names`` after spawn and must not infer
simulated bodies from fixed-joint metadata.

OpenUSD (``pxr``) is an Isaac Lab / Kit dependency and is imported lazily by the
inventory module only.  Importing this module in the base environment must
never import ``pxr``.
"""

from __future__ import annotations

from dataclasses import dataclass

from unigym.backends.isaaclab2.body_inventory import robot_body_inventory

__all__ = [
    "UsdRigidBody",
    "robot_rigid_bodies",
    "robot_relative_body_paths",
]


@dataclass(frozen=True)
class UsdRigidBody:
    """One authored rigid body under the robot asset root.

    Attributes:
        path: robot-relative prim path (``pelvis`` or
            ``left_hand/left_hand_index_0_link``).
        collision: whether any descendant prim carries a collision schema,
            including collision prims inside native instance prototypes.
    """

    path: str
    collision: bool


def robot_rigid_bodies(asset_path: str) -> list[UsdRigidBody]:
    """Enumerate the robot asset's authored rigid bodies in traversal order.

    The asset root is the stage's default prim.  Everything under the root that
    carries ``PhysicsRigidBodyAPI`` is included, including bodies without
    collision geometry and bodies whose fixed joints record a ``merged_into``
    parent.  This is the compatibility wrapper over the authored inventory, not
    a claim about the live PhysX body set.

    Raises:
        ValueError: The stage/root cannot be opened or the asset declares no
            rigid bodies.
    """
    inventory = robot_body_inventory(asset_path)
    return [UsdRigidBody(body.path, body.collision) for body in inventory.bodies]


def robot_relative_body_paths(asset_path: str) -> list[str]:
    """The robot-relative authored rigid-body paths in inventory order."""
    return [body.path for body in robot_rigid_bodies(asset_path)]
