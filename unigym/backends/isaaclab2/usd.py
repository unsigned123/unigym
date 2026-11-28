"""Compatibility facade over the IsaacLab 2 robot body inventory.

The isaaclab contact-sensor objects must exist BEFORE scene cloning, while the
post-spawn ``robot.data.body_names`` does not exist yet.  The USD traversal and
the authored/simulated population split are owned by
``unigym.backends.isaaclab2.body_inventory``; this module keeps the historical
``robot_rigid_bodies`` import path as a thin wrapper over the simulated
population until callers migrate.

OpenUSD (``pxr``) is an Isaac Lab / Kit dependency and is imported lazily by the
inventory module only.  Importing this module in the base environment must
never import ``pxr``.
"""

from __future__ import annotations

from dataclasses import dataclass

from unigym.backends.isaaclab2.body_inventory import (
    BodyPopulation,
    robot_body_inventory,
)

__all__ = [
    "UsdRigidBody",
    "robot_rigid_bodies",
    "robot_relative_body_paths",
]


@dataclass(frozen=True)
class UsdRigidBody:
    """One simulated rigid body under the robot asset root.

    Attributes:
        path: robot-relative prim path (``pelvis`` or
            ``left_hand/left_hand_index_0_link``).
        collision: whether any descendant prim carries a collision schema.
    """

    path: str
    collision: bool


def robot_rigid_bodies(asset_path: str) -> list[UsdRigidBody]:
    """Enumerate the robot asset's simulated rigid bodies in traversal order.

    The asset root is the stage's default prim.  Fixed-joint merged links are
    excluded (they do not exist as PhysX bodies); everything else under the
    root that carries ``PhysicsRigidBodyAPI`` is included, including bodies
    without collision geometry (their contact channels read zero).

    This is the compatibility wrapper over
    ``RobotBodyInventory.population(BodyPopulation.SIMULATED)``.

    Raises:
        ValueError: The stage/root cannot be opened or the asset declares no
            rigid bodies.
    """
    view = robot_body_inventory(asset_path).population(BodyPopulation.SIMULATED)
    return [UsdRigidBody(body.path, body.collision) for body in view.bodies]


def robot_relative_body_paths(asset_path: str) -> list[str]:
    """The robot-relative simulated rigid-body paths in inventory order."""
    return [body.path for body in robot_rigid_bodies(asset_path)]
