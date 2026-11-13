"""Lazy OpenUSD rigid-body inventory for pre-clone contact sensors.

The isaaclab contact-sensor objects must exist BEFORE scene cloning, while
the post-spawn ``robot.data.body_names`` does not exist yet.  This module owns
the pre-clone inventory: it walks the robot asset's USD subtree and reports
every rigid body with a robot-relative sensor path, its collision capability
and the fixed-joint merged links excluded.

OpenUSD (``pxr``) is an Isaac Lab / Kit dependency and is imported lazily
inside the traversal functions only.  Importing this module in the base
environment must never import ``pxr``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "UsdRigidBody",
    "robot_rigid_bodies",
    "robot_relative_body_paths",
]


@dataclass(frozen=True)
class UsdRigidBody:
    """One rigid body under the robot asset root.

    Attributes:
        path: robot-relative prim path (``pelvis`` or
            ``left_hand/left_hand_index_0_link``).
        collision: whether any descendant prim carries a collision schema.
    """

    path: str
    collision: bool


def _usd_modules():
    """Import the OpenUSD modules at the function boundary."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    return Sdf, Usd, UsdGeom, UsdPhysics


def _is_relative_to(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _has_collision(prim, Usd, UsdPhysics) -> bool:
    """True when any descendant carries an OpenUSD collision schema."""
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI):
            return True
        if any("Collision" in schema for schema in child.GetAppliedSchemas()):
            return True
    return False


def _fixed_merged_paths(stage, root, Sdf, Usd, UsdPhysics) -> set[str]:
    """Absolute prim paths merged into a parent by a fixed joint.

    PhysX merges the child side of a fixed joint into the parent at
    articulation build time.  ``body1`` is the child for the assembler joints
    produced by the USD pipeline; a target outside the robot root is the
    world and never merges a robot body.
    """
    root_path = str(root.GetPath())
    merged: set[str] = set()
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        targets: dict[str, list[str]] = {}
        for relation in ("physics:body0", "physics:body1"):
            rel = prim.GetRelationship(relation)
            targets[relation] = [str(path) for path in rel.GetTargets()] if rel else []
        body0 = targets["physics:body0"]
        body1 = targets["physics:body1"]
        if len(body0) != 1 or len(body1) != 1:
            continue
        parent, child = body0[0], body1[0]
        if not (_is_relative_to(parent, root_path) and _is_relative_to(child, root_path)):
            continue
        if child == root_path or parent == child:
            continue
        child_prim = stage.GetPrimAtPath(Sdf.Path(child))
        if child_prim is not None and child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            merged.add(child)
    return merged


def robot_rigid_bodies(asset_path: str) -> list[UsdRigidBody]:
    """Enumerate the robot asset's rigid bodies in USD traversal order.

    The asset root is the stage's default prim.  Fixed-joint merged links are
    excluded (they do not exist as PhysX bodies); everything else under the
    root that carries ``PhysicsRigidBodyAPI`` is included, including bodies
    without collision geometry (their contact channels read zero).

    Raises:
        ValueError: The stage/root cannot be opened or the asset declares no
            rigid bodies.
    """
    Sdf, Usd, UsdGeom, UsdPhysics = _usd_modules()
    del UsdGeom
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise ValueError(f"contact inventory: could not open USD asset {asset_path!r}")
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError(f"contact inventory: USD asset {asset_path!r} has no default prim")
    merged = _fixed_merged_paths(stage, root, Sdf, Usd, UsdPhysics)
    root_path = str(root.GetPath())
    bodies: list[UsdRigidBody] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path = str(prim.GetPath())
        if path in merged:
            continue
        relative = path[len(root_path) + 1 :]
        bodies.append(UsdRigidBody(relative, _has_collision(prim, Usd, UsdPhysics)))
    if not bodies:
        raise ValueError(
            f"contact inventory: USD asset {asset_path!r} declares no rigid bodies "
            f"under {root_path!r}"
        )
    return bodies


def robot_relative_body_paths(asset_path: str) -> list[str]:
    """The robot-relative rigid-body sensor paths in inventory order."""
    return [body.path for body in robot_rigid_bodies(asset_path)]
