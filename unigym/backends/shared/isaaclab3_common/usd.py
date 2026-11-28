"""Authored-only USD rigid-body inventory (test-only legacy traversal).

.. deprecated:: B4
   This is an authored, static USD view and is **not** runtime truth.  It is
   kept only because ``tests/backends/isaaclab3_physx/test_contacts.py`` still
   imports ``robot_rigid_bodies`` to compute an expected body count for the
   Cartpole asset.  New code must use the IsaacLab2 authored inventory
   (``unigym.backends.isaaclab2.body_inventory.robot_body_inventory``) for
   authored asset inspection and the live ``data.body_names`` for runtime body
   order.  The IsaacLab3 contact backend does not read this module.

This module walks the robot asset's USD subtree and reports every rigid body
with a robot-relative sensor path and its collision capability, applying the
static authored fixed-joint merge rule.  That rule is a static USD authoring
inference: it is not the live PhysX body universe.  Live articulation bodies
are measured from ``robot.data.body_names`` after spawn and may still include a
link that a USD fixed joint declares merged.

OpenUSD (``pxr``) is an Isaac Lab / Kit dependency and is imported lazily
inside the traversal functions only.  Importing this module in the base
environment must never import ``pxr``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "UsdRigidBody",
    "robot_rigid_bodies",
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

    This is the static authored fixed-joint rule used to build pre-clone
    sensor scopes.  PhysX does not necessarily merge that child into the
    parent, so this set is never the live articulation body universe.
    ``body1`` is the child for the assembler joints produced by the USD
    pipeline; a target outside the robot root is the world and never merges a
    robot body.
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

    The asset root is the stage's default prim.  Links that the static
    authored fixed-joint rule declares merged are excluded for pre-clone
    sensor scope construction; this is not a claim about the live PhysX body
    set.  Everything else under the root that carries ``PhysicsRigidBodyAPI``
    is included, including bodies without collision geometry (their contact
    channels read zero).

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
