"""USD-authored robot body inventory.

This module owns the single USD traversal for a robot asset and exposes the
authored rigid-body population.  Every prim under the robot default prim that
carries ``UsdPhysics.RigidBodyAPI`` is an authored body, including bodies that a
fixed joint declares merged into a parent.

The inventory is USD authoring truth only.  It deliberately exposes no
``SIMULATED`` population: live PhysX articulation bodies are a B2 measurement
from ``robot.data.body_names`` and must never be inferred by subtracting
fixed-joint children from the USD graph.  ``merged_into`` is kept as diagnostic
authoring metadata and never removes a body from this inventory.

OpenUSD (``pxr``) is an Isaac Lab / Kit dependency and is imported lazily inside
the traversal function only.  Importing this module in the base environment
must never import ``pxr``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "RigidBody",
    "RobotBodyInventory",
    "robot_body_inventory",
]


@dataclass(frozen=True)
class RigidBody:
    """One authored rigid body under the robot asset root.

    Attributes:
        path: robot-relative prim path (``pelvis`` or
            ``left_hand/left_hand_index_0_link``).  This is the durable body
            key; leaf prim names may collide across scopes.
        name: the USD prim name (``prim.GetName()``), kept for display and for
            unambiguous name lookup only.
        collision: whether any descendant prim carries a collision schema,
            including collision prims inside native instance prototypes.
        merged_into: the robot-relative parent path when a fixed joint declares
            this body merged into it, else ``None``.  Diagnostic authoring
            metadata only; it never removes this body from the inventory.
    """

    path: str
    name: str
    collision: bool
    merged_into: str | None


@dataclass(frozen=True)
class RobotBodyInventory:
    """The authored USD body inventory for one robot asset."""

    asset_path: Path
    bodies: tuple[RigidBody, ...]

    def __post_init__(self) -> None:
        """Reject duplicate durable body keys with both offending rows."""
        seen: dict[str, RigidBody] = {}
        for body in self.bodies:
            previous = seen.get(body.path)
            if previous is not None:
                raise ValueError(
                    "authored body inventory has duplicate path "
                    f"{body.path!r}: name {previous.name!r} and name {body.name!r}"
                )
            seen[body.path] = body

    def paths(self) -> tuple[str, ...]:
        """The durable robot-relative body paths in inventory order."""
        return tuple(body.path for body in self.bodies)

    def names(self) -> tuple[str, ...]:
        """The leaf body names in inventory order.

        Different scopes may legitimately share a leaf name; use :meth:`paths`
        for an unambiguous key.
        """
        return tuple(body.name for body in self.bodies)

    def resolve(self, key: str) -> RigidBody:
        """Return the body addressed by a durable path or a unique leaf name.

        Paths always win over names.  A leaf name that matches several bodies is
        ambiguous and rejected rather than resolved by traversal order.

        Raises:
            KeyError: No body has that path or name.
            ValueError: The key matches several bodies by leaf name.
        """
        for body in self.bodies:
            if body.path == key:
                return body
        matches = [body for body in self.bodies if body.name == key]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"authored body inventory has no body at path or name {key!r}")
        paths = ", ".join(repr(body.path) for body in matches)
        raise ValueError(
            f"authored body inventory name {key!r} is ambiguous across paths {paths}; "
            "use the full robot-relative path"
        )

    def contains(self, key: str) -> bool:
        """Whether any body path or leaf name equals ``key``."""
        return any(body.path == key or body.name == key for body in self.bodies)


def _usd_modules():
    """Import the OpenUSD modules at the function boundary."""
    from pxr import Sdf, Usd, UsdPhysics

    return Sdf, Usd, UsdPhysics


def _is_relative_to(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _relative_path(path: str, root_path: str) -> str:
    return path[len(root_path) + 1 :]


def _has_collision(prim, Usd, UsdPhysics) -> bool:
    """True when any descendant carries an OpenUSD collision schema.

    Collision prims live inside native instance prototypes in the composed G1
    asset, so the traversal must expand instance proxies.
    """
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if child.HasAPI(UsdPhysics.CollisionAPI):
            return True
        if any("Collision" in schema for schema in child.GetAppliedSchemas()):
            return True
    return False


def _fixed_merged_parents(stage, root, Sdf, Usd, UsdPhysics) -> dict[str, str]:
    """Absolute merged child path -> absolute parent path for fixed joints.

    A fixed joint records diagnostic authoring intent.  PhysX does not
    necessarily merge the child into the parent, so this map never removes a
    body from the authored inventory.

    ``body1`` is the child for the assembler joints produced by the USD
    pipeline; a target outside the robot root is the world and never merges a
    robot body.

    The first fixed joint in traversal order owns the merge when several joints
    name the same child; the merged child set is unchanged either way.
    """
    root_path = str(root.GetPath())
    merged: dict[str, str] = {}
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
            merged.setdefault(child, parent)
    return merged


def robot_body_inventory(asset_path: str | Path) -> RobotBodyInventory:
    """Build the authored body inventory for the robot asset at ``asset_path``.

    The asset root is the stage's default prim.  Every
    ``UsdPhysics.RigidBodyAPI`` prim in traversal order is listed, including
    bodies whose ``merged_into`` metadata names a fixed-joint parent.  Duplicate
    durable paths are rejected with both body names; duplicate leaf names are
    allowed but ambiguous name lookups are rejected.

    Raises:
        ValueError: The stage/root cannot be opened, the asset declares no rigid
            bodies, or two bodies share a durable path.
    """
    Sdf, Usd, UsdPhysics = _usd_modules()
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise ValueError(f"body inventory: could not open USD asset {asset_path!r}")
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError(f"body inventory: USD asset {asset_path!r} has no default prim")
    merged = _fixed_merged_parents(stage, root, Sdf, Usd, UsdPhysics)
    root_path = str(root.GetPath())
    bodies: list[RigidBody] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path = str(prim.GetPath())
        parent = merged.get(path)
        bodies.append(
            RigidBody(
                path=_relative_path(path, root_path),
                name=prim.GetName(),
                collision=_has_collision(prim, Usd, UsdPhysics),
                merged_into=None if parent is None else _relative_path(parent, root_path),
            )
        )
    if not bodies:
        raise ValueError(
            f"body inventory: USD asset {asset_path!r} declares no rigid bodies "
            f"under {root_path!r}"
        )
    return RobotBodyInventory(asset_path=Path(asset_path), bodies=tuple(bodies))
