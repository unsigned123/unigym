"""USD-derived robot body inventory with authored and simulated populations.

The robot asset declares more rigid bodies than PhysX simulates.  Every prim
carrying ``UsdPhysics.RigidBodyAPI`` is an authored body, but PhysX merges the
child side of a fixed joint into its parent at articulation build time.  This
module owns the single USD traversal and exposes the two populations:

* ``AUTHORED``: every rigid body under the robot default prim, in traversal
  order, including fixed-joint merged children.
* ``SIMULATED``: ``AUTHORED`` minus the fixed-joint merged children, which is
  the body universe PhysX articulations and contact sensors see.

OpenUSD (``pxr``) is an Isaac Lab / Kit dependency and is imported lazily inside
the traversal function only.  Importing this module in the base environment
must never import ``pxr``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "BodyPopulation",
    "BodyPopulationView",
    "RigidBody",
    "RobotBodyInventory",
    "robot_body_inventory",
]


class BodyPopulation(StrEnum):
    """The two body universes declared by one robot asset."""

    AUTHORED = "authored"
    SIMULATED = "simulated"


@dataclass(frozen=True)
class RigidBody:
    """One rigid body under the robot asset root.

    Attributes:
        path: robot-relative prim path (``pelvis`` or
            ``left_hand/left_hand_index_0_link``).
        name: the USD prim name (``prim.GetName()``), used for resolution.
        collision: whether any descendant prim carries a collision schema.
        merged_into: the robot-relative parent path when a fixed joint merges
            this body into a parent, else ``None``.
    """

    path: str
    name: str
    collision: bool
    merged_into: str | None


@dataclass(frozen=True)
class BodyPopulationView:
    """A frozen projection of one population over the inventory rows.

    The view carries its population so downstream code resolves body names
    without repeating a ``population=`` argument.
    """

    population: BodyPopulation
    bodies: tuple[RigidBody, ...]

    def __post_init__(self) -> None:
        """Reject duplicate resolved names inside this population."""
        seen: dict[str, str] = {}
        for body in self.bodies:
            previous = seen.get(body.name)
            if previous is not None:
                raise ValueError(
                    f"{self.population.value} body inventory has duplicate name "
                    f"{body.name!r}: {previous!r} and {body.path!r}"
                )
            seen[body.name] = body.path

    def names(self) -> tuple[str, ...]:
        """The resolved body names in inventory order."""
        return tuple(body.name for body in self.bodies)

    def resolve(self, name: str) -> RigidBody:
        """Return the body named ``name``.

        Raises:
            KeyError: No body in this population has that name.
        """
        for body in self.bodies:
            if body.name == name:
                return body
        raise KeyError(f"{self.population.value} body inventory has no body named {name!r}")

    def contains(self, name: str) -> bool:
        """Whether this population contains a body named ``name``."""
        return any(body.name == name for body in self.bodies)


@dataclass(frozen=True)
class RobotBodyInventory:
    """The single USD-derived inventory for one robot asset."""

    asset_path: Path
    bodies: tuple[RigidBody, ...]

    def population(self, population: BodyPopulation) -> BodyPopulationView:
        """Project the inventory onto one body population.

        Raises:
            ValueError: ``population`` is not a ``BodyPopulation`` member.
        """
        if population is BodyPopulation.AUTHORED:
            bodies = self.bodies
        elif population is BodyPopulation.SIMULATED:
            bodies = tuple(body for body in self.bodies if body.merged_into is None)
        else:
            raise ValueError(f"unknown body population {population!r}")
        return BodyPopulationView(population=population, bodies=bodies)

    def names(self, population: BodyPopulation) -> tuple[str, ...]:
        """The resolved body names for one population, in inventory order."""
        return self.population(population).names()


def _usd_modules():
    """Import the OpenUSD modules at the function boundary."""
    from pxr import Sdf, Usd, UsdPhysics

    return Sdf, Usd, UsdPhysics


def _is_relative_to(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _relative_path(path: str, root_path: str) -> str:
    return path[len(root_path) + 1 :]


def _has_collision(prim, Usd, UsdPhysics) -> bool:
    """True when any descendant carries an OpenUSD collision schema."""
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI):
            return True
        if any("Collision" in schema for schema in child.GetAppliedSchemas()):
            return True
    return False


def _fixed_merged_parents(stage, root, Sdf, Usd, UsdPhysics) -> dict[str, str]:
    """Absolute merged child path -> absolute parent path for fixed joints.

    PhysX merges the child side of a fixed joint into the parent at
    articulation build time.  ``body1`` is the child for the assembler joints
    produced by the USD pipeline; a target outside the robot root is the world
    and never merges a robot body.

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
    """Build the single body inventory for the robot asset at ``asset_path``.

    The asset root is the stage's default prim.  ``AUTHORED`` lists every
    ``UsdPhysics.RigidBodyAPI`` prim in traversal order; ``SIMULATED`` excludes
    the fixed-joint merged children.  Both populations reject duplicate prim
    names with both paths.

    Raises:
        ValueError: The stage/root cannot be opened or the asset declares no
            rigid bodies.
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
    inventory = RobotBodyInventory(asset_path=Path(asset_path), bodies=tuple(bodies))
    inventory.population(BodyPopulation.AUTHORED)
    inventory.population(BodyPopulation.SIMULATED)
    return inventory
