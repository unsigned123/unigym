"""Backend-neutral robot profile data.

A :class:`RobotProfile` is authoring data only: named joint sets, the
standby pose, feet, body parts, declared assets and named action faces.  It
carries no backend object and no backend switch; a backend that needs an
engine-specific realization builds it from this profile at its own boundary
(``BackendInfo.robot_realizer``).

The profile supplies the robot facts consumed by controls, observations and
the mujoco-family backends.  The isaaclab backends consume it through their
realizer, which owns actuator tables, prim paths and spawn conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from unigym._typing import JsonValue

__all__ = ["RobotAsset", "RobotProfile"]

_ASSET_FORMATS = frozenset({"usd", "mjcf", "urdf"})


@dataclass(frozen=True)
class RobotAsset:
    """One declared robot asset reference.

    Attributes:
        format: asset format (``usd`` / ``mjcf`` / ``urdf``).
        ref: resource reference as declared by the author (a pack-relative
            or absolute path/URL); resolution belongs to the composition
            boundary, not to this record.
    """

    format: str
    ref: str

    def __post_init__(self) -> None:
        if self.format not in _ASSET_FORMATS:
            raise ValueError(
                f"RobotAsset format {self.format!r} is not one of " f"{sorted(_ASSET_FORMATS)}"
            )
        if not self.ref:
            raise ValueError(f"RobotAsset({self.format!r}) requires a non-empty ref")


@dataclass(frozen=True)
class RobotProfile:
    """The backend-neutral authoring profile of one robot.

    Attributes:
        name: the profile kind (the registered robot factory name, e.g.
            ``g1``); an articulation may be renamed without changing it.
        joint_sets: the named joint vocabulary
            (``{set_name: (joint names in action order)}``).  ``body`` is
            the primary policy/body order; extra sets (``hand`` /
            ``right_hand`` / ...) are selected explicitly by
            ``ControlGroup.joint_set``.  This is the only authoring source
            of joint names.
        default_joint_pos: joint name -> default angle (rad), i.e. the
            policy standby pose (must match the policy's q0 by name).
        foot_bodies: foot body names in policy order.
        body_parts: cross-robot body-part metadata (feet/base/joints/...).
        assets: declared asset formats and references.
        action_faces: neutral named action surfaces beyond the primary body
            face (``{face_name: (joint names in term input order)}``); this
            is authoring data, not a backend capability.
        full_joint_order: optional explicit full-articulation joint order
            (document order) used by :attr:`full_joint_names`; ``None``
            derives the set union in declaration order.

    Compatibility read-only properties (derived from the data above):

        joint_order: the ``body`` set as a list.
        full_joint_names: :attr:`full_joint_order` or the union of every
            declared set.  Published by the ObsCatalog joint keys.
    """

    name: str
    joint_sets: Mapping[str, tuple[str, ...]]
    default_joint_pos: Mapping[str, float]
    foot_bodies: tuple[str, ...]
    body_parts: Mapping[str, JsonValue] = field(default_factory=dict)
    assets: tuple[RobotAsset, ...] = ()
    action_faces: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    full_joint_order: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RobotProfile requires a non-empty name")
        sets: dict[str, tuple[str, ...]] = {}
        for set_name, joints in dict(self.joint_sets or {}).items():
            names = tuple(str(joint) for joint in (joints or ()))
            if not names:
                raise ValueError(f"RobotProfile joint set {set_name!r} is empty")
            if len(set(names)) != len(names):
                raise ValueError(f"RobotProfile joint set {set_name!r} contains duplicate names")
            sets[str(set_name)] = names
        if not sets:
            raise ValueError("RobotProfile must declare at least one joint set")
        faces: dict[str, tuple[str, ...]] = {}
        for face, joints in dict(self.action_faces or {}).items():
            names = tuple(str(joint) for joint in (joints or ()))
            if not names:
                raise ValueError(f"RobotProfile action face {face!r} is empty")
            if len(set(names)) != len(names):
                raise ValueError(f"RobotProfile action face {face!r} contains duplicate names")
            faces[str(face)] = names
        full_order = self.full_joint_order
        if full_order is not None:
            full_order = tuple(str(joint) for joint in full_order)
            if not full_order:
                raise ValueError("RobotProfile full_joint_order must not be empty")
            if len(set(full_order)) != len(full_order):
                raise ValueError("RobotProfile full_joint_order contains duplicate names")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "joint_sets", sets)
        default_pos = dict(self.default_joint_pos or {})
        object.__setattr__(
            self,
            "default_joint_pos",
            {str(joint): float(value) for joint, value in default_pos.items()},
        )
        object.__setattr__(self, "foot_bodies", tuple(str(body) for body in self.foot_bodies or ()))
        object.__setattr__(self, "body_parts", dict(self.body_parts or {}))
        object.__setattr__(self, "assets", tuple(self.assets or ()))
        object.__setattr__(self, "action_faces", faces)
        object.__setattr__(self, "full_joint_order", full_order)

    # ------------------------------------------------------------------ #
    # read-only compatibility properties (derived data; no backend fields)
    # ------------------------------------------------------------------ #
    @property
    def joint_order(self) -> list[str]:
        """The ``body`` joint set (Isaac Lab / policy order)."""
        return list(self.joint_sets.get("body", ()))

    @property
    def full_joint_names(self) -> list[str]:
        """Every declared joint in set-declaration order (or explicit order)."""
        if self.full_joint_order is not None:
            return list(self.full_joint_order)
        union: list[str] = []
        for joints in self.joint_sets.values():
            for joint in joints:
                if joint not in union:
                    union.append(joint)
        return union

    @property
    def num_joints(self) -> int:
        return len(self.joint_order)
