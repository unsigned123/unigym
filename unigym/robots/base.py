"""Backend-neutral robot profile data.

A :class:`RobotProfile` is authoring data only: named joint sets, the
standby pose, the canonical base/feet/body-set facts, the per-format body
bindings, declared assets and named action faces.  It carries no backend
object and no backend switch; a backend that needs an engine-specific
realization builds it from this profile at its own boundary
(``BackendInfo.robot_realizer``).

The profile supplies the robot facts consumed by controls, observations and
the mujoco-family backends.  The isaaclab backends consume it through their
realizer, which owns actuator tables, prim paths and spawn conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, NewType

from unigym.assets.asset_ref import AssetRef, AssetRefError

__all__ = [
    "BodyBinding",
    "BodyBindingError",
    "CanonicalBodyId",
    "RobotProfile",
    "normalize_canonical_body_id",
    "validate_body_bindings",
]

CanonicalBodyId = NewType("CanonicalBodyId", str)
"""A backend-neutral semantic body id (runtime value is ``str``)."""


class BodyBindingError(ValueError):
    """A malformed canonical body id or body binding (fail-fast, no alias)."""


def normalize_canonical_body_id(value: object, what: str) -> CanonicalBodyId:
    """Validate and normalize one canonical body id.

    Canonical ids are semantic names, never backend paths: non-empty, with no
    surrounding or internal whitespace, no path separators, no wildcard and
    no empty dot-separated segment.
    """
    if not isinstance(value, str) or not value:
        raise BodyBindingError(f"{what} must be a non-empty string, got {value!r}")
    if value != value.strip():
        raise BodyBindingError(f"{what} must not have leading/trailing whitespace, got {value!r}")
    if any(character.isspace() for character in value):
        raise BodyBindingError(f"{what} must not contain whitespace, got {value!r}")
    if "/" in value or "\\" in value:
        raise BodyBindingError(
            f"{what} must be a semantic body id without path separators, got {value!r}"
        )
    if "*" in value:
        raise BodyBindingError(f"{what} must not contain the wildcard '*', got {value!r}")
    segments = value.split(".")
    if any(not segment for segment in segments):
        raise BodyBindingError(f"{what} must have non-empty dot-separated segments, got {value!r}")
    return CanonicalBodyId(value)


@dataclass(frozen=True)
class BodyBinding:
    """One canonical body id bound to per-format native names or prim paths.

    Attributes:
        canonical: the backend-neutral semantic body id.
        names: asset format -> native name / prim path.  The format
            vocabulary is owned by :mod:`unigym.assets.asset_ref`; this
            record only requires a non-empty string mapping.
    """

    canonical: CanonicalBodyId
    names: Mapping[str, str]

    def __post_init__(self) -> None:
        canonical = normalize_canonical_body_id(self.canonical, "BodyBinding canonical")
        object.__setattr__(self, "canonical", canonical)
        if not isinstance(self.names, Mapping) or not self.names:
            raise BodyBindingError(
                f"BodyBinding {canonical!r} names must be a non-empty mapping of "
                f"format -> native name, got {self.names!r}"
            )
        names: dict[str, str] = {}
        for asset_format, native_name in self.names.items():
            if not isinstance(asset_format, str) or not asset_format.strip():
                raise BodyBindingError(
                    f"BodyBinding {canonical!r} format must be a non-empty string, "
                    f"got {asset_format!r}"
                )
            if asset_format != asset_format.strip():
                raise BodyBindingError(
                    f"BodyBinding {canonical!r} format {asset_format!r} must not "
                    f"have leading/trailing whitespace"
                )
            if not isinstance(native_name, str) or not native_name.strip():
                raise BodyBindingError(
                    f"BodyBinding {canonical!r} native name for format "
                    f"{asset_format!r} must be a non-empty string, got {native_name!r}"
                )
            if native_name != native_name.strip():
                raise BodyBindingError(
                    f"BodyBinding {canonical!r} native name for format "
                    f"{asset_format!r} must not have leading/trailing whitespace, "
                    f"got {native_name!r}"
                )
            names[asset_format] = native_name
        object.__setattr__(self, "names", names)


def validate_body_bindings(
    bindings: Mapping[CanonicalBodyId, BodyBinding],
    *,
    required_ids: Iterable[CanonicalBodyId] = (),
    formats: Iterable[str] | None = None,
) -> None:
    """Validate one canonical-id -> binding map against required ids/formats.

    Every mapping key must equal its binding's ``canonical``.  Duplicate
    native names are rejected per format only.  With ``formats=None`` the
    duplicate scan covers the formats present; with explicit ``formats`` any
    unknown format key is rejected and every required id must carry every
    listed format.
    """
    if not isinstance(bindings, Mapping):
        raise BodyBindingError(f"body bindings must be a mapping, got {bindings!r}")
    records: dict[CanonicalBodyId, BodyBinding] = {}
    for key, binding in bindings.items():
        if not isinstance(binding, BodyBinding):
            raise BodyBindingError(f"body bindings[{key!r}] must be a BodyBinding, got {binding!r}")
        if key != binding.canonical:
            raise BodyBindingError(
                f"body bindings key {key!r} does not match its record canonical "
                f"{binding.canonical!r}"
            )
        records[binding.canonical] = binding

    required = tuple(
        normalize_canonical_body_id(required_id, "required body id") for required_id in required_ids
    )
    missing_ids = sorted(str(required_id) for required_id in required if required_id not in records)
    if missing_ids:
        raise BodyBindingError(f"body bindings are missing required ids {missing_ids}")

    if formats is None:
        allowed_formats: tuple[str, ...] | None = None
    else:
        allowed_formats = tuple(str(asset_format) for asset_format in formats)
        allowed = set(allowed_formats)
        for binding in records.values():
            unknown = sorted(set(binding.names) - allowed)
            if unknown:
                raise BodyBindingError(
                    f"BodyBinding {binding.canonical!r} declares unknown formats "
                    f"{unknown}; allowed formats are {list(allowed_formats)}"
                )

    seen: dict[str, dict[str, CanonicalBodyId]] = {}
    for binding in records.values():
        for asset_format, native_name in binding.names.items():
            if allowed_formats is not None and asset_format not in allowed_formats:
                continue
            owners = seen.setdefault(asset_format, {})
            previous = owners.get(native_name)
            if previous is not None and previous != binding.canonical:
                raise BodyBindingError(
                    f"duplicate {asset_format} native name {native_name!r} for bodies "
                    f"{previous!r} and {binding.canonical!r}"
                )
            owners[native_name] = binding.canonical

    if allowed_formats is not None:
        missing_formats = []
        for required_id in required:
            binding = records[required_id]
            missing = [fmt for fmt in allowed_formats if fmt not in binding.names]
            if missing:
                missing_formats.append((required_id, missing))
        if missing_formats:
            details = ", ".join(
                f"{required_id!r} missing {missing}" for required_id, missing in missing_formats
            )
            raise BodyBindingError(
                f"body bindings lack required formats for ids: {details}; "
                f"declared formats are {list(allowed_formats)}"
            )


def _canonical_ids(value: object, what: str) -> tuple[CanonicalBodyId, ...]:
    """Normalize a sequence of canonical body ids."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{what} must be a sequence of body ids, got {value!r}")
    return tuple(
        normalize_canonical_body_id(entry, f"{what}[{index}]") for index, entry in enumerate(value)
    )


@dataclass(frozen=True)
class RobotProfile:
    """The backend-neutral authoring profile of one robot.

    Attributes:
        name: the profile kind (the registered robot factory name, e.g.
            ``g1``); an articulation may be renamed without changing it.
        joint_sets: the named joint vocabulary
            (``{set_name: (joint names in action order)}``).  ``body`` is
            the primary policy/body order; extra sets (curated groups such
            as ``legs`` / ``waist`` / ``arms`` and explicit faces such as
            ``hand`` / ``right_hand``) are selected by name.
        default_joint_pos: joint name -> default angle (rad), i.e. the
            policy standby pose (must match the policy's q0 by name).
        assets: the neutral :class:`AssetRef` (or ``None`` when the asset is
            owned by a scene/pack resource).
        action_faces: neutral named action surfaces beyond the primary body
            face (``{face_name: (joint names in term input order)}``); this
            is authoring data, not a backend capability.
        base_body: the canonical id of the robot's base link (``None`` when
            the robot declares none).
        foot_bodies: semantic foot body ids (never a sensor declaration).
        body_sets: named canonical body groups (``grasp.left`` ...).
        body_bindings: canonical id -> per-format native name/path metadata.
        base_height: the free-base spawn height in metres (``None`` when the
            robot declares none); the mujoco family and the isaaclab G1
            realizer consume it as their default spawn height.
        full_joint_order: optional explicit full-articulation joint order
            (document order) used by :attr:`full_joint_names`; ``None``
            derives the set union in declaration order.

    Read-only properties (derived from the data above):

        joint_order: the ``body`` set as a list.
        full_joint_names: :attr:`full_joint_order` or the union of every
            declared set.  Published by the ObsCatalog joint keys.
    """

    name: str
    joint_sets: Mapping[str, tuple[str, ...]]
    default_joint_pos: Mapping[str, float]
    assets: AssetRef | None
    action_faces: Mapping[str, tuple[str, ...]]
    base_body: CanonicalBodyId | None
    foot_bodies: tuple[CanonicalBodyId, ...]
    body_sets: Mapping[str, tuple[CanonicalBodyId, ...]]
    body_bindings: Mapping[CanonicalBodyId, BodyBinding]
    base_height: float | None = None
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

        base_height = self.base_height
        if base_height is not None:
            if isinstance(base_height, bool) or not isinstance(base_height, (int, float)):
                raise ValueError(
                    f"RobotProfile base_height must be a finite number or None, "
                    f"got {base_height!r}"
                )
            base_height = float(base_height)
            if not math.isfinite(base_height):
                raise ValueError(f"RobotProfile base_height must be finite, got {base_height!r}")

        assets = self.assets
        if assets is not None:
            try:
                assets = AssetRef.from_mapping(assets, "RobotProfile assets")
            except AssetRefError as exc:
                raise ValueError(str(exc)) from exc
        base_body = self.base_body
        if base_body is not None:
            base_body = normalize_canonical_body_id(base_body, "RobotProfile base_body")
        foot_bodies = _canonical_ids(self.foot_bodies, "RobotProfile foot_bodies")
        body_sets: dict[str, tuple[CanonicalBodyId, ...]] = {}
        for set_name, bodies in dict(self.body_sets or {}).items():
            name = str(set_name)
            if not name:
                raise ValueError("RobotProfile body_sets names must be non-empty")
            members = _canonical_ids(bodies, f"RobotProfile body_sets[{name!r}]")
            if not members:
                raise ValueError(f"RobotProfile body set {name!r} is empty")
            if len(set(members)) != len(members):
                raise ValueError(f"RobotProfile body set {name!r} contains duplicate body ids")
            body_sets[name] = members
        bindings = dict(self.body_bindings or {})
        required = list(foot_bodies)
        if base_body is not None:
            required.insert(0, base_body)
        for members in body_sets.values():
            required.extend(members)
        validate_body_bindings(bindings, required_ids=required, formats=None)

        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "joint_sets", sets)
        default_pos = dict(self.default_joint_pos or {})
        object.__setattr__(
            self,
            "default_joint_pos",
            {str(joint): float(value) for joint, value in default_pos.items()},
        )
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "action_faces", faces)
        object.__setattr__(self, "base_body", base_body)
        object.__setattr__(self, "foot_bodies", foot_bodies)
        object.__setattr__(self, "body_sets", body_sets)
        object.__setattr__(self, "body_bindings", bindings)
        object.__setattr__(self, "base_height", base_height)
        object.__setattr__(self, "full_joint_order", full_order)

    # ------------------------------------------------------------------ #
    # read-only derived properties (no backend fields)
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
