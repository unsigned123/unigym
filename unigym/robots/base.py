"""Backend-neutral robot profile data.

A :class:`RobotProfile` is authoring data only: named joint sets, the
standby pose, the canonical base/feet/body-set facts, the per-format body
bindings, declared assets and named extra action sets.  It carries no backend
object and no backend switch; a backend that needs an engine-specific
realization builds it from this profile at its own boundary
(``BackendInfo.robot_realizer``).

The profile supplies the robot facts consumed by controls, observations and
the mujoco-family backends.  The isaaclab backends consume it through their
realizer, which owns actuator tables, prim paths and spawn conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping, NewType

if TYPE_CHECKING:
    from unigym.actuators.ir import ActuationGroup

from unigym._typing import JsonValue
from unigym.assets.asset_ref import ASSET_FORMATS, AssetRef, AssetRefError

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


def _freeze_asset_option(value: object, where: str) -> object:
    """Validate one JSON option value and return a defensive copy.

    Nested mappings become read-only ``MappingProxyType`` copies at every
    level.  JSON lists are copied and their items validated but stay lists:
    the value must remain dumpable as JSON/YAML, so sequence immutability is
    by documentation only.  Anything that is not a JSON mapping/list/scalar is
    rejected instead of stored as opaque metadata.
    """
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} keys must be strings, got {key!r}")
            frozen[key] = _freeze_asset_option(item, f"{where}[{key!r}]")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return [_freeze_asset_option(item, f"{where}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(
        f"{where} must be JSON-compatible (mapping, list or scalar), " f"got {type(value).__name__}"
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
            as ``legs`` / ``waist`` / ``arms`` and explicit named sets such as
            ``hand`` / ``right_hand``) are selected by name.
        default_joint_pos: joint name -> default angle (rad), i.e. the
            policy standby pose (must match the policy's q0 by name).
        assets: the neutral :class:`AssetRef` (or ``None`` when the asset is
            owned by a scene/pack resource).
        asset_backends: per-format backend allow-list
            (``{format: (backend names)}``).  An absent format means no
            restriction beyond the backend's declared format support; an
            empty mapping keeps the historical backend-only gate.  Every key
            must name a non-empty resource in :attr:`assets`.
        asset_options: opaque per-format import-boundary metadata
            (``{format: mapping}``).  Values must be JSON-compatible; nested
            mappings are stored as read-only copies, while JSON lists are
            copied but stay lists (treat sequences as read-only too; they stay
            mutable so a profile can be dumped back to JSON/YAML).  Core
            validates and stores it but does not interpret it in this slice; a
            realizer may expose it to its own import boundary through
            :func:`unigym.robots.assets.asset_options_for`.
        extra_action_sets: neutral named action sets beyond the primary body
            group (``{set_name: (joint names in term input order)}``); this
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
        actuation: the neutral :class:`~unigym.actuators.ir.ActuationGroup`
            records that physically drive this robot's joints.  Empty is
            legal (a passive robot, or a run that declares explicit
            actuation); a non-empty tuple must address declared joints and
            cover every non-passive declared joint exactly once.

    Read-only properties (derived from the data above):

        joint_order: the ``body`` set as a list.
        full_joint_names: :attr:`full_joint_order` or the union of every
            declared set.  Published by the observation schema joint keys.
    """

    name: str
    joint_sets: Mapping[str, tuple[str, ...]]
    default_joint_pos: Mapping[str, float]
    assets: AssetRef | None
    extra_action_sets: Mapping[str, tuple[str, ...]]
    base_body: CanonicalBodyId | None
    foot_bodies: tuple[CanonicalBodyId, ...]
    body_sets: Mapping[str, tuple[CanonicalBodyId, ...]]
    body_bindings: Mapping[CanonicalBodyId, BodyBinding]
    asset_backends: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    asset_options: Mapping[str, Mapping[str, JsonValue]] = field(default_factory=dict)
    base_height: float | None = None
    full_joint_order: tuple[str, ...] | None = None
    actuation: tuple[ActuationGroup, ...] = ()

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
        extra_sets: dict[str, tuple[str, ...]] = {}
        for set_name, joints in dict(self.extra_action_sets or {}).items():
            names = tuple(str(joint) for joint in (joints or ()))
            if not names:
                raise ValueError(f"RobotProfile extra action set {set_name!r} is empty")
            if len(set(names)) != len(names):
                raise ValueError(
                    f"RobotProfile extra action set {set_name!r} contains duplicate names"
                )
            extra_sets[str(set_name)] = names
        full_order = self.full_joint_order
        if full_order is not None:
            full_order = tuple(str(joint) for joint in full_order)
            if not full_order:
                raise ValueError("RobotProfile full_joint_order must not be empty")
            if len(set(full_order)) != len(full_order):
                raise ValueError("RobotProfile full_joint_order contains duplicate names")
        universe: list[str] = list(full_order or ())
        if not universe:
            for joints in sets.values():
                for joint in joints:
                    if joint not in universe:
                        universe.append(joint)

        raw_actuation = self.actuation or ()
        actuation: tuple[ActuationGroup, ...] = ()
        if raw_actuation:
            from unigym.actuators.ir import (
                ActuationError,
                normalize_actuation_groups,
                validate_actuation_groups,
            )

            try:
                actuation = normalize_actuation_groups(raw_actuation)
                validate_actuation_groups(
                    actuation,
                    joints=tuple(universe),
                    joint_sets=sets,
                )
            except ActuationError as exc:
                raise ValueError(f"RobotProfile actuation: {exc}") from exc

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

        if self.asset_backends is not None and not isinstance(self.asset_backends, Mapping):
            raise ValueError(
                f"RobotProfile asset_backends must be a mapping, "
                f"got {type(self.asset_backends).__name__}"
            )
        raw_backends = dict(self.asset_backends or {})
        unknown_backends = sorted(
            str(asset_format) for asset_format in raw_backends if asset_format not in ASSET_FORMATS
        )
        if unknown_backends:
            raise ValueError(
                f"RobotProfile asset_backends declares unknown format(s) "
                f"{unknown_backends}; allowed formats are {list(ASSET_FORMATS)}"
            )
        asset_backends: dict[str, tuple[str, ...]] = {}
        for asset_format, backends in raw_backends.items():
            key = str(asset_format)
            if assets is None or not getattr(assets, key, None):
                raise ValueError(
                    f"RobotProfile asset_backends[{key!r}] requires a declared " f"{key!r} asset"
                )
            if isinstance(backends, (str, bytes)) or not isinstance(backends, Iterable):
                raise ValueError(
                    f"RobotProfile asset_backends[{key!r}] must be a sequence of "
                    f"backend names, got {backends!r}"
                )
            raw_names = tuple(backends)
            if not raw_names:
                raise ValueError(f"RobotProfile asset_backends[{key!r}] must not be empty")
            if any(not isinstance(name, str) or not name.strip() for name in raw_names):
                raise ValueError(
                    f"RobotProfile asset_backends[{key!r}] backend names must be "
                    f"non-empty strings, got {raw_names!r}"
                )
            names = tuple(str(name) for name in raw_names)
            if len(set(names)) != len(names):
                raise ValueError(
                    f"RobotProfile asset_backends[{key!r}] backend names must be "
                    f"unique, got {names!r}"
                )
            asset_backends[key] = names

        if self.asset_options is not None and not isinstance(self.asset_options, Mapping):
            raise ValueError(
                f"RobotProfile asset_options must be a mapping, "
                f"got {type(self.asset_options).__name__}"
            )
        raw_options = dict(self.asset_options or {})
        unknown_options = sorted(
            str(asset_format) for asset_format in raw_options if asset_format not in ASSET_FORMATS
        )
        if unknown_options:
            raise ValueError(
                f"RobotProfile asset_options declares unknown format(s) "
                f"{unknown_options}; allowed formats are {list(ASSET_FORMATS)}"
            )
        asset_options: dict[str, Mapping[str, JsonValue]] = {}
        for asset_format, options in raw_options.items():
            key = str(asset_format)
            if assets is None or not getattr(assets, key, None):
                raise ValueError(
                    f"RobotProfile asset_options[{key!r}] requires a declared " f"{key!r} asset"
                )
            if not isinstance(options, Mapping):
                raise ValueError(
                    f"RobotProfile asset_options[{key!r}] must be a mapping, " f"got {options!r}"
                )
            copied: dict[str, object] = {}
            for name, value in options.items():
                if not isinstance(name, str):
                    raise ValueError(
                        f"RobotProfile asset_options[{key!r}] keys must be "
                        f"strings, got {name!r}"
                    )
                copied[name] = _freeze_asset_option(
                    value, f"RobotProfile asset_options[{key!r}][{name!r}]"
                )
            asset_options[key] = MappingProxyType(copied)

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
        object.__setattr__(self, "extra_action_sets", extra_sets)
        object.__setattr__(self, "base_body", base_body)
        object.__setattr__(self, "foot_bodies", foot_bodies)
        object.__setattr__(self, "body_sets", body_sets)
        object.__setattr__(self, "body_bindings", bindings)
        object.__setattr__(self, "asset_backends", asset_backends)
        object.__setattr__(self, "asset_options", asset_options)
        object.__setattr__(self, "base_height", base_height)
        object.__setattr__(self, "full_joint_order", full_order)
        object.__setattr__(self, "actuation", actuation)

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
