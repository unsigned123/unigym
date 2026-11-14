"""Explicit control presets for the experimental simulation facade.

A :class:`ControlSpec` is an immutable declaration of one or more control
faces.  Every face keeps both halves of the existing control contract next to
each other: the action face (``name`` / ``joints`` / ``q0``) and the actuation
gains (``kp`` / ``kd`` plus optional limits).  The builder lowers the faces to
the existing ``controls[]`` and ``articulations[].actuation[]`` tables; this
module never imports a backend or an engine.

Only :meth:`Control.position_pd` is exposed in the first spike.  Its drive is
``position_pd`` and its target is the existing ``position`` kind, so the
``DRIVE_TARGET`` table stays owned by :mod:`unigym.runtime.controls`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from unigym.runtime.controls import DRIVE_TARGET

from .errors import SimulationConfigError

__all__ = ["Control", "ControlFaceSpec", "ControlSpec"]

_GAIN_NAMES = frozenset({"kp", "kd"})


def _finite(value: object, what: str) -> float:
    """Return one finite real scalar or fail with field and value named."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationConfigError(f"{what} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SimulationConfigError(f"{what} must be finite, got {value!r}")
    return number


def _non_empty_name(value: object, what: str) -> str:
    """Validate one non-empty, non-whitespace string."""
    if not isinstance(value, str) or not value.strip():
        raise SimulationConfigError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _gain(value: object, what: str) -> float | Mapping[str, float]:
    """Normalize one scalar-or-name-mapped gain (non-empty map, finite values)."""
    if isinstance(value, Mapping):
        if not value:
            raise SimulationConfigError(f"{what} must not be an empty mapping")
        gains: dict[str, float] = {}
        for name, raw in value.items():
            joint = _non_empty_name(name, f"{what} key")
            gains[joint] = _finite(raw, f"{what}[{joint!r}]")
        return MappingProxyType(gains)
    return _finite(value, what)


def _limit(value: object, what: str) -> float | Mapping[str, float] | None:
    """Normalize an optional scalar-or-name-mapped limit."""
    if value is None:
        return None
    return _gain(value, what)


def _joint_names(value: object, what: str) -> tuple[str, ...]:
    """Normalize explicit joint names (non-empty, duplicate-free)."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SimulationConfigError(f"{what} must be a sequence of joint names, got {value!r}")
    names = tuple(value)
    if not names:
        raise SimulationConfigError(f"{what} must not be empty")
    for name in names:
        _non_empty_name(name, f"{what} entry")
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise SimulationConfigError(f"{what} contains duplicate joint names {duplicates}")
    return names


@dataclass(frozen=True)
class ControlFaceSpec:
    """One immutable control face: action joints plus explicit actuation gains.

    Attributes:
        name: the action face and ``NamedAction`` key.
        drive: one of :data:`unigym.runtime.controls.CONTROL_DRIVES`.
        target: the action kind; must equal ``DRIVE_TARGET[drive]``.
        joint_set: a profile joint-set name, or ``None``.
        joints: explicit joint names, or ``None``.  Exactly one of
            ``joint_set`` / ``joints`` is set.
        q0: explicit per-joint standby overrides; unspecified joints use the
            profile defaults at articulation declaration.
        gains: exactly ``kp`` and ``kd``, each a finite scalar or a non-empty
            finite per-joint mapping.
        effort_limit: optional finite scalar or per-joint mapping.
        velocity_limit: optional finite scalar or per-joint mapping.
    """

    name: str
    drive: str
    target: str
    joint_set: str | None
    joints: tuple[str, ...] | None
    q0: Mapping[str, float]
    gains: Mapping[str, float | Mapping[str, float]]
    effort_limit: float | Mapping[str, float] | None = None
    velocity_limit: float | Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        name = _non_empty_name(self.name, "control face name")
        if self.drive not in DRIVE_TARGET:
            raise SimulationConfigError(
                f"control face {name!r} drive must be one of {sorted(DRIVE_TARGET)}, "
                f"got {self.drive!r}"
            )
        expected_target = DRIVE_TARGET[self.drive]
        if self.target != expected_target:
            raise SimulationConfigError(
                f"control face {name!r} declares drive {self.drive!r} with target "
                f"{self.target!r}; drive {self.drive!r} requires target "
                f"{expected_target!r}"
            )

        joint_set = self.joint_set
        if joint_set is not None:
            joint_set = _non_empty_name(joint_set, f"control face {name!r} joint_set")
        joints = self.joints
        if joints is not None:
            joints = _joint_names(joints, f"control face {name!r} joints")
        if (joint_set is None) == (joints is None):
            raise SimulationConfigError(
                f"control face {name!r} must declare exactly one of joint_set or "
                f"non-empty joints (got joint_set={joint_set!r}, joints={joints!r})"
            )

        q0: dict[str, float] = {}
        if not isinstance(self.q0, Mapping):
            raise SimulationConfigError(
                f"control face {name!r} q0 must be a mapping, got {self.q0!r}"
            )
        for joint, raw in self.q0.items():
            key = _non_empty_name(joint, f"control face {name!r} q0 key")
            q0[key] = _finite(raw, f"control face {name!r} q0[{key!r}]")

        if not isinstance(self.gains, Mapping):
            raise SimulationConfigError(
                f"control face {name!r} gains must be a mapping, got {self.gains!r}"
            )
        unknown = sorted(set(self.gains) - _GAIN_NAMES)
        missing = sorted(_GAIN_NAMES - set(self.gains))
        if unknown or missing:
            raise SimulationConfigError(
                f"control face {name!r} gains must be exactly {sorted(_GAIN_NAMES)}; "
                f"missing {missing}, unknown {unknown}"
            )
        gains = {
            key: _gain(self.gains[key], f"control face {name!r} gains[{key!r}]")
            for key in ("kp", "kd")
        }

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "joint_set", joint_set)
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "q0", MappingProxyType(q0))
        object.__setattr__(self, "gains", MappingProxyType(gains))
        object.__setattr__(
            self, "effort_limit", _limit(self.effort_limit, f"control face {name!r} effort_limit")
        )
        object.__setattr__(
            self,
            "velocity_limit",
            _limit(self.velocity_limit, f"control face {name!r} velocity_limit"),
        )


@dataclass(frozen=True)
class ControlSpec:
    """A read-only mapping of face name -> :class:`ControlFaceSpec`."""

    faces: Mapping[str, ControlFaceSpec]

    def __post_init__(self) -> None:
        if not isinstance(self.faces, Mapping):
            raise SimulationConfigError(
                f"ControlSpec.faces must be a mapping, got {type(self.faces).__name__}"
            )
        faces: dict[str, ControlFaceSpec] = {}
        for name, face in self.faces.items():
            if not isinstance(face, ControlFaceSpec):
                raise SimulationConfigError(
                    f"ControlSpec face {name!r} must be a ControlFaceSpec, "
                    f"got {type(face).__name__}"
                )
            if name != face.name:
                raise SimulationConfigError(
                    f"ControlSpec face key {name!r} does not match its spec name " f"{face.name!r}"
                )
            faces[name] = face
        if not faces:
            raise SimulationConfigError("ControlSpec must declare at least one face")
        object.__setattr__(self, "faces", MappingProxyType(faces))

    @classmethod
    def from_faces(cls, faces: Sequence[ControlFaceSpec]) -> ControlSpec:
        """Build a spec from an ordered face sequence, rejecting duplicate names."""
        mapping: dict[str, ControlFaceSpec] = {}
        for face in faces:
            if not isinstance(face, ControlFaceSpec):
                raise SimulationConfigError(
                    f"ControlSpec.from_faces expects ControlFaceSpec entries, "
                    f"got {type(face).__name__}"
                )
            if face.name in mapping:
                raise SimulationConfigError(
                    f"ControlSpec declares duplicate face name {face.name!r}"
                )
            mapping[face.name] = face
        return cls(mapping)


class Control:
    """Factories for the canonical control presets of the facade."""

    @staticmethod
    def position_pd(
        *,
        joint_set: str | None = None,
        joints: Sequence[str] | None = None,
        face: str | None = None,
        q0: Mapping[str, float] | None = None,
        kp: float | Mapping[str, float],
        kd: float | Mapping[str, float],
        effort_limit: float | Mapping[str, float] | None = None,
        velocity_limit: float | Mapping[str, float] | None = None,
    ) -> ControlSpec:
        """Declare one ``position_pd`` control face.

        Exactly one of ``joint_set`` / ``joints`` is required.  ``face``
        defaults to ``joint_set``; an explicit-joint face must name itself.
        ``kp`` and ``kd`` are required and may be scalars or per-joint maps.
        """
        if (joint_set is None) == (joints is None):
            raise SimulationConfigError(
                "Control.position_pd requires exactly one of joint_set or joints "
                f"(got joint_set={joint_set!r}, joints={joints!r})"
            )
        if joint_set is not None:
            face_name = joint_set if face is None else _non_empty_name(face, "control face name")
        else:
            if face is None:
                raise SimulationConfigError(
                    "Control.position_pd(joints=...) requires an explicit face= name"
                )
            face_name = _non_empty_name(face, "control face name")
        resolved_joint_set = (
            None if joint_set is None else _non_empty_name(joint_set, "control joint_set")
        )
        resolved_joints = None if joints is None else _joint_names(joints, "control joints")
        if q0 is not None and not isinstance(q0, Mapping):
            raise SimulationConfigError(
                f"control face {face_name!r} q0 must be a mapping, got {q0!r}"
            )

        gains = {
            "kp": _gain(kp, f"control face {face_name!r} gains['kp']"),
            "kd": _gain(kd, f"control face {face_name!r} gains['kd']"),
        }
        return ControlSpec.from_faces(
            (
                ControlFaceSpec(
                    name=face_name,
                    drive="position_pd",
                    target="position",
                    joint_set=resolved_joint_set,
                    joints=resolved_joints,
                    q0=MappingProxyType(dict(q0) if q0 is not None else {}),
                    gains=MappingProxyType(gains),
                    effort_limit=effort_limit,
                    velocity_limit=velocity_limit,
                ),
            )
        )
