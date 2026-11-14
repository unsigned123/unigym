"""Identity-scoped declaration handles of the experimental simulation facade.

A handle is minted by one builder (or, in later slices, one simulation) and is
valid only for that owner.  Equality and hashing are object identity so handles
can key the typed observation mappings without a name lookup: two declarations
that happen to share a label remain distinct.

The handles are immutable.  Their mapping fields are stored as
:class:`~types.MappingProxyType`; the owned :class:`~unigym.robots.base.RobotProfile`
is the neutral profile produced by the pack/profile loader and is never
mutated here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from unigym._typing import JsonValue
from unigym.robots.base import CanonicalBodyId, RobotProfile, normalize_canonical_body_id

from .control import ControlSpec
from .errors import BindingError, ForeignHandleError, SimulationConfigError

if TYPE_CHECKING:
    from unigym.packs.packset import RobotRef

__all__ = [
    "ActionBinding",
    "ArticulationHandle",
    "BodyHandle",
    "CameraHandle",
]


def _check_owner(handle: BodyHandle, owner: object, what: str) -> None:
    """Reject a handle minted by a different builder/simulation."""
    if handle._owner is not owner:
        raise ForeignHandleError(
            f"{what} belongs to another builder/simulation; handles are scoped "
            f"to the owner that minted them"
        )


@dataclass(frozen=True, eq=False)
class ArticulationHandle:
    """One declared articulation and its resolved action faces.

    Attributes:
        name: the articulation label used in the run config.
        ref: the pack robot reference, or ``None`` for a config-mode handle.
        profile: the neutral, backend-independent profile.
        control: the explicit control preset, or ``None`` in config mode.
        faces: read-only face name -> resolved joint names mapping.
        _owner: the builder/simulation identity token (never a value).
    """

    name: str
    ref: RobotRef | None
    profile: RobotProfile
    control: ControlSpec | None
    faces: Mapping[str, tuple[str, ...]]
    _owner: object = field(repr=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise SimulationConfigError(
                f"articulation handle name must be non-empty, got {self.name!r}"
            )
        faces: dict[str, tuple[str, ...]] = {}
        for name, joints in self.faces.items():
            face_name = str(name)
            if not face_name:
                raise SimulationConfigError("articulation face names must be non-empty")
            face_joints = tuple(str(joint) for joint in joints)
            if not face_joints:
                raise SimulationConfigError(f"articulation face {face_name!r} has no joints")
            faces[face_name] = face_joints
        object.__setattr__(self, "faces", MappingProxyType(faces))

    def body(self, canonical_id: str) -> BodyHandle:
        """Resolve one canonical body id against the profile's declared vocabulary."""
        normalized = normalize_canonical_body_id(
            canonical_id, f"articulation {self.name!r} body id"
        )
        vocabulary = self._body_vocabulary()
        if normalized not in vocabulary:
            raise SimulationConfigError(
                f"articulation {self.name!r} declares no body id {normalized!r}; "
                f"declared body ids: {sorted(vocabulary)}"
            )
        return BodyHandle(articulation=self, canonical_id=normalized, _owner=self._owner)

    def bind_action(
        self,
        face: str,
        *,
        kind: str,
        order: Sequence[str] | None = None,
    ) -> ActionBinding:
        """Bind one face to an explicit action kind and column order."""
        from unigym.runtime.actions import LEGAL_ACTION_KINDS

        if face not in self.faces:
            raise BindingError(
                f"articulation {self.name!r} has no action face {face!r}; "
                f"declared faces: {sorted(self.faces)}"
            )
        if kind not in LEGAL_ACTION_KINDS:
            raise BindingError(
                f"action binding kind must be one of {list(LEGAL_ACTION_KINDS)}, " f"got {kind!r}"
            )
        face_joints = self.faces[face]
        if order is None:
            joint_names = face_joints
        else:
            if isinstance(order, (str, bytes)):
                raise BindingError(
                    f"action binding order must be a sequence of joint names, got {order!r}"
                )
            joint_names = tuple(order)
            if len(joint_names) != len(face_joints) or set(joint_names) != set(face_joints):
                raise BindingError(
                    f"action binding order for face {face!r} must be a permutation of "
                    f"{list(face_joints)}; got {list(joint_names)}"
                )
        return ActionBinding(
            articulation=self,
            face=face,
            joint_names=joint_names,
            kind=kind,
            _owner=self._owner,
        )

    def _body_vocabulary(self) -> frozenset[str]:
        """The profile's declared canonical body ids (base, feet, sets, bindings)."""
        profile = self.profile
        ids: set[str] = set()
        if profile.base_body is not None:
            ids.add(str(profile.base_body))
        for foot in profile.foot_bodies:
            ids.add(str(foot))
        for members in profile.body_sets.values():
            for member in members:
                ids.add(str(member))
        for canonical in profile.body_bindings:
            ids.add(str(canonical))
        return frozenset(ids)


@dataclass(frozen=True, eq=False)
class BodyHandle:
    """One canonical body id owned by an :class:`ArticulationHandle`."""

    articulation: ArticulationHandle
    canonical_id: CanonicalBodyId
    _owner: object = field(repr=False)

    def __post_init__(self) -> None:
        normalized = normalize_canonical_body_id(self.canonical_id, "body handle canonical id")
        object.__setattr__(self, "canonical_id", normalized)


@dataclass(frozen=True, eq=False)
class CameraHandle:
    """One declared camera entry (the existing ``cameras[]`` schema)."""

    name: str
    entry: Mapping[str, JsonValue]
    _owner: object = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SimulationConfigError(f"camera handle name must be non-empty, got {self.name!r}")
        if not isinstance(self.entry, Mapping):
            raise SimulationConfigError(
                f"camera handle {self.name!r} entry must be a mapping, "
                f"got {type(self.entry).__name__}"
            )
        object.__setattr__(self, "entry", MappingProxyType(dict(self.entry)))


@dataclass(frozen=True)
class ActionBinding:
    """An action face lowered for ``NamedAction``: columns plus an explicit kind."""

    articulation: ArticulationHandle
    face: str
    joint_names: tuple[str, ...]
    kind: str
    _owner: object = field(repr=False)

    @property
    def dim(self) -> int:
        """Per-env action dimension (== number of bound joint columns)."""
        return len(self.joint_names)
