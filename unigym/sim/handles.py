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

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from unigym._typing import JsonValue
from unigym.robots.base import CanonicalBodyId, RobotProfile, normalize_canonical_body_id

from .control import ControlSpec
from .errors import ForeignHandleError, SimulationConfigError

if TYPE_CHECKING:
    from unigym.packs.packset import RobotRef
    from unigym.runtime.actions import OrderedActionGroup

__all__ = [
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
    """One declared articulation and its resolved ordered action groups.

    Attributes:
        name: the articulation label used in the run config.
        ref: the pack robot reference, or ``None`` for a config-mode handle.
        profile: the neutral, backend-independent profile.
        control: the explicit control preset, or ``None`` in config mode.
        action_groups: read-only group name -> :class:`OrderedActionGroup`
            mapping; group identity is the enclosing articulation plus the key.
        _owner: the builder/simulation identity token (never a value).
    """

    name: str
    ref: RobotRef | None
    profile: RobotProfile
    control: ControlSpec | None
    action_groups: Mapping[str, OrderedActionGroup]
    _owner: object = field(repr=False)

    def __post_init__(self) -> None:
        from unigym.runtime.actions import OrderedActionGroup

        if not self.name or not self.name.strip():
            raise SimulationConfigError(
                f"articulation handle name must be non-empty, got {self.name!r}"
            )
        if not isinstance(self.action_groups, Mapping):
            raise SimulationConfigError(
                f"articulation {self.name!r} action_groups must be a mapping, "
                f"got {type(self.action_groups).__name__}"
            )
        groups: dict[str, OrderedActionGroup] = {}
        for name, group in self.action_groups.items():
            group_name = str(name)
            if not group_name:
                raise SimulationConfigError("articulation action group names must be non-empty")
            if not isinstance(group, OrderedActionGroup):
                raise SimulationConfigError(
                    f"articulation {self.name!r} action group {group_name!r} must be "
                    f"an OrderedActionGroup, got {type(group).__name__}"
                )
            if group.name != group_name:
                raise SimulationConfigError(
                    f"articulation {self.name!r} action group key {group_name!r} "
                    f"does not match its group name {group.name!r}"
                )
            if group.articulation != self.name:
                raise SimulationConfigError(
                    f"articulation {self.name!r} owns action group {group_name!r} "
                    f"declared for articulation {group.articulation!r}"
                )
            groups[group_name] = group
        object.__setattr__(self, "action_groups", MappingProxyType(groups))

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
