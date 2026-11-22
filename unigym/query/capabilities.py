"""Runtime-query capability declarations.

Capabilities describe which query operations a backend can answer.  They are
part of the query contract, not the env capability bits: a query operation
never adds a ``BackendCapabilities`` field and never adds a per-operation
trait.  The single backend trait is ``Trait.RUNTIME_QUERY``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from .errors import QueryCapabilityError

__all__ = [
    "ActuatorCaps",
    "ContactCaps",
    "Feature",
    "GeometryCaps",
    "JacobianCaps",
    "ModelCaps",
    "QueryCapabilities",
    "ScratchCaps",
    "ScratchIsolation",
    "ScratchMode",
    "StateCaps",
    "Support",
]

Support = Literal["native", "partial", "synthetic", "unsupported"]
ScratchMode = Literal["copy", "transaction"]
ScratchIsolation = Literal["bit_exact", "restored"]

_SUPPORTS = frozenset({"native", "partial", "synthetic", "unsupported"})
_SCRATCH_MODES = frozenset({"copy", "transaction"})
_SCRATCH_ISOLATIONS = frozenset({"bit_exact", "restored"})
_GROUP_ORDER = (
    "model",
    "state",
    "scratch",
    "jacobian",
    "geometry",
    "contacts",
    "actuators",
)
_MEMBERS: dict[str, tuple[str, ...]] = {
    "model": ("ir", "body", "geom", "site", "joint", "actuator"),
    "state": (
        "body_pose",
        "site_pose",
        "com_pose",
        "root_pose",
        "joint_position",
        "joint_velocity",
        "root_state",
        "com_state",
        "object_state",
        "object_com_state",
    ),
    "scratch": (
        "copy",
        "live_untouched",
        "set_root_state",
        "set_joint_state",
        "forward",
        "body_pose",
        "site_pose",
        "com_pose",
        "geometry",
        "jacobian",
    ),
    "jacobian": ("body", "site", "com", "point", "object"),
    "geometry": ("distance", "aabb"),
    "contacts": ("manifold", "jacobian", "wrench_to_joint"),
    "actuators": ("facts", "state"),
}


@dataclass(frozen=True)
class Feature:
    """One operation's support level and an optional teaching detail."""

    support: Support
    detail: str = ""

    def __post_init__(self) -> None:
        """Reject an unknown support level at construction."""
        if self.support not in _SUPPORTS:
            raise ValueError(
                f"Feature.support must be one of {sorted(_SUPPORTS)}, got {self.support!r}"
            )


_NOT_DECLARED = Feature("unsupported", "not declared")


@dataclass(frozen=True)
class ModelCaps:
    """Static model-fact operations."""

    ir: Feature
    body: Feature
    geom: Feature
    site: Feature
    joint: Feature
    actuator: Feature


@dataclass(frozen=True)
class StateCaps:
    """Live state-read operations."""

    body_pose: Feature
    site_pose: Feature
    com_pose: Feature
    root_pose: Feature
    joint_position: Feature
    joint_velocity: Feature
    root_state: Feature = _NOT_DECLARED
    com_state: Feature = _NOT_DECLARED
    object_state: Feature = _NOT_DECLARED
    object_com_state: Feature = _NOT_DECLARED


@dataclass(frozen=True)
class ScratchCaps:
    """Per-env scratch-copy operations."""

    copy: Feature
    live_untouched: Feature
    set_root_state: Feature
    set_joint_state: Feature
    forward: Feature
    body_pose: Feature
    site_pose: Feature
    com_pose: Feature
    geometry: Feature
    jacobian: Feature
    mode: ScratchMode = "copy"
    isolation: ScratchIsolation = "bit_exact"

    def __post_init__(self) -> None:
        """Reject an unknown scratch mode or isolation level at construction."""
        if self.mode not in _SCRATCH_MODES:
            raise ValueError(
                f"ScratchCaps.mode must be one of {sorted(_SCRATCH_MODES)}, got {self.mode!r}"
            )
        if self.isolation not in _SCRATCH_ISOLATIONS:
            raise ValueError(
                f"ScratchCaps.isolation must be one of {sorted(_SCRATCH_ISOLATIONS)}, "
                f"got {self.isolation!r}"
            )


@dataclass(frozen=True)
class JacobianCaps:
    """Jacobian target kinds."""

    body: Feature
    site: Feature
    com: Feature
    point: Feature
    object: Feature = _NOT_DECLARED


@dataclass(frozen=True)
class GeometryCaps:
    """Geometry operations."""

    distance: Feature
    aabb: Feature


@dataclass(frozen=True)
class ContactCaps:
    """Group-based contact operations and the deferred Jacobian/wrench pair."""

    manifold: Feature
    jacobian: Feature
    wrench_to_joint: Feature


@dataclass(frozen=True)
class ActuatorCaps:
    """Actuator-fact operations (P0A declares them unsupported)."""

    facts: Feature
    state: Feature


Caps = (
    ModelCaps | StateCaps | ScratchCaps | JacobianCaps | GeometryCaps | ContactCaps | ActuatorCaps
)


def _feature_map(caps: Caps, group: str) -> dict[str, Feature]:
    """Return ``{member: Feature}`` for one capability group."""
    out: dict[str, Feature] = {}
    for member in _MEMBERS[group]:
        value = getattr(caps, member, None)
        if isinstance(value, Feature):
            out[member] = value
    return out


@dataclass(frozen=True)
class QueryCapabilities:
    """The complete on-demand query capability declaration of one backend."""

    api_version: int
    backend: str
    model: ModelCaps
    state: StateCaps
    scratch: ScratchCaps
    jacobian: JacobianCaps
    geometry: GeometryCaps
    contacts: ContactCaps
    actuators: ActuatorCaps
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity/version and freeze the note mapping."""
        if self.api_version < 1:
            raise ValueError(f"api_version must be >= 1, got {self.api_version}")
        if not self.backend:
            raise ValueError("backend must be a non-empty name")
        object.__setattr__(self, "notes", MappingProxyType(dict(self.notes)))

    def _caps(self, group: str) -> Caps:
        """Return one capability group by name."""
        return getattr(self, group)

    def _feature(self, op: str) -> Feature:
        """Resolve one ``<group>.<operation>`` string to its feature."""
        group, dot, member = op.partition(".")
        if not dot or group not in _GROUP_ORDER:
            raise QueryCapabilityError(
                self.backend,
                op,
                "unknown",
                f"expected one of: {list(self.ops())}",
            )
        feature = _feature_map(self._caps(group), group).get(member)
        if feature is None:
            raise QueryCapabilityError(
                self.backend,
                op,
                "unknown",
                f"group {group!r} has operations {list(_MEMBERS[group])}",
            )
        return feature

    def supports(self, op: str) -> Support:
        """Return the declared support level of one operation."""
        return self._feature(op).support

    def require(self, *ops: str) -> None:
        """Raise :class:`QueryCapabilityError` for the first unsupported op."""
        missing = [(op, self._feature(op)) for op in ops if self.supports(op) == "unsupported"]
        if not missing:
            return
        op, feature = missing[0]
        detail = feature.detail
        if len(missing) > 1:
            extra = [name for name, _ in missing[1:]]
            detail = f"{detail}; also unsupported: {extra}".lstrip("; ")
        raise QueryCapabilityError(self.backend, op, feature.support, detail)

    def ops(self) -> tuple[str, ...]:
        """Return every declared ``<group>.<operation>`` name."""
        return tuple(f"{group}.{member}" for group in _GROUP_ORDER for member in _MEMBERS[group])
