"""Actuator law descriptors: the pure, backend-neutral law contract.

A descriptor names one control law and declares what its authoring payload may
contain, which neutral drives it serves, which compiled plants the MuJoCo
family must expose, and which backends implement it.  Descriptors are pure
data and safe to load during ``SPEC``; the backend factory that actually
materializes the law lives in :mod:`unigym.actuators.implementations` and is
imported much later.

This module is standard-library only (no numpy, no engine, no backend).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from unigym.provenance import (
    OriginTable,
    ProviderOrigin,
    RegistrySnapshot,
)
from unigym.registration import PackFace, require_registration

__all__ = [
    "ACTUATOR_LAW_ORIGINS",
    "ACTUATOR_LAW_REGISTRY",
    "BUILTIN_LAWS",
    "COMPUTED_TORQUE_PD",
    "DIRECT_TORQUE",
    "DIRECT_VELOCITY",
    "NATIVE_SERVO",
    "STANDARD_PD",
    "ActuatorLawDescriptor",
    "actuator_law",
    "attribute_actuator_law",
    "actuator_law_origin",
    "register_actuator_law",
    "restore_actuator_laws",
    "snapshot_actuator_laws",
]


def _empty_compiled_plant() -> dict[str, str]:
    return {}


@dataclass(frozen=True)
class ActuatorLawDescriptor:
    """The declarative contract of one actuator law.

    ``config_keys`` are the group-spec keys the law accepts, ``gain_keys`` the
    subset of ``gains`` it accepts, and ``required_gains`` those that must be
    present.  ``drives`` names the neutral control drives it serves.
    ``backends`` is ``None`` for "every backend" or the explicit capability
    set.  ``compiled_plant`` maps a backend to the compiled actuator class it
    requires.  ``backend_owned_limits`` names backends whose compiled
    actuator owns effort/velocity limits.
    """

    name: str
    config_keys: frozenset[str] = frozenset()
    gain_keys: frozenset[str] = frozenset()
    required_gains: tuple[str, ...] = ()
    allows_parts: bool = False
    drives: frozenset[str] = frozenset()
    backends: frozenset[str] | None = None
    compiled_plant: Mapping[str, str] = field(default_factory=_empty_compiled_plant)
    backend_owned_limits: frozenset[str] = frozenset()
    allows_command: bool = False
    requires_implementation: bool = False
    implementation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"actuator law name must be a non-empty string, got {self.name!r}")
        object.__setattr__(self, "config_keys", frozenset(str(k) for k in self.config_keys))
        object.__setattr__(self, "gain_keys", frozenset(str(k) for k in self.gain_keys))
        object.__setattr__(self, "required_gains", tuple(str(k) for k in self.required_gains))
        object.__setattr__(self, "drives", frozenset(str(d) for d in self.drives))
        if self.backends is not None:
            object.__setattr__(self, "backends", frozenset(str(b) for b in self.backends))
        object.__setattr__(
            self,
            "compiled_plant",
            MappingProxyType({str(k): str(v) for k, v in dict(self.compiled_plant).items()}),
        )
        object.__setattr__(
            self, "backend_owned_limits", frozenset(str(b) for b in self.backend_owned_limits)
        )
        missing = sorted(set(self.required_gains) - set(self.gain_keys))
        if missing:
            raise ValueError(
                f"actuator law {self.name!r} requires gains {missing} that are not in "
                f"its gain_keys {sorted(self.gain_keys)}"
            )
        if not set(self.required_gains) <= set(self.gain_keys):
            raise ValueError(f"actuator law {self.name!r} required_gains must be gain keys")
        if self.implementation_id is not None and not str(self.implementation_id).strip():
            raise ValueError(
                f"actuator law {self.name!r} implementation_id must be a non-empty string"
            )


ACTUATOR_LAW_REGISTRY: dict[str, ActuatorLawDescriptor] = {}
"""Registered actuator law descriptors by name."""

ACTUATOR_LAW_ORIGINS = OriginTable()
"""Private provenance side table beside :data:`ACTUATOR_LAW_REGISTRY`."""

ActuatorLawSnapshot = RegistrySnapshot[ActuatorLawDescriptor]
"""The origin-aware actuator-law registry snapshot type."""


_MJ_FAMILY = frozenset({"mujoco", "mjwarp"})

STANDARD_PD = ActuatorLawDescriptor(
    name="standard_pd",
    config_keys=frozenset({"kp", "kd"}),
    gain_keys=frozenset({"kp", "kd"}),
    required_gains=("kp", "kd"),
    drives=frozenset({"position_pd"}),
    backend_owned_limits=_MJ_FAMILY,
)

COMPUTED_TORQUE_PD = ActuatorLawDescriptor(
    name="computed_torque_pd",
    config_keys=frozenset({"kp", "kd", "parts"}),
    gain_keys=frozenset({"kp", "kd"}),
    required_gains=("kp", "kd"),
    allows_parts=True,
    drives=frozenset({"position_pd"}),
    backends=_MJ_FAMILY,
    backend_owned_limits=_MJ_FAMILY,
)

NATIVE_SERVO = ActuatorLawDescriptor(
    name="native_servo",
    config_keys=frozenset({"kp", "kd"}),
    drives=frozenset({"position_servo"}),
    backends=_MJ_FAMILY,
    compiled_plant={"mujoco": "position_servo", "mjwarp": "position_servo"},
    backend_owned_limits=_MJ_FAMILY,
)

DIRECT_TORQUE = ActuatorLawDescriptor(
    name="direct_torque",
    drives=frozenset({"torque"}),
    backends=_MJ_FAMILY,
    compiled_plant={"mujoco": "motor", "mjwarp": "motor"},
    backend_owned_limits=_MJ_FAMILY,
)

DIRECT_VELOCITY = ActuatorLawDescriptor(
    name="direct_velocity",
    drives=frozenset({"velocity"}),
    backends=_MJ_FAMILY,
    compiled_plant={"mujoco": "velocity_servo", "mjwarp": "velocity_servo"},
    backend_owned_limits=_MJ_FAMILY,
)

BUILTIN_LAWS = (
    STANDARD_PD,
    COMPUTED_TORQUE_PD,
    NATIVE_SERVO,
    DIRECT_TORQUE,
    DIRECT_VELOCITY,
)


def _install_core_laws() -> None:
    """Install the core descriptors regardless of the pack phase.

    Core built-ins are not pack registrations: they must be present even when
    ``unigym.actuator_law`` is first imported at a late phase (for example a
    materialization in a fresh process).  Pack laws still go through
    :func:`register_actuator_law` and its gate.
    """
    for descriptor in BUILTIN_LAWS:
        if descriptor.name in ACTUATOR_LAW_REGISTRY:
            continue
        ACTUATOR_LAW_REGISTRY[descriptor.name] = descriptor
        ACTUATOR_LAW_ORIGINS.set(descriptor.name, ProviderOrigin(kind="core"))


_install_core_laws()


def register_actuator_law(
    descriptor: ActuatorLawDescriptor,
    *,
    origin: ProviderOrigin | None = None,
) -> ActuatorLawDescriptor:
    """Register one law descriptor (fail-fast on duplicates)."""
    if not isinstance(descriptor, ActuatorLawDescriptor):
        raise TypeError(
            f"actuator law registration must be an ActuatorLawDescriptor, got "
            f"{type(descriptor).__name__}"
        )
    require_registration(PackFace.ACTUATOR_LAW, name=descriptor.name)
    existing = ACTUATOR_LAW_REGISTRY.get(descriptor.name)
    if existing is not None:
        raise ValueError(
            f"actuator law {descriptor.name!r} is already registered; one descriptor "
            "owns exactly one name"
        )
    if origin is not None and not isinstance(origin, ProviderOrigin):
        raise TypeError(
            f"actuator law {descriptor.name!r}: origin must be a ProviderOrigin, got "
            f"{type(origin).__name__}"
        )
    ACTUATOR_LAW_REGISTRY[descriptor.name] = descriptor
    if origin is None:
        ACTUATOR_LAW_ORIGINS.remove(descriptor.name)
    else:
        ACTUATOR_LAW_ORIGINS.set(descriptor.name, origin)
    return descriptor


def actuator_law(name: str) -> ActuatorLawDescriptor:
    """Return the registered descriptor for *name* (fail-fast when absent)."""
    try:
        return ACTUATOR_LAW_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown actuator law {name!r}; registered laws are "
            f"{sorted(ACTUATOR_LAW_REGISTRY)} | hint: declare the law in a pack "
            "manifest (kind: actuator_law) during ACTIVATE/SPEC"
        ) from None


def attribute_actuator_law(name: str, origin: ProviderOrigin) -> None:
    """Label one already-registered law with *origin*."""
    if name in ACTUATOR_LAW_REGISTRY:
        ACTUATOR_LAW_ORIGINS.set(name, origin)


def actuator_law_origin(name: str) -> ProviderOrigin:
    """Return the origin of one registered law (never ``None``)."""
    return ACTUATOR_LAW_ORIGINS.origin(name)


def snapshot_actuator_laws() -> ActuatorLawSnapshot:
    """Capture the law registry and its origins together."""
    return RegistrySnapshot.capture(ACTUATOR_LAW_REGISTRY, ACTUATOR_LAW_ORIGINS)


def restore_actuator_laws(snapshot: ActuatorLawSnapshot) -> None:
    """Restore a snapshot captured by :func:`snapshot_actuator_laws`."""
    snapshot.restore(ACTUATOR_LAW_REGISTRY, ACTUATOR_LAW_ORIGINS)
