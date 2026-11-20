"""Actuator implementation registry: late-bound backend factories.

A law descriptor is pure data.  The factory that turns it into a live
``ActuatorBase`` (MuJoCo family) or a native actuator cfg (IsaacLab family) is
an *implementation* and is materialized late:

* ``mujoco`` / ``mjwarp`` at ``BACKEND_BUILD``;
* ``isaaclab2`` / ``isaaclab3_*`` at ``ASSEMBLE``.

Implementation modules therefore never import during ``ACTIVATE``/``SPEC``.
The core five laws' factories are registered lazily through
:func:`ensure_core_implementations` at a legal phase (bootstrap or the
backend's own materialization phase), so importing this module mutates no
registry.

This module is internal: ``unigym.pack_api`` deliberately does not export the
registration helper.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from unigym.actuator_law import (
    ActuatorLawDescriptor,
    actuator_law,
    actuator_law_origin,
)
from unigym.provenance import OriginTable, ProviderOrigin, RegistrySnapshot
from unigym.registration import PackFace, require_registration

from . import descriptors as _descriptors  # noqa: F401  (registers core laws)
from .base import ACTUATOR_REGISTRY

__all__ = [
    "ACTUATOR_IMPLEMENTATION_ORIGINS",
    "ACTUATOR_IMPLEMENTATION_REGISTRY",
    "ActuatorBuildContext",
    "ActuatorImplementation",
    "default_implementation_kind",
    "ensure_core_implementations",
    "get_actuator_implementation",
    "has_actuator_implementation",
    "register_actuator_implementation",
    "restore_actuator_implementations",
    "snapshot_actuator_implementations",
]


@dataclass(frozen=True)
class ActuatorBuildContext:
    """Everything a late-bound implementation factory may consume.

    ``params`` is the law-authoring payload (for example ``kp``/``kd``), already
    validated by the law descriptor.  ``group`` is the resolved neutral group
    when one exists (MuJoCo), or ``None`` for a cfg-only backend.
    """

    law: ActuatorLawDescriptor
    backend: str
    group: object | None = None
    joint_names: tuple[str, ...] = ()
    params: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    extra: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_names", tuple(str(n) for n in self.joint_names))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))


@dataclass(frozen=True)
class ActuatorImplementation:
    """One ``(law, backend)`` implementation registration."""

    law: str
    backend: str
    factory: Callable[[ActuatorBuildContext], object]
    kind: str
    origin: ProviderOrigin
    entrypoint: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError(
                f"actuator implementation {self.law!r}@{self.backend!r} factory must be "
                f"callable, got {type(self.factory).__name__}"
            )


ACTUATOR_IMPLEMENTATION_REGISTRY: dict[tuple[str, str], ActuatorImplementation] = {}
"""``(law, backend)`` -> implementation."""

ACTUATOR_IMPLEMENTATION_ORIGINS = OriginTable()
"""Private provenance side table keyed as ``"law@backend"``."""

ActuatorImplementationSnapshot = RegistrySnapshot[ActuatorImplementation]


def _key(law: str, backend: str) -> str:
    return f"{law}@{backend}"


def default_implementation_kind(backend: str) -> str:
    """The default implementation kind for one backend."""
    if backend in ("mujoco", "mjwarp"):
        return "mujoco_ctrl"
    if backend.startswith("isaaclab"):
        return "isaaclab_cfg"
    return "factory"


def register_actuator_implementation(
    implementation: ActuatorImplementation | str,
    backend: str | None = None,
    factory: Callable[[ActuatorBuildContext], object] | None = None,
    *,
    law: str | None = None,
    kind: str | None = None,
    entrypoint: str | None = None,
    origin: ProviderOrigin | None = None,
) -> ActuatorImplementation:
    """Register one late-bound implementation (internal).

    Accepts either an :class:`ActuatorImplementation` or the flat
    ``(law, backend, factory)`` form.  The backend-specific registration
    window is enforced here.
    """
    if isinstance(implementation, ActuatorImplementation):
        record = implementation
        if backend is not None or factory is not None or law is not None:
            raise TypeError(
                "register_actuator_implementation does not accept extra arguments "
                "when given an ActuatorImplementation"
            )
    else:
        law_name = implementation if law is None else law
        if backend is None or factory is None:
            raise TypeError("register_actuator_implementation requires (law, backend, factory)")
        if origin is None:
            law_origin = actuator_law_origin(law_name)
            origin = law_origin
        if not isinstance(origin, ProviderOrigin):
            raise TypeError(
                f"actuator implementation {law_name!r}@{backend!r}: origin must be a "
                f"ProviderOrigin, got {type(origin).__name__}"
            )
        record = ActuatorImplementation(
            law=law_name,
            backend=backend,
            factory=factory,
            kind=kind or default_implementation_kind(backend),
            origin=origin,
            entrypoint=entrypoint,
        )
    # The law must be declared before its implementation.
    actuator_law(record.law)
    require_registration(
        PackFace.ACTUATOR_IMPLEMENTATION,
        name=_key(record.law, record.backend),
        backend=record.backend,
    )
    key = (record.law, record.backend)
    existing = ACTUATOR_IMPLEMENTATION_REGISTRY.get(key)
    if existing is not None:
        raise ValueError(
            f"actuator implementation {record.law!r}@{record.backend!r} is already "
            f"registered (kind {existing.kind!r}); one (law, backend) pair owns "
            "exactly one implementation"
        )
    if not isinstance(record.origin, ProviderOrigin):
        raise TypeError(
            f"actuator implementation {record.law!r}@{record.backend!r}: origin must "
            f"be a ProviderOrigin"
        )
    ACTUATOR_IMPLEMENTATION_REGISTRY[key] = record
    ACTUATOR_IMPLEMENTATION_ORIGINS.set(_key(record.law, record.backend), record.origin)
    return record


def has_actuator_implementation(law: str, backend: str) -> bool:
    """True when ``(law, backend)`` has a registered implementation."""
    return (law, backend) in ACTUATOR_IMPLEMENTATION_REGISTRY


def get_actuator_implementation(law: str, backend: str) -> ActuatorImplementation:
    """Return the implementation for ``(law, backend)`` or fail with a hint."""
    try:
        return ACTUATOR_IMPLEMENTATION_REGISTRY[(law, backend)]
    except KeyError:
        available = sorted(f"{name}@{be}" for name, be in ACTUATOR_IMPLEMENTATION_REGISTRY)
        raise ValueError(
            f"actuator law {law!r} has no implementation for backend {backend!r}; "
            f"registered implementations are {available} | hint: declare a "
            "kind: actuator_implementation provider for this (law, backend) or use "
            "a built-in law"
        ) from None


def snapshot_actuator_implementations() -> ActuatorImplementationSnapshot:
    """Capture the implementation registry and its origins together."""
    return RegistrySnapshot.capture(
        ACTUATOR_IMPLEMENTATION_REGISTRY, ACTUATOR_IMPLEMENTATION_ORIGINS
    )


def restore_actuator_implementations(snapshot: ActuatorImplementationSnapshot) -> None:
    """Restore a snapshot captured by :func:`snapshot_actuator_implementations`."""
    snapshot.restore(ACTUATOR_IMPLEMENTATION_REGISTRY, ACTUATOR_IMPLEMENTATION_ORIGINS)


# --------------------------------------------------------------------------- #
# core built-in implementations (bootstrap registrations)
# --------------------------------------------------------------------------- #
def _mujoco_factory(law_name: str) -> Callable[[ActuatorBuildContext], object]:
    def factory(_context: ActuatorBuildContext) -> object:
        cls = ACTUATOR_REGISTRY.get(law_name)
        if cls is None:  # pragma: no cover - core table is complete
            raise ValueError(f"no MuJoCo class registered for law {law_name!r}")
        return cls()

    return factory


def _isaaclab_standard_pd_factory(context: ActuatorBuildContext) -> object:
    """Return the standard_pd cfg kwargs (the plan translates them natively)."""
    gains = dict(context.params)
    return {
        "joint_names_expr": list(context.joint_names),
        "stiffness": gains.get("kp"),
        "damping": gains.get("kd"),
    }


_MUJOCO_LAWS = (
    "standard_pd",
    "computed_torque_pd",
    "native_servo",
    "direct_torque",
    "direct_velocity",
)
_ISAACLAB_BACKENDS = ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton")
_ALL_BACKENDS = ("mujoco", "mjwarp", *_ISAACLAB_BACKENDS)


def ensure_core_implementations(backend: str | None = None) -> None:
    """Register the core factories for *backend* (or all) at the current phase.

    This is the only core-implementation mutation point.  Callers pass the
    backend whose materialization phase is active: ``mujoco``/``mjwarp`` at
    ``BACKEND_BUILD``, IsaacLab at ``ASSEMBLE``.  ``backend=None`` is intended
    for bootstrap/test setup and registers every core factory.
    """
    backends = _ALL_BACKENDS if backend is None else (backend,)
    if backend is not None and backend not in _ALL_BACKENDS:
        return
    for be in backends:
        laws = _MUJOCO_LAWS if be in ("mujoco", "mjwarp") else ("standard_pd",)
        for law_name in laws:
            if (law_name, be) in ACTUATOR_IMPLEMENTATION_REGISTRY:
                continue
            factory = (
                _mujoco_factory(law_name)
                if be in ("mujoco", "mjwarp")
                else _isaaclab_standard_pd_factory
            )
            register_actuator_implementation(
                law_name,
                be,
                factory,
                origin=ProviderOrigin(kind="core"),
            )
