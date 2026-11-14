"""Backend-neutral robot profile registry.

Every factory returns a :class:`~unigym.robots.base.RobotProfile`; nothing in
this package imports a backend or an engine.  ``ROBOT_REGISTRY`` is the
authoritative registry: robot data and factories come from external packs, not
from a hard-coded core table.  Pack code registers through
:func:`register_robot`; resource-backed profiles are registered by the pack
runtime.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from unigym._typing import JsonValue
from unigym.provenance import (
    OriginTable,
    ProviderInfo,
    ProviderOrigin,
    RegistrySnapshot,
)

from .base import (
    BodyBinding,
    BodyBindingError,
    CanonicalBodyId,
    RobotProfile,
    normalize_canonical_body_id,
    validate_body_bindings,
)

if TYPE_CHECKING:
    from .data import load_robot_profile

_ROBOT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")

ROBOT_REGISTRY: dict[str, Callable[..., RobotProfile]] = {}
"""Registered robot factories by name."""

_ROBOT_ORIGINS = OriginTable()
"""Private provenance side table beside :data:`ROBOT_REGISTRY`."""

RobotRegistrySnapshot = RegistrySnapshot[Callable[..., RobotProfile]]
"""The origin-aware robot registry snapshot type."""


def _load(mod_name: str, fn_name: str) -> Callable[..., RobotProfile]:
    import importlib

    mod = importlib.import_module(f".{mod_name}", __name__)
    return getattr(mod, fn_name)


def _resolve_robot(name: str) -> Callable[..., RobotProfile]:
    """Resolve one name from the external robot registry."""
    external = ROBOT_REGISTRY.get(name)
    if external is not None:
        return external
    raise AttributeError(f"unknown robot {name!r}; registered robots are {sorted(ROBOT_REGISTRY)}")


def _factory_label(factory: object) -> str:
    """Return a stable ``module.qualname`` label for an error message."""
    module = getattr(factory, "__module__", None)
    qualname = getattr(factory, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    return repr(factory)


def validate_robot_registration(name: str, factory: Callable[..., RobotProfile]) -> None:
    """Validate one proposed external registration without mutating the registry.

    The policy owner is this module: the pack runtime uses the same check
    before it registers a whole provider set, so a later invalid item leaves
    no earlier registration behind.

    Raises:
        ValueError: The name is malformed, already registered, or the factory
            is not callable or is already registered under another name.
    """
    if not isinstance(name, str) or not _ROBOT_NAME_RE.fullmatch(name):
        raise ValueError(
            f"robot name {name!r} must be lowercase snake_case, optionally "
            f"dot-namespaced (e.g. 'acme.arm')"
        )
    existing = ROBOT_REGISTRY.get(name)
    if existing is not None:
        raise ValueError(
            f"robot name {name!r} is already registered to {_factory_label(existing)}; "
            f"cannot register {_factory_label(factory)} under the same name"
        )
    if not callable(factory):
        raise ValueError(
            f"robot factory for {name!r} must be callable (a function or class), "
            f"got {type(factory).__name__}"
        )
    alias = next((key for key, value in ROBOT_REGISTRY.items() if value is factory), None)
    if alias is not None:
        raise ValueError(
            f"robot factory {_factory_label(factory)} is already registered as "
            f"{alias!r}; one factory owns exactly one name, so it cannot also "
            f"register as {name!r}"
        )


def register_robot(
    name: str,
    factory: Callable[..., RobotProfile],
    *,
    origin: ProviderOrigin | None = None,
) -> Callable[..., RobotProfile]:
    """Register one external robot factory in the process-global registry.

    The name is the manifest item id used verbatim: lowercase snake_case with
    optional dot namespacing.  It is never trimmed.  Duplicates always fail,
    and one factory owns exactly one name.

    Args:
        name: Robot name a run config selects in the articulation ``name``.
        factory: A callable factory (function or class) returning a
            :class:`RobotProfile`; a profile instance is not a factory.
        origin: Internal/runtime-only provenance; pack authors never pass it.
            A registration without an origin reports
            :meth:`ProviderOrigin.unknown`.

    Returns:
        *factory*, so the helper can be used as a decorator.

    Raises:
        ValueError: The name or factory violates the registration contract.
    """
    validate_robot_registration(name, factory)
    if origin is not None and not isinstance(origin, ProviderOrigin):
        raise TypeError(
            f"robot {name!r}: origin must be a ProviderOrigin, got {type(origin).__name__}"
        )
    ROBOT_REGISTRY[name] = factory
    if origin is None:
        _ROBOT_ORIGINS.remove(name)
    else:
        _ROBOT_ORIGINS.set(name, origin)
    return factory


def robot_origin(name: str) -> ProviderOrigin:
    """Return the origin of one registered robot (never ``None``).

    A registered robot with no recorded origin returns
    :meth:`ProviderOrigin.unknown`.  An unknown name raises the existing
    unknown-robot ``AttributeError``.
    """
    if name not in ROBOT_REGISTRY:
        raise AttributeError(
            f"unknown robot {name!r}; registered robots are {sorted(ROBOT_REGISTRY)}"
        )
    return _ROBOT_ORIGINS.origin(name)


def robot_providers(kind: str | None = None) -> tuple[ProviderInfo, ...]:
    """Return the registered robots as live :class:`ProviderInfo` records.

    ``kind`` selects the provider kind; the robot registry owns only
    ``"robot"``, so any other kind returns an empty tuple (``None`` returns
    every record).  Records are ordered by name.
    """
    if kind is not None and kind != "robot":
        return ()
    return tuple(
        ProviderInfo(name=name, kind="robot", origin=_ROBOT_ORIGINS.origin(name))
        for name in sorted(ROBOT_REGISTRY)
    )


def attribute_robot(name: str, origin: ProviderOrigin) -> None:
    """Attach runtime-owned provenance to an already-registered robot.

    This is the pack runtime's activation-window attribution boundary, not
    pack-author API: pack code registers through :func:`register_robot` and
    never passes an origin here.  It may only label an entry whose current
    origin is absent or ``unknown``; a ``manifest`` or ``core`` origin is never
    relabeled.

    Raises:
        ValueError: The robot is unknown, or its current origin must not be
            relabeled.
        TypeError: *origin* is not a :class:`ProviderOrigin`.
    """
    if name not in ROBOT_REGISTRY:
        raise ValueError(
            f"robot {name!r} is not registered (registered robots: {sorted(ROBOT_REGISTRY)})"
        )
    if not isinstance(origin, ProviderOrigin):
        raise TypeError(
            f"robot {name!r}: origin must be a ProviderOrigin, got {type(origin).__name__}"
        )
    existing = _ROBOT_ORIGINS.get(name)
    if existing is not None and existing.kind != "unknown":
        raise ValueError(
            f"robot {name!r} already has a {existing.kind!r} origin; "
            "only an unattributed registration may be labeled"
        )
    _ROBOT_ORIGINS.set(name, origin)


def snapshot_robots() -> dict[str, Callable[..., RobotProfile]]:
    """Return a shallow copy of the current robot factories.

    This is the legacy factory-only snapshot: the origin side table is not
    captured.  Use :func:`snapshot_robot_registry` when rollback must restore
    provenance as well.
    """
    return dict(ROBOT_REGISTRY)


def restore_robots(snapshot: Mapping[str, Callable[..., RobotProfile]]) -> None:
    """Restore the robot factories to exactly *snapshot*.

    This is the legacy factory-only restore: the origin side table is left
    unchanged.  Use :func:`restore_robot_registry` for an origin-aware,
    total restore.
    """
    ROBOT_REGISTRY.clear()
    ROBOT_REGISTRY.update(snapshot)


def snapshot_robot_registry() -> RobotRegistrySnapshot:
    """Return a total snapshot of the robot factories and their origins."""
    return RegistrySnapshot.capture(ROBOT_REGISTRY, _ROBOT_ORIGINS)


def restore_robot_registry(snapshot: RobotRegistrySnapshot) -> None:
    """Restore the robot factories and origins to exactly *snapshot*."""
    if not isinstance(snapshot, RegistrySnapshot):
        raise TypeError(
            f"restore_robot_registry expects a RobotRegistrySnapshot, got {type(snapshot).__name__}"
        )
    snapshot.restore(ROBOT_REGISTRY, _ROBOT_ORIGINS)


def __getattr__(name: str) -> Any:
    if name == "load_robot_profile":
        return _load("data", "load_robot_profile")
    if name in ROBOT_REGISTRY:
        return _resolve_robot(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_robot_profile(section: dict[str, JsonValue]) -> RobotProfile:
    """Build a neutral :class:`RobotProfile` from one articulation declaration.

    ``section`` is the projection ``{name, params}`` of an
    :class:`~unigym.scenes.articulation.ArticulationSpec`; it is authoring
    data (joint order / q0 / assets / action faces), never a policy and never
    a backend object.

    Resolution goes through the external registry.  A missing name raises
    ``AttributeError`` so the existing ``Run(robot=...)`` error path is
    unchanged.
    """
    raw_params = section.get("params")
    params: dict[str, JsonValue] = dict(raw_params) if isinstance(raw_params, dict) else {}
    name_value = section["name"]
    if not isinstance(name_value, str):
        raise TypeError(f"robot profile name must be a string, got {type(name_value).__name__}")
    factory = _resolve_robot(name_value)
    profile = factory(**params)
    if not isinstance(profile, RobotProfile):
        raise TypeError(
            f"robot factory {name_value!r} must return a RobotProfile, "
            f"got {type(profile).__name__}"
        )
    if profile.name != name_value:
        raise ValueError(
            f"robot factory {name_value!r} returned profile {profile.name!r}; "
            f"the registry name and the profile name must match"
        )
    return profile


__all__ = [
    "ROBOT_REGISTRY",
    "BodyBinding",
    "BodyBindingError",
    "CanonicalBodyId",
    "ProviderInfo",
    "ProviderOrigin",
    "RobotProfile",
    "RobotRegistrySnapshot",
    "attribute_robot",
    "build_robot_profile",
    "load_robot_profile",
    "normalize_canonical_body_id",
    "register_robot",
    "restore_robot_registry",
    "restore_robots",
    "robot_origin",
    "robot_providers",
    "snapshot_robot_registry",
    "snapshot_robots",
    "validate_body_bindings",
]
