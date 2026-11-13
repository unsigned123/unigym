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


def register_robot(name: str, factory: Callable[..., RobotProfile]) -> Callable[..., RobotProfile]:
    """Register one external robot factory in the process-global registry.

    The name is the manifest item id used verbatim: lowercase snake_case with
    optional dot namespacing.  It is never trimmed.  Duplicates always fail,
    and one factory owns exactly one name.

    Args:
        name: Robot name a run config selects in the articulation ``name``.
        factory: A callable factory (function or class) returning a
            :class:`RobotProfile`; a profile instance is not a factory.

    Returns:
        *factory*, so the helper can be used as a decorator.

    Raises:
        ValueError: The name or factory violates the registration contract.
    """
    validate_robot_registration(name, factory)
    ROBOT_REGISTRY[name] = factory
    return factory


def snapshot_robots() -> dict[str, Callable[..., RobotProfile]]:
    """Return a shallow copy of the current external robot registry."""
    return dict(ROBOT_REGISTRY)


def restore_robots(snapshot: Mapping[str, Callable[..., RobotProfile]]) -> None:
    """Restore the external robot registry to exactly *snapshot*."""
    ROBOT_REGISTRY.clear()
    ROBOT_REGISTRY.update(snapshot)


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
    "RobotProfile",
    "build_robot_profile",
    "load_robot_profile",
    "normalize_canonical_body_id",
    "register_robot",
    "restore_robots",
    "snapshot_robots",
    "validate_body_bindings",
]
