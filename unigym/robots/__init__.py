"""Backend-neutral robot profile registry (lazy-loaded factories).

Every factory returns a :class:`~unigym.robots.base.RobotProfile`; nothing in
this package imports a backend or an engine.  The factory modules are still
loaded on first attribute access (PEP 562) so the registry table stays the
single, explicit name -> factory map.
"""

from __future__ import annotations

from typing import Any

from unigym._typing import JsonValue

from .base import (
    BodyBinding,
    BodyBindingError,
    CanonicalBodyId,
    RobotProfile,
    normalize_canonical_body_id,
    validate_body_bindings,
)

# registry name -> (module, factory function)
_FACTORIES = {
    "g1": ("g1", "make_g1_profile"),
    "gr1": ("gr1", "make_gr1_profile"),
}


def _load(mod_name: str, fn_name: str):
    import importlib

    mod = importlib.import_module(f".{mod_name}", __name__)
    return getattr(mod, fn_name)


def __getattr__(name: str) -> Any:
    if name == "ROBOT_REGISTRY":
        registry = {key: _load(mod, fn) for key, (mod, fn) in _FACTORIES.items()}
        globals()["ROBOT_REGISTRY"] = registry
        return registry
    if name in _FACTORIES:
        mod, fn = _FACTORIES[name]
        return _load(mod, fn)
    for mod, fn in _FACTORIES.values():  # also expose make_*_profile directly
        if name == fn:
            return _load(mod, fn)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_robot_profile(section: dict[str, JsonValue]) -> RobotProfile:
    """Build a neutral :class:`RobotProfile` from one articulation declaration.

    ``section`` is the projection ``{name, params}`` of an
    :class:`~unigym.scenes.articulation.ArticulationSpec`; it is authoring
    data (joint order / q0 / assets / action faces), never a policy and never
    a backend object.
    """
    raw_params = section.get("params")
    params: dict[str, JsonValue] = dict(raw_params) if isinstance(raw_params, dict) else {}
    name_value = section["name"]
    if not isinstance(name_value, str):
        raise TypeError(f"robot profile name must be a string, got {type(name_value).__name__}")
    factory = __getattr__(name_value)
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
    "BodyBinding",
    "BodyBindingError",
    "CanonicalBodyId",
    "RobotProfile",
    "build_robot_profile",
    "normalize_canonical_body_id",
    "validate_body_bindings",
]
