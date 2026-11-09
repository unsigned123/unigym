"""Robot recipe registry (isaaclab-dependent factories; lazy-loaded).

The factory modules (g1) import isaaclab at module level, so they are loaded
on first attribute access (PEP 562) — i.e. only after AppLauncher in
scripts/collect.py.  Pure-constant modules such as
``unigym.robots.g1_constants`` stay importable WITHOUT isaaclab
(local unit tests; importing this package no longer pulls isaaclab).
"""
from __future__ import annotations

from typing import Any

from unigym._typing import JsonValue

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


def build_robot_recipe(section: dict[str, JsonValue],
                       with_isaaclab: bool = False) -> RobotRecipe:
    """Build a :class:`RobotRecipe` from one articulation declaration.

    ``section`` is the projection ``{name, params, body_parts}`` of an
    :class:`~unigym.scenes.articulation.ArticulationSpec`; it is authoring
    data (joint order / q0 / feet / body parts / asset), never a policy.
    """
    raw_params = section.get("params")
    params: dict[str, JsonValue] = (
        dict(raw_params) if isinstance(raw_params, dict) else {})
    if "body_parts" in section:
        params["body_parts"] = section["body_parts"]
    name_value = section["name"]
    if not isinstance(name_value, str):
        raise TypeError(
            f"robot recipe name must be a string, got "
            f"{type(name_value).__name__}")
    factory = __getattr__(name_value)
    # with_isaaclab: isaaclab backends need the ArticulationCfg (built after
    # AppLauncher); the mujoco backend passes False -> articulation_cfg=None.
    return factory(**params, with_isaaclab=with_isaaclab)


# typing-only import (kept last; base.py pulls isaaclab).  Under `from
# __future__ import annotations` evaluation is lazy, but a real import here
# would defeat the laziness — use TYPE_CHECKING.
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from .base import RobotRecipe
