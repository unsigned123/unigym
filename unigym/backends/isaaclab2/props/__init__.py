"""IsaacLab prop backend face.

The neutral ``unigym.props`` manager resolves this object through
:attr:`BackendInfo.prop_backend`; it owns the type -> handler registry, the
scene-cfg builder (``scene_fields``) and the lazy cfg-class resolver.
"""
from __future__ import annotations

from typing import Any

from unigym.props.base import PropHandler, PropSpec

from .handlers import (BallProp, BoxProp, FreeArticulationProp, GenericProp,
                       PrimitiveProp, PropArticulation, UsdArticulationProp)
from .spawn import (_build_actuators, _build_urdf_spawn, _build_usd_spawn,
                    _resolve)

#: type -> handler class (the isaaclab handler registry)
_REGISTRY: dict[str, type[PropHandler]] = {
    "sphere": BallProp,
    "rigid_object": GenericProp,
    "usd": GenericProp,
    "articulation": PropArticulation,
    "usd_articulation": UsdArticulationProp,
    "free_articulation": FreeArticulationProp,
    "container": BoxProp,
    "table": BoxProp,
    "box": BoxProp,
    "cylinder": PrimitiveProp,
    "capsule": PrimitiveProp,
}

#: public compatibility name -> handler class
_CLASSES: dict[str, type[PropHandler]] = {
    cls.__name__: cls
    for cls in (BallProp, GenericProp, PropArticulation, UsdArticulationProp,
                FreeArticulationProp, BoxProp, PrimitiveProp)
}


class IsaacLabPropBackend:
    """Prop backend face for ``isaaclab2`` (scene-cfg-first form)."""

    name = "isaaclab2"
    #: scene-cfg-first form; MjSpec injection is a no-op
    injects_mjspec = False
    handler_registry = _REGISTRY

    def handler_for(self, spec: PropSpec) -> type[PropHandler]:
        """Return the handler class declared for ``spec.type`` (fail fast)."""
        try:
            return _REGISTRY[spec.type]
        except KeyError:
            known = ", ".join(sorted(_REGISTRY))
            raise ValueError(
                f"prop {spec.name!r}: unknown type {spec.type!r} "
                f"(known: {known}); a plane is static scene geometry and "
                f"cannot be a dynamic prop body") from None

    def compat(self, name: str) -> type[PropHandler]:
        """Resolve a legacy public handler class name."""
        return _CLASSES[name]

    def build_handler(self, spec: PropSpec, *, num_envs: int, device: str,
                      base_dir: str | None = None) -> PropHandler:
        """Instantiate the declared handler for one spec."""
        return self.handler_for(spec)(
            spec, num_envs=num_envs, device=device, base_dir=base_dir)

    def resolve_classes(self, classes=None) -> dict:
        """The lazy isaaclab cfg-class table (compat for the director)."""
        return _resolve(classes)

    def scene_fields(self, manager, *, classes=None) -> dict:
        """``{scene_field: cfg}`` for the isaaclab scene cfg (pre-clone)."""
        c = self.resolve_classes(classes)
        fields: dict[str, Any] = {}
        for handler in manager._handlers:
            name, cfg = handler.setup_scene(classes=c)
            fields[f"prop_{name}"] = cfg
        return fields

    def inject_spec(self, manager, mjspec) -> None:
        """No-op: the isaaclab form is scene-cfg-first (see ``scene_fields``)."""
        del manager, mjspec


def build_isaaclab_prop_backend() -> IsaacLabPropBackend:
    """Registry factory: the isaaclab prop backend face."""
    return IsaacLabPropBackend()


__all__ = [
    "IsaacLabPropBackend",
    "build_isaaclab_prop_backend",
    "_build_actuators",
    "_build_urdf_spawn",
    "_build_usd_spawn",
    "_resolve",
]
