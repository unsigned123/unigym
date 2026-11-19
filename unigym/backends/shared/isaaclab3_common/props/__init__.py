"""Shared IsaacLab 3 prop backend face.

The neutral ``unigym.props`` manager resolves this object through
:attr:`BackendInfo.prop_backend`; it owns the type -> handler registry, the
scene-cfg builder (``scene_fields``) and the lazy cfg-class resolver.  It is
shared by the ``isaaclab3_physx`` / ``isaaclab3_newton`` identities and never
imports a concrete backend.

The handler modules import torch, so both the registry and the legacy
class-name table are built lazily: importing ``props.spawn`` (which the
import-isolation test does) must not load torch through this package
``__init__``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from unigym.props.base import PropHandler, PropSpec

from .spawn import _build_actuators, _build_urdf_spawn, _build_usd_spawn, _resolve

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


def _torch_view(value: object) -> Tensor:
    """Return the explicit torch payload of an IsaacLab 3 ProxyArray."""
    import torch

    payload = getattr(value, "torch", None)
    if payload is None:
        payload = value
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"expected a torch payload, got {type(payload).__name__}")
    return payload


class _PropAssetDataLike(Protocol):
    """IsaacLab 3 data view surface the prop handlers read."""

    joint_pos: Tensor
    root_pos_w: Tensor
    root_quat_w: Tensor


class _PropAssetLike(Protocol):
    """Spawned RigidObject / Articulation surface the prop handlers use."""

    joint_names: Sequence[str]
    data: _PropAssetDataLike

    def write_root_pose_to_sim_index(self, *, root_pose: Tensor, env_ids=None) -> object: ...

    def write_root_velocity_to_sim_index(
        self, *, root_velocity: Tensor, env_ids=None
    ) -> object: ...

    def write_joint_state_to_sim_index(
        self, *, position: Tensor, velocity: Tensor, joint_ids=None, env_ids=None
    ) -> object: ...

    def write_joint_stiffness_to_sim(
        self, value: Tensor, joint_ids=None, env_ids=None
    ) -> object: ...

    def write_joint_damping_to_sim(self, value: Tensor, joint_ids=None, env_ids=None) -> object: ...

    def write_joint_friction_coefficient_to_sim(
        self, value: Tensor, joint_ids=None, env_ids=None
    ) -> object: ...


@lru_cache(maxsize=1)
def _handler_classes() -> dict[str, type[PropHandler]]:
    """Import the handler classes once (torch is loaded only here)."""
    from .handlers import (
        BallProp,
        BoxProp,
        FreeArticulationProp,
        GenericProp,
        PrimitiveProp,
        PropArticulation,
        UsdArticulationProp,
    )

    return {
        "BallProp": BallProp,
        "BoxProp": BoxProp,
        "FreeArticulationProp": FreeArticulationProp,
        "GenericProp": GenericProp,
        "PrimitiveProp": PrimitiveProp,
        "PropArticulation": PropArticulation,
        "UsdArticulationProp": UsdArticulationProp,
    }


@lru_cache(maxsize=1)
def _registry() -> dict[str, type[PropHandler]]:
    """The shared IsaacLab 3 type -> handler registry (built lazily)."""
    classes = _handler_classes()
    return {
        "sphere": classes["BallProp"],
        "rigid_object": classes["GenericProp"],
        "usd": classes["GenericProp"],
        "articulation": classes["PropArticulation"],
        "usd_articulation": classes["UsdArticulationProp"],
        "free_articulation": classes["FreeArticulationProp"],
        "container": classes["BoxProp"],
        "table": classes["BoxProp"],
        "box": classes["BoxProp"],
        "cylinder": classes["PrimitiveProp"],
        "capsule": classes["PrimitiveProp"],
    }


class IsaacLabPropBackend:
    """Prop backend face for the shared IsaacLab 3 library (scene-cfg-first).

    ``name`` is NOT an engine decision owned here: it stays empty by default
    and is supplied by the owning backend identity / caller.  This module must
    not hardcode a concrete backend name.
    """

    name = ""
    #: scene-cfg-first form; MjSpec injection is a no-op
    injects_mjspec = False

    def __init__(self, name: str | None = None):
        if name is not None:
            self.name = str(name)

    @property
    def handler_registry(self) -> dict[str, type[PropHandler]]:
        """Type -> handler class registry (materialized on first access)."""
        return _registry()

    def handler_for(self, spec: PropSpec) -> type[PropHandler]:
        """Return the handler class declared for ``spec.type`` (fail fast)."""
        registry = _registry()
        try:
            return registry[spec.type]
        except KeyError:
            known = ", ".join(sorted(registry))
            raise ValueError(
                f"prop {spec.name!r}: unknown type {spec.type!r} "
                f"(known: {known}); a plane is static scene geometry and "
                f"cannot be a dynamic prop body"
            ) from None

    def compat(self, name: str) -> type[PropHandler]:
        """Resolve a legacy public handler class name."""
        return _handler_classes()[name]

    def build_handler(
        self, spec: PropSpec, *, num_envs: int, device: str, base_dir: str | None = None
    ) -> PropHandler:
        """Instantiate the declared handler for one spec."""
        return self.handler_for(spec)(spec, num_envs=num_envs, device=device, base_dir=base_dir)

    def resolve_classes(self, classes=None) -> dict:
        """The lazy IsaacLab 3 cfg-class table (compat for the director)."""
        return _resolve(classes)

    def scene_fields(self, manager, *, classes=None) -> dict:
        """``{scene_field: cfg}`` for the IsaacLab 3 scene cfg (pre-clone)."""
        c = self.resolve_classes(classes)
        fields: dict[str, Any] = {}
        for handler in manager._handlers:
            name, cfg = handler.setup_scene(classes=c)
            fields[f"prop_{name}"] = cfg
        return fields

    def inject_spec(self, manager, mjspec) -> None:
        """No-op: the IsaacLab form is scene-cfg-first (see ``scene_fields``)."""
        del manager, mjspec


def build_isaaclab_prop_backend(name: str | None = None) -> IsaacLabPropBackend:
    """Registry factory: the shared IsaacLab 3 prop backend face.

    The owning backend identity passes its canonical name (or leaves it
    empty); the shared library supplies no concrete backend identity.
    """
    return IsaacLabPropBackend(name=name)


_HANDLER_NAMES = frozenset(
    {
        "BallProp",
        "BoxProp",
        "FreeArticulationProp",
        "GenericProp",
        "PrimitiveProp",
        "PropArticulation",
        "UsdArticulationProp",
    }
)


def __getattr__(name: str):
    """Lazily re-export the handler classes (torch stays unloaded until used)."""
    if name in _HANDLER_NAMES:
        return _handler_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IsaacLabPropBackend",
    "build_isaaclab_prop_backend",
    "_build_actuators",
    "_build_urdf_spawn",
    "_build_usd_spawn",
    "_resolve",
]
