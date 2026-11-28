"""Composition bindings: the lazy construction callables of every backend.

This module owns every concrete backend factory.  Each factory performs its
engine import *inside* the function, so importing this module (or
``unigym.backends``) never loads an engine.  Only composition code may read
:data:`BACKEND_BINDINGS`; neutral layers consume
:class:`unigym.capabilities.specs.BackendSpec` metadata instead.

The dataclass is deliberately flat: it carries exactly the construction faces
that currently cross the composition boundary, not a universal backend
context.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

from unigym.capabilities.traits import (
    CANONICAL_BACKENDS,
    ISAACLAB2_MODEL_EDIT_FIELDS,
    MUJOCO_MODEL_EDIT_FIELDS,
    ModelEditField,
)

if TYPE_CHECKING:
    from unigym.query.facade import QueryBackend

__all__ = ["BACKEND_BINDINGS", "BackendBindings", "ModelEditProvider"]


@dataclass(frozen=True)
class ModelEditProvider:
    """A backend's model-edit writer declaration (engine import stays lazy).

    Attributes:
        fields: The exact field set this provider can write.
        loader: Zero-argument factory returning the provider class/object;
            called only when a model-edit transaction opens.
    """

    fields: frozenset[ModelEditField]
    loader: Callable[[], object] = field(repr=False)


@dataclass(frozen=True)
class BackendBindings:
    """The construction callables of one backend (all imports stay lazy)."""

    scene_engine: Callable[[], object] | None = None
    env_class: Callable[[], type] | None = None
    live_provider: Callable[[], type] | None = None
    model_provider: Callable[[], Callable[[], Any]] | None = None
    model_edit_provider: ModelEditProvider | None = None
    query_backend: Callable[[], QueryBackend] | None = None
    model_handle: Callable[[], Callable[[object], object | None]] | None = None
    prop_backend: Callable[[], object] | None = None
    camera_backend: Callable[[], Callable[..., object]] | None = None
    director_ops: Callable[[], object] | None = None
    contact_backend: Callable[[], object] | None = None
    env_kwargs: Callable[[], Callable[..., dict[str, object]]] | None = None
    build_terrain: Callable[[], Callable[..., object]] | None = None
    build_events: Callable[[], Callable[..., object]] | None = None
    content_router: Callable[[], Callable[..., object]] | None = None
    scene_ops: Callable[[], object] | None = None
    kit_launcher: Callable[[], Callable[..., tuple[Any, Any]]] | None = None
    robot_realizer: Callable[[], Callable[..., object]] | None = None


def _mujoco_scene_engine() -> object:
    """Construct the MuJoCo scene engine (lazy: no engine import here)."""
    from unigym.backends.shared.mjfamily.scene.scene_engine import (
        MujocoSceneEngine,
    )

    return MujocoSceneEngine()


def _mujoco_env() -> type:
    """Load the MuJoCo env class lazily (engine import stays inside)."""
    from unigym.backends.mujoco.env import MujocoEnv

    return MujocoEnv


def _mjwarp_env() -> type:
    """Load the MuJoCo Warp env class lazily."""
    from unigym.backends.mjwarp.env import MjwarpEnv

    return MjwarpEnv


def _isaaclab2_env() -> type:
    """Load the IsaacLab 2.x env class lazily."""
    from unigym.backends.isaaclab2.env import IsaaclabEnv

    return IsaaclabEnv


def _isaaclab3_physx_env() -> type:
    """Load the IsaacLab 3.x PhysX env class lazily."""
    from unigym.backends.isaaclab3_physx.env import Isaaclab3PhysxEnv

    return Isaaclab3PhysxEnv


def _isaaclab3_newton_env() -> type:
    """Load the IsaacLab 3.x Newton env class lazily."""
    from unigym.backends.isaaclab3_newton.env import Isaaclab3NewtonEnv

    return Isaaclab3NewtonEnv


def _mujoco_live() -> type:
    """Load the MuJoCo-family live binder lazily (shared with mjwarp)."""
    from unigym.backends.mujoco.live import MujocoLiveBackend

    return MujocoLiveBackend


def _mujoco_model_provider() -> Callable[[], Any]:
    """Load the MuJoCo model provider class lazily."""
    from unigym.backends.shared.mjfamily.model.provider import MujocoModelProvider

    return MujocoModelProvider


def _mjwarp_model_provider() -> Callable[[], Any]:
    """Load the mjwarp-labelled model provider class lazily."""
    from unigym.backends.mjwarp.model import MjwarpModelProvider

    return MjwarpModelProvider


def _isaaclab2_model_provider() -> Callable[[], Any]:
    """Load the IsaacLab model provider class lazily."""
    from unigym.backends.isaaclab2.model import IsaacLabModelProvider

    return IsaacLabModelProvider


def _mujoco_model_edit_provider() -> object:
    """Load the MuJoCo model-edit writer lazily."""
    from unigym.backends.mujoco.model_edit import MujocoModelEditProvider

    return MujocoModelEditProvider


def _mjwarp_model_edit_provider() -> object:
    """Load the mjwarp model-edit writer lazily."""
    from unigym.backends.mjwarp.model_edit import MjwarpModelEditProvider

    return MjwarpModelEditProvider


def _isaaclab2_model_edit_provider() -> object:
    """Load the IsaacLab2 model-edit writer lazily."""
    from unigym.backends.isaaclab2.model_edit import IsaacLab2ModelEditProvider

    return IsaacLab2ModelEditProvider


def _mujoco_model_handle() -> Callable[[object], object | None]:
    """Load the MuJoCo compiled-model handle resolver lazily."""
    from unigym.backends.mujoco.model_handle import model_handle

    return model_handle


def _mujoco_query_backend() -> QueryBackend:
    """Load the MuJoCo runtime-query class lazily."""
    from unigym.backends.mujoco.query import MujocoQuery

    return MujocoQuery


def _mjwarp_query_backend() -> QueryBackend:
    """Load the mjWarp runtime-query class lazily."""
    from unigym.backends.mjwarp.query import MjwarpQuery

    return MjwarpQuery


def _isaaclab2_query_backend() -> QueryBackend:
    """Load the IsaacLab 2.x runtime-query class lazily."""
    from unigym.backends.isaaclab2.query import Isaaclab2Query

    return Isaaclab2Query


def _isaaclab3_query_backend() -> QueryBackend:
    """Load the IsaacLab 3.x runtime-query class lazily."""
    from unigym.backends.shared.isaaclab3_common.query import Isaaclab3Query

    return Isaaclab3Query


def _mjwarp_model_handle() -> Callable[[object], object | None]:
    """Load the mjwarp compiled-model handle resolver lazily."""
    from unigym.backends.mjwarp.model_handle import model_handle

    return model_handle


def _mujoco_rig_factory() -> Callable[..., object]:
    """Load the MuJoCo camera rig builder lazily."""
    from unigym.backends.mujoco.cameras import build_mj_rig

    return build_mj_rig


def _mjwarp_rig_factory() -> Callable[..., object]:
    """Load the MuJoCo-Warp camera rig builder lazily."""
    from unigym.backends.mjwarp.cameras import build_mjwarp_rig

    return build_mjwarp_rig


def _isaaclab2_rig_factory() -> Callable[..., object]:
    """Load the IsaacLab camera rig builder lazily."""
    from unigym.backends.isaaclab2.cameras.rigs import build_isaaclab_rig

    return build_isaaclab_rig


def _mujoco_prop_backend() -> object:
    """Load the MuJoCo prop backend face lazily."""
    from unigym.backends.shared.mjfamily.props.props import build_mujoco_prop_backend

    return build_mujoco_prop_backend()


def _isaaclab2_prop_backend() -> object:
    """Load the IsaacLab prop backend face lazily."""
    from unigym.backends.isaaclab2.props import build_isaaclab_prop_backend

    return build_isaaclab_prop_backend()


def _mujoco_director_ops() -> object:
    """Load the MuJoCo director ops class lazily."""
    from unigym.backends.shared.mjfamily.director.ops import MujocoDirectorOps

    return MujocoDirectorOps


def _mjwarp_director_ops() -> object:
    """Load the MuJoCo-Warp director ops class lazily."""
    from unigym.backends.mjwarp.director_ops import MjwarpDirectorOps

    return MjwarpDirectorOps


def _isaaclab2_director_ops() -> object:
    """Load the IsaacLab director ops class lazily."""
    from unigym.backends.isaaclab2.director_ops import IsaacLabDirectorOps

    return IsaacLabDirectorOps


def _mujoco_contact_backend() -> object:
    """Load the MuJoCo contact backend class lazily."""
    from unigym.backends.mujoco.contacts import MujocoContactBackend

    return MujocoContactBackend


def _isaaclab2_contact_backend() -> object:
    """Load the IsaacLab contact backend class lazily."""
    from unigym.backends.isaaclab2.contacts import IsaacLabContactBackend

    return IsaacLabContactBackend


# --------------------------------------------------------------------------- #
# construction faces (lazy: engine imports stay in the packages)
# --------------------------------------------------------------------------- #
def _mujoco_env_kwargs() -> Callable[..., dict[str, object]]:
    from unigym.backends.mujoco.build import env_kwargs

    return env_kwargs


def _mjwarp_env_kwargs() -> Callable[..., dict[str, object]]:
    from unigym.backends.mjwarp.build import env_kwargs

    return env_kwargs


def _mujoco_content_router() -> Callable[..., object]:
    from unigym.backends.mujoco.build import route_content

    return route_content


def _isaaclab2_scene_engine() -> object:
    """Construct the IsaacLab asset-scene engine (lazy: no engine import here)."""
    from unigym.backends.isaaclab2.asset_scene.engine import AssetSceneEngine

    return AssetSceneEngine()


def _isaaclab2_env_kwargs() -> Callable[..., dict[str, object]]:
    from unigym.backends.isaaclab2.build import env_kwargs

    return env_kwargs


def _isaaclab2_build_terrain() -> Callable[..., object]:
    from unigym.backends.isaaclab2.build import build_terrain

    return build_terrain


def _isaaclab2_build_events() -> Callable[..., object]:
    from unigym.backends.isaaclab2.build import build_events

    return build_events


def _mujoco_scene_ops() -> object:
    from unigym.backends.mujoco.scene import MujocoSceneOps

    return MujocoSceneOps


def _isaaclab2_scene_ops() -> object:
    from unigym.backends.isaaclab2.scene import IsaacLabSceneOps

    return IsaacLabSceneOps


def _isaaclab2_kit_launcher() -> Callable[..., tuple[Any, Any]]:
    from unigym.backends.isaaclab2.launcher import kit_launcher

    return kit_launcher


def _isaaclab2_robot_realizer() -> Callable[..., object]:
    from unigym.backends.isaaclab2.robots import build_robot_realization

    return build_robot_realization


def _isaaclab3_faces(backend: str) -> object:
    """Import one IsaacLab 3 backend's lazy face module.

    Called only from inside a face factory, so the engine-owned module (and
    any engine import it triggers) never loads when this module is imported.
    """
    return importlib.import_module(f"unigym.backends.{backend}.registry_faces")


def _isaaclab3_face(backend: str, face: str) -> Callable[[], object]:
    """Return a binding factory that resolves one face from *backend*.

    Args:
        backend: the canonical IsaacLab 3 identity
            (``isaaclab3_physx`` / ``isaaclab3_newton``).
        face: the zero-argument factory name in that backend's
            ``registry_faces`` module.
    """

    def factory() -> object:
        return getattr(_isaaclab3_faces(backend), face)()

    return factory


def _isaaclab3_model_provider(backend: str) -> Callable[[], Callable[[], Any]]:
    """Return a binding factory for the IsaacLab 3 model-provider face."""

    def factory() -> Callable[[], Any]:
        return getattr(_isaaclab3_faces(backend), "model_provider")()

    return factory


def _isaaclab3_callable_seam(backend: str, name: str) -> Callable[[], Callable[..., object]]:
    """Return a binding factory for an exported backend callable seam.

    ``build_terrain`` / ``build_events`` are the callables themselves (not
    zero-argument factories), so the binding factory resolves the module
    attribute without calling it.
    """

    def factory() -> Callable[..., object]:
        return getattr(_isaaclab3_faces(backend), name)

    return factory


def _isaaclab3_provided_callable(backend: str, name: str) -> Callable[[], Callable[..., object]]:
    """Return a binding factory for a zero-argument callable face factory.

    ``camera_backend`` / ``robot_realizer`` export a zero-argument factory
    returning the actual builder/realizer, so the binding factory calls it
    exactly once.
    """

    def factory() -> Callable[..., object]:
        return getattr(_isaaclab3_faces(backend), name)()

    return factory


def _isaaclab3_env_kwargs(
    backend: str,
) -> Callable[[], Callable[..., dict[str, object]]]:
    """Return a binding factory for the IsaacLab 3 ``env_kwargs`` seam."""

    def factory() -> Callable[..., dict[str, object]]:
        return getattr(_isaaclab3_faces(backend), "env_kwargs")

    return factory


def _isaaclab3_kit_launcher(backend: str) -> Callable[[], Callable[..., tuple[Any, Any]]]:
    """Return a binding factory for the IsaacLab 3 kit-launcher seam."""

    def factory() -> Callable[..., tuple[Any, Any]]:
        return getattr(_isaaclab3_faces(backend), "kit_launcher")

    return factory


BACKEND_BINDINGS: Mapping[str, BackendBindings] = MappingProxyType(
    {
        "mujoco": BackendBindings(
            scene_engine=_mujoco_scene_engine,
            env_class=_mujoco_env,
            live_provider=_mujoco_live,
            model_provider=_mujoco_model_provider,
            model_edit_provider=ModelEditProvider(
                MUJOCO_MODEL_EDIT_FIELDS, _mujoco_model_edit_provider
            ),
            query_backend=_mujoco_query_backend,
            model_handle=_mujoco_model_handle,
            prop_backend=_mujoco_prop_backend,
            camera_backend=_mujoco_rig_factory,
            director_ops=_mujoco_director_ops,
            contact_backend=_mujoco_contact_backend,
            env_kwargs=_mujoco_env_kwargs,
            content_router=_mujoco_content_router,
            scene_ops=_mujoco_scene_ops,
        ),
        "mjwarp": BackendBindings(
            scene_engine=_mujoco_scene_engine,
            env_class=_mjwarp_env,
            live_provider=_mujoco_live,
            model_provider=_mjwarp_model_provider,
            model_edit_provider=ModelEditProvider(
                MUJOCO_MODEL_EDIT_FIELDS, _mjwarp_model_edit_provider
            ),
            query_backend=_mjwarp_query_backend,
            model_handle=_mjwarp_model_handle,
            prop_backend=_mujoco_prop_backend,
            camera_backend=_mjwarp_rig_factory,
            director_ops=_mjwarp_director_ops,
            contact_backend=_mujoco_contact_backend,
            env_kwargs=_mjwarp_env_kwargs,
            content_router=_mujoco_content_router,
            scene_ops=_mujoco_scene_ops,
        ),
        "isaaclab2": BackendBindings(
            scene_engine=_isaaclab2_scene_engine,
            env_class=_isaaclab2_env,
            model_provider=_isaaclab2_model_provider,
            model_edit_provider=ModelEditProvider(
                ISAACLAB2_MODEL_EDIT_FIELDS, _isaaclab2_model_edit_provider
            ),
            query_backend=_isaaclab2_query_backend,
            prop_backend=_isaaclab2_prop_backend,
            camera_backend=_isaaclab2_rig_factory,
            director_ops=_isaaclab2_director_ops,
            contact_backend=_isaaclab2_contact_backend,
            env_kwargs=_isaaclab2_env_kwargs,
            build_terrain=_isaaclab2_build_terrain,
            build_events=_isaaclab2_build_events,
            scene_ops=_isaaclab2_scene_ops,
            kit_launcher=_isaaclab2_kit_launcher,
            robot_realizer=_isaaclab2_robot_realizer,
        ),
        "isaaclab3_physx": BackendBindings(
            scene_engine=_isaaclab3_face("isaaclab3_physx", "scene_engine"),
            env_class=_isaaclab3_physx_env,
            model_provider=_isaaclab3_model_provider("isaaclab3_physx"),
            query_backend=_isaaclab3_query_backend,
            prop_backend=_isaaclab3_face("isaaclab3_physx", "prop_backend"),
            camera_backend=_isaaclab3_provided_callable("isaaclab3_physx", "camera_backend"),
            director_ops=_isaaclab3_face("isaaclab3_physx", "director_ops"),
            contact_backend=_isaaclab3_face("isaaclab3_physx", "contact_backend"),
            env_kwargs=_isaaclab3_env_kwargs("isaaclab3_physx"),
            build_terrain=_isaaclab3_callable_seam("isaaclab3_physx", "build_terrain"),
            build_events=_isaaclab3_callable_seam("isaaclab3_physx", "build_events"),
            scene_ops=_isaaclab3_face("isaaclab3_physx", "scene_ops"),
            kit_launcher=_isaaclab3_kit_launcher("isaaclab3_physx"),
            robot_realizer=_isaaclab3_provided_callable("isaaclab3_physx", "robot_realizer"),
        ),
        "isaaclab3_newton": BackendBindings(
            scene_engine=_isaaclab3_face("isaaclab3_newton", "scene_engine"),
            env_class=_isaaclab3_newton_env,
            model_provider=_isaaclab3_model_provider("isaaclab3_newton"),
            query_backend=_isaaclab3_query_backend,
            prop_backend=_isaaclab3_face("isaaclab3_newton", "prop_backend"),
            camera_backend=_isaaclab3_provided_callable("isaaclab3_newton", "camera_backend"),
            director_ops=_isaaclab3_face("isaaclab3_newton", "director_ops"),
            contact_backend=_isaaclab3_face("isaaclab3_newton", "contact_backend"),
            env_kwargs=_isaaclab3_env_kwargs("isaaclab3_newton"),
            build_terrain=_isaaclab3_callable_seam("isaaclab3_newton", "build_terrain"),
            build_events=_isaaclab3_callable_seam("isaaclab3_newton", "build_events"),
            scene_ops=_isaaclab3_face("isaaclab3_newton", "scene_ops"),
            kit_launcher=_isaaclab3_kit_launcher("isaaclab3_newton"),
            robot_realizer=_isaaclab3_provided_callable("isaaclab3_newton", "robot_realizer"),
        ),
    }
)

if tuple(BACKEND_BINDINGS) != CANONICAL_BACKENDS:
    raise RuntimeError(
        "backend bindings order does not match "
        "unigym.capabilities.CANONICAL_BACKENDS: "
        f"{tuple(BACKEND_BINDINGS)} != {CANONICAL_BACKENDS}"
    )
