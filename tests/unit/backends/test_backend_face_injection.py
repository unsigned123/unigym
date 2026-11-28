"""Injected faces are mandatory and resolve exactly once (no double factory)."""

from __future__ import annotations

import types
from dataclasses import replace

import pytest

from unigym.backends.compose import (
    BackendRuntime,
    backend_runtime,
    validate_backend_runtime,
)
from unigym.capabilities.traits import CANONICAL_BACKENDS
from unigym.contacts.manager import build_contact_manager
from unigym.props import build_prop_manager
from unigym.runtime.scene import build_scene_ops


def test_prop_manager_requires_an_injected_prop_face() -> None:
    """A prop build without a composition face fails fast (no registry lookup)."""
    with pytest.raises(ValueError, match="has no prop backend face"):
        build_prop_manager(
            {"types": [{"name": "crate", "type": "rigid_object", "urdf_path": "/tmp/x.urdf"}]},
            num_envs=1,
            env_spacing=1.0,
            backend="mujoco",
        )


def test_contact_manager_requires_an_injected_contact_face() -> None:
    """A contact build without a composition face fails fast."""
    with pytest.raises(NotImplementedError, match="contacts are not implemented"):
        build_contact_manager(
            [
                {
                    "name": "feet",
                    "sensor_refs": [{"owner": "robot", "body": "foot"}],
                    "frame": "world",
                }
            ],
            backend="mujoco",
            articulation_names=("arm",),
        )


def test_scene_ops_requires_an_injected_scene_face() -> None:
    """A scene face build without a composition face fails fast."""
    with pytest.raises(ValueError, match="declares no scene_ops face"):
        build_scene_ops(object(), "mujoco", robot_name="robot")


def test_isaaclab3_callable_faces_resolve_the_builder_exactly_once(monkeypatch) -> None:
    """A factory-shaped face must be called once: binding -> builder/realizer."""
    import unigym.backends.bindings as bindings

    def builder(spec, render_groups=None):
        return ("rig", spec, render_groups)

    def realizer(profile, *, spec, params, asset_plan=None):
        return ("realized", profile, spec, params, asset_plan)

    fake = types.SimpleNamespace(
        camera_backend=lambda: builder,
        robot_realizer=lambda: realizer,
    )
    monkeypatch.setattr(bindings, "_isaaclab3_faces", lambda _backend: fake)

    for backend in ("isaaclab3_physx", "isaaclab3_newton"):
        runtime = backend_runtime(backend)

        camera_face = runtime.bindings.camera_backend
        assert camera_face is not None
        rig_builder = camera_face()
        assert rig_builder is builder
        assert rig_builder("spec", render_groups=None) == ("rig", "spec", None)

        realizer_face = runtime.bindings.robot_realizer
        assert realizer_face is not None
        realize = realizer_face()
        assert realize is realizer
        assert realize("profile", spec="spec", params={"x": 1}, asset_plan=None) == (
            "realized",
            "profile",
            "spec",
            {"x": 1},
            None,
        )


def test_content_router_and_scene_engine_resolve_one_level() -> None:
    """``content_router`` returns the router; ``scene_engine`` returns the engine."""
    runtime = backend_runtime("mujoco")
    router_face = runtime.bindings.content_router
    assert router_face is not None
    assert callable(router_face())
    engine_face = runtime.bindings.scene_engine
    assert engine_face is not None
    assert not callable(engine_face())


def test_content_fill_and_content_router_must_agree() -> None:
    """The spec flag and the injected router are one invariant, not two."""

    def with_fill(runtime: BackendRuntime, value: bool) -> BackendRuntime:
        return BackendRuntime(
            spec=replace(runtime.spec, content_fill=value), bindings=runtime.bindings
        )

    no_router = backend_runtime("isaaclab2")
    assert not no_router.spec.content_fill and no_router.bindings.content_router is None
    with pytest.raises(RuntimeError, match="content_fill is declared but content_router is None"):
        validate_backend_runtime(with_fill(no_router, True))

    router = backend_runtime("mujoco")
    assert router.spec.content_fill and router.bindings.content_router is not None
    with pytest.raises(RuntimeError, match="content_router is declared without content_fill"):
        validate_backend_runtime(with_fill(router, False))

    for name in CANONICAL_BACKENDS:
        validate_backend_runtime(backend_runtime(name))
