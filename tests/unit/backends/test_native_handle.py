"""Contracts for the external-only native escape hatch."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from unigym._typing import JsonValue
from unigym.capabilities.traits import (
    ISAACLAB2_TRAITS,
    ISAACLAB3_NEWTON_TRAITS,
    ISAACLAB3_PHYSX_TRAITS,
    MJWARP_TRAITS,
    MUJOCO_TRAITS,
    Trait,
)
from unigym.env.backends.base import BackendCapabilities
from unigym.env.native import NativeHandle
from unigym.run.api import Run
from unigym.run.instance import RunInstance


def _handle() -> NativeHandle:
    model = object()
    return NativeHandle("mujoco", {"model": model, "data": object()})


def test_native_handle_exposes_identity_objects_and_metadata() -> None:
    """Item/attribute/require access all return the same stored identity."""
    model = object()
    data = object()
    handle = NativeHandle("mujoco", {"model": model, "data": data})

    assert handle.backend == "mujoco"
    assert handle.version == 1
    assert handle.unstable is True
    assert handle.keys() == ("model", "data")
    assert handle["model"] is model
    assert handle.data is data
    assert handle.require("model") is model
    assert set(dir(handle)) >= {"model", "data"}


def test_native_handle_is_not_a_mapping() -> None:
    """The handle is a deliberate non-Mapping escape hatch."""
    assert not isinstance(_handle(), Mapping)


def test_native_handle_unknown_name_errors_name_and_available() -> None:
    """Unknown names fail clearly through both access paths."""
    handle = _handle()

    with pytest.raises(KeyError, match=r"bogus.*\['model', 'data'\]"):
        _ = handle["bogus"]
    with pytest.raises(AttributeError, match=r"bogus.*\['model', 'data'\]"):
        _ = handle.bogus
    with pytest.raises(KeyError, match=r"bogus.*\['model', 'data'\]"):
        handle.require("bogus")


def test_native_handle_private_attributes_stay_attributes() -> None:
    """Dunder/private lookup must not masquerade as a native object name."""
    with pytest.raises(AttributeError):
        _ = _handle()._missing


def test_native_handle_is_a_declared_trait_on_the_three_backends() -> None:
    """Every registered backend declares NATIVE_HANDLE after the C2 trait pass."""
    assert Trait.NATIVE_HANDLE in MUJOCO_TRAITS
    assert Trait.NATIVE_HANDLE in MJWARP_TRAITS
    assert Trait.NATIVE_HANDLE in ISAACLAB2_TRAITS
    assert Trait.NATIVE_HANDLE in ISAACLAB3_PHYSX_TRAITS
    assert Trait.NATIVE_HANDLE in ISAACLAB3_NEWTON_TRAITS

    assert Trait.NATIVE_HANDLE in BackendCapabilities(native_handle=True).traits()
    assert Trait.NATIVE_HANDLE not in BackendCapabilities().traits()


def _cfg() -> dict[str, JsonValue]:
    """A small valid config; ``build`` is stubbed in these unit tests."""
    return {
        "run_name": "r",
        "seed": 0,
        "env": {"backend": "mujoco", "num_envs": 1},
        "world_base": {"mjcf": "model.xml"},
        "collect": {"steps": 1},
        "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
        "action_spec": [
            {
                "name": "arm",
                "articulation": "arm",
                "joints": ["shoulder"],
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
        "observations": {"robot": ["arm"]},
    }


class _NativeEnv:
    """The minimum env face RunInstance.native touches."""

    def __init__(self, handle: NativeHandle) -> None:
        self.handle = handle
        self.calls: list[str] = []

    def native(self, backend: str) -> NativeHandle:
        self.calls.append(backend)
        return self.handle


def _instance(backend: str, env: object) -> RunInstance:
    """Build a RunInstance with only the fields the native path reads."""
    return RunInstance(
        spec=SimpleNamespace(backend=backend),
        profile=None,
        adapter=None,
        sampler=None,
        term_group=None,
        env=env,
        bridge=None,
    )


def test_run_native_delegates_to_the_live_backend(monkeypatch) -> None:
    """Run.native -> RunInstance.native -> env.native with the same name."""
    run = Run(_cfg())
    handle = _handle()
    env = _NativeEnv(handle)
    instance = _instance("mujoco", env)
    monkeypatch.setattr(run, "build", lambda inspect=False: instance)

    assert run.native("mujoco") is handle
    assert env.calls == ["mujoco"]


def test_run_native_rejects_a_backend_mismatch() -> None:
    """The requested backend must equal the run's live backend."""
    instance = _instance("mujoco", _NativeEnv(_handle()))

    with pytest.raises(ValueError, match=r"mjwarp.*mujoco"):
        instance.native("mjwarp")


def test_run_native_rejects_the_removed_isaaclab_aliases() -> None:
    """Selection is canonical-only: the removed aliases never resolve."""
    instance = _instance("isaaclab2", _NativeEnv(_handle()))

    for alias in ("isaaclab", "isaaclab3"):
        with pytest.raises(ValueError, match=r"not a canonical backend"):
            instance.native(alias)

    canonical_env = _NativeEnv(_handle())
    canonical_instance = _instance("isaaclab2", canonical_env)

    assert canonical_instance.native("isaaclab2") is canonical_env.handle
    assert canonical_env.calls == ["isaaclab2"]


def test_run_native_delegates_for_isaaclab3_physx(monkeypatch) -> None:
    """The C2 trait pass lights NATIVE_HANDLE up for isaaclab3_physx."""
    run = Run(_cfg())
    env = _NativeEnv(_handle())
    instance = _instance("isaaclab3_physx", env)
    monkeypatch.setattr(run, "build", lambda inspect=False: instance)

    assert instance.native("isaaclab3_physx") is env.handle
    assert run.native("isaaclab3_physx") is env.handle
    assert env.calls == ["isaaclab3_physx", "isaaclab3_physx"]


_CORE_ROOT = Path(__file__).resolve().parents[2] / "unigym"
_NATIVE_CALL_FILES = {
    "env/native.py",
    "run/api.py",
    "run/instance.py",
    "backends/mujoco/env.py",
    "backends/mjwarp/env.py",
    "backends/isaaclab2/env.py",
}


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    """Return the nearest enclosing function/method name, if any."""
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return None


def test_core_does_not_call_the_native_escape_hatch() -> None:
    """No core module calls ``.native(...)`` outside its own native method."""
    offenders: list[str] = []
    for path in sorted(_CORE_ROOT.rglob("*.py")):
        relative = path.relative_to(_CORE_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "native":
                continue
            allowed = (
                relative in _NATIVE_CALL_FILES and _enclosing_function(node, parents) == "native"
            )
            if not allowed:
                offenders.append(f"{relative}:{node.lineno}")

    assert offenders == [], f"core calls env.native outside the escape hatch: {offenders}"
