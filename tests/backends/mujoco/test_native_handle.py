"""MuJoCo native handle identity and backend-name contract."""

from __future__ import annotations

import pytest

from unigym.env.native import NativeHandle


def test_mujoco_native_returns_the_live_handles(tiny_env_factory) -> None:
    """model/data/datas are the backend's identity references, not copies."""
    env = tiny_env_factory(num_envs=2)

    before = env.mj_data.qpos.copy()
    handle = env.native("mujoco")

    assert isinstance(handle, NativeHandle)
    assert handle.backend == "mujoco"
    assert handle.keys() == ("model", "data", "datas")
    assert handle["model"] is env.mj_model
    assert handle["data"] is env.mj_data
    assert handle["datas"] is env.mj_datas
    assert handle.model is env.mj_model
    assert handle.require("data") is env.mj_data
    assert env.mj_data.qpos.tolist() == before.tolist()


def test_mujoco_native_rejects_another_backend_name(tiny_env_factory) -> None:
    """The exact backend name is validated without a registry lookup."""
    env = tiny_env_factory()

    with pytest.raises(ValueError, match=r"mujoco.*mjwarp"):
        env.native("mjwarp")


def test_mujoco_backend_capabilities_project_the_trait() -> None:
    """The capability bit and the registry trait cannot drift."""
    from unigym.backends.mujoco.env import BACKEND_CAPABILITIES
    from unigym.backends.registry import Trait

    assert Trait.NATIVE_HANDLE in BACKEND_CAPABILITIES.traits()
