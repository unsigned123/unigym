"""MuJoCo Warp native handle identity and backend-name contract."""

from __future__ import annotations

import pytest

from unigym.env.native import NativeHandle


def test_mjwarp_native_returns_the_live_handles(tiny_env_factory) -> None:
    """Every handle is the backend's identity reference, including warp."""
    env = tiny_env_factory(num_envs=2)

    handle = env.native("mjwarp")

    assert isinstance(handle, NativeHandle)
    assert handle.backend == "mjwarp"
    assert handle.keys() == ("model", "data", "datas", "warp_model", "warp_data")
    assert handle["model"] is env.mj_model
    assert handle["data"] is env.mj_data
    assert handle["datas"] is env.mj_datas
    assert handle["warp_model"] is env._wm
    assert handle["warp_data"] is env._wdata
    assert handle.warp_model is env._wm
    assert handle.require("warp_data") is env._wdata


def test_mjwarp_native_rejects_another_backend_name(tiny_env_factory) -> None:
    """The exact backend name is validated without a registry lookup."""
    env = tiny_env_factory()

    with pytest.raises(ValueError, match=r"mjwarp.*mujoco"):
        env.native("mujoco")


def test_mjwarp_backend_capabilities_project_the_trait() -> None:
    """The capability bit and the registry trait cannot drift."""
    from unigym.backends.mjwarp.env import BACKEND_CAPABILITIES
    from unigym.backends.registry import Trait

    assert Trait.NATIVE_HANDLE in BACKEND_CAPABILITIES.traits()
