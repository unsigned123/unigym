"""IsaacLab 2 native handle construction and identity contract."""

from __future__ import annotations

import pytest

from unigym.env.native import NativeHandle

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]


def test_isaaclab2_native_returns_the_live_handles(cartpole_env) -> None:
    """sim/scene/robot are identity references on a constructed env."""
    handle = cartpole_env.native("isaaclab2")

    assert isinstance(handle, NativeHandle)
    assert handle.backend == "isaaclab2"
    assert handle.keys() == ("sim", "scene", "robot")
    assert handle["sim"] is cartpole_env.sim
    assert handle["scene"] is cartpole_env.scene
    assert handle["robot"] is cartpole_env.robot
    assert handle.sim is cartpole_env.sim
    assert handle.require("scene") is cartpole_env.scene

    from unigym.backends.isaaclab2.env import BACKEND_CAPABILITIES
    from unigym.backends.registry import Trait

    assert Trait.NATIVE_HANDLE in BACKEND_CAPABILITIES.traits()


def test_isaaclab2_native_rejects_another_backend_name(cartpole_env) -> None:
    """The exact backend name is validated without a registry lookup."""
    with pytest.raises(ValueError, match=r"isaaclab2.*mujoco"):
        cartpole_env.native("mujoco")
