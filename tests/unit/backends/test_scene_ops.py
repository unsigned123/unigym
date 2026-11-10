"""The scene face requires an explicit, caller-owned robot name."""

from __future__ import annotations

import pytest

from unigym.backends.isaaclab2.scene import IsaacLabSceneOps
from unigym.backends.mujoco.scene import MujocoSceneOps
from unigym.runtime.scene import SceneFace, build_scene_ops

_OPS = (MujocoSceneOps, IsaacLabSceneOps)


@pytest.mark.parametrize("ops_factory", _OPS)
@pytest.mark.parametrize("robot_name", [None, "", "   ", " g1 "])
def test_scene_ops_constructor_requires_explicit_robot_name(ops_factory, robot_name):
    """A missing or empty robot name fails fast instead of falling back to G1."""
    with pytest.raises(ValueError, match="robot_name"):
        ops_factory(object(), robot_name=robot_name)


@pytest.mark.parametrize("ops_factory", _OPS)
def test_scene_ops_constructor_accepts_explicit_robot_name(ops_factory):
    """A non-empty caller-owned name is stored verbatim."""
    ops = ops_factory(object(), robot_name="robot")

    assert ops.robot_name == "robot"


@pytest.mark.parametrize("robot_name", [None, "", "   ", " g1 "])
def test_build_scene_ops_requires_explicit_robot_name(robot_name):
    """The factory validates the name before importing or touching a backend."""
    with pytest.raises(ValueError, match="robot_name"):
        build_scene_ops(object(), "mujoco", robot_name=robot_name)


def test_build_scene_ops_forwards_explicit_robot_name():
    """The factory forwards a valid name to the backend ops object."""
    ops = build_scene_ops(object(), "mujoco", robot_name="robot")

    assert ops.robot_name == "robot"


@pytest.mark.parametrize("robot_name", [None, "", "   ", " g1 "])
def test_scene_face_requires_explicit_robot_name(robot_name):
    """The cmd-16 face rejects construction without an explicit name."""
    with pytest.raises(ValueError, match="robot_name"):
        SceneFace(object(), "mujoco", robot_name=robot_name)
