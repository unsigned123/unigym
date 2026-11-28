"""The scene face requires an explicit, caller-owned robot name."""

from __future__ import annotations

import types

import pytest

from unigym.backends.compose import backend_runtime
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
    """The factory forwards a valid name to the injected backend ops object."""
    ops = build_scene_ops(
        object(),
        "mujoco",
        robot_name="robot",
        scene_ops_factory=backend_runtime("mujoco").bindings.scene_ops,
    )

    assert ops.robot_name == "robot"


@pytest.mark.parametrize("robot_name", [None, "", "   ", " g1 "])
def test_scene_face_requires_explicit_robot_name(robot_name):
    """The cmd-16 face rejects construction without an explicit name."""
    with pytest.raises(ValueError, match="robot_name"):
        SceneFace(object(), "mujoco", robot_name=robot_name)


def test_isaaclab_scene_contacts_flatten_sensors_by_ref_with_perms():
    """Scene tree reads the B2 per-ref sensor shape and applies its permutation."""

    class _Sensor:
        def __init__(self, names):
            self.body_names = list(names)

    class _Backend:
        _sensors = {
            "all": {
                "mode": "general",
                "sensors_by_ref": [[_Sensor(["tail", "root"]), _Sensor(["left"])]],
            }
        }
        _perm = {"all": [[1, 2, 0]]}

    spec = types.SimpleNamespace(name="all", sensor_refs=(), filter_refs=())
    cm = types.SimpleNamespace(backend=_Backend(), specs=[spec])
    env = types.SimpleNamespace(
        robot=types.SimpleNamespace(
            data=types.SimpleNamespace(body_names=["root", "left", "tail"])
        ),
        contact_manager=cm,
    )
    ops = IsaacLabSceneOps(env, robot_name="robot")

    assert ops._resolved_sensor_names(cm) == {"all": ["root", "left", "tail"]}
    assert ops._contacts_tree() == {"all": {"members": ["root", "left", "tail"]}}
