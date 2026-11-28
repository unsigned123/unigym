"""The scene face requires an explicit, caller-owned robot name."""

from __future__ import annotations

import types

import pytest

from unigym.backends.compose import backend_runtime
from unigym.backends.isaaclab2.scene import IsaacLabSceneOps
from unigym.backends.mujoco.scene import MujocoSceneOps
from unigym.backends.shared.isaaclab3_common.scene import IsaacLabSceneOps as IsaacLab3SceneOps
from unigym.runtime.scene import SceneFace, build_scene_ops

_OPS = (MujocoSceneOps, IsaacLabSceneOps)
_ISAACLAB_OPS = (IsaacLabSceneOps, IsaacLab3SceneOps)


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


@pytest.mark.parametrize("ops_factory", _ISAACLAB_OPS)
def test_isaaclab_full_tree_prop_links_come_from_live_data_only(ops_factory):
    """A stale asset-level ``body_names`` is never a scene-link fallback."""
    prop_asset = types.SimpleNamespace(
        body_names=["stale_asset_only_body"],
        data=types.SimpleNamespace(body_names=["prop_root", "prop_tip"]),
        joint_names=[],
    )
    handler = types.SimpleNamespace(
        spec=types.SimpleNamespace(name="box", type="rigid"),
        _asset=prop_asset,
    )
    env = types.SimpleNamespace(
        articulations={
            "robot": types.SimpleNamespace(base_body="base", joint_names=("j1",)),
        },
        robot=types.SimpleNamespace(data=types.SimpleNamespace(body_names=["base"])),
        prop_manager=types.SimpleNamespace(_handlers=[handler]),
    )

    doc = ops_factory(env, robot_name="robot").full_tree()

    assert doc["props"]["box"]["links"] == ["prop_root", "prop_tip"]


@pytest.mark.parametrize("ops_factory", _ISAACLAB_OPS)
@pytest.mark.parametrize("prop_data", [None, types.SimpleNamespace()])
def test_isaaclab_full_tree_missing_prop_live_names_reports_empty_links(ops_factory, prop_data):
    """A prop with no live data block documents empty links, not a stale fallback."""
    prop_asset = types.SimpleNamespace(
        body_names=["stale_asset_only_body"],
        data=prop_data,
        joint_names=[],
    )
    handler = types.SimpleNamespace(
        spec=types.SimpleNamespace(name="box", type="rigid"),
        _asset=prop_asset,
    )
    env = types.SimpleNamespace(
        articulations={
            "robot": types.SimpleNamespace(base_body="base", joint_names=("j1",)),
        },
        robot=types.SimpleNamespace(data=types.SimpleNamespace(body_names=["base"])),
        prop_manager=types.SimpleNamespace(_handlers=[handler]),
    )

    doc = ops_factory(env, robot_name="robot").full_tree()

    assert doc["props"]["box"]["links"] == []


@pytest.mark.parametrize("ops_factory", _ISAACLAB_OPS)
def test_isaaclab_full_tree_missing_robot_live_names_reports_empty_links(ops_factory):
    """The robot link list is the live ``data.body_names``; empty stays empty."""
    env = types.SimpleNamespace(
        articulations={
            "robot": types.SimpleNamespace(base_body="base", joint_names=("j1",)),
        },
        robot=types.SimpleNamespace(data=types.SimpleNamespace(body_names=[])),
        prop_manager=types.SimpleNamespace(_handlers=[]),
    )

    doc = ops_factory(env, robot_name="robot").full_tree()

    assert doc["robot"]["links"] == []
