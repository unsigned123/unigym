"""File-backed asset scenes construct on the real IsaacLab 2 dog lane."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unigym.assets.asset_ref import AssetRef
from unigym.backends.isaaclab2.asset_scene.names import asset_field, asset_prim_path
from unigym.scenes.scene_ir import AssetInstance, Pose, Scene

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

ARTICULATION = "cartpole"
JOINT_ORDER = ("slider_to_cart", "cart_to_pole")
SCENE = "bench"
ASSET_ID = "item"


def _asset_ref(asset_id: str, asset_format: str) -> AssetRef:
    reference = f"pack:isaaclab_cartpole/assets/{asset_id}"
    if asset_format == "urdf":
        return AssetRef(urdf=reference)
    return AssetRef(usd=reference)


def _asset_scene(
    asset_id: str,
    *,
    asset_format: str = "usd",
    kind: str = "static_rigid",
    collision: bool = True,
    xyz: tuple[float, float, float] = (0.2, 0.3, 0.5),
) -> Scene:
    return Scene(
        name=SCENE,
        asset_instances=(
            AssetInstance(
                id=ASSET_ID,
                asset=_asset_ref(asset_id, asset_format),
                pose=Pose(xyz=xyz),
                source="tests/backends/isaaclab2/test_asset_scene_kit.py:1",
                kind=kind,
                collision=collision,
            ),
        ),
    )


def _asset_plan(cartpole_pack):
    from unigym.assets.plan import AssetPlan
    from unigym.packs.runtime import AssetContext

    return AssetPlan(AssetContext(cartpole_pack))


def _provider(scene: Scene) -> str:
    from unigym.world import World

    preset = World.inline(scene).fragment()["scene_preset"]
    assert isinstance(preset, dict)
    provider = preset["provider"]
    assert isinstance(provider, str)
    return provider


def _resolved(cartpole_pack, asset_id: str) -> Path:
    return cartpole_pack.resource_for("assets", asset_id)


def _zero_action(env):
    from unigym.runtime.actions import NamedAction

    return {ARTICULATION: NamedAction.zeros(env.num_envs, list(JOINT_ORDER), "position")}


def _advance(env, steps: int) -> None:
    for _ in range(steps):
        env.step(_zero_action(env))


@pytest.mark.parametrize("num_envs", [1, 2], ids=["n1", "n2"])
def test_static_usd_field_clones_one_prim_per_env(num_envs, cartpole_pack, env_factory):
    """A static USD becomes one AssetBase field cloned once per environment."""
    xyz = (0.2, 0.3, 0.5)
    scene = _asset_scene("static_cup.usda", xyz=xyz)
    provider = _provider(scene)
    plan = _asset_plan(cartpole_pack)
    resolved = _resolved(cartpole_pack, "static_cup.usda")

    assert plan.resolve("pack:isaaclab_cartpole/assets/static_cup.usda") == str(resolved)

    env = env_factory(
        num_envs,
        scene_provider=provider,
        asset_plan=plan,
    )

    field = asset_field(SCENE, ASSET_ID)
    assert field in env.scene.extras
    expected_paths = [
        f"/World/envs/env_{index}/Scene_{SCENE}_{ASSET_ID}" for index in range(num_envs)
    ]

    import isaaclab.sim as sim_utils
    from isaaclab.sim.views import XformPrimView

    pattern = f"/World/envs/env_.*/Scene_{SCENE}_{ASSET_ID}"
    assert sorted(sim_utils.find_matching_prim_paths(pattern)) == sorted(expected_paths)

    view = XformPrimView(pattern, device="cpu")
    assert view.count == num_envs
    assert sorted(view.prim_paths) == sorted(expected_paths)
    positions = view.get_world_poses()[0].tolist()
    origins = env.scene.env_origins.tolist()

    import numpy as np

    for path, position in zip(view.prim_paths, positions):
        index = int(path.split("/env_", 1)[1].split("/", 1)[0])
        origin = origins[index]
        np.testing.assert_allclose(
            position,
            [origin[0] + xyz[0], origin[1] + xyz[1], origin[2] + xyz[2]],
            atol=1e-4,
        )

    _advance(env, 3)
    after = view.get_world_poses()[0].tolist()
    np.testing.assert_allclose(after, positions, atol=1e-6)


def test_static_usd_with_collision_disabled_constructs(cartpole_pack, env_factory):
    """The collision toggle still builds a native static field."""
    scene = _asset_scene("static_cup.usda", collision=False)
    env = env_factory(
        1,
        scene_provider=_provider(scene),
        asset_plan=_asset_plan(cartpole_pack),
    )

    view = env.scene[asset_field(SCENE, ASSET_ID)]
    assert view.count == 1


@pytest.mark.parametrize("num_envs", [1, 2], ids=["n1", "n2"])
def test_dynamic_usd_rigid_object_falls_under_gravity(num_envs, cartpole_pack, env_factory):
    """A dynamic USD becomes one RigidObject per env and its z decreases."""
    scene = _asset_scene("dynamic_cube.usda", kind="dynamic_rigid", xyz=(0.0, 0.0, 2.0))
    env = env_factory(
        num_envs,
        scene_provider=_provider(scene),
        asset_plan=_asset_plan(cartpole_pack),
    )

    field = asset_field(SCENE, ASSET_ID)
    assert field in env.scene.rigid_objects
    rigid = env.scene[field]
    assert rigid.num_instances == num_envs

    start = rigid.data.root_pos_w[:, 2].tolist()
    _advance(env, 20)
    end = rigid.data.root_pos_w[:, 2].tolist()

    assert start == pytest.approx([2.0] * num_envs, abs=1e-4)
    assert all(fell < initial - 0.05 for initial, fell in zip(start, end))


def test_static_urdf_is_refused_before_kit_construction(cartpole_pack, env_factory):
    """A static URDF has no IsaacLab static form and fails with the named error."""
    scene = _asset_scene(
        "jointless_cube.urdf",
        asset_format="urdf",
        kind="static_rigid",
    )

    with pytest.raises(ValueError, match="static_rigid URDF"):
        env_factory(
            1,
            scene_provider=_provider(scene),
            asset_plan=_asset_plan(cartpole_pack),
        )


def test_jointed_urdf_is_refused_as_articulation_in_scene(cartpole_pack, env_factory):
    """A URDF with joints is refused before any conversion or articulation."""
    scene = _asset_scene(
        "jointed_cube.urdf",
        asset_format="urdf",
        kind="dynamic_rigid",
    )

    with pytest.raises(ValueError, match=r"declares joints.*jointless URDF"):
        env_factory(
            1,
            scene_provider=_provider(scene),
            asset_plan=_asset_plan(cartpole_pack),
        )


def test_dynamic_jointless_urdf_constructs_or_fails_honestly(cartpole_pack, env_factory):
    """A jointless URDF either yields a RigidObject or names the conversion failure."""
    scene = _asset_scene(
        "jointless_cube.urdf",
        asset_format="urdf",
        kind="dynamic_rigid",
        xyz=(0.0, 0.0, 1.0),
    )
    plan = _asset_plan(cartpole_pack)

    try:
        env = env_factory(
            1,
            scene_provider=_provider(scene),
            asset_plan=plan,
        )
    except Exception as exc:  # noqa: BLE001 - the conversion outcome is environment-owned
        message = str(exc).lower()
        assert any(token in message for token in ("urdf", "rigid", "articulation")), message
        return

    field = asset_field(SCENE, ASSET_ID)
    assert field in env.scene.rigid_objects
    assert env.scene[field].num_instances == 1


def test_asset_scene_writes_the_five_build_artifacts(tmp_path, cartpole_pack, env_factory):
    """Composition lands scene/resolution/payload/compose/describe on disk."""
    scene = _asset_scene("static_cup.usda")
    resolved = _resolved(cartpole_pack, "static_cup.usda")
    env = env_factory(
        1,
        scene_provider=_provider(scene),
        asset_plan=_asset_plan(cartpole_pack),
        artifact_dir=tmp_path,
    )

    assert set(env.artifact_paths) == {
        "scene",
        "resolution",
        "payload_index",
        "compose",
        "describe",
    }
    for name in env.artifact_paths:
        assert Path(env.artifact_paths[name]).is_file(), name

    scene_doc = json.loads((tmp_path / "scene.json").read_text(encoding="utf-8"))
    assert scene_doc["name"] == SCENE
    assert json.loads((tmp_path / "resolution.json").read_text(encoding="utf-8"))
    assert json.loads((tmp_path / "payload_index.json").read_text(encoding="utf-8"))

    compose_doc = json.loads((tmp_path / "compose.txt").read_text(encoding="utf-8"))
    (field_doc,) = compose_doc["fields"]
    assert field_doc["field"] == asset_field(SCENE, ASSET_ID)
    assert field_doc["prim_path"] == asset_prim_path(SCENE, ASSET_ID)
    assert field_doc["resolved_path"] == str(resolved)

    describe_text = (tmp_path / "describe.json").read_text(encoding="utf-8")
    describe_doc = json.loads(describe_text)
    assert "resolved_path" not in describe_text
    assert describe_doc["ext"]["scene"]["fields"] == [
        {
            "field": asset_field(SCENE, ASSET_ID),
            "prim_path": asset_prim_path(SCENE, ASSET_ID),
            "kind": "static_rigid",
            "format": "usd",
        }
    ]


def test_asset_scene_env_closes_cleanly(cartpole_pack, env_factory):
    """The composed asset scene keeps the idempotent closed lifecycle."""
    scene = _asset_scene("static_cup.usda")
    env = env_factory(
        1,
        scene_provider=_provider(scene),
        asset_plan=_asset_plan(cartpole_pack),
    )
    action = _zero_action(env)

    env.close()
    env.close()

    with pytest.raises(RuntimeError) as excinfo:
        env.step(action)
    assert str(excinfo.value) == "isaaclab2 backend: environment is closed"
