"""The programmatic ``Run(robot=RobotProfile)`` path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from unigym.actuators.ir import ActuationGroup
from unigym.assets import release
from unigym.assets.asset_ref import AssetRef
from unigym.pack_api import discover
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.world import World

TINY_ARM = Path(__file__).resolve().parents[2] / "_support" / "data" / "tiny_arm.xml"
JOINTS = ("shoulder", "elbow")
ACTION_SPEC = OrderedActionSpec.from_mapping(
    [
        {
            "name": "tiny",
            "articulation": "tiny",
            "joints": list(JOINTS),
            "target": "position",
            "role": "body",
        }
    ]
)
POLICIES = (PolicyBinding.from_mapping({"adapter": "manual", "groups": [["tiny", "tiny"]]}),)


def _profile(*, assets: AssetRef | None) -> RobotProfile:
    return RobotProfile(
        name="tiny",
        joint_sets={"body": JOINTS},
        default_joint_pos={"shoulder": 0.1, "elbow": -0.2},
        assets=assets,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
        actuation=(
            ActuationGroup(
                name="body",
                law="standard_pd",
                joint_set="body",
                gains={"kp": 20.0, "kd": 1.0},
            ),
        ),
    )


def _floor_world(tmp_path: Path) -> Path:
    path = tmp_path / "world.xml"
    path.write_text(
        '<mujoco model="floor"><worldbody>'
        '<geom name="world_floor" type="plane" size="2 2 0.1"/>'
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    return path


def test_run_profile_path_builds_with_profile_assets_and_actuation(tmp_path: Path) -> None:
    """The profile supplies the asset, actuation, topology and standby pose."""
    profile = _profile(assets=AssetRef(mjcf=str(TINY_ARM)))
    run = Run(
        world=World.base(mjcf=_floor_world(tmp_path)),
        robot=profile,
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        steps=1,
        inspect=True,
    )
    try:
        assert run.spec.robot_profiles == {"tiny": profile}
        articulation = run.spec.articulations[0]
        assert articulation.name == "tiny"
        assert articulation.actuation is None  # profile actuation, not an inline copy

        run.build(inspect=True)
        control = run.controls[("tiny", "tiny")]
        assert control.joint_names == JOINTS
        assert control.q0.tolist() == pytest.approx([0.1, -0.2])
        view = run.articulations[0]
        assert view["name"] == "tiny"
        assert view["joints"] == list(JOINTS)
    finally:
        run.close()


def test_run_profile_path_requires_explicit_action_and_policies(tmp_path: Path) -> None:
    """No action layout is inferred from the body joint set on the profile path."""
    with pytest.raises(NotImplementedError, match="explicit action_spec"):
        Run(
            world=World.base(mjcf=_floor_world(tmp_path)),
            robot=_profile(assets=None),
            steps=1,
        )


def test_profile_release_assets_resolve_only_at_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``release:`` declaration stays in the spec and resolves for the build."""
    monkeypatch.setattr(release, "_resolver", lambda: lambda asset_id: TINY_ARM)
    profile = _profile(assets=AssetRef(mjcf="release:tiny-arm"))
    run = Run(
        world=World.base(mjcf=_floor_world(tmp_path)),
        robot=profile,
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        steps=1,
        inspect=True,
    )
    try:
        assert run.spec.articulations[0].model is not None
        assert run.spec.articulations[0].model.mjcf == "release:tiny-arm"

        run.build(inspect=True)
        control = run.controls[("tiny", "tiny")]
        assert control.joint_names == JOINTS
    finally:
        run.close()


def test_mujoco_build_does_not_resolve_an_unselected_release_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing USD release must not break a MuJoCo build that only needs MJCF."""
    requested: list[str] = []

    def resolver(asset_id: str) -> Path:
        requested.append(asset_id)
        return tmp_path / "absent-usd" / f"{asset_id}.usd"

    monkeypatch.setattr(release, "_resolver", lambda: resolver)
    profile = _profile(assets=AssetRef(mjcf=str(TINY_ARM), usd="release:g1-29dof-usd"))
    run = Run(
        world=World.base(mjcf=_floor_world(tmp_path)),
        robot=profile,
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        steps=1,
        inspect=True,
    )
    try:
        instance = run.build(inspect=True)
        assert requested == []
        assert instance.spec.articulations[0].model is not None
        assert instance.spec.articulations[0].model.usd == "release:g1-29dof-usd"
        assert instance.spec.articulations[0].model.mjcf == str(TINY_ARM)
    finally:
        run.close()


def _write_profile_pack(root: Path) -> None:
    """Write one content pack whose robot profile declares an MJCF asset."""
    root.mkdir(parents=True, exist_ok=True)
    robot = {
        "schema": "unigym.robot/1",
        "name": "tiny",
        "joint_sets": {"body": list(JOINTS)},
        "default_joint_pos": {"shoulder": 0.1, "elbow": -0.2},
        "assets": {
            "mjcf": str(TINY_ARM),
            "usd": "release:g1-29dof-usd",
        },
        "base_body": "base",
        "foot_bodies": [],
        "body_sets": {},
        "body_bindings": {"base": {"mjcf": "base"}},
        "extra_action_sets": {},
        "actuation": [
            {
                "name": "body",
                "law": "standard_pd",
                "joint_set": "body",
                "gains": {"kp": 20.0, "kd": 1.0},
            }
        ],
    }
    robot_path = root / "robots" / "tiny.yaml"
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    robot_path.write_text(yaml.safe_dump(robot, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema": "unigym.pack/1",
        "name": "profile_assets",
        "version": "1.0.0",
        "api": 1,
        "trust": "content",
        "provides": {
            "providers": [{"id": "tiny", "kind": "robot", "resource": "robots/tiny.yaml"}]
        },
    }
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def test_injected_profile_wins_over_a_pack_profile_with_the_same_id(
    tmp_path: Path,
) -> None:
    """An explicit ``Run(robot=profile)`` profile is used as-is, never pack-resolved."""
    pack_root = tmp_path / "pack"
    _write_profile_pack(pack_root)
    runtime = discover([pack_root])
    profile = replace(
        _profile(assets=AssetRef(mjcf=str(TINY_ARM))),
        default_joint_pos={"shoulder": 0.42, "elbow": -0.2},
    )
    run = Run(
        world=World.base(mjcf=_floor_world(tmp_path)),
        robot=profile,
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        steps=1,
        pack_runtime=runtime,
        discover_runtime=False,
        inspect=True,
    )
    try:
        resolved = run.spec.robot_profiles
        assert resolved is not None
        assert resolved["tiny"] is profile
        assert resolved["tiny"].default_joint_pos["shoulder"] == pytest.approx(0.42)
    finally:
        run.close()


def test_pack_profiles_cover_every_articulation_sharing_a_selector(
    tmp_path: Path,
) -> None:
    """Two articulations with one ``params.profile`` both receive the pack profile."""
    pack_root = tmp_path / "pack"
    _write_profile_pack(pack_root)
    runtime = discover([pack_root])
    config = {
        "run_name": "shared_profile_selector",
        "seed": 0,
        "env": {
            "backend": "mujoco",
            "num_envs": 1,
            "episode_length_s": 20.0,
            "decimation": 4,
            "sim_dt": 0.005,
        },
        "world_base": {"mjcf": str(_floor_world(tmp_path))},
        "collect": {"steps": 1},
        "adapter": {"name": "manual"},
        "command": {"name": "fixed", "dim": 0, "values": []},
        "articulations": [
            {"name": "left", "params": {"profile": "tiny"}},
            {"name": "right", "params": {"profile": "tiny"}},
        ],
        "action_spec": [
            {
                "name": "left",
                "articulation": "left",
                "joints": list(JOINTS),
                "target": "position",
                "role": "body",
            },
            {
                "name": "right",
                "articulation": "right",
                "joints": list(JOINTS),
                "target": "position",
                "role": "body",
            },
        ],
        "policies": [
            {
                "adapter": "manual",
                "groups": [["left", "left"], ["right", "right"]],
            }
        ],
    }
    run = Run(
        dict(config),
        pack_runtime=runtime,
        discover_runtime=False,
        inspect=True,
    )
    try:
        resolved = run.spec.robot_profiles
        assert resolved is not None
        assert set(resolved) == {"left", "right"}
        assert resolved["left"].name == "tiny"
        assert resolved["right"] is resolved["left"]
    finally:
        run.close()


def test_config_mode_composes_the_selected_profile_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A G1-like pack profile composes MJCF and never touches its USD release."""
    requested: list[str] = []

    def resolver(asset_id: str) -> Path:
        requested.append(asset_id)
        raise AssertionError(f"unselected USD release {asset_id!r} was resolved")

    monkeypatch.setattr(release, "_resolver", lambda: resolver)
    pack_root = tmp_path / "pack"
    _write_profile_pack(pack_root)
    runtime = discover([pack_root])
    config_path = tmp_path / "run.yaml"
    config = {
        "run_name": "config_profile_assets",
        "seed": 0,
        "env": {
            "backend": "mujoco",
            "num_envs": 1,
            "episode_length_s": 20.0,
            "decimation": 4,
            "sim_dt": 0.005,
        },
        "world_base": {"mjcf": str(_floor_world(tmp_path))},
        "collect": {"steps": 1},
        "adapter": {"name": "manual"},
        "command": {"name": "fixed", "dim": 0, "values": []},
        "articulations": [{"name": "tiny"}],
        "action_spec": [
            {
                "name": "tiny",
                "articulation": "tiny",
                "joints": list(JOINTS),
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [{"adapter": "manual", "groups": [["tiny", "tiny"]]}],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run = Run.from_config(
        config_path,
        pack_runtime=runtime,
        discover_runtime=False,
        inspect=True,
    )
    try:
        instance = run.build(inspect=True)
        model = instance.spec.articulations[0].model
        assert model is not None
        assert model.mjcf == str(TINY_ARM)
        assert model.usd == "release:g1-29dof-usd"
        assert requested == []
        view = run.articulations[0]
        assert view["name"] == "tiny"
        assert view["joints"] == list(JOINTS)
    finally:
        run.close()
