"""Declaration-only ``SimulationBuilder`` contract over an explicit PackSet."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from unigym.camera import Camera
from unigym.packs.packset import PackSet
from unigym.robots.base import RobotProfile
from unigym.sim import (
    Control,
    ControlSpec,
    DeferredFeatureError,
    NoControlError,
    NoWorldError,
    SimulationBuilder,
    SimulationConfigError,
)
from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.world import World
from tests._support.sim_fixtures import write_sim_pack


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Restore the robot registry snapshot after each builder contract test."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)


def _builder(pack_set: PackSet, **overrides: object) -> SimulationBuilder:
    """A valid builder over *pack_set* with optional field overrides."""
    values: dict[str, object] = {
        "backend": "mujoco",
        "dt": 0.02,
        "physics_dt": 0.005,
        "pack_set": pack_set,
    }
    values.update(overrides)
    return SimulationBuilder(**values)


def _control() -> ControlSpec:
    """The canonical one-group body control used by most contract cases."""
    return Control.position_pd(joint_set="body", kp=10.0, kd=0.5)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"backend": "gazebo"}, "not registered"),
        ({"dt": 0.0}, "dt must be finite and > 0"),
        ({"dt": float("nan")}, "dt must be finite and > 0"),
        ({"physics_dt": 0.005, "substeps": 4}, "exactly one of physics_dt or substeps"),
        ({"physics_dt": None}, "exactly one of physics_dt or substeps"),
        ({"physics_dt": 0.003}, "must equal physics_dt \\* substeps"),
        ({"num_envs": 0}, "num_envs"),
        ({"num_envs": True}, "num_envs"),
        ({"num_envs": 1.5}, "num_envs"),
        ({"num_envs": "2"}, "num_envs"),
        ({"steps": 0}, "steps"),
        ({"episode_length_s": 0.0}, "episode_length_s"),
        ({"name": "  "}, "name"),
        ({"seed": -1}, "seed"),
        ({"device": ""}, "device"),
        ({"artifacts": "yes"}, "artifacts"),
    ],
)
def test_constructor_validation_table(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    """Every constructor field fails fast with its field named."""
    pack_set, _ = write_sim_pack(tmp_path)

    with pytest.raises(SimulationConfigError, match=match):
        _builder(pack_set, **overrides)


def test_constructor_derives_substeps_from_physics_dt(tmp_path: Path) -> None:
    """dt = physics_dt * substeps is the one accepted arithmetic."""
    pack_set, _ = write_sim_pack(tmp_path)

    builder = _builder(pack_set)

    assert builder.pack_runtime is pack_set.to_runtime()


def test_constructor_derives_physics_dt_from_substeps(tmp_path: Path) -> None:
    """The substeps form derives the engine step from the policy period."""
    pack_set, _ = write_sim_pack(tmp_path)

    builder = _builder(pack_set, physics_dt=None, substeps=4)

    assert builder.pack_runtime is pack_set.to_runtime()


def test_num_envs_two_is_accepted(tmp_path: Path) -> None:
    """A declared multi-env run is accepted and carries the no-reset policy."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set, num_envs=2)
    builder.world = World.base(mjcf="world.xml")
    builder.add_articulation(ref, control=_control())

    cfg = builder._to_cfg()

    assert cfg["env"]["num_envs"] == 2
    assert cfg["runner"] == {"auto_reset": False}


def _write_config(tmp_path: Path, runner_lines: tuple[str, ...]) -> Path:
    """A minimal valid run config with an explicit reset policy section."""
    path = tmp_path / "run.yaml"
    path.write_text(
        "\n".join(
            [
                "run_name: policy",
                "seed: 0",
                "env:",
                "  backend: mujoco",
                "  num_envs: 1",
                "  episode_length_s: 10.0",
                "  decimation: 4",
                "  sim_dt: 0.005",
                "world_base: {mjcf: world.xml}",
                "collect: {steps: 4}",
                "articulations: [{name: arm, joints: [{name: shoulder}]}]",
                "action_spec:",
                "  - {name: arm, articulation: arm, joints: [shoulder],",
                "     target: position, role: body}",
                "policies:",
                "  - {adapter: manual, groups: [[arm, arm]]}",
                *runner_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "runner_lines",
    [
        ("runner: {auto_reset: true}",),
        (),
    ],
)
def test_from_config_refuses_an_auto_reset_policy(
    tmp_path: Path, runner_lines: tuple[str, ...]
) -> None:
    """A foreign auto-reset policy is refused, never silently overridden."""
    builder = SimulationBuilder.from_config(_write_config(tmp_path, runner_lines))

    with pytest.raises(SimulationConfigError, match="runner.auto_reset") as excinfo:
        builder.build()

    message = str(excinfo.value)
    assert "runner.auto_reset=True" in message
    assert "never silently overrides" in message


def test_world_and_terrain_are_owned_by_the_world(tmp_path: Path) -> None:
    """The builder stores the world and exposes its terrain face as sugar."""
    pack_set, _ = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    world = World()

    assert builder.world is None
    with pytest.raises(NoWorldError):
        _ = builder.terrain
    builder.world = world
    assert builder.world is world
    assert builder.terrain is world.terrain


def test_add_articulation_requires_an_explicit_control(tmp_path: Path) -> None:
    """There is no implicit default control."""
    pack_set, ref = write_sim_pack(tmp_path)

    with pytest.raises(NoControlError):
        _builder(pack_set).add_articulation(ref)


def test_add_articulation_resolves_groups_and_owns_the_handle(tmp_path: Path) -> None:
    """The handle carries the profile, control and resolved action groups."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    control = _control()

    handle = builder.add_articulation(ref, name="arm", control=control)

    assert builder.articulations == (handle,)
    assert handle.name == "arm"
    assert handle.control is control
    assert handle.profile is ref
    assert handle.profile.name == ref.name
    assert set(handle.action_groups) == {"body"}
    assert handle.action_groups["body"].joints == ("shoulder", "elbow")


def test_add_articulation_accepts_explicit_joints(tmp_path: Path) -> None:
    """Explicit group joints are kept in declaration order."""
    pack_set, ref = write_sim_pack(tmp_path)
    control = Control.position_pd(joints=["elbow", "shoulder"], group="arm", kp=1.0, kd=0.1)

    handle = _builder(pack_set).add_articulation(ref, control=control)

    assert set(handle.action_groups) == {"arm"}
    assert handle.action_groups["arm"].joints == ("elbow", "shoulder")


def test_add_articulation_rejects_duplicate_labels(tmp_path: Path) -> None:
    """Two articulations cannot share one label."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    control = _control()
    builder.add_articulation(ref, control=control)

    with pytest.raises(SimulationConfigError, match="already declared"):
        builder.add_articulation(ref, control=control)


def test_add_articulation_rejects_an_undeclared_joint_set(tmp_path: Path) -> None:
    """A control group may only select a profile joint set."""
    pack_set, ref = write_sim_pack(tmp_path)
    control = Control.position_pd(joint_set="legs", kp=1.0, kd=0.1)

    with pytest.raises(SimulationConfigError, match="does not declare"):
        _builder(pack_set).add_articulation(ref, control=control)


def test_add_articulation_rejects_an_unknown_explicit_joint(tmp_path: Path) -> None:
    """An explicit group joint must exist on the profile."""
    pack_set, ref = write_sim_pack(tmp_path)
    control = Control.position_pd(joints=["wheel"], group="arm", kp=1.0, kd=0.1)

    with pytest.raises(SimulationConfigError, match="'wheel'"):
        _builder(pack_set).add_articulation(ref, control=control)


def test_add_articulation_requires_a_q0_for_every_group_joint(tmp_path: Path) -> None:
    """A joint without a control override or a profile default has no q0."""
    pack_set, ref = write_sim_pack(tmp_path, default_joint_pos={"shoulder": 0.0, "finger": 0.0})

    with pytest.raises(SimulationConfigError, match="no q0 for joints \\['elbow'\\]"):
        _builder(pack_set).add_articulation(ref, control=_control())


def test_add_articulation_rejects_overlapping_groups(tmp_path: Path) -> None:
    """Action groups of one articulation must be disjoint."""
    pack_set, ref = write_sim_pack(tmp_path)
    left = Control.position_pd(joints=["shoulder"], group="left", kp=1.0, kd=0.1)
    right = Control.position_pd(joints=["shoulder"], group="right", kp=1.0, kd=0.1)
    control = ControlSpec.from_groups((left.groups["left"], right.groups["right"]))

    with pytest.raises(SimulationConfigError, match="overlap on joints"):
        _builder(pack_set).add_articulation(ref, control=control)


def test_add_camera_declares_the_existing_entry(tmp_path: Path) -> None:
    """A Camera.world entry becomes one builder-owned camera handle."""
    pack_set, _ = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    entry = Camera.world("top", pos=(1.0, 0.0, 2.0), lookat=(0.0, 0.0, 0.0))

    handle = builder.add_camera(entry)

    assert builder.cameras == (handle,)
    assert handle.name == "top"
    assert handle.entry["rig"] == entry["rig"]


def test_add_camera_name_override_and_duplicate_rejection(tmp_path: Path) -> None:
    """The override wins and camera names are unique."""
    pack_set, _ = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    entry = Camera.world("top", pos=(1.0, 0.0, 2.0), lookat=(0.0, 0.0, 0.0))

    handle = builder.add_camera(entry, name="overview")

    assert handle.name == "overview"
    with pytest.raises(SimulationConfigError, match="already declared"):
        builder.add_camera(entry, name="overview")


def test_add_camera_requires_a_name(tmp_path: Path) -> None:
    """A camera entry without a name cannot key the camera mapping."""
    pack_set, _ = write_sim_pack(tmp_path)

    with pytest.raises(SimulationConfigError, match="non-empty 'name'"):
        _builder(pack_set).add_camera({"rig": {"type": "world"}})


def test_add_articulation_requires_a_robot_profile(tmp_path: Path) -> None:
    """The builder consumes a complete RobotProfile, never an opaque handle."""
    pack_set, _ref = write_sim_pack(tmp_path)

    with pytest.raises(SimulationConfigError, match="expects a RobotProfile"):
        _builder(pack_set).add_articulation(object(), control=_control())


def test_body_attach_is_deferred(tmp_path: Path) -> None:
    """A canonical body attach names the supported native-mount form."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    body = builder.add_articulation(ref, control=_control()).body("base")
    entry = Camera.world("top", pos=(1.0, 0.0, 2.0), lookat=(0.0, 0.0, 0.0))

    with pytest.raises(DeferredFeatureError, match="mount=<native link name>"):
        builder.add_camera(entry, attach=body)


def test_build_without_world_is_refused_without_sealing(tmp_path: Path) -> None:
    """The live transition never half-builds; the builder stays declarable."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    builder.add_articulation(ref, control=_control())

    with pytest.raises(NoWorldError, match="builder.world"):
        builder.build()

    builder.world = World.base(mjcf="world.xml")
    assert builder.world is not None


def test_fixture_pack_set_is_explicit(tmp_path: Path) -> None:
    """The fixture pack set exposes exactly the written pack and profile."""
    pack_set, profile = write_sim_pack(tmp_path)

    assert any(
        profile.name == declared.name for pack in pack_set.packs() for declared in pack.robots()
    )
    assert isinstance(profile, RobotProfile)
    assert profile.assets is None
