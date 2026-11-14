"""Declaration-only ``SimulationBuilder`` contract over an explicit PackSet."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from unigym.camera import Camera
from unigym.packs.packset import PackSet, RobotRef
from unigym.sim import (
    Control,
    ControlSpec,
    DeferredFeatureError,
    ForeignHandleError,
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
    """The canonical one-face body control used by most contract cases."""
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


def test_add_articulation_resolves_faces_and_owns_the_handle(tmp_path: Path) -> None:
    """The handle carries the ref, profile, control and resolved face joints."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    control = _control()

    handle = builder.add_articulation(ref, name="arm", control=control)

    assert builder.articulations == (handle,)
    assert handle.name == "arm"
    assert handle.ref == ref
    assert handle.control is control
    assert handle.profile.name == ref.id
    assert handle.faces == {"body": ("shoulder", "elbow")}


def test_add_articulation_accepts_explicit_joints(tmp_path: Path) -> None:
    """Explicit face joints are kept in declaration order."""
    pack_set, ref = write_sim_pack(tmp_path)
    control = Control.position_pd(joints=["elbow", "shoulder"], face="arm", kp=1.0, kd=0.1)

    handle = _builder(pack_set).add_articulation(ref, control=control)

    assert handle.faces == {"arm": ("elbow", "shoulder")}


def test_add_articulation_rejects_duplicate_labels(tmp_path: Path) -> None:
    """Two articulations cannot share one label."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    control = _control()
    builder.add_articulation(ref, control=control)

    with pytest.raises(SimulationConfigError, match="already declared"):
        builder.add_articulation(ref, control=control)


def test_add_articulation_rejects_an_undeclared_joint_set(tmp_path: Path) -> None:
    """A control face may only select a profile joint set."""
    pack_set, ref = write_sim_pack(tmp_path)
    control = Control.position_pd(joint_set="legs", kp=1.0, kd=0.1)

    with pytest.raises(SimulationConfigError, match="does not declare"):
        _builder(pack_set).add_articulation(ref, control=control)


def test_add_articulation_rejects_an_unknown_explicit_joint(tmp_path: Path) -> None:
    """An explicit face joint must exist on the profile."""
    pack_set, ref = write_sim_pack(tmp_path)
    control = Control.position_pd(joints=["wheel"], face="arm", kp=1.0, kd=0.1)

    with pytest.raises(SimulationConfigError, match="'wheel'"):
        _builder(pack_set).add_articulation(ref, control=control)


def test_add_articulation_requires_a_q0_for_every_face_joint(tmp_path: Path) -> None:
    """A joint without a control override or a profile default has no q0."""
    pack_set, ref = write_sim_pack(tmp_path, default_joint_pos={"shoulder": 0.0, "finger": 0.0})

    with pytest.raises(SimulationConfigError, match="no q0 for joints \\['elbow'\\]"):
        _builder(pack_set).add_articulation(ref, control=_control())


def test_add_articulation_rejects_overlapping_faces(tmp_path: Path) -> None:
    """Faces of one articulation must be disjoint."""
    pack_set, ref = write_sim_pack(tmp_path)
    left = Control.position_pd(joints=["shoulder"], face="left", kp=1.0, kd=0.1)
    right = Control.position_pd(joints=["shoulder"], face="right", kp=1.0, kd=0.1)
    control = ControlSpec.from_faces((left.faces["left"], right.faces["right"]))

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


def test_foreign_pack_ref_is_refused_before_loading(tmp_path: Path) -> None:
    """A ref from another pack scope is a foreign handle, not a pack error."""
    pack_set, ref = write_sim_pack(tmp_path)

    with pytest.raises(ForeignHandleError, match="not in this pack set"):
        _builder(pack_set).add_articulation(replace(ref, pack="other_pack"), control=_control())


def test_mismatched_pack_version_is_refused(tmp_path: Path) -> None:
    """The ref version must match the set's manifest version."""
    pack_set, ref = write_sim_pack(tmp_path)

    with pytest.raises(ForeignHandleError, match="version"):
        _builder(pack_set).add_articulation(replace(ref, pack_version="9.9.9"), control=_control())


def test_body_attach_is_deferred(tmp_path: Path) -> None:
    """A canonical body attach names the supported native-mount form."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    body = builder.add_articulation(ref, control=_control()).body("base")
    entry = Camera.world("top", pos=(1.0, 0.0, 2.0), lookat=(0.0, 0.0, 0.0))

    with pytest.raises(DeferredFeatureError, match="mount=<native link name>"):
        builder.add_camera(entry, attach=body)


def test_build_is_not_implemented_in_this_slice(tmp_path: Path) -> None:
    """The live transition belongs to the next slice and never half-builds."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    builder.add_articulation(ref, control=_control())

    with pytest.raises(NotImplementedError, match="declaration-only"):
        builder.build()


def test_fixture_pack_set_is_explicit(tmp_path: Path) -> None:
    """The fixture pack set exposes exactly the written pack and ref."""
    pack_set, ref = write_sim_pack(tmp_path)

    assert [pack.name for pack in pack_set.packs()] == [ref.pack]
    assert isinstance(ref, RobotRef)
    assert ref.is_data is True
