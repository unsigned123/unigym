"""``SimulationBuilder`` config assembly into the existing run-config schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from unigym.camera import Camera
from unigym.packs.packset import PackSet
from unigym.run.api import Run
from unigym.run.spec import RunSpec
from unigym.scenes import Scene
from unigym.scenes.terrain import Terrain
from unigym.sim import (
    Control,
    ControlSpec,
    DeferredFeatureError,
    NoWorldError,
    SimulationBuilder,
    SimulationConfigError,
)
from unigym.world import Box, World
from tests._support.sim_fixtures import write_sim_pack


def _builder(pack_set: PackSet) -> SimulationBuilder:
    """A valid builder with a real world asset and one body control."""
    builder = SimulationBuilder(
        backend="mujoco",
        dt=0.02,
        physics_dt=0.005,
        pack_set=pack_set,
    )
    builder.world = World.base(mjcf="world.xml")
    return builder


def _body_control() -> ControlSpec:
    """The canonical one-face body preset (shoulder, elbow)."""
    return Control.position_pd(joint_set="body", kp=10.0, kd=0.5)


def test_to_cfg_emits_the_base_config_keys(tmp_path: Path) -> None:
    """The builder owns the run-name, env arithmetic and the manual adapter."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    builder.add_articulation(ref, control=_body_control())

    cfg = builder._to_cfg()

    assert cfg["run_name"] == "simulation"
    assert cfg["output_root"] == "data_out"
    assert cfg["seed"] == 0
    assert cfg["mode"] == "collect"
    assert cfg["adapter"] == {"name": "manual"}
    assert cfg["command"] == {"name": "fixed", "dim": 0, "values": []}
    assert cfg["collect"] == {"steps": 1000, "shard_steps": 1000}
    assert cfg["runner"] == {"auto_reset": False}
    assert cfg["env"] == {
        "backend": "mujoco",
        "num_envs": 1,
        "episode_length_s": 20.0,
        "decimation": 4,
        "sim_dt": 0.005,
        "params": {},
    }


def test_to_cfg_emits_explicit_articulation_and_control_tables(tmp_path: Path) -> None:
    """Faces become one standard_pd actuation group and one explicit control group."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    builder.add_articulation(ref, control=_body_control())

    cfg = builder._to_cfg()

    assert cfg["articulations"] == [
        {
            "name": ref.id,
            "actuation": [
                {
                    "name": "body",
                    "law": "standard_pd",
                    "joints": ["shoulder", "elbow"],
                    "gains": {"kp": 10.0, "kd": 0.5},
                }
            ],
            "passive_joints": ["finger"],
        }
    ]
    assert cfg["controls"] == [
        {
            "name": "body",
            "adapter": "manual",
            "articulation": ref.id,
            "joints": ["shoulder", "elbow"],
            "drive": "position_pd",
            "target": "position",
        }
    ]


def test_explicit_joints_keep_declaration_order(tmp_path: Path) -> None:
    """An explicit-joint face emits its joints without resolving a joint_set."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    control = Control.position_pd(joints=["elbow", "shoulder"], face="arm", kp=1.0, kd=0.1)
    builder.add_articulation(ref, control=control)

    cfg = builder._to_cfg()

    assert cfg["articulations"][0]["actuation"][0]["joints"] == ["elbow", "shoulder"]
    assert cfg["controls"][0]["joints"] == ["elbow", "shoulder"]
    assert cfg["controls"][0]["name"] == "arm"


def test_q0_and_limits_are_emitted_when_declared(tmp_path: Path) -> None:
    """Explicit q0 overrides and limits ride their declared tables."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    control = Control.position_pd(
        joint_set="body",
        q0={"shoulder": 0.25},
        kp=10.0,
        kd=0.5,
        effort_limit=100.0,
        velocity_limit=5.0,
    )
    builder.add_articulation(ref, control=control)

    cfg = builder._to_cfg()

    actuation = cfg["articulations"][0]["actuation"][0]
    assert actuation["effort_limit"] == 100.0
    assert actuation["velocity_limit"] == 5.0
    assert cfg["controls"][0]["q0"] == {"shoulder": 0.25}


def test_per_joint_gains_and_limits_keep_their_maps(tmp_path: Path) -> None:
    """A per-joint gain/limit map is emitted as a JSON mapping."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    control = Control.position_pd(
        joint_set="body",
        kp={"shoulder": 10.0, "elbow": 12.0},
        kd=0.5,
        effort_limit={"shoulder": 100.0, "elbow": 110.0},
    )
    builder.add_articulation(ref, control=control)

    cfg = builder._to_cfg()

    actuation = cfg["articulations"][0]["actuation"][0]
    assert actuation["gains"]["kp"] == {"shoulder": 10.0, "elbow": 12.0}
    assert actuation["effort_limit"] == {"shoulder": 100.0, "elbow": 110.0}


def test_passive_joints_are_the_profile_union_minus_driven(tmp_path: Path) -> None:
    """The profile's undriven joints become passive_joints."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    builder.add_articulation(ref, control=_body_control())

    cfg = builder._to_cfg()

    assert cfg["articulations"][0]["passive_joints"] == ["finger"]


def test_a_renamed_articulation_keeps_its_profile_selector(tmp_path: Path) -> None:
    """A custom label emits the existing params.profile selector explicitly."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    builder.add_articulation(ref, name="arm", control=_body_control())

    cfg = builder._to_cfg()

    assert cfg["articulations"][0] == {
        "name": "arm",
        "params": {"profile": ref.id},
        "actuation": [
            {
                "name": "body",
                "law": "standard_pd",
                "joints": ["shoulder", "elbow"],
                "gains": {"kp": 10.0, "kd": 0.5},
            }
        ],
        "passive_joints": ["finger"],
    }
    assert cfg["controls"][0]["articulation"] == "arm"


def test_cameras_are_deep_copied_declared_entries(tmp_path: Path) -> None:
    """Declared cameras are emitted verbatim and isolated from later mutation."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    builder.add_articulation(ref, control=_body_control())
    entry = Camera.world("top", pos=(1.0, 0.0, 2.0), lookat=(0.0, 0.0, 0.0))
    builder.add_camera(entry)

    cfg = builder._to_cfg()

    assert cfg["cameras"] == [entry]
    assert cfg["cameras"] is not entry
    assert cfg["cameras"][0] is not entry


def test_two_envs_are_emitted_with_the_no_reset_policy(tmp_path: Path) -> None:
    """``num_envs=2`` reaches ``env.num_envs`` and the runner never auto-resets."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = SimulationBuilder(
        backend="mujoco",
        dt=0.02,
        physics_dt=0.005,
        num_envs=2,
        pack_set=pack_set,
    )
    builder.world = World.base(mjcf="world.xml")
    builder.add_articulation(ref, control=_body_control())

    cfg = builder._to_cfg()

    assert cfg["env"]["num_envs"] == 2
    assert cfg["runner"] == {"auto_reset": False}


def test_cameras_at_two_envs_are_refused_before_the_engine_is_built(tmp_path: Path) -> None:
    """Declared camera frames at ``N > 1`` are a deferred feature at the boundary."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = SimulationBuilder(
        backend="mujoco",
        dt=0.02,
        physics_dt=0.005,
        num_envs=2,
        pack_set=pack_set,
    )
    builder.world = World.base(mjcf="world.xml")
    builder.add_articulation(ref, control=_body_control())
    builder.add_camera(Camera.world("top", pos=(1.0, 0.0, 2.0), lookat=(0.0, 0.0, 0.0)))

    with pytest.raises(DeferredFeatureError, match="camera frames at N > 1"):
        builder.build()

    assert builder._sim is None


def test_one_env_keeps_the_camera_table(tmp_path: Path) -> None:
    """Cameras at ``N == 1`` keep working exactly as in the first spike."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    builder.add_articulation(ref, control=_body_control())
    builder.add_camera(Camera.world("top", pos=(1.0, 0.0, 2.0), lookat=(0.0, 0.0, 0.0)))

    cfg = builder._to_cfg()

    assert cfg["env"]["num_envs"] == 1
    assert [entry["name"] for entry in cfg["cameras"]] == ["top"]


def _spec_for(builder: SimulationBuilder) -> RunSpec:
    """Normalize the builder's declarations through the one ``Run`` seam."""
    return Run(
        builder._to_cfg(),
        world=builder.world,
        pack_runtime=builder.pack_runtime,
        discover_runtime=False,
        no_artifacts=True,
    ).spec


def test_world_fragment_is_merged_by_run_once(tmp_path: Path) -> None:
    """The builder emits declarations only; ``Run(cfg, world=...)`` merges the world."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    builder.world = World.base(mjcf="world.xml")
    builder.add_articulation(ref, control=_body_control())

    cfg = builder._to_cfg()

    assert "world_base" not in cfg
    spec = _spec_for(builder)
    assert spec.world_base == {"mjcf": "world.xml"}
    assert [entry.name for entry in spec.articulations] == [ref.id]
    assert [entry.name for entry in spec.controls] == ["body"]


def test_inline_terrain_reaches_the_run_spec_through_the_world_handoff(tmp_path: Path) -> None:
    """An inline terrain world reaches ``spec.terrain`` via the one Run merge."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = SimulationBuilder(backend="isaaclab2", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    builder.world = World.inline(Scene(name="hill", terrain=Terrain(size=(4.0, 4.0))))
    builder.add_articulation(ref, control=_body_control())

    cfg = builder._to_cfg()

    assert "scene_preset" not in cfg
    spec = _spec_for(builder)
    assert isinstance(spec.terrain, Terrain)
    assert spec.terrain.size == (4.0, 4.0)


def test_world_objects_are_merged_exactly_once(tmp_path: Path) -> None:
    """A dynamic world object joins ``objects`` once, with the builder's articulation."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    world = World.base(mjcf="world.xml")
    world.add(Box((0.2, 0.2, 0.2), name="crate", pos=(0.0, 0.0, 0.1)), dynamic=True)
    builder.world = world
    builder.add_articulation(ref, control=_body_control())

    spec = _spec_for(builder)

    assert [entry.name for entry in spec.articulations] == [ref.id]
    props = spec.props
    assert isinstance(props, dict)
    types = props["types"]
    assert isinstance(types, list)
    assert [entry["name"] for entry in types] == ["crate"]


def test_missing_world_is_refused(tmp_path: Path) -> None:
    """There is no implicit world."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    builder.add_articulation(ref, control=_body_control())

    with pytest.raises(NoWorldError, match="builder.world"):
        builder._to_cfg()


def test_empty_world_fragment_is_refused(tmp_path: Path) -> None:
    """A world that declares no asset cannot become a run config."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    builder.world = World()
    builder.add_articulation(ref, control=_body_control())

    with pytest.raises(NoWorldError, match="no world asset"):
        builder._to_cfg()


def test_world_fragment_may_not_own_articulations_or_controls(tmp_path: Path) -> None:
    """The builder is the single owner of the articulation/control tables."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    world = World.base(mjcf="world.xml")
    world.articulation("other", model={"mjcf": "other.xml"})
    builder.world = world
    builder.add_articulation(ref, control=_body_control())

    with pytest.raises(SimulationConfigError, match="must not declare"):
        builder._to_cfg()


def test_at_least_one_articulation_is_required(tmp_path: Path) -> None:
    """A world without an articulation is not a simulation."""
    pack_set, _ = write_sim_pack(tmp_path)
    builder = _builder(pack_set)

    with pytest.raises(SimulationConfigError, match="at least one articulation"):
        builder._to_cfg()


def test_two_articulations_may_not_share_a_default_face_name(tmp_path: Path) -> None:
    """A duplicate default face is refused at the declaration boundary."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    builder.add_articulation(ref, name="a", control=_body_control())

    with pytest.raises(SimulationConfigError) as excinfo:
        builder.add_articulation(ref, name="b", control=_body_control())

    message = str(excinfo.value)
    assert "'body'" in message
    assert "'a'" in message
    assert "'b'" in message
    assert "declare a distinct face=" in message
    assert tuple(handle.name for handle in builder.articulations) == ("a",)


def test_two_articulations_with_distinct_faces_emit_unique_control_names(
    tmp_path: Path,
) -> None:
    """An explicit face= per articulation keeps the global control table unique."""
    pack_set, ref = write_sim_pack(tmp_path)
    builder = _builder(pack_set)
    builder.add_articulation(
        ref,
        name="a",
        control=Control.position_pd(joint_set="body", face="a", kp=10.0, kd=0.5),
    )
    builder.add_articulation(
        ref,
        name="b",
        control=Control.position_pd(joint_set="body", face="b", kp=10.0, kd=0.5),
    )

    cfg = builder._to_cfg()

    assert [entry["name"] for entry in cfg["articulations"]] == ["a", "b"]
    assert [entry["name"] for entry in cfg["controls"]] == ["a", "b"]
