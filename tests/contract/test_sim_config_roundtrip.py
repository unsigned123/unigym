"""``to_config`` / ``from_config`` preserve the authoring tables, not handles."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest
import yaml

from unigym.errors import ConfigError
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.runner.state import read_extra
from unigym.sim import ArticulationHandle, Simulation, SimulationBuilder
from unigym.world import World
from tests._support.fakes.runner import FakeRunner
from tests._support.sim_fixtures import write_sim_pack

_CONFIG = {
    "run_name": "roundtrip",
    "seed": 7,
    "env": {
        "backend": "mujoco",
        "num_envs": 1,
        "episode_length_s": 20.0,
        "decimation": 4,
        "sim_dt": 0.005,
    },
    "world_base": {"mjcf": "world.xml"},
    "collect": {"steps": 10},
    "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
    "action_spec": [
        {
            "name": "arm",
            "articulation": "arm",
            "joints": ["shoulder"],
            "target": "position",
            "role": "body",
        }
    ],
    "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
    "cameras": [{"name": "top", "rig": {"type": "world"}}],
}


def _write_config(tmp_path: Path) -> Path:
    """Write the authoring config the round-trip starts from."""
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(_CONFIG, sort_keys=False), encoding="utf-8")
    return path


def _handle(owner: object) -> ArticulationHandle:
    """One profile-backed handle for the direct Simulation seam."""
    profile = RobotProfile(
        name="arm",
        joint_sets={"body": ("shoulder",)},
        default_joint_pos={"shoulder": 0.0},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    return ArticulationHandle(
        name="arm",
        ref=None,
        profile=profile,
        control=None,
        faces={},
        _owner=owner,
    )


def _simulation(source: Path, *, num_envs: int = 1) -> Simulation:
    """A pure Simulation seam over the source Run (no engine is started)."""
    run = Run.from_config(source)
    runner = FakeRunner(num_envs=num_envs)
    owner = object()
    return Simulation(
        backend="mujoco",
        num_envs=num_envs,
        world=World.from_config(yaml.safe_load(source.read_text(encoding="utf-8"))),
        articulations=(_handle(owner),),
        cameras=(),
        runner=runner,
        env=runner.env,
        run=run,
        owner=owner,
        state=read_extra(runner.env),
        obs_cache=dict(runner.env.obs_cache),
        t=0,
    )


def _tables(spec) -> tuple[object, ...]:
    """The semantic tables the round-trip must preserve."""
    return (
        spec.run_name,
        tuple(item.name for item in spec.articulations),
        tuple(
            (group.name, group.articulation, tuple(group.joints), group.target, group.role)
            for group in spec.action_spec.groups
        ),
        tuple((binding.adapter, tuple(binding.groups)) for binding in spec.policies),
        spec.cameras,
        spec.world_base,
    )


def test_to_config_round_trips_the_authoring_tables(tmp_path: Path) -> None:
    """The emitted yaml reloads to the same labels, controls, cameras and world."""
    source = _write_config(tmp_path)
    sim = _simulation(source)
    try:
        written = sim.to_config(tmp_path / "written.yaml")
        again = sim.to_config(tmp_path / "written-again.yaml")
    finally:
        sim.close()

    assert Path(written).read_text(encoding="utf-8") == Path(again).read_text(encoding="utf-8")
    original = Run.from_config(source).spec
    reloaded = Run.from_config(written).spec
    assert _tables(reloaded) == _tables(original)

    builder = SimulationBuilder.from_config(written)
    assert builder.articulations == ()
    assert builder.cameras == ()


def test_multienv_and_reset_policy_round_trip(tmp_path: Path) -> None:
    """num_envs=2 and runner.auto_reset: false survive to_config -> from_config."""
    cfg = {key: value for key, value in _CONFIG.items() if key != "cameras"}
    cfg["env"] = {**_CONFIG["env"], "num_envs": 2}
    cfg["runner"] = {"auto_reset": False}
    source = tmp_path / "multienv.yaml"
    source.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    sim = _simulation(source, num_envs=2)
    try:
        written = sim.to_config(tmp_path / "multienv-written.yaml")
    finally:
        sim.close()

    document = yaml.safe_load(Path(written).read_text(encoding="utf-8"))
    assert document["env"]["num_envs"] == 2
    assert document["runner"] == {"auto_reset": False}
    reloaded = Run.from_config(written).spec
    assert reloaded.num_envs == 2
    assert reloaded.runner == {"auto_reset": False}


def test_to_config_serializes_no_facade_handles(tmp_path: Path) -> None:
    """The authoring document carries tables only, never handle objects."""
    source = _write_config(tmp_path)
    sim = _simulation(source)
    try:
        written = sim.to_config(tmp_path / "written.yaml")
    finally:
        sim.close()

    text = Path(written).read_text(encoding="utf-8")
    document = yaml.safe_load(text)

    assert "ArticulationHandle" not in text
    assert "_owner" not in text
    assert "handles" not in document
    assert document["articulations"] == _CONFIG["articulations"]
    assert document["action_spec"] == _CONFIG["action_spec"]
    assert document["policies"] == _CONFIG["policies"]
    assert document["cameras"] == _CONFIG["cameras"]
    assert document["world_base"] == _CONFIG["world_base"]


def _forbidden_discovery(*args: object, **kwargs: object) -> NoReturn:
    """Fail the test if the ambient discovery path is touched."""
    raise AssertionError("SimulationBuilder.from_config must not discover")


def test_from_config_without_pack_set_refuses_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pack-needing config names the explicit facade seam, never discovery."""
    cfg = {
        "run_name": "no-pack",
        "seed": 0,
        "env": {
            "backend": "mujoco",
            "num_envs": 1,
            "episode_length_s": 10.0,
            "decimation": 4,
            "sim_dt": 0.005,
        },
        "collect": {"steps": 1},
        "scene_preset": "missing.preset",
    }
    path = tmp_path / "no-pack.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr("unigym.pack_api.discover", _forbidden_discovery)

    with pytest.raises(ConfigError, match="pack_set=.*pack_runtime="):
        SimulationBuilder.from_config(path)


def test_from_config_uses_the_explicit_pack_set(tmp_path: Path) -> None:
    """A pack-referencing config is loaded through the caller's explicit pack set."""
    pack_set, ref = write_sim_pack(tmp_path / "pack")
    cfg = dict(_CONFIG)
    cfg["articulations"] = [{"name": "body", "params": {"profile": ref.id}}]
    cfg["action_spec"] = [
        {
            "name": "body",
            "articulation": "body",
            "joints": ["shoulder", "elbow"],
            "target": "position",
            "role": "body",
        }
    ]
    cfg["policies"] = [{"adapter": "manual", "groups": [["body", "body"]]}]
    path = tmp_path / "pack.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    builder = SimulationBuilder.from_config(path, pack_set=pack_set)

    assert builder.pack_runtime is pack_set.to_runtime()
