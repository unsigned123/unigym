"""Config-mode builders refuse declarations and read one normalized RunSpec."""

from __future__ import annotations

from pathlib import Path

import pytest

from unigym.camera import Camera
from unigym.sim import (
    ConfigModeError,
    Control,
    SimulationBuilder,
    SimulationConfigError,
)
from unigym.world import World
from tests._support.sim_fixtures import write_sim_pack


def _config_text(*, include_timing: bool = True) -> str:
    """A minimal valid run config; timing keys are omitted on request."""
    env = "env: {backend: mujoco, num_envs: 1, episode_length_s: 20.0"
    if include_timing:
        env += ", decimation: 4, sim_dt: 0.005"
    env += "}"
    return "\n".join(
        [
            "run_name: loaded",
            "seed: 3",
            env,
            "world_base: {mjcf: world.xml}",
            "collect: {steps: 12}",
            "articulations: [{name: arm, joints: [{name: shoulder}]}]",
            "action_spec:",
            "  - {name: arm, articulation: arm, joints: [shoulder],",
            "     target: position, role: body}",
            "policies:",
            "  - {adapter: manual, groups: [[arm, arm]]}",
            "",
        ]
    )


def _config_builder(tmp_path: Path) -> SimulationBuilder:
    """A config-mode builder over one valid yaml document."""
    path = tmp_path / "run.yaml"
    path.write_text(_config_text(), encoding="utf-8")
    return SimulationBuilder.from_config(path)


@pytest.mark.parametrize("operation", ["world", "articulation", "camera"])
def test_config_mode_declaration_is_refused(tmp_path: Path, operation: str) -> None:
    """Every declaration-mode operation raises ConfigModeError before mutation."""
    builder = _config_builder(tmp_path)
    _, ref = write_sim_pack(tmp_path / "pack")

    if operation == "world":
        with pytest.raises(ConfigModeError, match="from_config"):
            builder.world = World.base(mjcf="other.xml")
    elif operation == "articulation":
        with pytest.raises(ConfigModeError, match="add_articulation"):
            builder.add_articulation(
                ref,
                control=Control.position_pd(joint_set="body", kp=1.0, kd=0.1),
            )
    else:
        with pytest.raises(ConfigModeError, match="add_camera"):
            builder.add_camera(Camera.world("top", pos=(1.0, 0.0, 2.0), lookat=(0.0, 0.0, 0.0)))

    assert builder.world is None
    assert builder.articulations == ()
    assert builder.cameras == ()


def test_from_config_requires_the_explicit_engine_timing(tmp_path: Path) -> None:
    """dt and physics_dt come from env.sim_dt/decimation, never a default."""
    path = tmp_path / "run.yaml"
    path.write_text(_config_text(include_timing=False), encoding="utf-8")

    with pytest.raises(SimulationConfigError, match="env.sim_dt"):
        SimulationBuilder.from_config(path)


def test_config_mode_without_pack_set_has_no_pack_runtime(tmp_path: Path) -> None:
    """A pack-less config-mode builder refuses an implicit runtime."""
    builder = _config_builder(tmp_path)

    with pytest.raises(SimulationConfigError, match="pass pack_set="):
        _ = builder.pack_runtime


def test_from_config_needs_at_least_one_manual_control_group(tmp_path: Path) -> None:
    """A non-manual adapter yields no external action table, so it is refused."""
    path = tmp_path / "run.yaml"
    path.write_text(
        _config_text().replace("adapter: manual", "adapter: policy"),
        encoding="utf-8",
    )

    with pytest.raises(SimulationConfigError, match="manual policy"):
        SimulationBuilder.from_config(path)


def test_from_config_release_asset_does_not_require_a_pack_set(tmp_path: Path) -> None:
    """A ``release:`` asset resolves through its provider, not a pack runtime."""
    text = _config_text().replace(
        "articulations: [{name: arm, joints: [{name: shoulder}]}]",
        "articulations: [{name: arm, model: {mjcf: 'release:arm-mjcf'}, "
        "joints: [{name: shoulder}]}]",
    )
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")

    builder = SimulationBuilder.from_config(path)

    assert builder.world is None
