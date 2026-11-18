"""Backend-level typed observations: sites, multi-art views and snapshots."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unigym.env.obs import ObservationConfig
from unigym.runtime.actions import NamedAction


def _site_mjcf(tiny_mjcf: Path) -> Path:
    """Add one named site under the tip body of the shared fixture."""
    target = tiny_mjcf.with_name("tiny_arm_site.xml")
    text = tiny_mjcf.read_text(encoding="utf-8")
    text = text.replace(
        '<geom name="tip_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03" mass="0.5"/>',
        '<geom name="tip_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03" mass="0.5"/>'
        '<site name="grip" pos="0 0 -0.1"/>',
    )
    target.write_text(text, encoding="utf-8")
    return target


def test_selected_site_is_published_in_the_record(tiny_env_factory, tiny_mjcf):
    env = tiny_env_factory(
        mjcf=_site_mjcf(tiny_mjcf),
        observations=ObservationConfig(robot=("tiny",), site=("grip",)),
    )
    try:
        step = env.step(
            (NamedAction("tiny", torch.zeros(1, 2), ("elbow", "shoulder"), "position"),)
        )
        assert "site/grip/pos" in step.record
        assert tuple(step.record["site/grip/pos"].shape) == (1, 3)
        assert step.obs.site("grip").pos_w.shape == (1, 3)
    finally:
        env.close()


def test_full_joint_velocity_is_typed_while_policy_leaf_stays_subset(tiny_env_factory):
    """Passive joints have a typed velocity; the artifact leaf stays policy-only."""
    env = tiny_env_factory(control_joints=("shoulder",))
    try:
        full_names = env.articulations["tiny"].joint_names
        velocity = torch.tensor(
            [[float(index + 1) for index in range(len(full_names))]],
            dtype=torch.float32,
        )
        env.set_joint_state(
            None,
            articulation="tiny",
            joint_names=full_names,
            position=torch.zeros_like(velocity),
            velocity=velocity,
        )
        env.settle(None)
        robot = env.obs.snapshot().robot("tiny")
        shoulder = full_names.index("shoulder")

        assert robot.policy_joint_names == ("shoulder",)
        assert robot.joint_names == full_names
        assert robot.policy_joint_names != robot.joint_names
        assert robot.joint_vel.shape == (1, 1)
        assert robot.joint_vel_full.shape == (1, len(full_names))
        assert robot.joint_vel_full.tolist() == velocity.tolist()
        assert robot.joint_vel.tolist() == [[velocity[0, shoulder].item()]]
        assert torch.equal(robot.joints(("shoulder",), leaf="joint_vel_full"), robot.joint_vel)
        record = env.record()
        assert "articulation/tiny/joint_vel" in record
        assert "articulation/tiny/joint_vel_full" not in record
    finally:
        env.close()


def test_unknown_site_fails_at_build(tiny_env_factory, tiny_mjcf):
    with pytest.raises(ValueError, match="observations.site 'missing'"):
        tiny_env_factory(
            mjcf=tiny_mjcf,
            observations=ObservationConfig(robot=("tiny",), site=("missing",)),
        )


def test_rewind_restores_the_typed_observation_registry(tiny_env_factory):
    env = tiny_env_factory()
    try:
        action = NamedAction("tiny", torch.zeros(1, 2), ("elbow", "shoulder"), "position")
        env.step((action,))
        engine_snapshot = env.snapshot()
        obs_snapshot = env.obs.rewind_state()
        before = env.obs.robot("tiny").joint_pos_rel.clone()

        env.step((action,))
        env.restore(engine_snapshot)
        env.obs.restore_rewind_state(obs_snapshot)

        assert torch.equal(env.obs.robot("tiny").joint_pos_rel, before)
    finally:
        env.close()
