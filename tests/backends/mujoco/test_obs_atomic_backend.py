"""Backend-level typed observations: sites, multi-art views and snapshots."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unigym.env.obs import ObservationConfig
from unigym.sites import SiteSpec
from unigym.props.asset_ir import AssetDefinition, Body, Geom, Site
from unigym.props.templates import (
    PropTemplateResult,
    register_prop_template,
    restore_prop_templates,
    snapshot_prop_templates,
)
from unigym.runtime.actions import NamedAction

_HALF = 2**-0.5


def _site_mjcf(tiny_mjcf: Path) -> Path:
    """Add one oriented site under the tip body of the shared fixture."""
    target = tiny_mjcf.with_name("tiny_arm_site.xml")
    text = tiny_mjcf.read_text(encoding="utf-8")
    text = text.replace(
        '<geom name="tip_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03" mass="0.5"/>',
        '<geom name="tip_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03" mass="0.5"/>'
        f'<site name="grip" pos="0 0 -0.1" quat="{_HALF} 0 0 {_HALF}"/>',
    )
    target.write_text(text, encoding="utf-8")
    return target


def test_selected_native_site_is_published_as_a_full_pose(tiny_env_factory, tiny_mjcf):
    env = tiny_env_factory(
        mjcf=_site_mjcf(tiny_mjcf),
        observations=ObservationConfig(robot=("tiny",), site=("grip",)),
    )
    try:
        step = env.step(
            (NamedAction("tiny", torch.zeros(1, 2), ("elbow", "shoulder"), "position"),)
        )
        assert "site/grip/pos" in step.record
        assert "site/grip/quat" in step.record
        assert tuple(step.record["site/grip/pos"].shape) == (1, 3)
        assert tuple(step.record["site/grip/quat"].shape) == (1, 4)
        assert step.obs.site("grip").pos_w.shape == (1, 3)
        assert step.obs.site("grip").quat_w[0].tolist() == pytest.approx(
            [_HALF, 0.0, 0.0, _HALF], abs=1e-6
        )
    finally:
        env.close()


def test_robot_profile_site_uses_body_offset_without_a_native_site(tiny_env_factory, tiny_mjcf):
    site = SiteSpec(
        "tcp",
        "base",
        pos=(0.1, 0.2, 0.3),
        quat_wxyz=(_HALF, 0.0, 0.0, _HALF),
    )
    env = tiny_env_factory(
        mjcf=tiny_mjcf,
        sites={"tcp": site},
        observations=ObservationConfig(robot=("tiny",), site=("tcp",)),
    )
    try:
        step = env.step(
            (NamedAction("tiny", torch.zeros(1, 2), ("elbow", "shoulder"), "position"),)
        )
        base_pos = step.obs.robot("tiny").base_pos_w
        offset = step.obs.site("tcp").pos_w - base_pos
        assert offset[0].tolist() == pytest.approx([0.1, 0.2, 0.3], abs=2e-3)
        assert step.obs.site("tcp").quat_w[0].tolist() == pytest.approx(
            [_HALF, 0.0, 0.0, _HALF], abs=1e-6
        )
    finally:
        env.close()


def test_native_site_disagreeing_with_the_profile_spec_fails_at_build(tiny_env_factory, tiny_mjcf):
    site = SiteSpec("grip", "base", pos=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="disagrees with articulation"):
        tiny_env_factory(
            mjcf=_site_mjcf(tiny_mjcf),
            sites={"grip": site},
            observations=ObservationConfig(robot=("tiny",), site=("grip",)),
        )


def test_generated_prop_site_reads_the_native_mjcf_pose(tiny_env_factory, tiny_mjcf):
    templates = snapshot_prop_templates()
    asset = AssetDefinition(
        name="cup",
        root=Body(
            "cup",
            geoms=(Geom("cup_geom", "box", (0.03, 0.03, 0.03), mass=0.1),),
            sites=(Site("tcp", pos=(0.0, 0.0, 0.05), quat=(_HALF, 0.0, 0.0, _HALF)),),
        ),
    )

    def provider(name, params):
        """Return one generated free prop carrying an oriented site."""
        del params
        return PropTemplateResult(
            spec={"name": name, "type": "free_articulation", "joints": {}},
            asset=asset,
        )

    register_prop_template(
        "site_prop",
        params={"unused": {"type": "string", "required": False}},
    )(provider)
    try:
        from unigym.props.base import PropManager

        plan = SimpleNamespace(context=None, resolve_definition=lambda value: value)
        manager = PropManager(
            {"types": [{"name": "cup", "template": "site_prop", "params": {}}]},
            num_envs=1,
            env_spacing=4.0,
            device="cpu",
            backend="mujoco",
            asset_plan=plan,
        )
        env = tiny_env_factory(
            mjcf=tiny_mjcf,
            prop_manager=manager,
            observations=ObservationConfig(robot=("tiny",), site=("tcp",)),
        )
        try:
            step = env.step(
                (NamedAction("tiny", torch.zeros(1, 2), ("elbow", "shoulder"), "position"),)
            )
            assert step.obs.site("tcp").quat_w[0].tolist() == pytest.approx(
                [_HALF, 0.0, 0.0, _HALF], abs=1e-6
            )
            assert tuple(step.record["site/tcp/pos"].shape) == (1, 3)
        finally:
            env.close()
    finally:
        restore_prop_templates(templates)


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


def test_native_site_publishes_batched_poses(tiny_env_factory, tiny_mjcf):
    """N>1 envs publish one site pose row per env."""
    env = tiny_env_factory(
        num_envs=2,
        mjcf=_site_mjcf(tiny_mjcf),
        observations=ObservationConfig(robot=("tiny",), site=("grip",)),
    )
    try:
        step = env.step(
            (NamedAction("tiny", torch.zeros(2, 2), ("elbow", "shoulder"), "position"),)
        )
        assert tuple(step.record["site/grip/pos"].shape) == (2, 3)
        assert tuple(step.record["site/grip/quat"].shape) == (2, 4)
        assert step.obs.site("grip").quat_w.shape == (2, 4)
    finally:
        env.close()


def test_empty_site_selection_does_not_build_the_asset_inventory(
    tiny_env_factory, tiny_mjcf, monkeypatch
):
    """No selected site means no inventory scan of unrelated assets."""
    from unigym.backends import obs_schema

    def fail(*args, **kwargs):
        raise AssertionError("site inventory was built for an empty selection")

    monkeypatch.setattr(obs_schema, "build_site_inventory", fail)
    env = tiny_env_factory(
        mjcf=tiny_mjcf,
        observations=ObservationConfig(robot=("tiny",)),
    )
    try:
        env.step((NamedAction("tiny", torch.zeros(1, 2), ("elbow", "shoulder"), "position"),))
    finally:
        env.close()
