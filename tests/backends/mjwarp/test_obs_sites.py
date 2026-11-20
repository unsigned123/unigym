"""mjWarp site observations use the shared native/body pose reader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unigym.env.obs import ObservationConfig
from unigym.sites import SiteSpec
from unigym.runtime.actions import NamedAction

_HALF = 2**-0.5


def _site_mjcf(tiny_mjcf: Path) -> Path:
    """Add one oriented site under the tip body of the shared fixture."""
    target = tiny_mjcf.with_name("tiny_arm_mjwarp_site.xml")
    text = tiny_mjcf.read_text(encoding="utf-8")
    text = text.replace(
        '<geom name="tip_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03" mass="0.5"/>',
        '<geom name="tip_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03" mass="0.5"/>'
        f'<site name="grip" pos="0 0 -0.1" quat="{_HALF} 0 0 {_HALF}"/>',
    )
    target.write_text(text, encoding="utf-8")
    return target


def test_native_site_publishes_position_and_orientation(tiny_env_factory, tiny_mjcf):
    env = tiny_env_factory(
        mjcf=_site_mjcf(tiny_mjcf),
        observations=ObservationConfig(robot=("tiny",), site=("grip",)),
    )
    try:
        step = env.step(
            (NamedAction("tiny", torch.zeros(1, 2), ("elbow", "shoulder"), "position"),)
        )
        assert step.obs.site("grip").pos_w.shape == (1, 3)
        assert step.obs.site("grip").quat_w[0].tolist() == pytest.approx(
            [_HALF, 0.0, 0.0, _HALF], abs=1e-6
        )
        assert "site/grip/quat" in step.record
    finally:
        env.close()


def test_profile_site_uses_body_offset_without_a_native_site(tiny_env_factory, tiny_mjcf):
    site = SiteSpec("tcp", "base", pos=(0.1, 0.2, 0.3), quat_wxyz=(_HALF, 0.0, 0.0, _HALF))
    env = tiny_env_factory(
        mjcf=tiny_mjcf,
        sites={"tcp": site},
        observations=ObservationConfig(robot=("tiny",), site=("tcp",)),
    )
    try:
        step = env.step(
            (NamedAction("tiny", torch.zeros(1, 2), ("elbow", "shoulder"), "position"),)
        )
        offset = step.obs.site("tcp").pos_w - step.obs.robot("tiny").base_pos_w
        assert offset[0].tolist() == pytest.approx([0.1, 0.2, 0.3], abs=2e-3)
        assert step.obs.site("tcp").quat_w[0].tolist() == pytest.approx(
            [_HALF, 0.0, 0.0, _HALF], abs=1e-6
        )
    finally:
        env.close()
