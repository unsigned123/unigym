"""``World.flat()`` resolution: the run layer picks the backend's own ground."""

from __future__ import annotations

import pytest

from unigym.actuators.ir import ActuationGroup
from unigym.errors import ConfigError
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.world import World

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


def _profile() -> RobotProfile:
    return RobotProfile(
        name="tiny",
        joint_sets={"body": JOINTS},
        default_joint_pos={"shoulder": 0.0, "elbow": 0.0},
        assets=None,
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


def _spec(backend: str) -> Run:
    return Run(
        world=World.flat(),
        robot=_profile(),
        spawn={"pos": (0.0, 0.0, 0.4)},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        observations={"robot": ["tiny"]},
        backend=backend,
        inspect=True,
    )


def test_world_flat_resolves_to_the_packaged_mujoco_ground() -> None:
    run = _spec("mujoco")
    try:
        cfg = run.spec.cfg
        assert cfg["world_base"]["mjcf"].endswith("flat_ground.xml")
        assert cfg["articulations"][0]["spawn"] == {"pos": [0.0, 0.0, 0.4]}
    finally:
        run.close()


def test_world_flat_resolves_to_a_native_flat_terrain_preset() -> None:
    """IsaacLab2 has no world_base consumer; it gets its own flat preset."""
    run = _spec("isaaclab2")
    try:
        cfg = run.spec.cfg
        assert "world_base" not in cfg
        assert cfg["terrain"] == {"name": "flat", "params": {}}
    finally:
        run.close()


def test_world_flat_fails_loudly_on_a_backend_without_any_ground() -> None:
    """The isaaclab3 stub consumes neither world_base nor terrain: no silence."""
    run = _spec("isaaclab3")
    try:
        with pytest.raises(ConfigError, match="isaaclab3.*no flat-ground resolution"):
            run.spec
    finally:
        run.close()
