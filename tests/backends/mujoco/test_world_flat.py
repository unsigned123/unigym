"""World.flat() on the real MuJoCo lane: neutral ground + run-owned robot.

The world owns no robot asset and declares no explicit ``world_base``.  The
run layer resolves the flat declaration to the packaged ground MJCF and
attaches the profile asset exactly once.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from unigym.assets.asset_ref import AssetRef
from unigym.errors import ConfigError
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.world import World

ACTUATION = [
    {
        "name": "body",
        "law": "standard_pd",
        "joint_set": "body",
        "gains": {"kp": 20.0, "kd": 1.0},
    }
]

ACTION_SPEC = OrderedActionSpec.from_mapping(
    [
        {
            "name": "legs",
            "articulation": "flatbot",
            "joints": ["hinge"],
            "target": "position",
            "role": "body",
        }
    ]
)
POLICIES = (PolicyBinding.from_mapping({"adapter": "manual", "groups": [["flatbot", "legs"]]}),)

ROBOT_MJCF = """<mujoco model="flatbot">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.6">
      <freejoint name="root"/>
      <geom name="base_geom" type="box" size="0.05 0.05 0.05"/>
      <body name="link" pos="0 0 -0.1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="link_geom" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""

#: The same robot with an explicit contact-solver block, mirroring the shape
#: real robot assets ship (the values belong to the asset, not to the world).
ROBOT_OPTION = '<option cone="elliptic" impratio="100"/>'
ROBOT_WITH_OPTION_MJCF = ROBOT_MJCF.replace(
    '<compiler angle="radian"/>', f'<compiler angle="radian"/>\n  {ROBOT_OPTION}'
)

WORLD_WITH_CONFLICTING_OPTION_MJCF = """<mujoco model="conflict ground">
  <compiler angle="radian"/>
  <option cone="pyramidal"/>
  <worldbody>
    <geom name="world_ground" type="plane" size="0 0 0.05"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def robot_asset(tmp_path: Path) -> Path:
    """A robot-only MJCF: no floor, no world asset, one body chain."""
    path = tmp_path / "flatbot.xml"
    path.write_text(ROBOT_MJCF, encoding="utf-8")
    return path


@pytest.fixture
def optioned_robot_asset(tmp_path: Path) -> Path:
    """A robot-only MJCF that authors its own contact-solver options."""
    path = tmp_path / "flatbot_optioned.xml"
    path.write_text(ROBOT_WITH_OPTION_MJCF, encoding="utf-8")
    return path


def _profile(asset: Path) -> RobotProfile:
    return RobotProfile(
        name="flatbot",
        joint_sets={"body": ("hinge",)},
        default_joint_pos={"hinge": 0.0},
        assets=AssetRef(mjcf=str(asset)),
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
        actuation=ACTUATION,
    )


def test_world_flat_owns_no_robot_and_no_explicit_world_base(robot_asset: Path) -> None:
    """The flat declaration is empty: ground is resolved by the run layer."""
    world = World.flat()

    assert world.is_flat is True
    assert world.fragment() == {}
    assert world.static_declarations() == []
    assert not hasattr(world, "articulation")


def test_world_flat_builds_with_a_profile_asset_on_mujoco(robot_asset: Path) -> None:
    """MuJoCo receives the packaged ground once and the robot asset once."""
    run = Run(
        world=World.flat(),
        robot=_profile(robot_asset),
        spawn={"pos": (0.0, 0.0, 0.4)},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        backend="mujoco",
        inspect=True,
    )
    try:
        instance = run.build()
        model = instance.env.mj_model

        assert run.cfg["world_base"]["mjcf"].endswith("flat_ground.xml")
        assert [entry["name"] for entry in run.cfg["articulations"]] == ["flatbot"]
        assert run.cfg["articulations"][0]["spawn"] == {"pos": [0.0, 0.0, 0.4]}

        bodies = [model.body(i).name for i in range(model.nbody)]
        geoms = [model.geom(i).name for i in range(model.ngeom)]
        assert "flat_ground" in geoms
        assert bodies.count("base") == 1
        assert bodies.count("link") == 1
    finally:
        run.close()


def test_world_flat_inherits_the_single_articulation_contact_options(
    optioned_robot_asset: Path,
) -> None:
    """The flat ground declares no <option>: the robot asset stays the source.

    Parity: the composed model's ``opt`` matches the model loaded straight
    from the robot asset, so composition is not silently changing physics.
    """
    run = Run(
        world=World.flat(),
        robot=_profile(optioned_robot_asset),
        spawn={"pos": (0.0, 0.0, 0.4)},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        backend="mujoco",
        inspect=True,
    )
    try:
        model = run.build().env.mj_model
        source = mujoco.MjModel.from_xml_path(str(optioned_robot_asset))

        assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
        assert model.opt.impratio == 100.0
        assert model.opt.cone == source.opt.cone
        assert model.opt.impratio == source.opt.impratio
    finally:
        run.close()


def test_world_flat_rejects_conflicting_world_and_articulation_options(
    tmp_path: Path, optioned_robot_asset: Path
) -> None:
    """An attribute authored by both sides with different values fails fast."""
    world_asset = tmp_path / "conflict_ground.xml"
    world_asset.write_text(WORLD_WITH_CONFLICTING_OPTION_MJCF, encoding="utf-8")

    run = Run(
        world=World.base(mjcf=str(world_asset)),
        robot=_profile(optioned_robot_asset),
        spawn={"pos": (0.0, 0.0, 0.4)},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        backend="mujoco",
        inspect=True,
    )
    try:
        with pytest.raises(ConfigError, match="conflicting <option>.*cone"):
            run.build()
    finally:
        run.close()
