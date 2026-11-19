"""Build-time actuation plan: driven coverage, law matrix and hard rejects."""

from __future__ import annotations

import pytest

from unigym.actuators import ActuationError, ActuationGroup
from unigym.robots.base import RobotProfile
from unigym.run.build import (
    _build_actuation_plans,
    _reject_removed_actuator_sources,
)
from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec
from unigym.scenes.articulation import ArticulationSpec, JointSpec


def _profile(name: str = "arm", sets: dict | None = None) -> RobotProfile:
    return RobotProfile(
        name=name,
        joint_sets=({"body": ("shoulder", "elbow")} if sets is None else sets),
        default_joint_pos={},
        assets=None,
        extra_action_sets={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )


def _action(name: str, art: str, joints: tuple[str, ...], target: str = "position"):
    return OrderedActionGroup(name=name, articulation=art, joints=joints, target=target)


def _spec(*groups: OrderedActionGroup) -> OrderedActionSpec:
    return OrderedActionSpec(tuple(groups))


def _group(name: str = "body", joints=("shoulder", "elbow"), law="standard_pd") -> ActuationGroup:
    gains = {"kp": 1.0, "kd": 1.0} if law in ("standard_pd", "computed_torque_pd") else {}
    return ActuationGroup(name=name, law=law, joints=joints, gains=gains)


def test_plan_covers_exactly_the_driven_joints_in_declaration_order():
    """The per-articulation plan resolves against the action-driven joints."""
    spec = ArticulationSpec(
        name="arm",
        joints=(JointSpec("shoulder"), JointSpec("elbow"), JointSpec("finger")),
        passive_joints=("finger",),
        actuation=(_group(),),
    )

    plans = _build_actuation_plans(
        (spec,),
        {"arm": _profile()},
        _spec(_action("body", "arm", ("shoulder", "elbow"))),
        "mujoco",
    )

    assert [row.joint_names for row in plans["arm"]] == [("shoulder", "elbow")]


def test_plan_fails_when_a_driven_articulation_omits_actuation():
    """The hard cut: driven joints require an explicit actuation declaration."""
    spec = ArticulationSpec(name="arm", joints=(JointSpec("shoulder"),))

    with pytest.raises(ValueError, match="declares driven joints but no `actuation`"):
        _build_actuation_plans(
            (spec,),
            {"arm": _profile()},
            _spec(_action("body", "arm", ("shoulder",))),
            "mujoco",
        )


def test_plan_requires_a_profile_for_joint_set_addressing():
    """A profile-less articulation cannot resolve a joint_set group."""
    spec = ArticulationSpec(
        name="arm",
        joints=(JointSpec("shoulder"),),
        actuation=(
            ActuationGroup(
                name="body", law="standard_pd", joint_set="body", gains={"kp": 1.0, "kd": 1.0}
            ),
        ),
    )

    with pytest.raises(ActuationError, match="joint_set 'body' is unknown"):
        _build_actuation_plans(
            (spec,),
            {},
            _spec(_action("body", "arm", ("shoulder",))),
            "mujoco",
        )


def test_plan_is_per_articulation_and_ignores_non_action_joints():
    """Each articulation gets its own plan from its action groups only."""
    left = ArticulationSpec(
        name="left",
        joints=(JointSpec("shoulder"),),
        actuation=(_group(joints=("shoulder",)),),
    )
    right = ArticulationSpec(
        name="right",
        joints=(JointSpec("shoulder"),),
        actuation=(_group(joints=("shoulder",)),),
    )
    profiles = {
        "left": _profile("left", {"body": ("shoulder",)}),
        "right": _profile("right", {"body": ("shoulder",)}),
    }

    plans = _build_actuation_plans(
        (left, right),
        profiles,
        _spec(
            _action("left", "left", ("shoulder",)),
            _action("right", "right", ("shoulder",)),
        ),
        "mujoco",
    )

    assert set(plans) == {"left", "right"}


@pytest.mark.parametrize(
    ("target", "law"),
    [("torque", "direct_torque"), ("velocity", "direct_velocity")],
)
def test_plan_accepts_direct_drive_laws_on_mujoco_family(target, law):
    """torque/velocity targets resolve to their explicit direct-drive law."""
    spec = ArticulationSpec(
        name="arm",
        joints=(JointSpec("shoulder"),),
        actuation=(_group(law=law, joints=("shoulder",)),),
    )

    plans = _build_actuation_plans(
        (spec,),
        {"arm": _profile()},
        _spec(_action("body", "arm", ("shoulder",), target=target)),
        "mujoco",
    )

    assert [row.group.law for row in plans["arm"]] == [law]


@pytest.mark.parametrize("backend", ["isaaclab2", "isaaclab3_physx", "isaaclab3_newton"])
@pytest.mark.parametrize(
    "law",
    ["computed_torque_pd", "direct_torque", "direct_velocity"],
)
def test_plan_backend_gate_rejects_non_standard_laws(backend, law):
    """IsaacLab support is standard_pd only; direct drive is not faked."""
    spec = ArticulationSpec(
        name="g1",
        joints=(JointSpec("shoulder"),),
        actuation=(_group(law=law, joints=("shoulder",)),),
    )

    with pytest.raises(ActuationError, match="not supported by backend"):
        _build_actuation_plans(
            (spec,),
            {"g1": _profile("g1")},
            _spec(_action("g1", "g1", ("shoulder",))),
            backend,
        )


def test_plan_backend_gate_accepts_mujoco_family_non_standard_laws():
    """mujoco/mjwarp accept computed_torque_pd and native_servo plans."""
    ct = ArticulationSpec(
        name="arm",
        joints=(JointSpec("shoulder"),),
        actuation=(_group(law="computed_torque_pd", joints=("shoulder",)),),
    )
    servo = ArticulationSpec(
        name="hand",
        joints=(JointSpec("finger"),),
        actuation=(ActuationGroup(name="hand", law="native_servo", joints=("finger",)),),
    )

    plans = _build_actuation_plans(
        (ct, servo),
        {
            "arm": _profile("arm", {"body": ("shoulder",)}),
            "hand": _profile("hand", {"body": ("finger",)}),
        },
        _spec(
            _action("arm", "arm", ("shoulder",)),
            _action("hand", "hand", ("finger",)),
        ),
        "mjwarp",
    )

    assert {name: row.group.law for name, rows in plans.items() for row in rows} == {
        "arm": "computed_torque_pd",
        "hand": "native_servo",
    }


def test_run_build_rejects_removed_actuator_and_gain_sources():
    """Top-level actuator and env gain knobs are hard rejects."""
    with pytest.raises(ValueError, match="top-level `actuator:`"):
        _reject_removed_actuator_sources({"actuator": {"type": "implicit_pd"}})

    for key in ("actuators", "kp", "kd", "hand_kp", "hand_kd"):
        with pytest.raises(ValueError, match="removed actuator/gain keys"):
            _reject_removed_actuator_sources({"env": {"params": {key: {}}}})
