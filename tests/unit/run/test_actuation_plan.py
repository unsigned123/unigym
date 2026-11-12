"""Build-time actuation plan: driven coverage, law matrix and hard rejects."""

from __future__ import annotations

import pytest

from unigym.actuators import ActuationError, ActuationGroup
from unigym.robots.base import RobotProfile
from unigym.run.build import (
    _build_actuation_plans,
    _reject_removed_actuator_sources,
)
from unigym.runtime.controllers import ControlRuntime
from unigym.scenes.articulation import ArticulationSpec, JointSpec


def _profile(name: str = "arm", sets: dict | None = None) -> RobotProfile:
    return RobotProfile(
        name=name,
        joint_sets=({"body": ("shoulder", "elbow")} if sets is None else sets),
        default_joint_pos={},
        assets=None,
        action_faces={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )


def _control(name: str, art: str, joints: tuple[str, ...], drive: str = "position_pd"):
    return ControlRuntime(
        name=name,
        action_face=name,
        adapter="manual",
        articulation=art,
        joint_names=joints,
        q0=(0.0,) * len(joints),
        drive=drive,
        target="position" if drive in ("position_pd", "position_servo") else drive,
    )


def _group(name: str = "body", joints=("shoulder", "elbow"), law="standard_pd") -> ActuationGroup:
    gains = {"kp": 1.0, "kd": 1.0} if law in ("standard_pd", "computed_torque_pd") else {}
    return ActuationGroup(name=name, law=law, joints=joints, gains=gains)


def test_plan_covers_exactly_the_driven_joints_in_declaration_order():
    """The per-articulation plan resolves against the control-driven joints."""
    spec = ArticulationSpec(
        name="arm",
        joints=(JointSpec("shoulder"), JointSpec("elbow"), JointSpec("finger")),
        passive_joints=("finger",),
        actuation=(_group(),),
    )

    plans = _build_actuation_plans(
        (spec,), {"arm": _profile()}, (_control("body", "arm", ("shoulder", "elbow")),), "mujoco"
    )

    assert [row.joint_names for row in plans["arm"]] == [("shoulder", "elbow")]


def test_plan_fails_when_a_driven_articulation_omits_actuation():
    """The hard cut: driven joints require an explicit actuation declaration."""
    spec = ArticulationSpec(name="arm", joints=(JointSpec("shoulder"),))

    with pytest.raises(ValueError, match="declares driven joints but no `actuation`"):
        _build_actuation_plans(
            (spec,), {"arm": _profile()}, (_control("body", "arm", ("shoulder",)),), "mujoco"
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
        _build_actuation_plans((spec,), {}, (_control("body", "arm", ("shoulder",)),), "mujoco")


def test_plan_is_per_articulation_and_excludes_prop_controls():
    """Each articulation gets its own plan; prop controls are ignored."""
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
    prop = ControlRuntime(
        name="door",
        action_face="door",
        adapter="manual",
        articulation="left",
        joint_names=("hinge",),
        q0=(0.0,),
        drive="position_servo",
        target="position",
        prop=True,
    )
    profiles = {
        "left": _profile("left", {"body": ("shoulder",)}),
        "right": _profile("right", {"body": ("shoulder",)}),
    }

    plans = _build_actuation_plans(
        (left, right),
        profiles,
        (
            _control("left", "left", ("shoulder",)),
            _control("right", "right", ("shoulder",)),
            prop,
        ),
        "mujoco",
    )

    assert set(plans) == {"left", "right"}


@pytest.mark.parametrize("drive", ["torque", "velocity", "effort", "current"])
def test_plan_rejects_direct_drive_controls(drive):
    """Direct drives fail at plan time with the D4 teaching error."""
    spec = ArticulationSpec(
        name="arm",
        joints=(JointSpec("shoulder"),),
        actuation=(_group(joints=("shoulder",)),),
    )

    with pytest.raises(ActuationError, match="no D4 actuation law"):
        _build_actuation_plans(
            (spec,),
            {"arm": _profile()},
            (_control("body", "arm", ("shoulder",), drive=drive),),
            "mujoco",
        )


@pytest.mark.parametrize("backend", ["isaaclab2", "isaaclab3"])
def test_plan_backend_gate_rejects_non_standard_laws(backend):
    """IsaacLab support is standard_pd only, including the G1 exception."""
    spec = ArticulationSpec(
        name="g1",
        joints=(JointSpec("shoulder"),),
        actuation=(_group(law="computed_torque_pd", joints=("shoulder",)),),
    )

    with pytest.raises(ActuationError, match="not supported by backend"):
        _build_actuation_plans(
            (spec,),
            {"g1": _profile("g1")},
            (_control("g1", "g1", ("shoulder",)),),
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
        (
            _control("arm", "arm", ("shoulder",)),
            _control("hand", "hand", ("finger",), drive="position_servo"),
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
