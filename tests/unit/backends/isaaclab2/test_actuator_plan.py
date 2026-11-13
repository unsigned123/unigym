"""Neutral actuation -> Isaac Lab actuator plan (fake cfg classes)."""

from __future__ import annotations

import pytest

from unigym.actuators import ActuationGroup, ResolvedActuationGroup
from unigym.backends.isaaclab2.actuators.plan import build_actuator_plan


class _ImplicitActuatorCfgDouble:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _resolved(
    name: str = "body",
    *,
    law: str = "standard_pd",
    joints: tuple[str, ...] = ("shoulder", "elbow"),
    gains: dict | None = None,
    parts: tuple[str, ...] | None = None,
    command: dict | None = None,
    effort_limit=None,
    velocity_limit=None,
) -> ResolvedActuationGroup:
    return ResolvedActuationGroup(
        articulation="arm",
        group=ActuationGroup(
            name=name,
            law=law,
            joints=joints,
            gains={"kp": 10.0, "kd": 1.0} if gains is None else gains,
            parts=parts,
            command=command or {},
            effort_limit=effort_limit,
            velocity_limit=velocity_limit,
        ),
        joint_names=joints,
    )


def _classes():
    return {"ImplicitActuatorCfg": _ImplicitActuatorCfgDouble}


def test_standard_pd_translates_to_one_implicit_actuator_cfg():
    """standard_pd maps stiffness/damping and the joint expression."""
    plan = build_actuator_plan((_resolved(),), classes=_classes())

    cfg = plan["body"]
    assert isinstance(cfg, _ImplicitActuatorCfgDouble)
    assert cfg.kwargs == {
        "joint_names_expr": ["shoulder", "elbow"],
        "stiffness": 10.0,
        "damping": 1.0,
    }


def test_standard_pd_maps_limits_to_sim_fields():
    """Effort/velocity limits become the Isaac Lab sim limit fields."""
    plan = build_actuator_plan(
        (_resolved(effort_limit=40.0, velocity_limit=8.0),),
        classes=_classes(),
    )

    kwargs = plan["body"].kwargs
    assert kwargs["effort_limit_sim"] == 40.0
    assert kwargs["velocity_limit_sim"] == 8.0


def test_standard_pd_can_omit_limits():
    """Absent limits stay absent (no invented default)."""
    kwargs = build_actuator_plan((_resolved(),), classes=_classes())["body"].kwargs

    assert "effort_limit_sim" not in kwargs
    assert "velocity_limit_sim" not in kwargs


@pytest.mark.parametrize(
    "law",
    ["computed_torque_pd", "native_servo", "direct_torque", "direct_velocity", "bogus"],
)
def test_unsupported_laws_fail_with_a_capability_error(law):
    """Only standard_pd has a native Isaac Lab actuator mapping; direct drive is rejected."""
    if law == "bogus":
        with pytest.raises(ValueError, match="not registered"):
            _resolved(law=law)
        return
    with pytest.raises(ValueError, match="standard_pd only"):
        build_actuator_plan((_resolved(law=law),), classes=_classes())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"parts": ("body",)}, "parts are only valid"),
        ({"command": {"mode": "x"}}, "non-empty `command`"),
        ({"gains": {"kp": 1.0}}, "missing gains"),
        ({"gains": {"kp": 1.0, "kd": 1.0, "bogus": 1.0}}, "unknown gains"),
    ],
)
def test_malformed_standard_pd_payload_fails(kwargs, match):
    """Parts/command/unknown/missing gains fail with a teaching error."""
    with pytest.raises(ValueError, match=match):
        build_actuator_plan((_resolved(**kwargs),), classes=_classes())


def test_injected_classes_table_must_provide_implicit_actuator_cfg():
    """The test injection face is explicit about the class key."""
    with pytest.raises(ValueError, match="ImplicitActuatorCfg"):
        build_actuator_plan((_resolved(),), classes={})
