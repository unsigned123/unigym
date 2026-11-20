"""IsaacLab2 cfg plan dispatches a pack law through its ASSEMBLE factory."""

from __future__ import annotations

import pytest

from unigym.actuator_law import (
    ActuatorLawDescriptor,
    register_actuator_law,
    restore_actuator_laws,
    snapshot_actuator_laws,
)
from unigym.actuators import ResolvedActuationGroup
from unigym.actuators.implementations import (
    register_actuator_implementation,
    restore_actuator_implementations,
    snapshot_actuator_implementations,
)
from unigym.actuators.ir import ActuationGroup
from unigym.backends.isaaclab2.actuators.plan import build_actuator_plan
from unigym.registration import (
    RegistrationError,
    RegistrationPhase,
    enter_phase,
    reset_registration,
)


class _ImplicitActuatorCfgDouble:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _resolved(
    law: str, gains: dict[str, float], *, params: dict | None = None
) -> tuple[ResolvedActuationGroup, ...]:
    return (
        ResolvedActuationGroup(
            articulation="arm",
            group=ActuationGroup(
                name="body", law=law, joints=("j1",), gains=gains, params=params or {}
            ),
            joint_names=("j1",),
        ),
    )


@pytest.fixture(autouse=True)
def _isolate():
    laws = snapshot_actuator_laws()
    implementations = snapshot_actuator_implementations()
    reset_registration()
    yield
    restore_actuator_laws(laws)
    restore_actuator_implementations(implementations)
    reset_registration()


def _register_pack_law() -> None:
    register_actuator_law(
        ActuatorLawDescriptor(
            name="acme.compliant_pd",
            config_keys=frozenset({"kp", "kd", "damping"}),
            gain_keys=frozenset({"kp", "kd"}),
            required_gains=("kp", "kd"),
            drives=frozenset({"position_pd"}),
            backends=frozenset({"isaaclab2"}),
            requires_implementation=True,
        )
    )

    def factory(context):
        return {
            "joint_names_expr": list(context.joint_names),
            "stiffness": context.params["kp"],
            "damping": context.params.get("damping", context.params["kd"]),
        }

    register_actuator_implementation("acme.compliant_pd", "isaaclab2", factory, kind="isaaclab_cfg")


def test_pack_law_builds_through_its_native_cfg_factory() -> None:
    _register_pack_law()
    plan = build_actuator_plan(
        _resolved("acme.compliant_pd", {"kp": 40.0, "kd": 1.0}),
        classes={"ImplicitActuatorCfg": _ImplicitActuatorCfgDouble},
        backend="isaaclab2",
    )
    cfg = plan["body"]
    assert isinstance(cfg, _ImplicitActuatorCfgDouble)
    assert cfg.kwargs == {
        "joint_names_expr": ["j1"],
        "stiffness": 40.0,
        "damping": 1.0,
    }


def test_pack_law_without_the_requested_backend_keeps_the_native_error() -> None:
    _register_pack_law()
    with pytest.raises(ValueError, match="no native Isaac Lab actuator mapping"):
        build_actuator_plan(
            _resolved("acme.compliant_pd", {"kp": 40.0, "kd": 1.0}),
            classes={"ImplicitActuatorCfg": _ImplicitActuatorCfgDouble},
            backend="isaaclab3_physx",
        )


def test_standard_pd_behaviour_and_messages_are_unchanged() -> None:
    plan = build_actuator_plan(
        _resolved("standard_pd", {"kp": 10.0, "kd": 0.5}),
        classes={"ImplicitActuatorCfg": _ImplicitActuatorCfgDouble},
    )
    assert plan["body"].kwargs == {
        "joint_names_expr": ["j1"],
        "stiffness": 10.0,
        "damping": 0.5,
    }
    with pytest.raises(ValueError, match="standard_pd only"):
        build_actuator_plan(
            _resolved("native_servo", {}),
            classes={"ImplicitActuatorCfg": _ImplicitActuatorCfgDouble},
        )


def test_implementation_registration_is_assemble_only_for_isaaclab() -> None:
    enter_phase(RegistrationPhase.SPEC)
    register_actuator_law(
        ActuatorLawDescriptor(
            name="acme.early",
            drives=frozenset({"position_pd"}),
            backends=frozenset({"isaaclab2"}),
        )
    )
    with pytest.raises(RegistrationError, match="ASSEMBLE"):
        register_actuator_implementation("acme.early", "isaaclab2", lambda context: None)
    reset_registration()
    enter_phase(RegistrationPhase.ASSEMBLE)
    register_actuator_implementation("acme.early", "isaaclab2", lambda context: None)


def test_pack_law_required_gains_can_come_from_params_only() -> None:
    from unigym.actuators.ir import validate_actuation_group

    _register_pack_law()
    group = ActuationGroup(
        name="body",
        law="acme.compliant_pd",
        joints=("j1",),
        params={"kp": 40.0, "kd": 1.0, "damping": 0.2},
    )
    assert group.gains == {}
    validate_actuation_group(group, backend="isaaclab2")

    plan = build_actuator_plan(
        (ResolvedActuationGroup(articulation="arm", group=group, joint_names=("j1",)),),
        classes={"ImplicitActuatorCfg": _ImplicitActuatorCfgDouble},
        backend="isaaclab2",
    )
    assert plan["body"].kwargs == {
        "joint_names_expr": ["j1"],
        "stiffness": 40.0,
        "damping": 0.2,
    }
