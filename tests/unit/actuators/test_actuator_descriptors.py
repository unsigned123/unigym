"""The five built-in law descriptors reproduce the legacy law surface."""

from __future__ import annotations

import pytest

from unigym.actuator_law import ACTUATOR_LAW_REGISTRY, ActuatorLawDescriptor
from unigym.actuators import (
    ActuationError,
    ActuationGroup,
    BACKEND_LAW_CAPABILITIES,
    validate_actuation_group,
)


@pytest.fixture(autouse=True)
def _isolate_actuator_registries():
    """Restore the law/implementation registries after each test."""
    from unigym.actuator_law import restore_actuator_laws, snapshot_actuator_laws
    from unigym.actuators.implementations import (
        restore_actuator_implementations,
        snapshot_actuator_implementations,
    )

    laws = snapshot_actuator_laws()
    implementations = snapshot_actuator_implementations()
    yield
    restore_actuator_laws(laws)
    restore_actuator_implementations(implementations)


_BUILTINS = {
    "standard_pd": (frozenset({"kp", "kd"}), ("kp", "kd"), False, frozenset({"position_pd"})),
    "computed_torque_pd": (
        frozenset({"kp", "kd"}),
        ("kp", "kd"),
        True,
        frozenset({"position_pd"}),
    ),
    "native_servo": (frozenset(), (), False, frozenset({"position_servo"})),
    "direct_torque": (frozenset(), (), False, frozenset({"torque"})),
    "direct_velocity": (frozenset(), (), False, frozenset({"velocity"})),
}


@pytest.mark.parametrize("name", sorted(_BUILTINS))
def test_builtin_descriptors_match_the_legacy_laws(name: str) -> None:
    descriptor = ACTUATOR_LAW_REGISTRY[name]
    gains, required, parts, drives = _BUILTINS[name]
    assert descriptor.gain_keys == gains
    assert descriptor.required_gains == required
    assert descriptor.allows_parts is parts
    assert descriptor.drives == drives
    assert descriptor.allows_command is False


def test_descriptors_drive_the_backend_capability_view() -> None:
    assert "computed_torque_pd" in BACKEND_LAW_CAPABILITIES["mujoco"]
    assert "native_servo" not in BACKEND_LAW_CAPABILITIES["isaaclab2"]
    assert BACKEND_LAW_CAPABILITIES["isaaclab3_physx"] == frozenset({"standard_pd"})


def test_compiled_plant_is_declared_by_descriptor() -> None:
    assert ACTUATOR_LAW_REGISTRY["native_servo"].compiled_plant == {
        "mujoco": "position_servo",
        "mjwarp": "position_servo",
    }
    assert ACTUATOR_LAW_REGISTRY["direct_torque"].compiled_plant["mujoco"] == "motor"
    assert ACTUATOR_LAW_REGISTRY["direct_velocity"].compiled_plant["mjwarp"] == "velocity_servo"


def test_pack_descriptor_gain_validation_is_descriptor_driven() -> None:
    descriptor = ActuatorLawDescriptor(
        name="acme.compliant_pd",
        config_keys=frozenset({"kp", "kd", "damping"}),
        gain_keys=frozenset({"kp", "kd"}),
        required_gains=("kp", "kd"),
        drives=frozenset({"position_pd"}),
        backends=frozenset({"mujoco"}),
    )
    from unigym.actuator_law import (
        register_actuator_law,
        restore_actuator_laws,
        snapshot_actuator_laws,
    )

    snapshot = snapshot_actuator_laws()
    register_actuator_law(descriptor)
    try:
        validate_actuation_group(
            ActuationGroup(
                name="body",
                law="acme.compliant_pd",
                joints=("j1",),
                gains={"kp": 1.0, "kd": 1.0},
            ),
            backend="mujoco",
        )
        with pytest.raises(ActuationError, match="unknown gains"):
            validate_actuation_group(
                ActuationGroup(
                    name="body",
                    law="acme.compliant_pd",
                    joints=("j1",),
                    gains={"kp": 1.0, "kd": 1.0, "bogus": 1.0},
                ),
                backend="mujoco",
            )
    finally:
        restore_actuator_laws(snapshot)


def test_descriptor_rejects_required_gain_outside_its_gain_set() -> None:
    with pytest.raises(ValueError, match="requires gains"):
        ActuatorLawDescriptor(
            name="bad",
            gain_keys=frozenset({"kp"}),
            required_gains=("kp", "kd"),
        )


def test_core_implementations_are_registered_and_return_actuator_base() -> None:
    from unigym.actuators import ActuatorBase
    from unigym.actuators.implementations import (
        ACTUATOR_IMPLEMENTATION_REGISTRY,
        ActuatorBuildContext,
        default_implementation_kind,
        ensure_core_implementations,
        get_actuator_implementation,
    )

    ensure_core_implementations()
    for law in (
        "standard_pd",
        "computed_torque_pd",
        "native_servo",
        "direct_torque",
        "direct_velocity",
    ):
        assert (law, "mujoco") in ACTUATOR_IMPLEMENTATION_REGISTRY
        assert (law, "mjwarp") in ACTUATOR_IMPLEMENTATION_REGISTRY

    context = ActuatorBuildContext(law=ACTUATOR_LAW_REGISTRY["standard_pd"], backend="mujoco")
    actuator = get_actuator_implementation("standard_pd", "mujoco").factory(context)
    assert isinstance(actuator, ActuatorBase)
    assert default_implementation_kind("mujoco") == "mujoco_ctrl"
    assert default_implementation_kind("isaaclab3_physx") == "isaaclab_cfg"


def test_missing_implementation_fails_with_the_registered_pairs() -> None:
    from unigym.actuators.implementations import get_actuator_implementation

    with pytest.raises(ValueError, match="no implementation for backend"):
        get_actuator_implementation("standard_pd", "isaaclab9")


def test_actuation_group_params_are_copied_frozen_json() -> None:
    from unigym.actuators import ActuationGroup

    group = ActuationGroup(
        name="body",
        law="standard_pd",
        joints=("j1",),
        gains={"kp": 1.0, "kd": 1.0},
        params={"damping": 0.2, "tags": ["a", "b"]},
    )
    assert group.params["damping"] == 0.2
    assert group.params["tags"] == ("a", "b")
    with pytest.raises(TypeError):
        group.params["damping"] = 0.3  # type: ignore[index]

    with pytest.raises(ValueError, match="must be JSON data"):
        ActuationGroup(
            name="body",
            law="standard_pd",
            joints=("j1",),
            gains={"kp": 1.0, "kd": 1.0},
            params={"bad": object()},
        )
