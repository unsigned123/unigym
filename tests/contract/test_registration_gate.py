"""Every pack-registerable registry is gated by the process-wide phases."""

from __future__ import annotations

import pytest

from unigym.adapters import ADAPTER_REGISTRY, register_adapter
from unigym.benchmarks.task_providers import PROVIDERS, register as register_task_provider
from unigym.initializers.base import INITIALIZER_REGISTRY, register as register_initializer
from unigym.props.templates import register_prop_template
from unigym.registration import (
    PackFace,
    RegistrationError,
    RegistrationPhase,
    RegistrationSealedError,
    enter_phase,
    reset_registration,
    seal_runtime,
)
from unigym.robots import ROBOT_REGISTRY, register_robot
from unigym.scenes.pipeline import register_scene_content_filler
from unigym.scenes.providers import register_inline_scene_provider, register_scene_provider
from unigym.terminations import TERMINATION_REGISTRY, register_termination


@pytest.fixture(autouse=True)
def _isolate_extra_registries():
    """Restore law/implementation/initializer/task-provider registries per test."""
    from unigym.actuator_law import restore_actuator_laws, snapshot_actuator_laws
    from unigym.actuators.implementations import (
        restore_actuator_implementations,
        snapshot_actuator_implementations,
    )
    from unigym.benchmarks.task_providers import (
        restore_task_providers,
        snapshot_task_providers,
    )
    from unigym.initializers.base import restore_initializers, snapshot_initializers

    laws = snapshot_actuator_laws()
    implementations = snapshot_actuator_implementations()
    initializers = snapshot_initializers()
    task_providers = snapshot_task_providers()
    yield
    restore_actuator_laws(laws)
    restore_actuator_implementations(implementations)
    restore_initializers(initializers)
    restore_task_providers(task_providers)


class _DummyAdapter:
    pass


def _factory():
    return None


def _scene(params, context):
    """A scene provider fixture."""


def _filler(*, catalog, generator_exists):
    return object()


def _termination(env):
    return None


def test_adapter_wrong_phase_and_sealed() -> None:
    enter_phase(RegistrationPhase.SPEC)
    with pytest.raises(RegistrationError, match="adapter"):
        register_adapter("acme.policy", _DummyAdapter)
    assert "acme.policy" not in ADAPTER_REGISTRY

    reset_registration()
    seal_runtime()
    with pytest.raises(RegistrationSealedError):
        register_adapter("acme.policy", _DummyAdapter)


def test_robot_wrong_phase() -> None:
    enter_phase(RegistrationPhase.BACKEND_BUILD)
    with pytest.raises(RegistrationError, match="robot"):
        register_robot("acme.arm", _factory)
    assert "acme.arm" not in ROBOT_REGISTRY


def test_termination_wrong_phase() -> None:
    enter_phase(RegistrationPhase.MANIFEST)
    with pytest.raises(RegistrationError, match="termination"):
        register_termination("acme.term", _termination)
    assert "acme.term" not in TERMINATION_REGISTRY


def test_scene_provider_and_content_filler_wrong_phase() -> None:
    enter_phase(RegistrationPhase.SPEC)
    with pytest.raises(RegistrationError, match="scene_provider"):
        register_scene_provider("acme.scene", _scene)
    with pytest.raises(RegistrationError, match="scene_content_filler"):
        register_scene_content_filler("acme.filler", _filler)


def test_inline_scene_provider_window_is_spec_only() -> None:
    from unigym.scenes.providers import restore_scene_providers, snapshot_scene_providers

    providers = snapshot_scene_providers()
    try:
        enter_phase(RegistrationPhase.MANIFEST)
        with pytest.raises(RegistrationError, match="scene_provider_inline"):
            register_inline_scene_provider("acme.inline", _scene)
        reset_registration()
        enter_phase(RegistrationPhase.SPEC)
        register_inline_scene_provider("acme.inline", _scene)
    finally:
        restore_scene_providers(providers)


def test_prop_template_wrong_phase() -> None:
    enter_phase(RegistrationPhase.SPEC)
    with pytest.raises(RegistrationError, match="prop_template"):
        register_prop_template("acme.cube", params={"size": {"type": "float", "default": 0.1}})


def test_initializer_is_legal_until_backend_build() -> None:
    class _Init:
        name = "acme.init"

        def apply(self):
            return {}

    enter_phase(RegistrationPhase.SPEC)
    register_initializer(_Init)
    assert INITIALIZER_REGISTRY["acme.init"] is _Init

    reset_registration()
    enter_phase(RegistrationPhase.RUNTIME)
    with pytest.raises(RegistrationSealedError):
        register_initializer(_Init)


def test_task_provider_wrong_phase() -> None:
    enter_phase(RegistrationPhase.SPEC)
    with pytest.raises(RegistrationError, match="task_provider"):

        @register_task_provider("acme.task")
        def _provider(slots, tables):
            """Provider fixture."""
            return {}

    assert "acme.task" not in PROVIDERS


def test_actuator_law_and_implementation_windows() -> None:
    from unigym.actuator_law import ActuatorLawDescriptor, register_actuator_law
    from unigym.actuators.implementations import register_actuator_implementation

    descriptor = ActuatorLawDescriptor(
        name="acme.law",
        gain_keys=frozenset({"kp"}),
        required_gains=("kp",),
        drives=frozenset({"position_pd"}),
        backends=frozenset({"mujoco"}),
    )
    enter_phase(RegistrationPhase.ACTIVATE)
    register_actuator_law(descriptor)

    enter_phase(RegistrationPhase.ASSEMBLE)
    with pytest.raises(RegistrationError, match="mujoco.*BACKEND_BUILD"):
        register_actuator_implementation("acme.law", "mujoco", _factory)

    enter_phase(RegistrationPhase.BACKEND_BUILD)
    register_actuator_implementation("acme.law", "mujoco", _factory)


def test_implementation_requires_a_registered_law() -> None:
    from unigym.registration import enter_phase as advance

    advance(RegistrationPhase.BACKEND_BUILD)
    from unigym.actuators.implementations import register_actuator_implementation

    with pytest.raises(ValueError, match="unknown actuator law"):
        register_actuator_implementation("acme.missing", "mujoco", _factory)


def test_face_enum_covers_gate_message() -> None:
    enter_phase(RegistrationPhase.SPEC)
    with pytest.raises(RegistrationError) as excinfo:
        register_robot("acme.arm", _factory)
    message = str(excinfo.value)
    assert PackFace.ROBOT.value in message
    assert RegistrationPhase.SPEC.value in message
