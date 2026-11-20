"""Registration phase primitives: order, windows, narrowing and sealing."""

from __future__ import annotations

import pytest

from unigym.registration import (
    ACTUATOR_IMPLEMENTATION_WINDOWS,
    FACE_WINDOWS,
    BackendRegistrationPolicy,
    PackFace,
    PhaseWindow,
    RegistrationError,
    RegistrationGate,
    RegistrationPhase,
    RegistrationSealedError,
    require_registration,
)


def test_phase_order_is_monotonic_forward_only() -> None:
    gate = RegistrationGate()
    for phase in (
        RegistrationPhase.MANIFEST,
        RegistrationPhase.ACTIVATE,
        RegistrationPhase.SPEC,
        RegistrationPhase.ASSEMBLE,
        RegistrationPhase.BACKEND_BUILD,
    ):
        gate.enter(phase)
        assert gate.phase is phase

    with pytest.raises(RegistrationError, match="monotonic"):
        gate.enter(RegistrationPhase.ACTIVATE)

    gate.at_least(RegistrationPhase.SPEC)
    assert gate.phase is RegistrationPhase.BACKEND_BUILD


def test_at_least_advances_and_never_regresses() -> None:
    gate = RegistrationGate()
    gate.at_least(RegistrationPhase.SPEC)
    assert gate.phase is RegistrationPhase.SPEC
    gate.at_least(RegistrationPhase.MANIFEST)
    assert gate.phase is RegistrationPhase.SPEC


def test_phase_window_contains_and_narrows() -> None:
    window = PhaseWindow(RegistrationPhase.ACTIVATE, RegistrationPhase.BACKEND_BUILD)
    assert window.contains(RegistrationPhase.ACTIVATE)
    assert window.contains(RegistrationPhase.SPEC)
    assert not window.contains(RegistrationPhase.MANIFEST)

    narrowed = window.narrow(PhaseWindow(RegistrationPhase.SPEC, RegistrationPhase.ASSEMBLE))
    assert narrowed == PhaseWindow(RegistrationPhase.SPEC, RegistrationPhase.ASSEMBLE)

    with pytest.raises(RegistrationError, match="do not intersect"):
        window.narrow(PhaseWindow(RegistrationPhase.MANIFEST, RegistrationPhase.MANIFEST))


def test_backend_policy_window_and_denied_faces() -> None:
    policy = BackendRegistrationPolicy(
        backend="mujoco",
        windows={
            PackFace.ACTUATOR_IMPLEMENTATION: ACTUATOR_IMPLEMENTATION_WINDOWS["mujoco"],
        },
        denied_faces=frozenset({PackFace.EXECUTOR}),
    )
    assert policy.window_for(PackFace.ACTUATOR_IMPLEMENTATION) == PhaseWindow(
        RegistrationPhase.BACKEND_BUILD, RegistrationPhase.BACKEND_BUILD
    )
    assert policy.window_for(PackFace.EXECUTOR) is None
    assert policy.window_for(PackFace.ROBOT) is None


def test_sealed_gate_rejects_every_phase_and_registration() -> None:
    gate = RegistrationGate()
    gate.seal_runtime()
    assert gate.sealed
    assert gate.phase is RegistrationPhase.RUNTIME

    with pytest.raises(RegistrationSealedError, match="sealed"):
        gate.enter(RegistrationPhase.RUNTIME)
    with pytest.raises(RegistrationSealedError, match="not allowed in phase RUNTIME"):
        gate.require(PackFace.ROBOT, name="acme.arm")

    gate.reset()
    assert gate.phase is None
    assert not gate.sealed


def test_bootstrap_allows_registration() -> None:
    gate = RegistrationGate()
    gate.require(PackFace.ADAPTER, name="acme.policy")
    gate.require(PackFace.ACTUATOR_IMPLEMENTATION, name="x@mujoco", backend="mujoco")


def test_face_windows_cover_the_plan_table() -> None:
    assert FACE_WINDOWS[PackFace.ACTUATOR_LAW] == PhaseWindow(
        RegistrationPhase.MANIFEST, RegistrationPhase.SPEC
    )
    assert FACE_WINDOWS[PackFace.INITIALIZER] == PhaseWindow(
        RegistrationPhase.ACTIVATE, RegistrationPhase.BACKEND_BUILD
    )
    assert FACE_WINDOWS[PackFace.ROBOT] == PhaseWindow(
        RegistrationPhase.ACTIVATE, RegistrationPhase.ACTIVATE
    )


def test_require_registration_names_face_window_and_backend_policy() -> None:
    gate = RegistrationGate(phase=RegistrationPhase.ASSEMBLE)
    with pytest.raises(RegistrationError, match="robot") as excinfo:
        gate.require(PackFace.ROBOT, name="acme.arm")
    message = str(excinfo.value)
    assert "ACTIVATE..ACTIVATE" in message
    assert "hint" in message

    with pytest.raises(RegistrationError, match="mujoco.*BACKEND_BUILD"):
        gate.require(
            PackFace.ACTUATOR_IMPLEMENTATION,
            name="acme.law@mujoco",
            backend="mujoco",
        )


def test_process_gate_delegate_uses_the_singleton() -> None:
    reset_phase = RegistrationGate(phase=RegistrationPhase.SPEC)
    del reset_phase
    # The module-level delegate is the process singleton; a SPEC phase blocks
    # an ACTIVATE-only face.
    from unigym import registration

    registration.reset_registration()
    registration.enter_phase(RegistrationPhase.SPEC)
    try:
        with pytest.raises(RegistrationError, match="scene_provider"):
            require_registration(PackFace.SCENE_PROVIDER, name="acme.scene")
    finally:
        registration.reset_registration()
