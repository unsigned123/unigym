"""The facade error taxonomy is importable, rooted and carries its message."""

from __future__ import annotations

import pytest

from unigym.errors import ConfigError, UniGymError
from unigym.sim import (
    SIM_API_VERSION,
    BindingError,
    BuilderSealedError,
    ConfigModeError,
    DeferredFeatureError,
    ForeignHandleError,
    MissingActionError,
    NoControlError,
    NoWorldError,
    Observation,
    ObservationError,
    PendingActionError,
    Simulation,
    SimulationClosedError,
    SimulationConfigError,
    SimulationError,
    SimulationFaultedError,
    StaleStateError,
)

_SIMULATION_ERRORS = (
    SimulationConfigError,
    ForeignHandleError,
    BindingError,
    PendingActionError,
    MissingActionError,
    StaleStateError,
    ObservationError,
    SimulationClosedError,
    SimulationFaultedError,
    BuilderSealedError,
    ConfigModeError,
    DeferredFeatureError,
)


@pytest.mark.parametrize("error", _SIMULATION_ERRORS)
def test_simulation_errors_derive_from_unigym_error(error) -> None:
    """Every runtime facade error is catchable as UniGymError/SimulationError."""
    assert issubclass(error, SimulationError)
    assert issubclass(error, UniGymError)
    assert not issubclass(error, ConfigError)


@pytest.mark.parametrize("error", (NoWorldError, NoControlError))
def test_declaration_errors_keep_the_config_base(error) -> None:
    """The two declaration errors mirror Run's ConfigError catch."""
    assert issubclass(error, ConfigError)
    assert issubclass(error, UniGymError)


@pytest.mark.parametrize(
    "error",
    (*_SIMULATION_ERRORS, NoWorldError, NoControlError),
)
def test_errors_preserve_expected_and_actual_values(error) -> None:
    """The message carrying expected/actual values survives construction."""
    message = "expected face 'body', got 'hand' (actual order ['hand', 'body'])"
    assert str(error(message)) == message


def test_public_facade_records_are_importable() -> None:
    """The live facade types resolve from the one experimental namespace."""
    assert Simulation.__name__ == "Simulation"
    assert Observation.__name__ == "Observation"
    assert issubclass(SimulationError, UniGymError)


def test_sim_api_version_is_four() -> None:
    """The ordered-action-group contract bump is the published facade version."""
    assert SIM_API_VERSION == 4
