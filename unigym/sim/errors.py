"""Error taxonomy of the experimental simulation facade.

Every facade failure is a :class:`~unigym.errors.UniGymError`.  Configuration
mistakes that prevent a declaration (no world, no control) keep the existing
:class:`~unigym.errors.ConfigError` base; every runtime-facing facade failure
shares :class:`SimulationError`.  The split lets callers catch the facade
without importing a backend while preserving the historical ``ConfigError``
catch for the two declaration errors that mirror ``Run``.
"""

from __future__ import annotations

from unigym.errors import ConfigError, UniGymError

__all__ = [
    "BindingError",
    "BuilderSealedError",
    "ConfigModeError",
    "DeferredFeatureError",
    "ForeignHandleError",
    "MissingActionError",
    "NoControlError",
    "NoWorldError",
    "ObservationError",
    "PendingActionError",
    "SimulationClosedError",
    "SimulationConfigError",
    "SimulationError",
    "SimulationFaultedError",
    "StaleStateError",
]


class SimulationError(UniGymError):
    """Base class for every facade failure that is not a declaration error."""


class SimulationConfigError(SimulationError):
    """A facade declaration is invalid (dt arithmetic, labels, backend name)."""


class ForeignHandleError(SimulationError):
    """A handle or binding belongs to another builder/simulation."""


class BindingError(SimulationError):
    """An action face, kind, order or shape is illegal."""


class PendingActionError(SimulationError):
    """A pending action is duplicated, missing or stale."""


class MissingActionError(SimulationError):
    """A required control face has no staged action."""


class StaleStateError(SimulationError):
    """The runner advanced out of band; the facade clock is behind."""


class ObservationError(SimulationError):
    """A state key is missing or invalid."""


class SimulationClosedError(SimulationError):
    """The simulation is closed."""


class SimulationFaultedError(SimulationError):
    """The simulation faulted during an engine tick."""


class BuilderSealedError(SimulationError):
    """The builder is sealed after a successful build."""


class ConfigModeError(SimulationError):
    """A config-mode builder cannot declare new objects."""


class DeferredFeatureError(SimulationError):
    """A capability declared out of the current spike scope."""


class NoWorldError(ConfigError):
    """``build()`` needs ``builder.world``."""


class NoControlError(ConfigError):
    """``add_articulation`` needs ``control=``."""
