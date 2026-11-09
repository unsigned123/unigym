"""Public error hierarchy of the unigym API (S1a).

Every user-facing failure derives from :class:`UniGymError`, so callers can
catch one base type without importing backend-specific modules.  The model
IR errors keep their historical names and additional builtin bases for
backward compatibility (a ``ModelIRError`` is still a ``ValueError``).
"""

from __future__ import annotations

__all__ = [
    "BackendError",
    "CapabilityError",
    "ConfigError",
    "UniGymError",
]


class UniGymError(Exception):
    """Base class for every error raised by unigym's public API."""


class ConfigError(UniGymError):
    """A run config is structurally invalid or cannot be normalized."""


class BackendError(UniGymError):
    """A backend failed to build, compile or bind its native model."""


class CapabilityError(UniGymError):
    """A requested capability is not provided by the selected backend."""
