"""Built-in actuator law descriptors.

The definitions live beside the registry in :mod:`unigym.actuator_law` so the
core laws are available even when the registry module is imported late in a
process.  This module is the stable import face used by the actuators package.
"""

from __future__ import annotations

from unigym.actuator_law import (
    BUILTIN_LAWS,
    COMPUTED_TORQUE_PD,
    DIRECT_TORQUE,
    DIRECT_VELOCITY,
    NATIVE_SERVO,
    STANDARD_PD,
)

__all__ = [
    "BUILTIN_LAWS",
    "COMPUTED_TORQUE_PD",
    "DIRECT_TORQUE",
    "DIRECT_VELOCITY",
    "NATIVE_SERVO",
    "STANDARD_PD",
]
