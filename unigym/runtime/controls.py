"""The explicit drive/target vocabulary shared by the control runtime.

Action authoring lives in :mod:`unigym.runtime.actions` (``OrderedActionGroup``
+ ``PolicyBinding``) and physical actuation in :mod:`unigym.actuators.ir`
(``ActuationGroup``).  This module keeps only the closed drive/target table and
its teaching error: the derived :class:`~unigym.runtime.controllers.ControlRuntime`
maps an action target plus a resolved actuation law onto one drive.

Drive semantics:

* ``position_pd``      -- software PD; ``target = position`` (absolute angle)
* ``position_servo``   -- engine position actuator; ``target = position``
* ``torque``           -- torque target; ``target = torque``
* ``velocity``         -- velocity target; ``target = velocity``
* ``effort``           -- effort/force magnitude; ``target = effort``
* ``current``          -- current target; ``target = current``
"""

from __future__ import annotations

__all__ = [
    "CONTROL_DRIVES",
    "CONTROL_TARGETS",
    "ControlError",
    "DRIVE_TARGET",
    "target_for_drive",
]

CONTROL_DRIVES = (
    "position_pd",
    "position_servo",
    "torque",
    "velocity",
    "effort",
    "current",
)
"""The closed drive set (``passive`` is not a group -- use passive_joints)."""

CONTROL_TARGETS = ("position", "velocity", "torque", "effort", "current")
"""The closed action-target / ``NamedAction.kind`` set."""

DRIVE_TARGET = {
    "position_pd": "position",
    "position_servo": "position",
    "torque": "torque",
    "velocity": "velocity",
    "effort": "effort",
    "current": "current",
}
"""The one legal target kind per drive (explicit, never inferred)."""


class ControlError(ValueError):
    """A malformed control declaration (fail-fast, no alias)."""


def target_for_drive(drive: str) -> str:
    """The explicit target kind mapped from *drive* (fail-fast)."""
    try:
        return DRIVE_TARGET[drive]
    except KeyError:
        raise ControlError(
            f"unknown control drive {drive!r}; legal drives are "
            f"{list(CONTROL_DRIVES)} (passive joints are declared in the "
            f"articulation's `passive_joints`, not as a group)"
        ) from None
