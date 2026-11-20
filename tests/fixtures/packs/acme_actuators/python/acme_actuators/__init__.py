"""Actuator implementation factory fixture (imported only at materialization)."""

import sys

_IMPORTED_AT = "materialize"
_IMPORTED_IN = set(sys.modules)


def make_mujoco(context):
    """Return a marker object; a real pack would return an ``ActuatorBase``."""
    return {
        "law": context.law.name,
        "backend": context.backend,
        "params": dict(context.params),
    }
