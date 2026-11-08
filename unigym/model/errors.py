"""Errors raised by the model IR and its live proxies (design v098 N3).

They live in their own module so the neutral IR and the backend-bound
live face share one import site.  ``StaleModelError`` is the lifecycle
error: a dynamic proxy read after ``env.close()`` must fail loudly rather
than return a frozen value (design section 9.11).  All three also derive
from the public :class:`unigym.errors.UniGymError` base (S1a).
"""
from __future__ import annotations

from unigym.errors import UniGymError

__all__ = [
    "ModelIRError",
    "ModelSetError",
    "StaleModelError",
    "UnknownExtensionError",
]


class ModelIRError(UniGymError, ValueError):
    """A structurally invalid model IR (bad reference, range or name)."""


class ModelSetError(UniGymError, RuntimeError):
    """A whole-scene read of a ``ModelSet`` that has no honest merge."""


class UnknownExtensionError(ModelIRError):
    """A serialized namespace extension this version cannot parse."""


class StaleModelError(UniGymError, RuntimeError):
    """A dynamic proxy was read after its native world was closed."""
