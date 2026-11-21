"""IsaacLab 3.x shared runtime-query implementation (P0B-2 model + pose).

The PhysX and Newton identities share one implementation package but own
separate runtime stacks and capability labels.  This module parameterizes the
shared ``unigym.backends.shared.isaaclab_query`` math by ``env.backend_name``
so a query handle always reports the concrete ``isaaclab3_physx`` or
``isaaclab3_newton`` identity.  IsaacLab 3.x live quaternions are ``xyzw`` at
the simulator boundary and are converted to repo-internal ``wxyz`` on read.

Only model facts from the published ``ModelIR`` and live body/root/COM/joint/
robot-profile-site reads are implemented.  Static actuator facts are a
facts-only subset; actuator runtime state, scratch, geometry, Jacobians and
contacts are declared unsupported and raise ``QueryCapabilityError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from unigym.backends.shared.isaaclab_query import (
    IsaacLabQuery,
    IsaacLabQueryEnv,
    isaaclab_query_capabilities,
)
from unigym.query.capabilities import QueryCapabilities

if TYPE_CHECKING:
    from unigym.env.backends.base import EnvBackend

__all__ = [
    "ISAACLAB3_NEWTON_QUERY_CAPABILITIES",
    "ISAACLAB3_PHYSX_QUERY_CAPABILITIES",
    "Isaaclab3Query",
    "isaaclab3_query_capabilities",
]

ISAACLAB3_PHYSX_QUERY_CAPABILITIES = isaaclab_query_capabilities("isaaclab3_physx")
ISAACLAB3_NEWTON_QUERY_CAPABILITIES = isaaclab_query_capabilities("isaaclab3_newton")

_CAPABILITIES: Mapping[str, QueryCapabilities] = {
    "isaaclab3_physx": ISAACLAB3_PHYSX_QUERY_CAPABILITIES,
    "isaaclab3_newton": ISAACLAB3_NEWTON_QUERY_CAPABILITIES,
}


def isaaclab3_query_capabilities(backend: str) -> QueryCapabilities:
    """Return the capability object declared for one IsaacLab 3 identity."""
    try:
        return _CAPABILITIES[backend]
    except KeyError:
        raise ValueError(
            f"unknown IsaacLab 3 backend {backend!r}; available: {sorted(_CAPABILITIES)}"
        ) from None


class Isaaclab3Query(IsaacLabQuery):
    """The IsaacLab 3.x runtime-query facade for the PhysX/Newton identities."""

    def __init__(self, env: EnvBackend) -> None:
        """Bind the query to one live IsaacLab 3 env and its concrete identity."""
        if not isinstance(env, IsaacLabQueryEnv):
            raise TypeError(f"Isaaclab3Query requires an IsaacLab 3 env, got {type(env).__name__}")
        backend = getattr(env, "backend_name", None)
        if not isinstance(backend, str) or backend not in _CAPABILITIES:
            raise TypeError(
                f"Isaaclab3Query requires an env with backend_name "
                f"'isaaclab3_physx' or 'isaaclab3_newton', got {backend!r}"
            )
        super().__init__(
            env,
            backend=backend,
            capabilities=_CAPABILITIES[backend],
            quat_order="xyzw",
        )
