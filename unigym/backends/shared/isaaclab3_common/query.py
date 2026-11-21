"""IsaacLab 3.x shared runtime-query implementation (model + pose + Jacobian).

The PhysX and Newton identities share one implementation package but own
separate runtime stacks and capability labels.  This module parameterizes the
shared ``unigym.backends.shared.isaaclab_query`` math by ``env.backend_name``
so a query handle always reports the concrete ``isaaclab3_physx`` or
``isaaclab3_newton`` identity.  IsaacLab 3.x live quaternions are ``xyzw`` at
the simulator boundary and are converted to repo-internal ``wxyz`` on read.

Model facts come from the published ``ModelIR``; live body/root/COM/joint/
robot-profile-site reads are native.  Static actuator facts are a facts-only
subset.  P1C-1 adds the native body/body-COM Jacobians and composes
site/world-point rows from the body link Jacobian; actuator runtime state,
scratch, geometry and contacts stay unsupported and raise
``QueryCapabilityError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from unigym.backends.shared.isaaclab_query import (
    IsaacLabJacobianQuery,
    IsaacLabQuery,
    IsaacLabQueryEnv,
    isaaclab_query_capabilities,
)
from unigym.query.capabilities import Feature, JacobianCaps, QueryCapabilities

if TYPE_CHECKING:
    from unigym.env.backends.base import EnvBackend

__all__ = [
    "ISAACLAB3_NEWTON_QUERY_CAPABILITIES",
    "ISAACLAB3_PHYSX_QUERY_CAPABILITIES",
    "Isaaclab3Query",
    "isaaclab3_query_capabilities",
]

_JACOBIAN_SCOPE = (
    "model facts + live pose + static actuator facts + native body/body-COM "
    "Jacobian with synthetic site/world-point rows; no scratch/geometry/contact/"
    "actuator-state"
)


def _isaaclab3_capabilities(backend: str) -> QueryCapabilities:
    """Return the P1C-1 capability declaration for one IsaacLab 3 identity."""
    base = isaaclab_query_capabilities(backend)
    notes = dict(base.notes)
    notes["scope"] = _JACOBIAN_SCOPE
    notes["jacobian"] = (
        "body and body COM are native engine rows; site/world-point rows are "
        "synthesized from the body link row plus the offset; a subtree COM row is "
        "not exposed by the engine"
    )
    return replace(
        base,
        jacobian=JacobianCaps(
            body=Feature(
                "native",
                "native body link Jacobian at the body origin, base DoFs first",
            ),
            site=Feature(
                "synthetic",
                "body link Jacobian composed with the robot-profile site local offset",
            ),
            com=Feature(
                "partial",
                "native body COM Jacobian; subtree COM is unsupported",
            ),
            point=Feature(
                "synthetic",
                "body link Jacobian composed with the world-point offset",
            ),
        ),
        notes=notes,
    )


ISAACLAB3_PHYSX_QUERY_CAPABILITIES = _isaaclab3_capabilities("isaaclab3_physx")
ISAACLAB3_NEWTON_QUERY_CAPABILITIES = _isaaclab3_capabilities("isaaclab3_newton")

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
        self.kinematics = IsaacLabJacobianQuery(self)
