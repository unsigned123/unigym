"""IsaacLab 2.x implementation of the runtime-query face (P0B-2 model + pose).

The shared math lives in ``unigym.backends.shared.isaaclab_query``; this module
binds it to ``IsaaclabEnv`` and the ``isaaclab2`` capability declaration.
IsaacLab 2.x live quaternions are already repo-internal ``wxyz``.

Only model facts from the published ``ModelIR`` and live body/root/COM/joint/
robot-profile-site reads are implemented.  Static actuator facts are a
facts-only subset; actuator runtime state, scratch, geometry, Jacobians and
contacts are declared unsupported and raise ``QueryCapabilityError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unigym.backends.shared.isaaclab_query import (
    IsaacLabQuery,
    IsaacLabQueryEnv,
    isaaclab_query_capabilities,
)

if TYPE_CHECKING:
    from unigym.env.backends.base import EnvBackend

__all__ = ["ISAACLAB2_QUERY_CAPABILITIES", "Isaaclab2Query"]

ISAACLAB2_QUERY_CAPABILITIES = isaaclab_query_capabilities("isaaclab2")


class Isaaclab2Query(IsaacLabQuery):
    """The IsaacLab 2.x runtime-query facade over the live articulation buffers."""

    backend = "isaaclab2"
    capabilities = ISAACLAB2_QUERY_CAPABILITIES

    def __init__(self, env: EnvBackend) -> None:
        """Bind the query to one live ``IsaaclabEnv``."""
        if not isinstance(env, IsaacLabQueryEnv):
            raise TypeError(f"Isaaclab2Query requires an IsaaclabEnv, got {type(env).__name__}")
        super().__init__(
            env,
            backend=self.backend,
            capabilities=self.capabilities,
            quat_order="wxyz",
        )
