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
site/world-point rows from the body link Jacobian.  P1C-2 adds the exclusive
per-env transaction scratch: root/joint writes go through the live state-write
face and close restores the captured state.  P1C-3 adds the PhysX filtered
pairwise contact manifold for groups that declare ``manifold: true``; the
Newton manifold, contact Jacobians and wrench-to-joint stay unsupported.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import TYPE_CHECKING

from unigym.backends.shared.isaaclab_query import (
    SCRATCH_CONTACT_DETAIL,
    IsaacLabContactQuery,
    IsaacLabJacobianQuery,
    IsaacLabQuery,
    IsaacLabQueryEnv,
    isaaclab_query_capabilities,
)
from unigym.query.capabilities import (
    ContactCaps,
    Feature,
    JacobianCaps,
    QueryCapabilities,
    ScratchCaps,
)
from unigym.query.facade import ScratchWorld

if TYPE_CHECKING:
    from unigym.env.backends.base import EnvBackend

__all__ = [
    "ISAACLAB3_NEWTON_QUERY_CAPABILITIES",
    "ISAACLAB3_PHYSX_QUERY_CAPABILITIES",
    "Isaaclab3Query",
    "isaaclab3_query_capabilities",
]

_SCOPE = (
    "model facts + live pose + static actuator facts + native body/body-COM "
    "Jacobian with synthetic site/world-point rows + transaction scratch + "
    "PhysX filtered pairwise contact manifold; no scratch geometry/contact/"
    "actuator-state"
)
_SCRATCH_TRANSACTION = (
    "exclusive per-env transaction over the live state; IsaacLab exposes no " "snapshot copy"
)
_SCRATCH_RESTORED = (
    "captured root/joint state is restored and forwarded on close; restored, " "not bit-exact"
)
_SCRATCH_ROOT = (
    "one free-root articulation through the live state-write face; a fixed root " "stays read-only"
)
_SCRATCH_JOINT = (
    "explicit absolute 1-DOF hinge/slide joint positions and optional "
    "velocities through the live state-write face"
)
_SCRATCH_FORWARD = "scene.write_data_to_sim() + sim.forward(); no implicit sync"
_SCRATCH_READ = "the live articulation buffers inside the active transaction"
_SCRATCH_GEOMETRY = "geometry and contact reads are not supported in an IsaacLab scratch"
_PHYSX_CONTACT = (
    "filtered pairwise PhysX contact manifolds for groups declared manifold: true; "
    "per-contact world position, separation and non-negative normal-force magnitude; "
    "the engine normal direction is not exposed (normal_w/frame_w are None), there are "
    "no geom ids or per-point friction, and no contact Jacobian/wrench"
)
_NEWTON_CONTACT = (
    "the IsaacLab 3 Newton filtered contact manifold is not implemented; "
    "no contact Jacobian/wrench either"
)


def _isaaclab3_capabilities(backend: str) -> QueryCapabilities:
    """Return the P1C-3 capability declaration for one IsaacLab 3 identity."""
    base = isaaclab_query_capabilities(backend)
    notes = dict(base.notes)
    notes["scope"] = _SCOPE
    notes["jacobian"] = (
        "body and body COM are native engine rows; site/world-point rows are "
        "synthesized from the body link row plus the offset; a subtree COM row is "
        "not exposed by the engine"
    )
    notes["scratch"] = (
        "exclusive per-env transaction; the live root/joint state is captured on "
        "enter, written and forwarded by explicit calls, then restored on close. "
        "There is no snapshot copy and no concurrent step/reset."
    )
    notes["contacts"] = (
        "PhysX filtered pairwise manifolds only, for contact groups that declare "
        "manifold: true; positions, separation and a non-negative force magnitude "
        "are exposed, the contact-normal direction is not; the Newton contact "
        "manifold and all contact Jacobian/wrench reads stay unsupported"
    )
    manifold = (
        Feature("partial", _PHYSX_CONTACT)
        if backend == "isaaclab3_physx"
        else Feature("unsupported", _NEWTON_CONTACT)
    )
    return replace(
        base,
        scratch=ScratchCaps(
            copy=Feature("synthetic", _SCRATCH_TRANSACTION),
            live_untouched=Feature("partial", _SCRATCH_RESTORED),
            set_root_state=Feature("synthetic", _SCRATCH_ROOT),
            set_joint_state=Feature("synthetic", _SCRATCH_JOINT),
            forward=Feature("synthetic", _SCRATCH_FORWARD),
            body_pose=Feature("synthetic", _SCRATCH_READ),
            site_pose=Feature("synthetic", _SCRATCH_READ),
            com_pose=Feature("synthetic", _SCRATCH_READ),
            geometry=Feature("unsupported", _SCRATCH_GEOMETRY),
            jacobian=Feature("synthetic", _SCRATCH_READ),
            contacts=Feature("unsupported", SCRATCH_CONTACT_DETAIL),
            mode="transaction",
            isolation="restored",
        ),
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
        contacts=ContactCaps(
            manifold=manifold,
            jacobian=base.contacts.jacobian,
            wrench_to_joint=base.contacts.wrench_to_joint,
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
        self.contacts = IsaacLabContactQuery(self)

    def scratch(self, env: int | None = None) -> AbstractContextManager[ScratchWorld]:
        """Open one exclusive transaction scratch for *env* (default env 0)."""
        self._require_open()
        return self._open_scratch(self._scratch_env_id(env))
