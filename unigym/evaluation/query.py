"""Read-only, env-scoped adapter over the runtime query facade.

:class:`EvalQuery` mirrors the read subset of
:class:`~unigym.query.facade.Query`.  It deliberately omits ``scratch()`` and
``close()``: an evaluation callback may not open a writable scratch world and
may not close the run-shared query handle.  Every read checks that the owning
:class:`~unigym.evaluation.context.EnvView` is still open; a per-env view fixes
the env selection and rejects any read or contact point outside its scope.

Query response types already validate and detach their tensors in the query
layer, so results returned here are safe copies.  The adapter itself is bound
to the view lifetime and is invalid after ``EnvView.close()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol

import torch

from ..model.ir import Actuator, Body, Geom, GeomContact, Joint, ModelIR, Site
from ..query.base import (
    AABB,
    ActuatorFacts,
    ActuatorRuntime,
    ComRef,
    ComState,
    ContactJacobian,
    ContactManifold,
    ContactPoint,
    DistanceResult,
    DistanceScan,
    EntityRef,
    EnvSelection,
    Jacobian,
    ObjectRef,
    PointRef,
    Pose,
    RigidState,
)
from ..query.capabilities import QueryCapabilities
from ..query.facade import (
    ActuatorQuery,
    ContactQuery,
    GeometryQuery,
    KinematicsQuery,
    ModelQuery,
    Query,
    StateQuery,
)
from .context import EvalViewError

if TYPE_CHECKING:
    from .context import EnvView

__all__ = ["EvalQuery"]


def _scoped_env(env: EnvSelection, scope: int | None) -> EnvSelection:
    """Fix *env* to *scope* for a per-env view and reject out-of-scope reads."""
    if scope is None:
        return env
    if env is None:
        return scope
    if isinstance(env, bool):
        raise EvalViewError(f"env selection must not be a bool, got {env!r}")
    if isinstance(env, int):
        selected = [env]
    elif isinstance(env, torch.Tensor):
        selected = [int(value) for value in env.detach().reshape(-1).tolist()]
    else:
        selected = [int(value) for value in env]
    if selected != [scope]:
        raise EvalViewError(
            f"env selection {selected} is outside this view's scope (env_id={scope})"
        )
    return scope


class _ScopedModel:
    """Read-only model subface; static facts carry no env axis."""

    __slots__ = ("_inner", "_view")

    def __init__(self, view: EnvView, inner: ModelQuery) -> None:
        self._view = view
        self._inner = inner

    def ir(self, articulation: str | None = None) -> ModelIR:
        self._view._ensure_open()
        return self._inner.ir(articulation)

    def body(self, name: str, *, articulation: str | None = None) -> Body:
        self._view._ensure_open()
        return self._inner.body(name, articulation=articulation)

    def geom(self, name: str, *, articulation: str | None = None) -> Geom:
        self._view._ensure_open()
        return self._inner.geom(name, articulation=articulation)

    def site(self, name: str, *, articulation: str | None = None) -> Site:
        self._view._ensure_open()
        return self._inner.site(name, articulation=articulation)

    def joint(self, name: str, *, articulation: str | None = None) -> Joint:
        self._view._ensure_open()
        return self._inner.joint(name, articulation=articulation)

    def actuator(self, name: str, *, articulation: str | None = None) -> Actuator:
        self._view._ensure_open()
        return self._inner.actuator(name, articulation=articulation)

    def geom_contact(
        self,
        name: str,
        *,
        articulation: str | None = None,
    ) -> GeomContact:
        self._view._ensure_open()
        return self._inner.geom_contact(name, articulation=articulation)


class _ScopedState:
    """Read-only live-state subface with the view's env selection fixed."""

    __slots__ = ("_inner", "_view")

    def __init__(self, view: EnvView, inner: StateQuery) -> None:
        self._view = view
        self._inner = inner

    def body_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> Pose:
        return self._inner.body_pose(
            name, articulation=articulation, env=_scoped_env(env, self._view.env_id)
        )

    def site_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> Pose:
        return self._inner.site_pose(
            name, articulation=articulation, env=_scoped_env(env, self._view.env_id)
        )

    def com_pose(
        self,
        ref: ComRef,
        *,
        frame: str = "world",
        env: EnvSelection = None,
    ) -> Pose:
        return self._inner.com_pose(ref, frame=frame, env=_scoped_env(env, self._view.env_id))

    def root_pose(self, articulation: str, *, env: EnvSelection = None) -> Pose:
        return self._inner.root_pose(articulation, env=_scoped_env(env, self._view.env_id))

    def root_state(self, articulation: str, *, env: EnvSelection = None) -> RigidState:
        return self._inner.root_state(articulation, env=_scoped_env(env, self._view.env_id))

    def com_state(
        self,
        ref: ComRef,
        *,
        frame: str = "world",
        env: EnvSelection = None,
    ) -> ComState:
        return self._inner.com_state(ref, frame=frame, env=_scoped_env(env, self._view.env_id))

    def object_state(
        self,
        ref: ObjectRef,
        *,
        env: EnvSelection = None,
    ) -> RigidState:
        return self._inner.object_state(ref, env=_scoped_env(env, self._view.env_id))

    def object_com_state(
        self,
        ref: ObjectRef,
        *,
        env: EnvSelection = None,
    ) -> ComState:
        return self._inner.object_com_state(ref, env=_scoped_env(env, self._view.env_id))

    def joint_position(
        self,
        articulation: str,
        *,
        joint_names: Sequence[str] | None = None,
        env: EnvSelection = None,
    ) -> torch.Tensor:
        return self._inner.joint_position(
            articulation,
            joint_names=joint_names,
            env=_scoped_env(env, self._view.env_id),
        )

    def joint_velocity(
        self,
        articulation: str,
        *,
        joint_names: Sequence[str] | None = None,
        env: EnvSelection = None,
    ) -> torch.Tensor:
        return self._inner.joint_velocity(
            articulation,
            joint_names=joint_names,
            env=_scoped_env(env, self._view.env_id),
        )


class _ScopedKinematics:
    """Read-only Jacobian subface with the view's env selection fixed."""

    __slots__ = ("_inner", "_view")

    def __init__(self, view: EnvView, inner: KinematicsQuery) -> None:
        self._view = view
        self._inner = inner

    def jacobian(
        self,
        target: EntityRef | ComRef | PointRef,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
        frame: str = "world",
        reference: str = "body_origin",
        rotational: bool = True,
    ) -> Jacobian:
        return self._inner.jacobian(
            target,
            articulation=articulation,
            env=_scoped_env(env, self._view.env_id),
            frame=frame,
            reference=reference,
            rotational=rotational,
        )

    def object_jacobian(
        self,
        ref: ObjectRef,
        *,
        reference: Literal["body_origin", "com"] = "body_origin",
        env: EnvSelection = None,
        frame: str = "world",
        rotational: bool = True,
    ) -> Jacobian:
        return self._inner.object_jacobian(
            ref,
            reference=reference,
            env=_scoped_env(env, self._view.env_id),
            frame=frame,
            rotational=rotational,
        )


class _EvalGeometryQuery(Protocol):
    """The geometry read subset of :class:`GeometryQuery`, without scratch."""

    def distance(
        self,
        a: EntityRef,
        b: EntityRef,
        *,
        max_distance_m: float = 1.0,
        env: EnvSelection = None,
    ) -> DistanceResult:
        """Return the signed distance between two geoms."""
        ...

    def aabb(self, entity: EntityRef, *, env: EnvSelection = None) -> AABB:
        """Return the world AABB of one geom or body."""
        ...

    def all_pairs(
        self,
        *,
        max_distance_m: float = 1.0,
        env: EnvSelection = None,
    ) -> DistanceScan:
        """Return every collision-geom pair's signed distance."""
        ...


class _ScopedGeometry:
    """Read-only geometry subface; no scratch world is reachable."""

    __slots__ = ("_inner", "_view")

    def __init__(self, view: EnvView, inner: GeometryQuery) -> None:
        self._view = view
        self._inner = inner

    def distance(
        self,
        a: EntityRef,
        b: EntityRef,
        *,
        max_distance_m: float = 1.0,
        env: EnvSelection = None,
    ) -> DistanceResult:
        return self._inner.distance(
            a,
            b,
            max_distance_m=max_distance_m,
            env=_scoped_env(env, self._view.env_id),
        )

    def aabb(self, entity: EntityRef, *, env: EnvSelection = None) -> AABB:
        return self._inner.aabb(entity, env=_scoped_env(env, self._view.env_id))

    def all_pairs(
        self,
        *,
        max_distance_m: float = 1.0,
        env: EnvSelection = None,
    ) -> DistanceScan:
        return self._inner.all_pairs(
            max_distance_m=max_distance_m,
            env=_scoped_env(env, self._view.env_id),
        )


class _ScopedContacts:
    """Read-only contact subface; contact points must belong to the scope."""

    __slots__ = ("_inner", "_view")

    def __init__(self, view: EnvView, inner: ContactQuery) -> None:
        self._view = view
        self._inner = inner

    def manifold(self, group: str, *, env: EnvSelection = None) -> ContactManifold:
        return self._inner.manifold(group, env=_scoped_env(env, self._view.env_id))

    def scan(self, *, env: EnvSelection = None) -> ContactManifold:
        return self._inner.scan(env=_scoped_env(env, self._view.env_id))

    def jacobian(self, point: ContactPoint, articulation: str) -> ContactJacobian:
        return self._inner.jacobian(self._scoped_point(point), articulation)

    def wrench_to_joint(self, point: ContactPoint, articulation: str) -> torch.Tensor:
        return self._inner.wrench_to_joint(self._scoped_point(point), articulation)

    def _scoped_point(self, point: ContactPoint) -> ContactPoint:
        scope = self._view.env_id
        if scope is not None and point.env_id != scope:
            raise EvalViewError(
                f"contact point env_id {point.env_id} is outside this view's "
                f"scope (env_id={scope})"
            )
        return point


class _ScopedActuators:
    """Read-only actuator subface with the view's env selection fixed."""

    __slots__ = ("_inner", "_view")

    def __init__(self, view: EnvView, inner: ActuatorQuery) -> None:
        self._view = view
        self._inner = inner

    def rows(self, articulation: str | None = None) -> tuple[ActuatorFacts, ...]:
        self._view._ensure_open()
        return self._inner.rows(articulation)

    def facts(self, name: str, *, articulation: str | None = None) -> ActuatorFacts:
        self._view._ensure_open()
        return self._inner.facts(name, articulation=articulation)

    def state(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> ActuatorRuntime:
        return self._inner.state(
            name,
            articulation=articulation,
            env=_scoped_env(env, self._view.env_id),
        )


class EvalQuery:
    """Read-only scoped adapter over a live :class:`Query`.

    The adapter exposes ``backend``, ``capabilities``, ``model``, ``state``,
    ``kinematics``, ``geometry``, ``contacts``, ``actuators`` and ``require``.
    It has no ``scratch()`` and no ``close()``; the run-shared query handle
    stays owned by the run/env and is never closed by the view.
    """

    __slots__ = (
        "_actuators",
        "_contacts",
        "_geometry",
        "_kinematics",
        "_model",
        "_owner",
        "_query",
        "_state",
    )

    def __init__(self, owner: EnvView, query: Query) -> None:
        self._owner = owner
        self._query = query
        self._model = _ScopedModel(owner, query.model)
        self._state = _ScopedState(owner, query.state)
        self._kinematics = _ScopedKinematics(owner, query.kinematics)
        self._geometry = _ScopedGeometry(owner, query.geometry)
        self._contacts = _ScopedContacts(owner, query.contacts)
        self._actuators = _ScopedActuators(owner, query.actuators)

    @property
    def backend(self) -> str:
        """The canonical backend name."""
        self._owner._ensure_open()
        return self._query.backend

    @property
    def capabilities(self) -> QueryCapabilities:
        """The backend's declared query capabilities."""
        self._owner._ensure_open()
        return self._query.capabilities

    @property
    def model(self) -> ModelQuery:
        """Static model facts."""
        self._owner._ensure_open()
        return self._model

    @property
    def state(self) -> StateQuery:
        """Live per-env state reads."""
        self._owner._ensure_open()
        return self._state

    @property
    def kinematics(self) -> KinematicsQuery:
        """Jacobian reads."""
        self._owner._ensure_open()
        return self._kinematics

    @property
    def geometry(self) -> _EvalGeometryQuery:
        """Distance and AABB reads."""
        self._owner._ensure_open()
        return self._geometry

    @property
    def contacts(self) -> ContactQuery:
        """Contact manifold and active-contact scan reads."""
        self._owner._ensure_open()
        return self._contacts

    @property
    def actuators(self) -> ActuatorQuery:
        """Actuator facts and live per-env runtime channels."""
        self._owner._ensure_open()
        return self._actuators

    def require(self, *ops: str) -> None:
        """Raise when any named operation is unsupported by the backend."""
        self._owner._ensure_open()
        self._query.require(*ops)
