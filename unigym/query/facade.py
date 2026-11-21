"""The runtime-query facade and its subface protocols.

``Query`` is the stable user entry (``run.query``); its subfaces group model
facts, live state, kinematics, geometry, contacts and actuators.  The contact
face is group-based: a manifold carries the active points, and the Jacobian and
wrench-to-joint operations map one point's constraint rows into an
articulation's DOF layout.  The actuator subface reads compiled facts and live
per-env control/force/length/velocity channels; backends that cannot answer an
operation declare it unsupported.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from ..model.ir import Actuator, Body, Geom, Joint, ModelIR, Site
from .base import (
    AABB,
    ActuatorFacts,
    ActuatorRuntime,
    ComRef,
    ContactJacobian,
    ContactManifold,
    ContactPoint,
    DistanceResult,
    EntityRef,
    EnvSelection,
    Jacobian,
    PointRef,
    Pose,
)
from .capabilities import QueryCapabilities

if TYPE_CHECKING:
    from unigym.env.backends.base import EnvBackend

__all__ = [
    "ActuatorQuery",
    "ContactQuery",
    "GeometryQuery",
    "KinematicsQuery",
    "ModelQuery",
    "Query",
    "QueryBackend",
    "ScratchWorld",
    "StateQuery",
]


class ModelQuery(Protocol):
    """Static model facts resolved from the published ModelIR."""

    def ir(self, articulation: str | None = None) -> ModelIR:
        """Return the whole-scene or one articulation's model IR."""
        ...

    def body(self, name: str, *, articulation: str | None = None) -> Body:
        """Return one body row by neutral name."""
        ...

    def geom(self, name: str, *, articulation: str | None = None) -> Geom:
        """Return one geom row by neutral name."""
        ...

    def site(self, name: str, *, articulation: str | None = None) -> Site:
        """Return one site row by neutral name."""
        ...

    def joint(self, name: str, *, articulation: str | None = None) -> Joint:
        """Return one joint row by neutral name."""
        ...

    def actuator(self, name: str, *, articulation: str | None = None) -> Actuator:
        """Return one actuator row by neutral name."""
        ...


class StateQuery(Protocol):
    """Live per-env state reads."""

    def body_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> Pose:
        """Return the world pose of one body."""
        ...

    def site_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> Pose:
        """Return the world pose of one site."""
        ...

    def com_pose(
        self,
        ref: ComRef,
        *,
        frame: str = "world",
        env: EnvSelection = None,
    ) -> Pose:
        """Return one body or subtree center of mass in world coordinates."""
        ...

    def root_pose(self, articulation: str, *, env: EnvSelection = None) -> Pose:
        """Return one articulation's free-root world pose."""
        ...

    def joint_position(
        self,
        articulation: str,
        *,
        joint_names: Sequence[str] | None = None,
        env: EnvSelection = None,
    ) -> torch.Tensor:
        """Return joint positions for one articulation."""
        ...

    def joint_velocity(
        self,
        articulation: str,
        *,
        joint_names: Sequence[str] | None = None,
        env: EnvSelection = None,
    ) -> torch.Tensor:
        """Return joint velocities for one articulation."""
        ...


class KinematicsQuery(Protocol):
    """Jacobian reads at bodies, sites, COM points and world points."""

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
        """Return the translational/rotational Jacobian of one target."""
        ...


class GeometryQuery(Protocol):
    """Distance and bounding-box geometry reads."""

    def distance(
        self,
        a: EntityRef,
        b: EntityRef,
        *,
        max_distance_m: float = 1.0,
        env: EnvSelection = None,
        scratch: ScratchWorld | None = None,
    ) -> DistanceResult:
        """Return the signed distance between two geoms."""
        ...

    def aabb(self, entity: EntityRef, *, env: EnvSelection = None) -> AABB:
        """Return the world AABB of one geom or body."""
        ...


class ContactQuery(Protocol):
    """Group-based contact manifolds and one point's constraint mapping."""

    def manifold(self, group: str, *, env: EnvSelection = None) -> ContactManifold:
        """Return one declared contact group's active points.

        ``group`` is required and resolves against the run's ``contacts:``
        groups; the group owns the body filter.  There is no all-contact
        world scan and no implicit default group.
        """
        ...

    def jacobian(self, point: ContactPoint, articulation: str) -> ContactJacobian:
        """Return the contact-frame constraint Jacobian rows for one point.

        ``point`` must come from a current manifold of the same env.
        """
        ...

    def wrench_to_joint(self, point: ContactPoint, articulation: str) -> torch.Tensor:
        """Return the contact force mapped to joint space as ``rows.T @ force``."""
        ...


class ActuatorQuery(Protocol):
    """Static actuator facts and per-env live runtime channels."""

    def rows(self, articulation: str | None = None) -> tuple[ActuatorFacts, ...]:
        """Return every selected actuator's facts in ``ModelIR`` order.

        ``articulation=None`` reads the merged whole-scene table; a
        multi-articulation run without a lossless merged IR raises
        ``QueryReferenceError`` instead of choosing one articulation.
        """
        ...

    def facts(self, name: str, *, articulation: str | None = None) -> ActuatorFacts:
        """Return one actuator's static facts by neutral name."""
        ...

    def state(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> ActuatorRuntime:
        """Return one actuator's per-env control, force, length and velocity."""
        ...


class ScratchWorld(Protocol):
    """An exclusive per-env scratch copy of the live world."""

    def set_root_state(
        self,
        name: str,
        *,
        pos_w: torch.Tensor,
        quat_wxyz: torch.Tensor,
        lin_vel_w: torch.Tensor | None = None,
        ang_vel_w: torch.Tensor | None = None,
    ) -> None:
        """Override one free root's pose and optional velocities."""
        ...

    def set_joint_state(
        self,
        articulation: str,
        joint_names: Sequence[str],
        position: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ) -> None:
        """Override absolute 1-DOF joint positions and optional velocities."""
        ...

    def forward(self) -> None:
        """Recompute derived quantities on the scratch copy."""
        ...

    def body_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
    ) -> Pose:
        """Return one body's scratch world pose."""
        ...

    def site_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
    ) -> Pose:
        """Return one site's scratch world pose."""
        ...

    def com_pose(self, ref: ComRef, *, frame: str = "world") -> Pose:
        """Return one COM reference on the scratch copy."""
        ...

    def geometry(self) -> GeometryQuery:
        """Return a geometry subface bound to the scratch copy."""
        ...

    def jacobian(
        self,
        target: EntityRef | ComRef | PointRef,
        *,
        articulation: str | None = None,
        frame: str = "world",
        reference: str = "body_origin",
        rotational: bool = True,
    ) -> Jacobian:
        """Return a Jacobian computed on the scratch copy."""
        ...

    def close(self) -> None:
        """Release the scratch copy (idempotent)."""
        ...


@runtime_checkable
class Query(Protocol):
    """The stable on-demand runtime query facade."""

    @property
    def backend(self) -> str:
        """The canonical backend name."""
        ...

    @property
    def capabilities(self) -> QueryCapabilities:
        """The backend's declared query capabilities."""
        ...

    @property
    def model(self) -> ModelQuery:
        """Static model facts."""
        ...

    @property
    def state(self) -> StateQuery:
        """Live per-env state reads."""
        ...

    @property
    def kinematics(self) -> KinematicsQuery:
        """Jacobian reads."""
        ...

    @property
    def geometry(self) -> GeometryQuery:
        """Distance and AABB reads."""
        ...

    @property
    def contacts(self) -> ContactQuery:
        """Group-based contact manifold reads."""
        ...

    @property
    def actuators(self) -> ActuatorQuery:
        """Actuator facts and live per-env runtime channels."""
        ...

    def scratch(self, env: int | None = None) -> AbstractContextManager[ScratchWorld]:
        """Open one exclusive scratch copy for *env* (default env 0)."""
        ...

    def require(self, *ops: str) -> None:
        """Raise when any named operation is unsupported."""
        ...

    def close(self) -> None:
        """Invalidate this query handle (idempotent)."""
        ...


class QueryBackend(Protocol):
    """A query backend class: construct it with the live env to get a Query."""

    def __call__(self, env: EnvBackend) -> Query:
        """Bind the query backend to one live env."""
        ...
