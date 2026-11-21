"""The runtime-query contract package.

Public entry points are ``run.query`` (trait-gated) and the optional
``env.query`` backend face.  Query is not an observation channel: it never
publishes to ``env.obs`` and never records.
"""

from __future__ import annotations

from .base import (
    AABB,
    ActuatorFacts,
    ActuatorRuntime,
    ActuatorSlotKind,
    ComRef,
    ContactJacobian,
    ContactManifold,
    ContactPoint,
    DistanceResult,
    DofLayout,
    EntityRef,
    EnvSelection,
    Jacobian,
    PointRef,
    Pose,
)
from .capabilities import (
    ActuatorCaps,
    ContactCaps,
    Feature,
    GeometryCaps,
    JacobianCaps,
    ModelCaps,
    QueryCapabilities,
    ScratchCaps,
    StateCaps,
    Support,
)
from .errors import (
    QueryCapabilityError,
    QueryError,
    QueryReferenceError,
    QueryShapeError,
    QueryStaleError,
)
from .facade import (
    ActuatorQuery,
    ContactQuery,
    GeometryQuery,
    KinematicsQuery,
    ModelQuery,
    Query,
    QueryBackend,
    ScratchWorld,
    StateQuery,
)

__all__ = [
    "AABB",
    "ActuatorCaps",
    "ActuatorFacts",
    "ActuatorQuery",
    "ActuatorRuntime",
    "ActuatorSlotKind",
    "ComRef",
    "ContactCaps",
    "ContactJacobian",
    "ContactManifold",
    "ContactPoint",
    "ContactQuery",
    "DistanceResult",
    "DofLayout",
    "EntityRef",
    "EnvSelection",
    "Feature",
    "GeometryCaps",
    "GeometryQuery",
    "Jacobian",
    "JacobianCaps",
    "KinematicsQuery",
    "ModelCaps",
    "ModelQuery",
    "PointRef",
    "Pose",
    "Query",
    "QueryBackend",
    "QueryCapabilities",
    "QueryCapabilityError",
    "QueryError",
    "QueryReferenceError",
    "QueryShapeError",
    "QueryStaleError",
    "ScratchCaps",
    "ScratchWorld",
    "StateCaps",
    "StateQuery",
    "Support",
]
