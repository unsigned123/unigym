"""Shared IsaacLab runtime-query implementation (model facts and live pose).

IsaacLab 2.x and the two IsaacLab 3.x identities expose the same articulation
runtime surface (``env.articulations`` / ``env.robots`` / ``env.model_set``), so
the query math owns here.  A backend subclass supplies its env type, its
canonical backend label, its capability object and the live quaternion order;
it never copies the model lookup, state read or site-composition code.

P0B-2 implements only the model facts carried by the published ``ModelIR`` and
live body/root/COM/joint/site reads.  Scratch, geometry, Jacobians, contacts
and actuator runtime are declared unsupported and raise
``QueryCapabilityError``; they belong to P1.

Engine-thread contract: the query reads the live articulation buffers at a
step boundary and adds no lock.  Every result is a detached CPU torch copy; no
observation is published and nothing is recorded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import NoReturn, Protocol, runtime_checkable

import torch

from unigym.backends.obs_schema import build_site_inventory
from unigym.backends.shared.isaaclab_bodies import body_binding_candidates
from unigym.core import quat as quat_math
from unigym.env.backends.base import normalize_state_env_ids
from unigym.model import Actuator, Body, Geom, Joint, ModelIR, ModelSet, Site
from unigym.query.base import ComRef, EntityRef, EnvSelection, PointRef, Pose
from unigym.query.capabilities import (
    ActuatorCaps,
    ContactCaps,
    Feature,
    GeometryCaps,
    JacobianCaps,
    ModelCaps,
    QueryCapabilities,
    ScratchCaps,
    StateCaps,
)
from unigym.query.errors import (
    QueryCapabilityError,
    QueryReferenceError,
    QueryShapeError,
    QueryStaleError,
)
from unigym.query.facade import ScratchWorld
from unigym.robots.base import RobotProfile, normalize_canonical_body_id
from unigym.runtime.controllers import ArticulationRuntime
from unigym.sites import SiteEntry

__all__ = [
    "IsaacLabQuery",
    "IsaacLabQueryEnv",
    "isaaclab_query_capabilities",
]

_P1_DETAIL = (
    "P0B-2 implements model facts and live pose only; "
    "the IsaacLab query does not implement this operation"
)
_GEOM_DETAIL = (
    "the IsaacLab ModelIR carries bodies/joints/actuators only; " "no geom facts are synthesized"
)
_SITE_MODEL_DETAIL = (
    "the IsaacLab ModelIR carries bodies/joints/actuators only; "
    "no site model facts are synthesized (use state.site_pose for robot-profile sites)"
)
_COM_DETAIL = (
    "body COM position only; the Pose quaternion is identity by contract and "
    "subtree COM is unsupported"
)
_SITE_STATE_DETAIL = (
    "robot-profile sites only (body + local offset); no USD prim or " "FrameTransformer fallback"
)
_ENGINE_THREAD = "engine thread only; query at a step boundary, no concurrent step/reset"
_COPIES = "detached CPU torch copies; no obs publication or record writes"


@runtime_checkable
class IsaacLabQueryEnv(Protocol):
    """The IsaacLab env surface the shared query implementation reads."""

    @property
    def num_envs(self) -> int:
        """Number of live cloned worlds."""
        ...

    @property
    def model_set(self) -> ModelSet:
        """The published per-articulation model views."""
        ...

    @property
    def articulations(self) -> Mapping[str, ArticulationRuntime]:
        """The live articulation runtime table."""
        ...

    @property
    def robots(self) -> Mapping[str, object]:
        """The live articulation assets keyed by articulation name."""
        ...


def isaaclab_query_capabilities(backend: str) -> QueryCapabilities:
    """Return the P0B-2 capability declaration for one IsaacLab identity.

    Args:
        backend: The canonical backend label (``isaaclab2``,
            ``isaaclab3_physx`` or ``isaaclab3_newton``).

    Returns:
        The honest model+pose capability object for that identity.
    """
    native = Feature("native")
    unsupported = Feature("unsupported", _P1_DETAIL)
    return QueryCapabilities(
        api_version=1,
        backend=backend,
        model=ModelCaps(
            ir=native,
            body=native,
            geom=Feature("unsupported", _GEOM_DETAIL),
            site=Feature("unsupported", _SITE_MODEL_DETAIL),
            joint=native,
            actuator=native,
        ),
        state=StateCaps(
            body_pose=native,
            site_pose=Feature("partial", _SITE_STATE_DETAIL),
            com_pose=Feature("partial", _COM_DETAIL),
            root_pose=native,
            joint_position=native,
            joint_velocity=native,
        ),
        scratch=ScratchCaps(
            copy=unsupported,
            live_untouched=unsupported,
            set_root_state=unsupported,
            set_joint_state=unsupported,
            forward=unsupported,
            body_pose=unsupported,
            site_pose=unsupported,
            com_pose=unsupported,
            geometry=unsupported,
            jacobian=unsupported,
        ),
        jacobian=JacobianCaps(
            body=unsupported,
            site=unsupported,
            com=unsupported,
            point=unsupported,
        ),
        geometry=GeometryCaps(
            distance=Feature("unsupported", _P1_DETAIL),
            aabb=Feature("unsupported", _P1_DETAIL),
        ),
        contacts=ContactCaps(
            manifold=Feature("unsupported", _P1_DETAIL),
            jacobian=Feature("unsupported", _P1_DETAIL),
            wrench_to_joint=Feature("unsupported", _P1_DETAIL),
        ),
        actuators=ActuatorCaps(
            facts=Feature("unsupported", _P1_DETAIL),
            state=Feature("unsupported", _P1_DETAIL),
        ),
        notes={
            "engine_thread": _ENGINE_THREAD,
            "returns": _COPIES,
            "scope": "P0B-2 model facts + live pose; no scratch/geometry/Jacobian/contact",
            "sites": "state.site_pose composes robot-profile sites from body + local offset",
        },
    )


def _as_names(value: object) -> tuple[str, ...]:
    """Return a name sequence from a list, tensor or IsaacLab ``ProxyArray``."""
    if value is None:
        return ()
    payload = getattr(value, "torch", None)
    if payload is not None:
        value = payload
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    raise QueryShapeError("names", "an iterable of names", type(value).__name__)


def _tensor_view(value: object, field: str) -> torch.Tensor:
    """Return the explicit ``.torch`` payload of an IsaacLab data view."""
    payload = getattr(value, "torch", None)
    if payload is not None:
        value = payload
    if not isinstance(value, torch.Tensor):
        raise QueryShapeError(field, "a torch.Tensor or ProxyArray", type(value).__name__)
    return value


def _select_env(selection: EnvSelection, num_envs: int) -> torch.Tensor:
    """Normalize ``env=None``/int/list/tensor to a 1-D int64 CPU id vector."""
    if selection is None:
        return normalize_state_env_ids(None, num_envs, device="cpu")
    if isinstance(selection, bool):
        raise QueryShapeError("env", "an int, a sequence of ints or None", repr(selection))
    if isinstance(selection, torch.Tensor):
        return normalize_state_env_ids(selection, num_envs, device="cpu")
    values = [selection] if isinstance(selection, int) else list(selection)
    if not values:
        return torch.empty(0, dtype=torch.int64)
    return normalize_state_env_ids(values, num_envs, device="cpu")


def _index_of(op: str, name: str, names: Sequence[str]) -> int:
    """Return ``names.index(name)`` or a teaching reference error."""
    try:
        return list(names).index(name)
    except ValueError:
        raise QueryReferenceError(op, name, names) from None


def _body_index(op: str, usd_name: str, names: Sequence[str]) -> int:
    """Resolve one profile USD body binding against the live body order."""
    matches = body_binding_candidates(names, usd_name)
    if len(matches) == 1:
        return matches[0]
    raise QueryReferenceError(
        op,
        usd_name,
        names,
        "the profile body binding does not resolve to exactly one live body",
    )


class _Subface:
    """Common open/name resolution for one query subface."""

    def __init__(self, query: IsaacLabQuery) -> None:
        """Bind the subface to its owning query handle."""
        self._query = query

    def _open(self) -> None:
        """Reject use after the query handle was closed."""
        self._query._require_open()


class _ModelQuery(_Subface):
    """Static model facts resolved from the published ``ModelIR``."""

    def ir(self, articulation: str | None = None) -> ModelIR:
        """Return the whole-scene or one articulation's model IR."""
        self._open()
        return self._query._ir(articulation)

    def body(self, name: str, *, articulation: str | None = None) -> Body:
        """Return one body row by neutral name."""
        self._open()
        return self._query._body_row(name, articulation)

    def geom(self, name: str, *, articulation: str | None = None) -> Geom:
        """Reject geom model facts: the IsaacLab ModelIR carries none."""
        del name, articulation
        self._open()
        self._query.capabilities.require("model.geom")
        raise AssertionError("model.geom is declared unsupported but require() returned")

    def site(self, name: str, *, articulation: str | None = None) -> Site:
        """Reject site model facts: the IsaacLab ModelIR carries none."""
        del name, articulation
        self._open()
        self._query.capabilities.require("model.site")
        raise AssertionError("model.site is declared unsupported but require() returned")

    def joint(self, name: str, *, articulation: str | None = None) -> Joint:
        """Return one joint row by neutral name."""
        self._open()
        return self._query._joint_row(name, articulation)

    def actuator(self, name: str, *, articulation: str | None = None) -> Actuator:
        """Return one actuator row by neutral name."""
        self._open()
        return self._query._actuator_row(name, articulation)


class _StateQuery(_Subface):
    """Live per-env body/root/COM/joint/site reads."""

    def body_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> Pose:
        """Return the world pose of one body."""
        self._open()
        art_name = self._query._resolve_body_articulation("state.body_pose", name, articulation)
        asset = self._query._asset_for("state.body_pose", art_name)
        data = self._query._live_data("state.body_pose", asset, art_name)
        names = self._query._body_names(asset, data)
        index = _index_of("state.body_pose", name, names)
        ids = _select_env(env, self._query._env.num_envs)
        pos = self._query._read_rows("state.body_pose", data, "body_pos_w", ids, index, 3, False)
        quat = self._query._read_rows("state.body_pose", data, "body_quat_w", ids, index, 4, True)
        return Pose(ids, pos, quat)

    def site_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> Pose:
        """Return one robot-profile site's world pose (body + local offset)."""
        self._open()
        inventory = self._query._site_inventory()
        entry = inventory.get(name)
        if entry is None:
            raise QueryReferenceError("state.site_pose", name, tuple(sorted(inventory)))
        if entry.kind != "articulation":
            raise QueryCapabilityError(
                self._query.backend,
                "state.site_pose",
                "partial",
                f"site {name!r} is a generated prop site; only robot-profile sites "
                f"are supported in P0B-2",
            )
        if articulation is not None and articulation != entry.owner:
            raise QueryReferenceError(
                "state.site_pose",
                name,
                tuple(sorted(inventory)),
                f"site {name!r} is owned by articulation {entry.owner!r}",
            )
        profile = self._query._profiles().get(entry.owner)
        if profile is None:
            raise QueryReferenceError(
                "state.site_pose",
                name,
                tuple(sorted(inventory)),
                f"articulation {entry.owner!r} has no profile",
            )
        canonical = normalize_canonical_body_id(entry.spec.body, f"query site {name!r} body")
        binding = profile.body_bindings.get(canonical)
        if binding is None or "usd" not in binding.names:
            raise QueryReferenceError(
                "state.site_pose",
                name,
                tuple(sorted(inventory)),
                f"site body {entry.spec.body!r} has no usd binding in profile " f"{entry.owner!r}",
            )
        asset = self._query._asset_for("state.site_pose", entry.owner)
        data = self._query._live_data("state.site_pose", asset, entry.owner)
        names = self._query._body_names(asset, data)
        index = _body_index("state.site_pose", binding.names["usd"], names)
        ids = _select_env(env, self._query._env.num_envs)
        body_pos = self._query._read_rows(
            "state.site_pose", data, "body_pos_w", ids, index, 3, False
        )
        body_quat = self._query._read_rows(
            "state.site_pose", data, "body_quat_w", ids, index, 4, True
        )
        local_pos = torch.tensor(entry.spec.pos, dtype=torch.float64).expand(body_pos.shape[0], -1)
        local_quat = torch.tensor(entry.spec.quat_wxyz, dtype=torch.float64)
        pos_w = body_pos + quat_math.rotate(body_quat, local_pos)
        quat_w = quat_math.normalize(quat_math.mul(body_quat, local_quat))
        return Pose(ids, pos_w, quat_w)

    def com_pose(
        self,
        ref: ComRef,
        *,
        frame: str = "world",
        env: EnvSelection = None,
    ) -> Pose:
        """Return one body's COM position in world coordinates."""
        self._open()
        if frame != "world":
            raise QueryCapabilityError(
                self._query.backend,
                "state.com_pose",
                "partial",
                f"frame={frame!r}; only 'world' is implemented",
            )
        if ref.subtree:
            raise QueryCapabilityError(
                self._query.backend,
                "state.com_pose",
                "partial",
                "subtree COM is not implemented in P0B-2; pass subtree=False",
            )
        art = self._query._articulation(ref.articulation)
        body_name = ref.body or str(getattr(art, "base_body", "") or "")
        if not body_name:
            raise QueryReferenceError(
                "state.com_pose",
                ref.articulation,
                (),
                "no base body is declared for this articulation",
            )
        ir = self._query._ir(ref.articulation)
        try:
            row = ir.bodies[body_name]
        except KeyError:
            raise QueryReferenceError("state.com_pose", body_name, ir.bodies.names()) from None
        if row.ipos is None:
            raise QueryCapabilityError(
                self._query.backend,
                "state.com_pose",
                "partial",
                f"body {body_name!r} has no published COM offset",
            )
        asset = self._query._asset_for("state.com_pose", ref.articulation)
        data = self._query._live_data("state.com_pose", asset, ref.articulation)
        names = self._query._body_names(asset, data)
        index = _index_of("state.com_pose", body_name, names)
        ids = _select_env(env, self._query._env.num_envs)
        body_pos = self._query._read_rows(
            "state.com_pose", data, "body_pos_w", ids, index, 3, False
        )
        body_quat = self._query._read_rows(
            "state.com_pose", data, "body_quat_w", ids, index, 4, True
        )
        local = torch.tensor(row.ipos, dtype=torch.float64).expand(body_pos.shape[0], -1)
        pos_w = body_pos + quat_math.rotate(body_quat, local)
        identity = torch.zeros((ids.numel(), 4), dtype=torch.float64)
        identity[:, 0] = 1.0
        return Pose(ids, pos_w, identity)

    def root_pose(self, articulation: str, *, env: EnvSelection = None) -> Pose:
        """Return one articulation's free-root world pose."""
        self._open()
        art = self._query._articulation(articulation)
        if str(getattr(art, "root", "free")) != "free":
            raise QueryReferenceError(
                "state.root_pose",
                articulation,
                (),
                "the articulation has a fixed root",
            )
        asset = self._query._asset_for("state.root_pose", articulation)
        data = self._query._live_data("state.root_pose", asset, articulation)
        ids = _select_env(env, self._query._env.num_envs)
        pos = self._query._read_env_rows("state.root_pose", data, "root_pos_w", ids, 3)
        quat = self._query._read_env_rows("state.root_pose", data, "root_quat_w", ids, 4, True)
        return Pose(ids, pos, quat)

    def joint_position(
        self,
        articulation: str,
        *,
        joint_names: Sequence[str] | None = None,
        env: EnvSelection = None,
    ) -> torch.Tensor:
        """Return joint positions for one articulation."""
        self._open()
        return self._joint_values(articulation, joint_names, env, velocity=False)

    def joint_velocity(
        self,
        articulation: str,
        *,
        joint_names: Sequence[str] | None = None,
        env: EnvSelection = None,
    ) -> torch.Tensor:
        """Return joint velocities for one articulation."""
        self._open()
        return self._joint_values(articulation, joint_names, env, velocity=True)

    def _joint_values(
        self,
        articulation: str,
        joint_names: Sequence[str] | None,
        env: EnvSelection,
        *,
        velocity: bool,
    ) -> torch.Tensor:
        """Read the requested joint columns in declared joint order."""
        op = "state.joint_velocity" if velocity else "state.joint_position"
        art = self._query._articulation(articulation)
        asset = self._query._asset_for(op, articulation)
        data = self._query._live_data(op, asset, articulation)
        declared = self._query._joint_names(asset, data) or tuple(
            getattr(art, "joint_names", ()) or ()
        )
        names = (
            tuple(getattr(art, "joint_names", ()) or ())
            if joint_names is None
            else tuple(str(name) for name in joint_names)
        )
        if not names:
            ids = _select_env(env, self._query._env.num_envs)
            return torch.empty((ids.numel(), 0), dtype=torch.float64)
        ir = self._query._ir(articulation)
        for name in names:
            row = ir.joints.get(name)
            if row is None:
                raise QueryReferenceError(op, name, ir.joints.names())
            if row.type not in ("hinge", "slide"):
                raise QueryCapabilityError(
                    self._query.backend,
                    op,
                    "unsupported",
                    f"joint {name!r} has type {row.type!r}; only 1-DOF "
                    f"hinge/slide joints are supported",
                )
        columns = [_index_of(op, name, declared) for name in names]
        ids = _select_env(env, self._query._env.num_envs)
        attr = "joint_vel" if velocity else "joint_pos"
        value = getattr(data, attr, None)
        if value is None:
            raise QueryCapabilityError(
                self._query.backend,
                op,
                "unsupported",
                f"live articulation data has no {attr!r}",
            )
        tensor = _tensor_view(value, attr).detach().to("cpu").to(torch.float64)
        if tensor.ndim != 2:
            raise QueryShapeError(attr, "shape (N, num_joints)", str(tuple(tensor.shape)))
        return tensor[ids][:, columns].clone()


class _UnsupportedKinematics(_Subface):
    """Jacobian reads are capability-gated."""

    def jacobian(
        self,
        target: EntityRef | ComRef | PointRef,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
        frame: str = "world",
        reference: str = "body_origin",
        rotational: bool = True,
    ) -> NoReturn:
        """Reject Jacobian reads."""
        del target, articulation, env, frame, reference, rotational
        self._open()
        raise QueryCapabilityError(self._query.backend, "jacobian", "unsupported", _P1_DETAIL)


class _UnsupportedGeometry(_Subface):
    """Distance and AABB reads are capability-gated."""

    def distance(
        self,
        a: EntityRef,
        b: EntityRef,
        *,
        max_distance_m: float = 1.0,
        env: EnvSelection = None,
        scratch: ScratchWorld | None = None,
    ) -> NoReturn:
        """Reject geometry distance reads."""
        del a, b, max_distance_m, env, scratch
        self._open()
        raise QueryCapabilityError(
            self._query.backend, "geometry.distance", "unsupported", _P1_DETAIL
        )

    def aabb(self, entity: EntityRef, *, env: EnvSelection = None) -> NoReturn:
        """Reject geometry AABB reads."""
        del entity, env
        self._open()
        raise QueryCapabilityError(self._query.backend, "geometry.aabb", "unsupported", _P1_DETAIL)


class _UnsupportedContacts(_Subface):
    """Contact reads are capability-gated until their own plans land."""

    def manifold(self, group: str, *, env: EnvSelection = None) -> NoReturn:
        """Reject a group-based contact manifold read on IsaacLab."""
        del group, env
        self._open()
        raise QueryCapabilityError(
            self._query.backend, "contacts.manifold", "unsupported", _P1_DETAIL
        )

    def jacobian(
        self,
        a: EntityRef,
        b: EntityRef | None = None,
        *,
        point: int = 0,
        env: EnvSelection = None,
        frame: str = "world",
    ) -> NoReturn:
        """Reject contact Jacobian reads."""
        del a, b, point, env, frame
        self._open()
        raise QueryCapabilityError(
            self._query.backend, "contacts.jacobian", "unsupported", _P1_DETAIL
        )


class _UnsupportedActuators(_Subface):
    """Actuator runtime facts are capability-gated."""

    def facts(self, name: str, *, articulation: str | None = None) -> NoReturn:
        """Reject actuator runtime facts."""
        del name, articulation
        self._open()
        raise QueryCapabilityError(
            self._query.backend, "actuators.facts", "unsupported", _P1_DETAIL
        )

    def state(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> NoReturn:
        """Reject actuator runtime state."""
        del name, articulation, env
        self._open()
        raise QueryCapabilityError(
            self._query.backend, "actuators.state", "unsupported", _P1_DETAIL
        )


class IsaacLabQuery:
    """The shared IsaacLab runtime-query facade over live articulation buffers.

    A concrete backend validates its own env type, then sets :attr:`backend`,
    :attr:`capabilities` and the live quaternion order before calling this
    constructor.  Capability, stale and unsupported-operation messages use the
    concrete backend label.
    """

    backend: str
    capabilities: QueryCapabilities

    def __init__(
        self,
        env: IsaacLabQueryEnv,
        *,
        backend: str,
        capabilities: QueryCapabilities,
        quat_order: str,
    ) -> None:
        """Bind one query handle to a live IsaacLab env."""
        if not isinstance(env, IsaacLabQueryEnv):
            raise TypeError(
                f"{type(self).__name__} requires an IsaacLab env, got {type(env).__name__}"
            )
        if not backend:
            raise ValueError(f"{type(self).__name__} requires a non-empty backend label")
        if capabilities.backend != backend:
            raise ValueError(
                f"{type(self).__name__} backend {backend!r} does not match the "
                f"capability object {capabilities.backend!r}"
            )
        if quat_order not in ("wxyz", "xyzw"):
            raise ValueError(
                f"{type(self).__name__} quat_order must be 'wxyz' or 'xyzw', " f"got {quat_order!r}"
            )
        self._env = env
        self.backend = backend
        self.capabilities = capabilities
        self._quat_order = quat_order
        self._closed = False
        self.model = _ModelQuery(self)
        self.state = _StateQuery(self)
        self.kinematics = _UnsupportedKinematics(self)
        self.geometry = _UnsupportedGeometry(self)
        self.contacts = _UnsupportedContacts(self)
        self.actuators = _UnsupportedActuators(self)

    def require(self, *ops: str) -> None:
        """Raise when any named query operation is unsupported."""
        self._require_open()
        self.capabilities.require(*ops)

    def scratch(self, env: int | None = None) -> AbstractContextManager[ScratchWorld]:
        """Reject scratch worlds: P0B-2 has no scratch copy."""
        del env
        self._require_open()
        raise QueryCapabilityError(self.backend, "scratch", "unsupported", _P1_DETAIL)

    def close(self) -> None:
        """Invalidate this query handle (idempotent)."""
        self._closed = True

    def _require_open(self) -> None:
        """Raise :class:`QueryStaleError` after this handle was closed."""
        if self._closed:
            raise QueryStaleError(
                f"query for backend {self.backend!r} is stale: the run/env/query is closed"
            )

    def _model_set(self) -> ModelSet:
        """Return the env's published model set."""
        return self._env.model_set

    def _ir(self, articulation: str | None) -> ModelIR:
        """Resolve the whole-scene or one articulation's published IR."""
        model_set = self._model_set()
        if articulation is not None:
            try:
                return model_set[articulation]
            except KeyError:
                raise QueryReferenceError("model.ir", articulation, model_set.names()) from None
        if model_set.merged is not None:
            return model_set.merged
        if len(model_set) == 1:
            return next(iter(model_set.articulations.values()))
        raise QueryReferenceError(
            "model.ir",
            None,
            model_set.names(),
            "pass articulation=... for a multi-articulation run",
        )

    def _articulation(self, name: str) -> ArticulationRuntime:
        """Return one live articulation runtime or a teaching reference error."""
        arts = self._env.articulations
        art = arts.get(name)
        if art is None:
            raise QueryReferenceError("articulation", name, tuple(arts))
        return art

    def _asset_for(self, op: str, articulation: str) -> object:
        """Return the live asset bound to one articulation runtime."""
        art = self._articulation(articulation)
        asset = getattr(art, "asset", None)
        if asset is None:
            raise QueryReferenceError(
                op,
                articulation,
                tuple(self._env.articulations),
                "the articulation runtime has no live asset",
            )
        return asset

    def _live_data(self, op: str, asset: object, articulation: str) -> object:
        """Return one live asset's data block."""
        data = getattr(asset, "data", None)
        if data is None:
            raise QueryReferenceError(
                op,
                articulation,
                tuple(self._env.articulations),
                "the live asset exposes no data block",
            )
        return data

    def _body_names(self, asset: object, data: object) -> tuple[str, ...]:
        """Return the live body name order for one asset."""
        return _as_names(getattr(data, "body_names", None)) or _as_names(
            getattr(asset, "body_names", None)
        )

    def _joint_names(self, asset: object, data: object) -> tuple[str, ...]:
        """Return the live joint name order for one asset."""
        return _as_names(getattr(data, "joint_names", None)) or _as_names(
            getattr(asset, "joint_names", None)
        )

    def _resolve_body_articulation(
        self,
        op: str,
        name: str,
        articulation: str | None,
    ) -> str:
        """Resolve the articulation that owns one body name."""
        if articulation is not None:
            self._articulation(articulation)
            return articulation
        names = tuple(self._env.articulations)
        if len(names) == 1:
            return names[0]
        matches = []
        for candidate in names:
            asset = self._asset_for(op, candidate)
            data = self._live_data(op, asset, candidate)
            if name in self._body_names(asset, data):
                matches.append(candidate)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise QueryReferenceError(op, name, names, "no articulation declares this body")
        raise QueryReferenceError(
            op,
            name,
            tuple(matches),
            "the body is ambiguous across articulations; pass articulation=...",
        )

    def _body_row(self, name: str, articulation: str | None) -> Body:
        """Return one body row from the resolved IR."""
        ir = self._ir(articulation)
        try:
            return ir.bodies[name]
        except KeyError:
            raise QueryReferenceError("body", name, ir.bodies.names()) from None

    def _joint_row(self, name: str, articulation: str | None) -> Joint:
        """Return one joint row from the resolved IR."""
        ir = self._ir(articulation)
        try:
            return ir.joints[name]
        except KeyError:
            raise QueryReferenceError("joint", name, ir.joints.names()) from None

    def _actuator_row(self, name: str, articulation: str | None) -> Actuator:
        """Return one actuator row from the resolved IR."""
        ir = self._ir(articulation)
        try:
            return ir.actuators[name]
        except KeyError:
            raise QueryReferenceError("actuator", name, ir.actuators.names()) from None

    def _profiles(self) -> Mapping[str, RobotProfile]:
        """Return the articulation-owned robot profiles (public runtime field)."""
        profiles: dict[str, RobotProfile] = {}
        for name, art in self._env.articulations.items():
            profile = getattr(art, "profile", None)
            if isinstance(profile, RobotProfile):
                profiles[name] = profile
        return profiles

    def _site_inventory(self) -> Mapping[str, SiteEntry]:
        """Return the run-wide asset-level site inventory."""
        definitions = getattr(getattr(self._env, "prop_manager", None), "asset_definitions", None)
        return build_site_inventory(self._profiles(), definitions or {})

    def _wxyz(self, quats: torch.Tensor) -> torch.Tensor:
        """Reorder a live quaternion tensor into repo-internal ``wxyz``."""
        if self._quat_order == "wxyz":
            return quats
        return quat_math.to_wxyz(quats)

    def _read_rows(
        self,
        op: str,
        data: object,
        attr: str,
        ids: torch.Tensor,
        index: int,
        width: int,
        quaternion: bool,
    ) -> torch.Tensor:
        """Read one row from a per-body ``(N, B, width)`` live tensor."""
        value = getattr(data, attr, None)
        if value is None:
            raise QueryCapabilityError(
                self.backend,
                op,
                "unsupported",
                f"live articulation data has no {attr!r}",
            )
        tensor = _tensor_view(value, attr).detach().to("cpu").to(torch.float64)
        if tensor.ndim != 3 or int(tensor.shape[-1]) != width:
            raise QueryShapeError(attr, f"shape (N, B, {width})", str(tuple(tensor.shape)))
        rows = tensor[ids][:, index]
        return self._wxyz(rows) if quaternion else rows

    def _read_env_rows(
        self,
        op: str,
        data: object,
        attr: str,
        ids: torch.Tensor,
        width: int,
        quaternion: bool = False,
    ) -> torch.Tensor:
        """Read one per-env ``(N, width)`` live tensor."""
        value = getattr(data, attr, None)
        if value is None:
            raise QueryCapabilityError(
                self.backend,
                op,
                "unsupported",
                f"live articulation data has no {attr!r}",
            )
        tensor = _tensor_view(value, attr).detach().to("cpu").to(torch.float64)
        if tensor.ndim != 2 or int(tensor.shape[-1]) != width:
            raise QueryShapeError(attr, f"shape (N, {width})", str(tuple(tensor.shape)))
        rows = tensor[ids]
        return self._wxyz(rows) if quaternion else rows
