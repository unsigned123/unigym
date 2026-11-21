"""Shared IsaacLab runtime-query implementation (model facts and live pose).

IsaacLab 2.x and the two IsaacLab 3.x identities expose the same articulation
runtime surface (``env.articulations`` / ``env.robots`` / ``env.model_set``), so
the query math owns here.  A backend subclass supplies its env type, its
canonical backend label, its capability object and the live quaternion order;
it never copies the model lookup, state read or site-composition code.

P0B-2 implements the model facts carried by the published ``ModelIR`` and live
body/root/COM/joint/site reads.  P1B-2 adds static actuator facts from the
published ``ModelIR`` actuator rows and USD attrs.  P1C-1 adds the IsaacLab 3
body/body-COM Jacobians (native engine rows) and composes site/world-point
Jacobians from the body link row.  P1C-2 adds the IsaacLab 3 transaction
scratch: IsaacLab has no snapshot copy, so a scratch writes the live state-write
face, forwards, reads, and restores the captured root/joint state on close.
Actuator runtime state, geometry and contacts stay unsupported and raise
``QueryCapabilityError``.  IsaacLab 2.x keeps the unsupported scratch and
kinematics subfaces.

Engine-thread contract: the query reads the live articulation buffers at a
step boundary and adds no lock.  Every result is a detached CPU torch copy; no
observation is published and nothing is recorded.  An active transaction
scratch owns its env exclusively: no concurrent scratch, step or reset.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import NoReturn, Protocol, runtime_checkable

import torch

from unigym.backends.obs_schema import build_site_inventory
from unigym.backends.shared.isaaclab_bodies import body_binding_candidates
from unigym.core import quat as quat_math
from unigym.env.backends.base import normalize_state_env_ids
from unigym.model import Actuator, Body, Geom, Joint, ModelIR, ModelSet, Site
from unigym.query.base import (
    ActuatorFacts,
    ComRef,
    ContactPoint,
    DofLayout,
    EntityRef,
    EnvSelection,
    Jacobian,
    PointRef,
    Pose,
)
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
from unigym.query.facade import GeometryQuery, ScratchWorld
from unigym.robots.base import RobotProfile, normalize_canonical_body_id
from unigym.runtime.controllers import ArticulationRuntime
from unigym.sites import SiteEntry

__all__ = [
    "IsaacLabQuery",
    "IsaacLabQueryEnv",
    "isaaclab_query_capabilities",
]

_P1_DETAIL = "the IsaacLab query does not implement this operation"
_BASE_DOF_SUFFIXES = ("px", "py", "pz", "rx", "ry", "rz")
_ONE_DOF_JOINT_TYPES = frozenset({"hinge", "slide"})
_ACTUATOR_FACTS_DETAIL = (
    "static IsaacLab actuator-group facts only; slot_kind='isaaclab_group', "
    "joint_names come from the published USD attrs, and ctrlrange/gear/gains/"
    "limits are not fabricated"
)
_ACTUATOR_STATE_DETAIL = (
    "IsaacLab actuator groups do not expose a single scalar ctrl/force/length/"
    "velocity channel compatible with the MuJoCo-family actuator row"
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
_SCRATCH_GEOMETRY_DETAIL = "geometry and contact reads are not supported in an IsaacLab scratch"


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


@runtime_checkable
class IsaacLabTransactionEnv(IsaacLabQueryEnv, Protocol):
    """The IsaacLab 3 live state-write face a transaction scratch drives."""

    def set_root_state(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        *,
        name: str,
        pos_w: torch.Tensor,
        quat_wxyz: torch.Tensor,
        lin_vel_w: torch.Tensor,
        ang_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        """Write one free root's world pose and velocity."""
        ...

    def set_joint_state(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        *,
        articulation: str,
        joint_names: Sequence[str],
        position: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write explicit joint positions and optional velocities."""
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
            facts=Feature("partial", _ACTUATOR_FACTS_DETAIL),
            state=Feature("unsupported", _ACTUATOR_STATE_DETAIL),
        ),
        notes={
            "engine_thread": _ENGINE_THREAD,
            "returns": _COPIES,
            "scope": (
                "P0B-2 model facts + live pose + static actuator facts; "
                "no scratch/geometry/Jacobian/contact/actuator-state"
            ),
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


def _scratch_vector(value: torch.Tensor, field: str, length: int) -> torch.Tensor:
    """Validate a scratch write vector and return a finite float64 CPU copy."""
    if not isinstance(value, torch.Tensor):
        raise QueryShapeError(field, f"a torch.Tensor of shape ({length},)", type(value).__name__)
    if not value.is_floating_point():
        raise QueryShapeError(field, "a floating tensor", str(value.dtype))
    if tuple(value.shape) != (length,):
        raise QueryShapeError(field, f"shape ({length},)", str(tuple(value.shape)))
    vector = value.detach().to("cpu").to(torch.float64)
    if not bool(torch.isfinite(vector).all()):
        raise QueryShapeError(field, "finite values", str(vector.tolist()))
    return vector


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


def _actuator_joint_names(ir: ModelIR, name: str) -> tuple[str, ...]:
    """Return one actuator group's published joint names (or empty)."""
    usd = ir.usd
    if usd is None:
        return ()
    entry = usd.attrs.get(f"actuator/{name}")
    if not isinstance(entry, Mapping):
        return ()
    raw = entry.get("joint_names")
    if raw is None:
        return ()
    field = f"actuator/{name}.joint_names"
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise QueryShapeError(field, "a sequence of names", type(raw).__name__)
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise QueryShapeError(field, "non-empty string names", repr(item))
        names.append(item)
    return tuple(names)


def _actuator_facts(ir: ModelIR, row: Actuator) -> ActuatorFacts:
    """Map one IsaacLab actuator-group row to facts-only neutral facts.

    IsaacLab actuator groups have no compiled single-actuator slot and no
    native scalar control range, gear or gain/bias parameters, so those fields
    stay ``None`` rather than being fabricated from a group configuration.
    """
    return ActuatorFacts(
        name=row.name,
        target=row.target or None,
        kind=row.kind,
        slot=None,
        slot_kind="isaaclab_group",
        ctrlrange=None,
        gear=None,
        gainprm=None,
        biasprm=None,
        joint_names=_actuator_joint_names(ir, row.name),
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
        owner, index, local_pos, local_quat = self._query._site_target(
            "state.site_pose", name, articulation
        )
        asset = self._query._asset_for("state.site_pose", owner)
        data = self._query._live_data("state.site_pose", asset, owner)
        ids = _select_env(env, self._query._env.num_envs)
        body_pos = self._query._read_rows(
            "state.site_pose", data, "body_pos_w", ids, index, 3, False
        )
        body_quat = self._query._read_rows(
            "state.site_pose", data, "body_quat_w", ids, index, 4, True
        )
        offset = local_pos.expand(body_pos.shape[0], -1)
        pos_w = body_pos + quat_math.rotate(body_quat, offset)
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


class IsaacLabJacobianQuery(_Subface):
    """IsaacLab 3 body/COM Jacobians plus composed site/world-point rows."""

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
        """Return one target's world-frame Jacobian in the articulation layout.

        Body and body-COM rows are read from the engine's native Jacobian.
        Site and world-point rows are composed from the body link row plus the
        offset from the body origin; the angular block is reference-point
        invariant and therefore stays the body link angular row.
        """
        self._open()
        if frame != "world":
            raise QueryCapabilityError(
                self._query.backend,
                "jacobian",
                "unsupported",
                f"frame={frame!r}; only 'world' is implemented for IsaacLab",
            )
        if reference != "body_origin":
            raise QueryCapabilityError(
                self._query.backend,
                "jacobian",
                "unsupported",
                f"reference={reference!r}; the target's own origin is the only "
                f"reference available for IsaacLab",
            )
        ids = _select_env(env, self._query._env.num_envs)
        if isinstance(target, ComRef):
            rows, layout = self._com_rows(target, articulation, ids)
            return self._finish(ids, rows, layout, frame, reference, rotational)
        if isinstance(target, PointRef):
            rows, layout = self._point_rows(target, articulation, ids)
            return self._finish(ids, rows, layout, frame, reference, rotational)
        if isinstance(target, EntityRef):
            if target.kind == "body":
                rows, layout = self._body_rows(target, articulation, ids)
                return self._finish(ids, rows, layout, frame, reference, rotational)
            if target.kind == "site":
                rows, layout = self._site_rows(target, articulation, ids)
                return self._finish(ids, rows, layout, frame, reference, rotational)
            raise QueryCapabilityError(
                self._query.backend,
                "jacobian.geom",
                "unsupported",
                "Jacobian targets are body/site/COM/world-point; a geom has no frame",
            )
        raise QueryShapeError("target", "an EntityRef, ComRef or PointRef", type(target).__name__)

    def _body_rows(
        self,
        target: EntityRef,
        articulation: str | None,
        ids: torch.Tensor,
    ) -> tuple[torch.Tensor, DofLayout]:
        """Read the native body link Jacobian row for one body."""
        op = "jacobian.body"
        art_name = self._resolve_body_target(op, target.articulation, articulation, target.name)
        art, asset, names, data = self._live_articulation(op, art_name)
        index = _index_of(op, target.name, names)
        return self._native_rows(
            op, "body_link_jacobian_w", art, asset, art_name, names, data, index, ids
        )

    def _com_rows(
        self,
        target: ComRef,
        articulation: str | None,
        ids: torch.Tensor,
    ) -> tuple[torch.Tensor, DofLayout]:
        """Read the native body center-of-mass Jacobian row."""
        op = "jacobian.com"
        if target.subtree:
            raise QueryCapabilityError(
                self._query.backend,
                op,
                "partial",
                "a subtree COM Jacobian is not exposed by IsaacLab; pass subtree=False",
            )
        art_name = self._resolve_entity_articulation(op, target.articulation, articulation)
        art, asset, names, data = self._live_articulation(op, art_name)
        body_name = target.body or str(getattr(art, "base_body", "") or "")
        if not body_name:
            raise QueryReferenceError(
                op,
                art_name,
                names,
                "no base body is declared for this articulation",
            )
        index = _index_of(op, body_name, names)
        return self._native_rows(
            op, "body_com_jacobian_w", art, asset, art_name, names, data, index, ids
        )

    def _site_rows(
        self,
        target: EntityRef,
        articulation: str | None,
        ids: torch.Tensor,
    ) -> tuple[torch.Tensor, DofLayout]:
        """Compose a robot-profile site row from the body link row + offset."""
        op = "jacobian.site"
        if target.articulation is not None and articulation is not None:
            if target.articulation != articulation:
                raise QueryReferenceError(
                    op,
                    target.articulation,
                    self._query._model_set().names(),
                    f"reference articulation {target.articulation!r} conflicts with "
                    f"articulation={articulation!r}",
                )
        selector = target.articulation if target.articulation is not None else articulation
        owner, index, local_pos, _ = self._query._site_target(op, target.name, selector)
        art, asset, names, data = self._live_articulation(op, owner)
        rows, layout = self._native_rows(
            op, "body_link_jacobian_w", art, asset, owner, names, data, index, ids
        )
        body_quat = self._query._read_rows(op, data, "body_quat_w", ids, index, 4, True)
        offset = local_pos.expand(body_quat.shape[0], -1)
        return self._shift_translation(rows, quat_math.rotate(body_quat, offset)), layout

    def _point_rows(
        self,
        target: PointRef,
        articulation: str | None,
        ids: torch.Tensor,
    ) -> tuple[torch.Tensor, DofLayout]:
        """Compose a world-point row from the body link row + world offset."""
        op = "jacobian.point"
        entity = target.entity
        art_name = self._resolve_body_target(op, entity.articulation, articulation, entity.name)
        art, asset, names, data = self._live_articulation(op, art_name)
        index = _index_of(op, entity.name, names)
        rows, layout = self._native_rows(
            op, "body_link_jacobian_w", art, asset, art_name, names, data, index, ids
        )
        body_pos = self._query._read_rows(op, data, "body_pos_w", ids, index, 3, False)
        offset = target.point_w.to(torch.float64) - body_pos
        return self._shift_translation(rows, offset), layout

    @staticmethod
    def _shift_translation(rows: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Shift a link-origin Jacobian's linear block to a point offset by *offset*."""
        columns = int(rows.shape[-1])
        translated = rows[:, 0:3, :] + torch.cross(
            rows[:, 3:6, :], offset[:, :, None].expand(-1, -1, columns), dim=1
        )
        return torch.cat((translated, rows[:, 3:6, :]), dim=1)

    def _native_rows(
        self,
        op: str,
        attr: str,
        art: ArticulationRuntime,
        asset: object,
        art_name: str,
        names: Sequence[str],
        data: object,
        index: int,
        ids: torch.Tensor,
    ) -> tuple[torch.Tensor, DofLayout]:
        """Read one engine Jacobian body row and build its ``DofLayout``.

        The native rows are ``(N, num_jacobi_bodies, 6, nv)`` with the six
        world-frame base DoFs first (floating base only) and the actuated
        joints in ``art.joint_names`` order.  A fixed base drops the fixed-root
        body row from the public body order, so the Jacobian row is the public
        index with the root omitted: indices before the root are unchanged and
        indices after it shift down by one.
        """
        joint_names = tuple(art.joint_names)
        ir = self._query._ir(art_name)
        for name in joint_names:
            row = ir.joints.get(name)
            if row is None:
                raise QueryReferenceError(
                    op,
                    name,
                    ir.joints.names(),
                    "the articulation joint is missing from the published IR",
                )
            if row.type not in _ONE_DOF_JOINT_TYPES:
                raise QueryCapabilityError(
                    self._query.backend,
                    op,
                    "unsupported",
                    f"articulation {art_name!r} joint {name!r} has type {row.type!r}; "
                    f"only hinge/slide joint columns are supported",
                )
        value = getattr(data, attr, None)
        if value is None:
            raise QueryCapabilityError(
                self._query.backend,
                op,
                "unsupported",
                f"live articulation data has no {attr!r}",
            )
        tensor = _tensor_view(value, attr).detach().to("cpu").to(torch.float64)
        if tensor.ndim != 4 or int(tensor.shape[2]) != 6:
            raise QueryShapeError(attr, "shape (N, B, 6, nv)", str(tuple(tensor.shape)))
        nv = int(tensor.shape[-1])
        base_columns = nv - len(joint_names)
        if base_columns not in (0, 6):
            raise QueryShapeError(
                attr,
                f"{len(joint_names)} joint columns plus 0 or 6 base DoFs",
                str(nv),
            )
        root_index = (
            self._fixed_root_index(op, art, asset, art_name, names) if base_columns == 0 else None
        )
        expected_bodies = len(names) if root_index is None else len(names) - 1
        if int(tensor.shape[1]) != expected_bodies:
            raise QueryShapeError(
                attr,
                f"shape (N, {expected_bodies}, 6, {nv})",
                str(tuple(tensor.shape)),
            )
        if root_index is not None and index == root_index:
            raise QueryCapabilityError(
                self._query.backend,
                op,
                "partial",
                f"body {names[index]!r} is the fixed root and has no Jacobian row",
            )
        jacobi_index = index if root_index is None or index < root_index else index - 1
        base_dofs: tuple[str, ...] = ()
        base_names: tuple[str, ...] = ()
        if base_columns == 6:
            base_body = self._base_body_name(art, art_name)
            base_dofs = tuple(f"{base_body}.{suffix}" for suffix in _BASE_DOF_SUFFIXES)
            base_names = (base_body,)
        layout = DofLayout(
            articulation=art_name,
            base_dofs=base_dofs,
            base_names=base_names,
            joint_names=joint_names,
            nv=nv,
            order=base_dofs + joint_names,
        )
        return tensor[ids][:, jacobi_index], layout

    def _live_articulation(
        self,
        op: str,
        art_name: str,
    ) -> tuple[ArticulationRuntime, object, tuple[str, ...], object]:
        """Return the live articulation, asset, body order and data block."""
        art = self._query._articulation(art_name)
        asset = self._query._asset_for(op, art_name)
        data = self._query._live_data(op, asset, art_name)
        return art, asset, self._query._body_names(asset, data), data

    def _fixed_root_index(
        self,
        op: str,
        art: ArticulationRuntime,
        asset: object,
        art_name: str,
        names: Sequence[str],
    ) -> int:
        """Return the fixed root's public body index, or raise a teaching error.

        The engine omits the backend root body (backend body row 0) from the
        public Jacobian body axis.  Resolve which public row that is from the
        live backend body name, falling back to the declared USD base binding
        only when the live backend body order is unavailable.  An unresolved
        or inconsistent root never silently maps to public row 0.
        """
        backend_names = _as_names(getattr(asset, "backend_body_names", None))
        if backend_names:
            root_name = backend_names[0]
            if root_name not in names:
                raise QueryCapabilityError(
                    self._query.backend,
                    op,
                    "unsupported",
                    f"articulation {art_name!r} is fixed-base but its backend root "
                    f"body {root_name!r} is not in the public body order {list(names)}",
                )
            return list(names).index(root_name)
        declared = str(getattr(art, "base_body", "") or "")
        if declared:
            if declared not in names:
                raise QueryReferenceError(
                    op,
                    declared,
                    names,
                    "the declared base body is not in the live body order",
                )
            return list(names).index(declared)
        raise QueryCapabilityError(
            self._query.backend,
            op,
            "unsupported",
            f"articulation {art_name!r} is fixed-base but its root body public "
            f"index is unresolved; neither the live backend body order nor a "
            f"declared base body identifies it",
        )

    def _base_body_name(self, art: ArticulationRuntime, art_name: str) -> str:
        """Return the declared canonical base body for a floating articulation.

        The published base labels name the articulation's declared canonical
        base body (``profile.base_body`` / ``ArticulationRuntime.base_body_id``),
        never the backend-native USD binding and never an inferred live row.
        An undeclared base body is a free-root mapping failure: the query raises
        instead of labelling the six base columns with an arbitrary body.
        """
        canonical = getattr(art, "base_body_id", None)
        declared = str(canonical) if canonical is not None else ""
        if declared:
            return declared
        raise QueryCapabilityError(
            self._query.backend,
            "jacobian",
            "unsupported",
            f"articulation {art_name!r} declares a free root but no base body is "
            f"declared; free root unresolved (set profile.base_body)",
        )

    def _resolve_body_target(
        self,
        op: str,
        reference: str | None,
        articulation: str | None,
        body_name: str,
    ) -> str:
        """Resolve one body target's articulation, disambiguating by live body names."""
        if reference is not None:
            if articulation is not None and articulation != reference:
                raise QueryReferenceError(
                    op,
                    reference,
                    self._query._model_set().names(),
                    f"reference articulation {reference!r} conflicts with "
                    f"articulation={articulation!r}",
                )
            self._query._articulation(reference)
            return reference
        return self._query._resolve_body_articulation(op, body_name, articulation)

    def _resolve_entity_articulation(
        self,
        op: str,
        reference: str | None,
        articulation: str | None,
    ) -> str:
        """Resolve one target's articulation before the ``articulation=`` keyword."""
        names = self._query._model_set().names()
        if reference is not None:
            if articulation is not None and articulation != reference:
                raise QueryReferenceError(
                    op,
                    reference,
                    names,
                    f"reference articulation {reference!r} conflicts with "
                    f"articulation={articulation!r}",
                )
            self._query._articulation(reference)
            return reference
        if articulation is not None:
            self._query._articulation(articulation)
            return articulation
        if len(names) == 1:
            return names[0]
        raise QueryReferenceError(
            op,
            None,
            names,
            "pass articulation=... for a multi-articulation run",
        )

    def _finish(
        self,
        ids: torch.Tensor,
        rows: torch.Tensor,
        layout: DofLayout,
        frame: str,
        reference: str,
        rotational: bool,
    ) -> Jacobian:
        """Build the public result, zeroing the angular block when not requested."""
        transl = rows[:, 0:3, :]
        rot = rows[:, 3:6, :] if rotational else torch.zeros_like(transl)
        return Jacobian(ids, transl, rot, layout, frame, reference)


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


class _UnsupportedScratchGeometry:
    """Distance and AABB reads are capability-gated inside a transaction scratch."""

    def __init__(self, world: _TransactionScratch) -> None:
        """Bind the stub to its owning transaction scratch."""
        self._world = world

    def distance(
        self,
        a: EntityRef,
        b: EntityRef,
        *,
        max_distance_m: float = 1.0,
        env: EnvSelection = None,
        scratch: ScratchWorld | None = None,
    ) -> NoReturn:
        """Reject geometry distance reads in a transaction scratch."""
        del a, b, max_distance_m, env, scratch
        self._world._open()
        raise QueryCapabilityError(
            self._world._query.backend,
            "scratch.geometry",
            "unsupported",
            _SCRATCH_GEOMETRY_DETAIL,
        )

    def aabb(self, entity: EntityRef, *, env: EnvSelection = None) -> NoReturn:
        """Reject geometry AABB reads in a transaction scratch."""
        del entity, env
        self._world._open()
        raise QueryCapabilityError(
            self._world._query.backend,
            "scratch.geometry",
            "unsupported",
            _SCRATCH_GEOMETRY_DETAIL,
        )


class _TransactionScratch:
    """One exclusive per-env transaction over the live IsaacLab state.

    IsaacLab exposes no snapshot/restore, so this is not a copy.  The captured
    root/joint state is the restore source; writes go through the env's live
    state-write face and every read observes the live buffers after an explicit
    :meth:`forward`.  Closing restores the captured state and forwards again,
    so the live world is ``restored`` rather than bit-exact untouched.  The
    owner (the query) keeps one active scratch per env; the caller must not
    step, reset or open another scratch while it is active.
    """

    def __init__(self, query: IsaacLabQuery, env_id: int) -> None:
        """Bind the transaction to one env of a live query handle."""
        env = query._env
        if not isinstance(env, IsaacLabTransactionEnv):
            raise QueryCapabilityError(
                query.backend,
                "scratch",
                "unsupported",
                "the env exposes no IsaacLab live state-write face",
            )
        self._query = query
        self._env: IsaacLabTransactionEnv = env
        self._env_id = env_id
        self._closed = False
        self._roots: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._joints: dict[str, tuple[tuple[str, ...], torch.Tensor, torch.Tensor]] = {}
        self._geometry = _UnsupportedScratchGeometry(self)

    def __enter__(self) -> _TransactionScratch:
        """Capture every articulation's live root/joint state for this env."""
        self._query._require_open()
        if self._env_id in self._query._active_scratch:
            raise QueryCapabilityError(
                self._query.backend,
                "scratch",
                "unsupported",
                f"env {self._env_id} already has an active scratch copy",
            )
        self._query._active_scratch.add(self._env_id)
        try:
            self._capture()
        except BaseException:
            self._query._active_scratch.discard(self._env_id)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Restore the captured state; preserve an in-flight body exception.

        A restore/flush failure never replaces the body exception.  It is
        attached to the body exception as a note so the transaction failure is
        still visible in the traceback.
        """
        del traceback
        try:
            self.close()
        except BaseException as restore_error:
            if exc_type is None or exc is None:
                raise
            exc.add_note(
                f"scratch restore failed while propagating "
                f"{type(exc).__name__}: {type(restore_error).__name__}: {restore_error}"
            )
        finally:
            self._query._active_scratch.discard(self._env_id)
        return False

    def set_root_state(
        self,
        name: str,
        *,
        pos_w: torch.Tensor,
        quat_wxyz: torch.Tensor,
        lin_vel_w: torch.Tensor | None = None,
        ang_vel_w: torch.Tensor | None = None,
    ) -> None:
        """Override one free-root articulation's pose and optional velocities."""
        self._open()
        art = self._query._articulation(name)
        asset = self._query._asset_for("scratch.set_root_state", name)
        if not self._root_is_free(art, asset):
            raise QueryCapabilityError(
                self._query.backend,
                "scratch.set_root_state",
                "unsupported",
                f"articulation {name!r} has a fixed root; set_root_state needs a free root",
            )
        if name not in self._roots:
            raise QueryCapabilityError(
                self._query.backend,
                "scratch.set_root_state",
                "unsupported",
                f"articulation {name!r} did not capture a free-root state",
            )
        pos = _scratch_vector(pos_w, "pos_w", 3)
        quat = _scratch_vector(quat_wxyz, "quat_wxyz", 4)
        norm = float(quat.norm())
        if norm <= 1e-12:
            raise QueryShapeError("quat_wxyz", "a non-zero quaternion", str(quat.tolist()))
        quat = quat / norm
        data = self._query._live_data("scratch.set_root_state", asset, name)
        linear = (
            self._row(data, "root_lin_vel_w", 3)
            if lin_vel_w is None
            else _scratch_vector(lin_vel_w, "lin_vel_w", 3)
        )
        angular = (
            self._row(data, "root_ang_vel_w", 3)
            if ang_vel_w is None
            else _scratch_vector(ang_vel_w, "ang_vel_w", 3)
        )
        self._env.set_root_state(
            [self._env_id],
            name=name,
            pos_w=self._on_device(pos),
            quat_wxyz=self._on_device(quat),
            lin_vel_w=self._on_device(linear),
            ang_vel_w=self._on_device(angular),
        )

    def set_joint_state(
        self,
        articulation: str,
        joint_names: Sequence[str],
        position: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ) -> None:
        """Override absolute 1-DOF joint positions and optional velocities."""
        self._open()
        art = self._query._articulation(articulation)
        asset = self._query._asset_for("scratch.set_joint_state", articulation)
        data = self._query._live_data("scratch.set_joint_state", asset, articulation)
        if isinstance(joint_names, (str, bytes)):
            raise QueryShapeError("joint_names", "a sequence of joint names", "a string")
        names = tuple(str(name) for name in joint_names)
        if not names:
            raise QueryShapeError("joint_names", "a non-empty sequence", "[]")
        if len(set(names)) != len(names):
            raise QueryShapeError("joint_names", "unique names", str(names))
        declared = self._query._joint_names(asset, data) or tuple(art.joint_names)
        missing = sorted(set(names) - set(declared))
        if missing:
            raise QueryReferenceError("scratch.set_joint_state", missing[0], declared)
        ir = self._query._ir(articulation)
        for name in names:
            row = ir.joints.get(name)
            if row is None:
                raise QueryReferenceError(
                    "scratch.set_joint_state",
                    name,
                    ir.joints.names(),
                )
            if row.type not in _ONE_DOF_JOINT_TYPES:
                raise QueryCapabilityError(
                    self._query.backend,
                    "scratch.set_joint_state",
                    "unsupported",
                    f"joint {name!r} has type {row.type!r}; only 1-DOF hinge/slide "
                    f"joints are supported",
                )
        positions = _scratch_vector(position, "position", len(names))
        velocities = None if velocity is None else _scratch_vector(velocity, "velocity", len(names))
        self._env.set_joint_state(
            [self._env_id],
            articulation=articulation,
            joint_names=names,
            position=self._on_device(positions),
            velocity=None if velocities is None else self._on_device(velocities),
        )

    def forward(self) -> None:
        """Flush the live writes and forward the simulator view."""
        self._open()
        self._flush()

    def body_pose(self, name: str, *, articulation: str | None = None) -> Pose:
        """Return one body's live pose inside the active transaction."""
        self._open()
        return self._query.state.body_pose(name, articulation=articulation, env=self._env_id)

    def site_pose(self, name: str, *, articulation: str | None = None) -> Pose:
        """Return one robot-profile site's live pose inside the transaction."""
        self._open()
        return self._query.state.site_pose(name, articulation=articulation, env=self._env_id)

    def com_pose(self, ref: ComRef, *, frame: str = "world") -> Pose:
        """Return one COM reference on the live transaction buffers."""
        self._open()
        return self._query.state.com_pose(ref, frame=frame, env=self._env_id)

    def geometry(self) -> GeometryQuery:
        """Reject geometry reads: a transaction scratch has no geometry copy."""
        self._open()
        return self._geometry

    def jacobian(
        self,
        target: EntityRef | ComRef | PointRef,
        *,
        articulation: str | None = None,
        frame: str = "world",
        reference: str = "body_origin",
        rotational: bool = True,
    ) -> Jacobian:
        """Return a Jacobian read from the live transaction buffers."""
        self._open()
        return self._query.kinematics.jacobian(
            target,
            articulation=articulation,
            env=self._env_id,
            frame=frame,
            reference=reference,
            rotational=rotational,
        )

    def close(self) -> None:
        """Restore the captured state, forward, and release the env guard."""
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        try:
            for name, (pos, quat, lin, ang) in self._roots.items():
                try:
                    self._restore_root(name, pos, quat, lin, ang)
                except BaseException as exc:
                    if error is None:
                        error = exc
            for name, (names, position, velocity) in self._joints.items():
                if not names:
                    continue
                try:
                    self._restore_joints(name, names, position, velocity)
                except BaseException as exc:
                    if error is None:
                        error = exc
        finally:
            try:
                self._flush()
            except BaseException as exc:
                if error is None:
                    error = exc
            self._query._active_scratch.discard(self._env_id)
        if error is not None:
            raise error

    def _capture(self) -> None:
        """Clone the selected env's root/joint state for every articulation."""
        for name in tuple(self._env.articulations):
            asset = self._query._asset_for("scratch", name)
            data = self._query._live_data("scratch", asset, name)
            art = self._query._articulation(name)
            if self._root_is_free(art, asset):
                self._roots[name] = (
                    self._row(data, "root_pos_w", 3),
                    self._row(data, "root_quat_w", 4),
                    self._row(data, "root_lin_vel_w", 3),
                    self._row(data, "root_ang_vel_w", 3),
                )
            names = self._query._joint_names(asset, data) or tuple(art.joint_names)
            self._joints[name] = (
                names,
                self._row(data, "joint_pos", len(names)),
                self._row(data, "joint_vel", len(names)),
            )

    def _root_is_free(self, art: ArticulationRuntime, asset: object) -> bool:
        """Return whether the live root accepts writes (declaration + asset fact).

        ``ArticulationRuntime.root`` defaults to ``free`` even for a fixed USD
        base, so the live asset's ``is_fixed_base`` is the binding fact: a
        fixed asset never enters the restore-root set.
        """
        if bool(getattr(asset, "is_fixed_base", False)):
            return False
        return str(getattr(art, "root", "free")) == "free"

    def _row(self, data: object, attr: str, width: int) -> torch.Tensor:
        """Clone one ``(1, width)`` row of a live per-env tensor."""
        value = getattr(data, attr, None)
        if value is None:
            raise QueryCapabilityError(
                self._query.backend,
                "scratch",
                "unsupported",
                f"live articulation data has no {attr!r}",
            )
        tensor = _tensor_view(value, attr)
        if tensor.ndim != 2 or int(tensor.shape[-1]) != width:
            raise QueryShapeError(attr, f"shape (N, {width})", str(tuple(tensor.shape)))
        return tensor[self._env_id : self._env_id + 1].detach().clone()

    def _on_device(self, value: torch.Tensor) -> torch.Tensor:
        """Reshape one write vector to ``(1, length)`` on the env device."""
        return value.reshape(1, -1).to(device=self._device())

    def _device(self) -> torch.device:
        """Return the env's compute device for the state-write face."""
        device = getattr(self._env, "device", None)
        if device is None:
            raise QueryCapabilityError(
                self._query.backend,
                "scratch",
                "unsupported",
                "the IsaacLab env exposes no device for state writes",
            )
        return torch.device(device)

    def _flush(self) -> None:
        """Run the live write flush and simulator forward."""
        scene = getattr(self._env, "scene", None)
        sim = getattr(self._env, "sim", None)
        if scene is None or not hasattr(scene, "write_data_to_sim"):
            raise QueryCapabilityError(
                self._query.backend,
                "scratch.forward",
                "unsupported",
                "the IsaacLab env exposes no scene.write_data_to_sim()",
            )
        if sim is None or not hasattr(sim, "forward"):
            raise QueryCapabilityError(
                self._query.backend,
                "scratch.forward",
                "unsupported",
                "the IsaacLab env exposes no sim.forward()",
            )
        scene.write_data_to_sim()
        sim.forward()

    def _restore_root(
        self,
        name: str,
        pos: torch.Tensor,
        quat_xyzw: torch.Tensor,
        lin: torch.Tensor,
        ang: torch.Tensor,
    ) -> None:
        """Write one captured free-root state back through the live face."""
        self._env.set_root_state(
            [self._env_id],
            name=name,
            pos_w=pos,
            quat_wxyz=self._query._wxyz(quat_xyzw),
            lin_vel_w=lin,
            ang_vel_w=ang,
        )

    def _restore_joints(
        self,
        name: str,
        joint_names: tuple[str, ...],
        position: torch.Tensor,
        velocity: torch.Tensor,
    ) -> None:
        """Write the captured 1-DOF hinge/slide joint state back.

        The scratch write face only addresses 1-DOF hinge/slide joints, so a
        captured joint whose IR row is missing or is another type is skipped:
        it cannot have been changed through the scratch.
        """
        ir = self._query._ir(name)
        writable: list[str] = []
        columns: list[int] = []
        for index, joint_name in enumerate(joint_names):
            row = ir.joints.get(joint_name)
            if row is None or row.type not in _ONE_DOF_JOINT_TYPES:
                continue
            writable.append(joint_name)
            columns.append(index)
        if not writable:
            return
        self._env.set_joint_state(
            [self._env_id],
            articulation=name,
            joint_names=tuple(writable),
            position=position[:, columns],
            velocity=velocity[:, columns],
        )

    def _open(self) -> None:
        """Reject reads and writes after the transaction context closed."""
        if self._closed:
            raise QueryStaleError("scratch world is closed")
        self._query._require_open()


class _UnsupportedContacts(_Subface):
    """Contact reads are capability-gated until their own plans land."""

    def manifold(self, group: str, *, env: EnvSelection = None) -> NoReturn:
        """Reject a group-based contact manifold read on IsaacLab."""
        del group, env
        self._open()
        raise QueryCapabilityError(
            self._query.backend, "contacts.manifold", "unsupported", _P1_DETAIL
        )

    def jacobian(self, point: ContactPoint, articulation: str) -> NoReturn:
        """Reject contact Jacobian reads."""
        del point, articulation
        self._open()
        raise QueryCapabilityError(
            self._query.backend, "contacts.jacobian", "unsupported", _P1_DETAIL
        )

    def wrench_to_joint(self, point: ContactPoint, articulation: str) -> NoReturn:
        """Reject contact wrench-to-joint reads."""
        del point, articulation
        self._open()
        raise QueryCapabilityError(
            self._query.backend, "contacts.wrench_to_joint", "unsupported", _P1_DETAIL
        )


class _ActuatorQuery(_Subface):
    """Static IsaacLab actuator-group facts; runtime state stays unsupported."""

    def rows(self, articulation: str | None = None) -> tuple[ActuatorFacts, ...]:
        """Return every selected actuator group's facts in ``ModelIR`` order."""
        self._open()
        ir = self._query._ir(articulation)
        return tuple(_actuator_facts(ir, row) for row in ir.actuators)

    def facts(self, name: str, *, articulation: str | None = None) -> ActuatorFacts:
        """Return one actuator group's static facts by neutral name."""
        self._open()
        ir = self._query._ir(articulation)
        try:
            row = ir.actuators[name]
        except KeyError:
            raise QueryReferenceError("actuators.facts", name, ir.actuators.names()) from None
        return _actuator_facts(ir, row)

    def state(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> NoReturn:
        """Reject actuator runtime state: an IsaacLab group has no scalar row."""
        del name, articulation, env
        self._open()
        raise QueryCapabilityError(
            self._query.backend,
            "actuators.state",
            "unsupported",
            _ACTUATOR_STATE_DETAIL,
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
        self._active_scratch: set[int] = set()
        self.model = _ModelQuery(self)
        self.state = _StateQuery(self)
        self.kinematics = _UnsupportedKinematics(self)
        self.geometry = _UnsupportedGeometry(self)
        self.contacts = _UnsupportedContacts(self)
        self.actuators = _ActuatorQuery(self)

    def require(self, *ops: str) -> None:
        """Raise when any named query operation is unsupported."""
        self._require_open()
        self.capabilities.require(*ops)

    def scratch(self, env: int | None = None) -> AbstractContextManager[ScratchWorld]:
        """Reject scratch worlds: the shared base has no scratch copy."""
        del env
        self._require_open()
        raise QueryCapabilityError(self.backend, "scratch", "unsupported", _P1_DETAIL)

    def _scratch_env_id(self, env: int | None) -> int:
        """Normalize the scratch env selector to one env id (default env 0)."""
        self._require_open()
        if env is None:
            return 0
        if isinstance(env, bool) or not isinstance(env, int):
            raise QueryShapeError("env", "an int env id or None", type(env).__name__)
        return int(_select_env(env, self._env.num_envs)[0])

    def _open_scratch(self, env_id: int) -> AbstractContextManager[ScratchWorld]:
        """Construct the transaction scratch used by an IsaacLab 3 subclass."""
        return _TransactionScratch(self, env_id)

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

    def _site_target(
        self,
        op: str,
        name: str,
        articulation: str | None,
    ) -> tuple[str, int, torch.Tensor, torch.Tensor]:
        """Resolve one robot-profile site to its owner, body row and local frame.

        Returns ``(articulation, body_index, local_pos, local_quat)`` where the
        local frame is expressed in the owning body frame.  Both the site-pose
        read and the composed site Jacobian use this single resolution.
        """
        inventory = self._site_inventory()
        entry = inventory.get(name)
        if entry is None:
            raise QueryReferenceError(op, name, tuple(sorted(inventory)))
        if entry.kind != "articulation":
            raise QueryCapabilityError(
                self.backend,
                op,
                "partial",
                f"site {name!r} is a generated prop site; only robot-profile sites "
                f"are supported",
            )
        if articulation is not None and articulation != entry.owner:
            raise QueryReferenceError(
                op,
                name,
                tuple(sorted(inventory)),
                f"site {name!r} is owned by articulation {entry.owner!r}",
            )
        profile = self._profiles().get(entry.owner)
        if profile is None:
            raise QueryReferenceError(
                op,
                name,
                tuple(sorted(inventory)),
                f"articulation {entry.owner!r} has no profile",
            )
        canonical = normalize_canonical_body_id(entry.spec.body, f"query site {name!r} body")
        binding = profile.body_bindings.get(canonical)
        if binding is None or "usd" not in binding.names:
            raise QueryReferenceError(
                op,
                name,
                tuple(sorted(inventory)),
                f"site body {entry.spec.body!r} has no usd binding in profile {entry.owner!r}",
            )
        asset = self._asset_for(op, entry.owner)
        data = self._live_data(op, asset, entry.owner)
        names = self._body_names(asset, data)
        index = _body_index(op, binding.names["usd"], names)
        local_pos = torch.tensor(entry.spec.pos, dtype=torch.float64)
        local_quat = torch.tensor(entry.spec.quat_wxyz, dtype=torch.float64)
        return entry.owner, index, local_pos, local_quat

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
