"""Shared MuJoCo-family runtime-query implementation.

``backends/mujoco`` and ``backends/mjwarp`` both speak the same
``MjModel``/``MjData`` contract, so the query math owns here.  A backend
subclass supplies its env type, backend label and capability object; it never
copies the geometry, Jacobian or scratch implementation.

Engine-thread contract: the query runs at a step boundary and adds no lock.
Live-state reads come from the backend's CPU ``MjData`` face.  For mjwarp that
face is the CPU **mirror** of the warp device data, so a caller must query
between steps, after the mirror sync.  Scratch uses a private ``mj_copyData``
of that mirror plus ``mj_forward``; it never mutates the live mirror and never
touches the warp device.  Every result is a detached CPU torch copy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, NoReturn, Protocol, runtime_checkable

import numpy as np
import torch

from unigym.core import quat as quat_math
from unigym.env.backends.base import normalize_state_env_ids
from unigym.model import Actuator, Body, Geom, Joint, ModelIR, ModelSet, Site
from unigym.query.base import (
    AABB,
    ComRef,
    DistanceResult,
    DofLayout,
    EntityRef,
    EnvSelection,
    Jacobian,
    PointRef,
    Pose,
)
from unigym.query.capabilities import QueryCapabilities
from unigym.query.errors import (
    QueryCapabilityError,
    QueryReferenceError,
    QueryShapeError,
    QueryStaleError,
)
from unigym.query.facade import ScratchWorld
from unigym.runtime.controllers import ArticulationRuntime

if TYPE_CHECKING:
    from unigym.backends.shared.mjfamily.props.props import MujocoPropBackend

import mujoco  # noqa: E402  (the backend env sets MUJOCO_GL before this import)

__all__ = ["MjFamilyEnv", "MjFamilyQuery"]


@runtime_checkable
class MjFamilyEnv(Protocol):
    """The MuJoCo-family env surface the shared query implementation reads.

    The members are read-only properties: both backends expose them as
    properties (`mj_model`, `mj_datas`, ...) or plain attributes, and the
    query only reads them.
    """

    @property
    def num_envs(self) -> int:
        """Number of live worlds."""
        ...

    @property
    def mj_model(self) -> mujoco.MjModel:
        """The compiled CPU model; ``None`` after teardown."""
        ...

    @property
    def mj_datas(self) -> Sequence[mujoco.MjData]:
        """One CPU ``MjData`` read face per env (the mirror for mjWarp)."""
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
    def prop_backend(self) -> MujocoPropBackend | None:
        """The optional prop runtime, or ``None`` when the env has no props."""
        ...


_BASE_DOF_SUFFIXES = ("px", "py", "pz", "rx", "ry", "rz")
_JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}
_ONE_DOF_JOINT_TYPES = frozenset(
    {int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)}
)


def _joint_type_name(joint_type: int) -> str:
    """Return the MuJoCo joint type name (or its integer spelling)."""
    return _JOINT_TYPE_NAMES.get(joint_type, str(joint_type))


def _subtree_body_ids(model: mujoco.MjModel, root: int) -> set[int]:
    """Return *root* and every compiled body below it."""
    if root < 0 or root >= model.nbody:
        return set()
    if root == 0:
        return set(range(model.nbody))
    ids = {root}
    for body_id in range(1, model.nbody):
        current = body_id
        while current != 0:
            parent = int(model.body_parentid[current])
            if parent == root:
                ids.add(body_id)
                break
            current = parent
    return ids


def _select_env(selection: EnvSelection, num_envs: int) -> torch.Tensor:
    """Normalize ``env=None``/int/list/tensor to a 1-D int64 env id vector."""
    if selection is None:
        return normalize_state_env_ids(None, num_envs, device="cpu")
    if isinstance(selection, bool):
        raise QueryShapeError("env", "an int, a sequence of ints or None", repr(selection))
    if isinstance(selection, int):
        selection = [selection]
    if isinstance(selection, torch.Tensor):
        if selection.numel() == 0:
            return torch.empty(0, dtype=torch.int64)
    else:
        values = list(selection)
        if not values:
            return torch.empty(0, dtype=torch.int64)
        selection = values
    return normalize_state_env_ids(selection, num_envs, device="cpu")


def _tensor(value: np.ndarray) -> torch.Tensor:
    """Return a detached float64 CPU copy of *value*."""
    return torch.tensor(np.array(value, dtype=np.float64, copy=True), dtype=torch.float64)


def _target_articulation(target: EntityRef | ComRef | PointRef) -> str | None:
    """Return the articulation named by the target reference, if any."""
    if isinstance(target, ComRef):
        return target.articulation
    if isinstance(target, PointRef):
        return target.entity.articulation
    return target.articulation


def _stack(values: list[np.ndarray], width: int) -> np.ndarray:
    """Stack per-env rows, preserving the ``(0, width)`` empty selection."""
    if not values:
        return np.empty((0, width), dtype=np.float64)
    return np.stack(values, axis=0).astype(np.float64, copy=False)


def _identity_quats(rows: int) -> np.ndarray:
    """Return ``rows`` identity ``(w, x, y, z)`` quaternions."""
    out = np.zeros((rows, 4), dtype=np.float64)
    out[:, 0] = 1.0
    return out


def _scratch_vector(value: torch.Tensor, field: str, length: int) -> np.ndarray:
    """Validate a scratch write vector and return a finite float64 array."""
    if not isinstance(value, torch.Tensor):
        raise QueryShapeError(field, f"a torch.Tensor of shape ({length},)", type(value).__name__)
    if tuple(value.shape) != (length,):
        raise QueryShapeError(field, f"shape ({length},)", str(tuple(value.shape)))
    arr = value.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(arr).all():
        raise QueryShapeError(field, "finite values", str(arr.tolist()))
    return arr


class _QueryPart:
    """Common env/name resolution for one query subface."""

    def __init__(
        self,
        query: MjFamilyQuery,
        *,
        data: mujoco.MjData | None = None,
        env_id: int | None = None,
        scratch: _ScratchWorld | None = None,
    ) -> None:
        self._query = query
        self._data = data
        self._env_id = env_id
        self._scratch = scratch

    @property
    def _env(self) -> MjFamilyEnv:
        return self._query._env

    def _open(self) -> None:
        self._query._require_open()
        if self._scratch is not None and self._scratch._closed:
            raise QueryStaleError("scratch world is closed")

    def _frames(self, env: EnvSelection) -> tuple[torch.Tensor, list[mujoco.MjData]]:
        if self._data is not None:
            if self._env_id is None:
                raise QueryShapeError("env_id", "a scratch env id", "None")
            env_id = int(self._env_id)
            ids = torch.tensor([env_id], dtype=torch.int64)
            if env is not None:
                requested = _select_env(env, self._env.num_envs)
                if not torch.equal(requested, ids):
                    raise QueryShapeError(
                        "env",
                        f"the scratch env {env_id}",
                        str(requested.tolist()),
                    )
            return ids, [self._data]
        ids = _select_env(env, self._env.num_envs)
        return ids, [self._env.mj_datas[int(index)] for index in ids.tolist()]

    def _model_set(self) -> ModelSet:
        return self._env.model_set

    def _ir(self, articulation: str | None) -> ModelIR:
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
        art = self._env.articulations.get(name)
        if art is None:
            raise QueryReferenceError("articulation", name, sorted(self._env.articulations))
        return art

    def _body_row(self, name: str, articulation: str | None) -> Body:
        ir = self._ir(articulation)
        try:
            return ir.bodies[name]
        except KeyError:
            raise QueryReferenceError("body", name, ir.bodies.names()) from None

    def _geom_row(self, name: str, articulation: str | None) -> Geom:
        ir = self._ir(articulation)
        try:
            return ir.geoms[name]
        except KeyError:
            raise QueryReferenceError("geom", name, ir.geoms.names()) from None

    def _site_row(self, name: str, articulation: str | None) -> Site:
        ir = self._ir(articulation)
        try:
            return ir.sites[name]
        except KeyError:
            raise QueryReferenceError("site", name, ir.sites.names()) from None

    def _joint_row(self, name: str, articulation: str | None) -> Joint:
        ir = self._ir(articulation)
        try:
            return ir.joints[name]
        except KeyError:
            raise QueryReferenceError("joint", name, ir.joints.names()) from None

    def _body_id(self, row: Body) -> int:
        if row.body_id is None:
            raise QueryReferenceError("body", row.name, (), "the model is not compiled")
        return int(row.body_id)

    def _geom_id(self, row: Geom) -> int:
        if row.geom_id is None:
            raise QueryReferenceError("geom", row.name, (), "the model is not compiled")
        return int(row.geom_id)

    def _site_id(self, row: Site) -> int:
        if row.site_id is None:
            raise QueryReferenceError("site", row.name, (), "the model is not compiled")
        return int(row.site_id)


class MjFamilyQuery:
    """The shared MuJoCo-family runtime-query facade.

    A concrete backend validates its own env type, then sets :attr:`backend`
    and :attr:`capabilities` before calling this constructor.  Capability,
    stale and unsupported-operation messages use the concrete backend label.
    """

    backend: str
    capabilities: QueryCapabilities

    def __init__(self, env: MjFamilyEnv) -> None:
        if not isinstance(env, MjFamilyEnv):
            raise TypeError(
                f"{type(self).__name__} requires a MuJoCo-family env, " f"got {type(env).__name__}"
            )
        self._env: MjFamilyEnv = env
        self._closed = False
        self._active_scratch: set[int] = set()
        self.model = _ModelQuery(self)
        self.state = _StateQuery(self)
        self.kinematics = _KinematicsQuery(self)
        self.geometry = _GeometryQuery(self)
        self.contacts = _UnsupportedContacts(self)
        self.actuators = _UnsupportedActuators(self)

    def require(self, *ops: str) -> None:
        """Raise when any named query operation is unsupported."""
        self._require_open()
        self.capabilities.require(*ops)

    def scratch(self, env: int | None = None) -> AbstractContextManager[_ScratchWorld]:
        """Open one exclusive scratch copy for *env* (default env 0)."""
        self._require_open()
        if env is None:
            env_id = 0
        else:
            if isinstance(env, bool) or not isinstance(env, int):
                raise QueryShapeError(
                    "env",
                    "an int env id or None",
                    type(env).__name__,
                )
            env_id = int(_select_env(env, self._env.num_envs)[0])
        return _ScratchContext(self, env_id)

    def close(self) -> None:
        """Invalidate this query handle (idempotent)."""
        self._closed = True

    def _require_open(self) -> None:
        # ``close`` nulls the env's live model handle; reading it is the
        # backend-neutral "env is gone" signal shared by both family envs.
        if self._closed or self._env.mj_model is None:
            raise QueryStaleError(
                f"query for backend {self.backend!r} is stale: " f"the run/env/query is closed"
            )


class _ModelQuery(_QueryPart):
    def ir(self, articulation: str | None = None) -> ModelIR:
        self._open()
        return self._ir(articulation)

    def body(self, name: str, *, articulation: str | None = None) -> Body:
        self._open()
        return self._body_row(name, articulation)

    def geom(self, name: str, *, articulation: str | None = None) -> Geom:
        self._open()
        return self._geom_row(name, articulation)

    def site(self, name: str, *, articulation: str | None = None) -> Site:
        self._open()
        return self._site_row(name, articulation)

    def joint(self, name: str, *, articulation: str | None = None) -> Joint:
        self._open()
        return self._joint_row(name, articulation)

    def actuator(self, name: str, *, articulation: str | None = None) -> Actuator:
        self._open()
        ir = self._ir(articulation)
        try:
            return ir.actuators[name]
        except KeyError:
            raise QueryReferenceError("actuator", name, ir.actuators.names()) from None


class _StateQuery(_QueryPart):
    def body_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> Pose:
        self._open()
        body_id = self._body_id(self._body_row(name, articulation))
        ids, datas = self._frames(env)
        positions = [np.asarray(data.xpos[body_id], dtype=np.float64) for data in datas]
        quats = [np.asarray(data.xquat[body_id], dtype=np.float64) for data in datas]
        return Pose(ids, _tensor(_stack(positions, 3)), _tensor(_stack(quats, 4)))

    def site_pose(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> Pose:
        self._open()
        site_id = self._site_id(self._site_row(name, articulation))
        ids, datas = self._frames(env)
        positions = [np.asarray(data.site_xpos[site_id], dtype=np.float64) for data in datas]
        quats = [_site_quat(data, site_id) for data in datas]
        return Pose(ids, _tensor(_stack(positions, 3)), _tensor(_stack(quats, 4)))

    def com_pose(
        self,
        ref: ComRef,
        *,
        frame: str = "world",
        env: EnvSelection = None,
    ) -> Pose:
        self._open()
        if frame != "world":
            raise QueryCapabilityError(
                self._query.backend,
                "state.com_pose",
                "unsupported",
                f"frame={frame!r}; only 'world' is implemented in P0A",
            )
        body_id = self._com_body_id(ref)
        ids, datas = self._frames(env)
        positions = [
            np.asarray(
                data.subtree_com[body_id] if ref.subtree else data.xipos[body_id],
                dtype=np.float64,
            )
            for data in datas
        ]
        return Pose(ids, _tensor(_stack(positions, 3)), _tensor(_identity_quats(len(datas))))

    def root_pose(self, articulation: str, *, env: EnvSelection = None) -> Pose:
        self._open()
        art = self._articulation(articulation)
        if art.free_qpos_adr is None:
            if art.root == "free":
                raise QueryCapabilityError(
                    self._query.backend,
                    "state.root_pose",
                    "unsupported",
                    f"articulation {articulation!r} declares a free root but "
                    f"the compiled free joint is unresolved",
                )
            raise QueryReferenceError(
                "root_pose",
                articulation,
                (),
                "the articulation has a fixed root",
            )
        adr = int(art.free_qpos_adr)
        ids, datas = self._frames(env)
        positions = [np.asarray(data.qpos[adr : adr + 3], dtype=np.float64) for data in datas]
        quats = [np.asarray(data.qpos[adr + 3 : adr + 7], dtype=np.float64) for data in datas]
        return Pose(ids, _tensor(_stack(positions, 3)), _tensor(_stack(quats, 4)))

    def joint_position(
        self,
        articulation: str,
        *,
        joint_names: Sequence[str] | None = None,
        env: EnvSelection = None,
    ) -> torch.Tensor:
        self._open()
        return self._joint_values(articulation, joint_names, env, velocity=False)

    def joint_velocity(
        self,
        articulation: str,
        *,
        joint_names: Sequence[str] | None = None,
        env: EnvSelection = None,
    ) -> torch.Tensor:
        self._open()
        return self._joint_values(articulation, joint_names, env, velocity=True)

    def _com_body_id(self, ref: ComRef) -> int:
        art = self._articulation(ref.articulation)
        name = ref.body
        if name is None:
            name = str(art.base_body_id or art.base_body or "")
        if not name:
            raise QueryReferenceError(
                "com_pose",
                ref.articulation,
                (),
                "no base body is declared for this articulation",
            )
        return self._body_id(self._body_row(name, ref.articulation))

    def _joint_values(
        self,
        articulation: str,
        joint_names: Sequence[str] | None,
        env: EnvSelection,
        *,
        velocity: bool,
    ) -> torch.Tensor:
        op = "state.joint_velocity" if velocity else "state.joint_position"
        art = self._articulation(articulation)
        names = tuple(art.joint_names if joint_names is None else joint_names)
        if not names:
            ids, _ = self._frames(env)
            return torch.empty((ids.numel(), 0), dtype=torch.float64)
        index = {name: position for position, name in enumerate(art.joint_names)}
        missing = sorted(set(names) - set(index))
        if missing:
            raise QueryReferenceError(op, missing[0], art.joint_names)
        for name in names:
            self._require_1dof(name, articulation, op)
        addresses = art.jdr_art if velocity else art.jadr_art
        if addresses is None:
            raise QueryShapeError(
                "jdr_art" if velocity else "jadr_art",
                "a compiled address table",
                "None",
            )
        rows = np.asarray(addresses, dtype=np.int64).reshape(-1)[[index[name] for name in names]]
        ids, datas = self._frames(env)
        values = [
            np.asarray(
                data.qvel[rows] if velocity else data.qpos[rows],
                dtype=np.float64,
            )
            for data in datas
        ]
        return _tensor(_stack(values, len(names)))

    def _require_1dof(self, name: str, articulation: str, op: str) -> None:
        row = self._joint_row(name, articulation)
        if row.type not in ("hinge", "slide"):
            raise QueryCapabilityError(
                self._query.backend,
                op,
                "unsupported",
                f"joint {name!r} has type {row.type!r}; only 1-DOF joints are supported",
            )


class _KinematicsQuery(_QueryPart):
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
        self._open()
        if frame != "world":
            raise QueryCapabilityError(
                self._query.backend,
                "jacobian",
                "unsupported",
                f"frame={frame!r}; only 'world' is implemented in P0A",
            )
        if reference != "body_origin":
            raise QueryCapabilityError(
                self._query.backend,
                "jacobian",
                "unsupported",
                f"reference={reference!r}; the target's own origin is the only "
                f"reference available in P0A",
            )
        if isinstance(target, ComRef) and target.subtree and rotational:
            raise QueryCapabilityError(
                self._query.backend,
                "jacobian.com",
                "unsupported",
                "a subtree COM has no defined rotational Jacobian; "
                "pass rotational=False for the translational block",
            )
        art_name = self._articulation_for(target, articulation)
        columns, layout = self._columns(art_name)
        ids, datas = self._frames(env)
        model = self._env.mj_model
        transl: list[np.ndarray] = []
        rot: list[np.ndarray] = []
        zeros = np.zeros((3, len(columns)), dtype=np.float64)
        for data in datas:
            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            self._fill(data, jacp, jacr, target, art_name)
            transl.append(np.asarray(jacp[:, columns], dtype=np.float64))
            rot.append(np.asarray(jacr[:, columns], dtype=np.float64) if rotational else zeros)
        return Jacobian(
            ids,
            torch.tensor(np.stack(transl, axis=0) if transl else np.empty((0, 3, layout.nv))),
            torch.tensor(np.stack(rot, axis=0) if rot else np.empty((0, 3, layout.nv))),
            layout,
            frame,
            reference,
        )

    def _articulation_for(
        self,
        target: EntityRef | ComRef | PointRef,
        articulation: str | None,
    ) -> str:
        model_set = self._model_set()
        names = model_set.names()
        reference = _target_articulation(target)
        if reference is not None:
            if articulation is not None and articulation != reference:
                raise QueryReferenceError(
                    "jacobian",
                    reference,
                    names,
                    f"reference articulation {reference!r} conflicts with "
                    f"articulation={articulation!r}",
                )
            if reference not in model_set:
                raise QueryReferenceError("jacobian", reference, names)
            return reference
        if articulation is not None:
            if articulation not in model_set:
                raise QueryReferenceError("jacobian", articulation, names)
            return articulation
        if len(model_set) == 1:
            return names[0]
        if isinstance(target, (EntityRef, PointRef)):
            entity = target if isinstance(target, EntityRef) else target.entity
            table = {"body": "bodies", "site": "sites", "geom": "geoms"}[entity.kind]
            matches = [name for name in names if entity.name in getattr(model_set[name], table)]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise QueryReferenceError(
                    "jacobian",
                    entity.name,
                    names,
                    "no articulation declares this entity",
                )
            raise QueryReferenceError(
                "jacobian",
                entity.name,
                matches,
                "the entity is ambiguous across articulations; pass articulation=...",
            )
        raise QueryReferenceError(
            "jacobian",
            None,
            names,
            "pass articulation=... for a multi-articulation run",
        )

    def _columns(self, articulation: str) -> tuple[np.ndarray, DofLayout]:
        art = self._articulation(articulation)
        model = self._env.mj_model
        ir = self._ir(articulation)
        base_id = self._compiled_base_body_id(art, ir)
        if base_id is None:
            if art.root == "free":
                raise QueryCapabilityError(
                    self._query.backend,
                    "jacobian",
                    "unsupported",
                    f"articulation {art.name!r} declares a free root but its "
                    f"compiled free root/base body is unresolved (declare "
                    f"profile.base_body or the articulation base body)",
                )
            raise QueryCapabilityError(
                self._query.backend,
                "jacobian",
                "unsupported",
                f"articulation {art.name!r} base body is unresolved; cannot "
                f"scope the compiled joint set",
            )
        if art.root == "free" and art.free_dof_adr is None:
            raise QueryCapabilityError(
                self._query.backend,
                "jacobian",
                "unsupported",
                f"articulation {art.name!r} declares a free root but the "
                f"compiled free joint is unresolved",
            )
        compiled = self._compiled_joints(model, base_id)
        columns: list[int] = []
        covered: set[int] = set()
        base_dofs: tuple[str, ...] = ()
        base_names: tuple[str, ...] = ()
        if art.free_dof_adr is not None:
            start = int(art.free_dof_adr)
            info = compiled.get(start)
            if info is None or info[1] != int(mujoco.mjtJoint.mjJNT_FREE):
                raise QueryCapabilityError(
                    self._query.backend,
                    "jacobian",
                    "unsupported",
                    f"articulation {art.name!r} declares a free root at dof "
                    f"{start} but the compiled joint is not free",
                )
            base_body = str(art.base_body_id or art.base_body or "root")
            base_dofs = tuple(f"{base_body}.{suffix}" for suffix in _BASE_DOF_SUFFIXES)
            base_names = (base_body,)
            columns.extend(range(start, start + 6))
            covered.update(columns)
        joint_names: list[str] = []
        for name in art.joint_names:
            row = ir.joints.get(name)
            if row is None or row.dof_adr is None:
                raise QueryReferenceError(
                    "jacobian",
                    name,
                    ir.joints.names(),
                    "the articulation joint is missing from the compiled IR",
                )
            dof = int(row.dof_adr)
            info = compiled.get(dof)
            if info is None:
                raise QueryReferenceError(
                    "jacobian",
                    name,
                    [value[2] for value in compiled.values()],
                    "the joint is not in the articulation subtree",
                )
            _, joint_type, compiled_name = info
            if joint_type not in _ONE_DOF_JOINT_TYPES:
                raise QueryCapabilityError(
                    self._query.backend,
                    "jacobian",
                    "unsupported",
                    f"articulation {art.name!r} joint {name!r} "
                    f"({compiled_name!r}) has compiled type "
                    f"{_joint_type_name(joint_type)!r}; only hinge/slide joints "
                    f"and the free root are supported",
                )
            columns.append(dof)
            joint_names.append(name)
            covered.add(dof)
        for dof, (_, joint_type, compiled_name) in compiled.items():
            if dof in covered:
                continue
            raise QueryCapabilityError(
                self._query.backend,
                "jacobian",
                "unsupported",
                f"articulation {art.name!r} has compiled joint "
                f"{compiled_name!r} (compiled type "
                f"{_joint_type_name(joint_type)!r}) outside the neutral "
                f"joint table; only hinge/slide joints and the free root "
                f"are supported",
            )
        order = base_dofs + tuple(joint_names)
        layout = DofLayout(
            articulation=articulation,
            base_dofs=base_dofs,
            base_names=base_names,
            joint_names=tuple(joint_names),
            nv=len(columns),
            order=order,
        )
        return np.asarray(columns, dtype=np.int64), layout

    def _compiled_base_body_id(self, art: ArticulationRuntime, ir: ModelIR) -> int | None:
        name = str(art.base_body_id or art.base_body or "")
        if name:
            row = ir.bodies.get(name)
            if row is not None and row.body_id is not None:
                return int(row.body_id)
        body_id = art.addresses.get("base_body_id")
        if isinstance(body_id, int) and body_id >= 0:
            return body_id
        return None

    def _compiled_joints(
        self,
        model: mujoco.MjModel,
        base_id: int | None,
    ) -> dict[int, tuple[int, int, str]]:
        if base_id is None:
            joint_ids = range(model.njnt)
        else:
            body_ids = _subtree_body_ids(model, base_id)
            joint_ids = [
                joint_id
                for joint_id in range(model.njnt)
                if int(model.jnt_bodyid[joint_id]) in body_ids
            ]
        compiled: dict[int, tuple[int, int, str]] = {}
        for joint_id in joint_ids:
            dof = int(model.jnt_dofadr[joint_id])
            compiled[dof] = (
                joint_id,
                int(model.jnt_type[joint_id]),
                str(model.joint(joint_id).name),
            )
        return compiled

    def _fill(
        self,
        data: mujoco.MjData,
        jacp: np.ndarray,
        jacr: np.ndarray,
        target: EntityRef | ComRef | PointRef,
        articulation: str,
    ) -> None:
        model = self._env.mj_model
        if isinstance(target, PointRef):
            body_id = self._body_id(self._body_row(target.entity.name, articulation))
            point = np.ascontiguousarray(target.point_w.numpy(), dtype=np.float64)
            mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
            return
        if isinstance(target, ComRef):
            body_id = self._com_body_id(target, articulation)
            if target.subtree:
                mujoco.mj_jacSubtreeCom(model, data, jacp, body_id)
            else:
                mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
            return
        if target.kind == "body":
            body_id = self._body_id(self._body_row(target.name, articulation))
            mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
            return
        if target.kind == "site":
            site_id = self._site_id(self._site_row(target.name, articulation))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            return
        raise QueryCapabilityError(
            self._query.backend,
            "jacobian.geom",
            "unsupported",
            "Jacobian targets are body/site/COM/world-point; a geom has no frame",
        )

    def _com_body_id(self, ref: ComRef, articulation: str) -> int:
        art = self._articulation(articulation)
        name = ref.body if ref.body is not None else str(art.base_body_id or art.base_body or "")
        if not name:
            raise QueryReferenceError("jacobian", articulation, (), "no base body is declared")
        return self._body_id(self._body_row(name, articulation))


class _GeometryQuery(_QueryPart):
    def distance(
        self,
        a: EntityRef,
        b: EntityRef,
        *,
        max_distance_m: float = 1.0,
        env: EnvSelection = None,
        scratch: ScratchWorld | None = None,
    ) -> DistanceResult:
        self._open()
        if scratch is not None:
            if env is not None:
                raise QueryShapeError(
                    "env/scratch",
                    "exactly one of env or scratch",
                    "both were supplied",
                )
            if self._data is not None:
                raise QueryShapeError(
                    "scratch",
                    "no scratch argument on a scratch-bound geometry face",
                    type(scratch).__name__,
                )
            return scratch.geometry().distance(a, b, max_distance_m=max_distance_m)
        limit = float(max_distance_m)
        if not np.isfinite(limit) or limit <= 0.0:
            raise QueryShapeError("max_distance_m", "a finite positive distance", repr(limit))
        geom_a = self._require_geom(a)
        geom_b = self._require_geom(b)
        ids, datas = self._frames(env)
        model = self._env.mj_model
        distances: list[float] = []
        closest_a: list[np.ndarray] = []
        closest_b: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        capped: list[bool] = []
        for data in datas:
            fromto = np.zeros(6, dtype=np.float64)
            raw = float(mujoco.mj_geomDistance(model, data, geom_a, geom_b, limit, fromto))
            delta = fromto[3:6] - fromto[0:3]
            norm = float(np.linalg.norm(delta))
            normal = delta / norm if norm > 1e-12 else np.zeros(3, dtype=np.float64)
            distances.append(raw)
            closest_a.append(fromto[0:3].copy())
            closest_b.append(fromto[3:6].copy())
            normals.append(normal)
            capped.append(raw >= limit)
        return DistanceResult(
            ids,
            _tensor(np.asarray(distances, dtype=np.float64)),
            _tensor(_stack(closest_a, 3)),
            _tensor(_stack(closest_b, 3)),
            _tensor(_stack(normals, 3)),
            torch.tensor(capped, dtype=torch.bool),
            mesh_exact=False,
        )

    def aabb(self, entity: EntityRef, *, env: EnvSelection = None) -> AABB:
        self._open()
        rows = self._entity_geoms(entity)
        ids, datas = self._frames(env)
        model = self._env.mj_model
        lower: list[np.ndarray] = []
        upper: list[np.ndarray] = []
        for data in datas:
            bounds = [self._geom_bounds(model, data, row) for row in rows]
            lower.append(np.min([bound[0] for bound in bounds], axis=0))
            upper.append(np.max([bound[1] for bound in bounds], axis=0))
        return AABB(ids, _tensor(_stack(lower, 3)), _tensor(_stack(upper, 3)))

    def _require_geom(self, entity: EntityRef) -> int:
        if entity.kind != "geom":
            raise QueryReferenceError(
                "geometry.distance",
                entity.name,
                (),
                "both operands must be geom references",
            )
        return self._geom_id(self._geom_row(entity.name, entity.articulation))

    def _entity_geoms(self, entity: EntityRef) -> list[Geom]:
        if entity.kind == "geom":
            return [self._geom_row(entity.name, entity.articulation)]
        if entity.kind == "body":
            ir = self._ir(entity.articulation)
            rows = [row for row in ir.geoms if row.body == entity.name]
            if not rows:
                raise QueryReferenceError(
                    "geometry.aabb",
                    entity.name,
                    ir.geoms.names(),
                    "the body has no geoms",
                )
            return rows
        raise QueryReferenceError(
            "geometry.aabb",
            entity.name,
            (),
            "AABB is defined for geom and body references",
        )

    def _geom_bounds(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        row: Geom,
    ) -> tuple[np.ndarray, np.ndarray]:
        geom_id = self._geom_id(row)
        size = np.asarray(row.size, dtype=np.float64)
        if row.type == "sphere":
            half = np.full(3, size[0], dtype=np.float64)
        elif row.type in ("capsule", "cylinder"):
            half = np.asarray([size[0], size[0], size[1] + size[0]], dtype=np.float64)
        elif row.type in ("ellipsoid", "box"):
            half = size.copy()
        elif row.type == "mesh":
            radius = float(model.geom_rbound[geom_id])
            half = np.full(3, radius, dtype=np.float64)
        else:
            raise QueryCapabilityError(
                self._query.backend,
                "geometry.aabb",
                "unsupported",
                f"geom {row.name!r} has type {row.type!r} with no finite AABB",
            )
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        world_half = np.abs(rotation) @ half
        return center - world_half, center + world_half


class _UnsupportedContacts:
    def __init__(self, query: MjFamilyQuery) -> None:
        self._query = query

    def manifold(
        self,
        a: EntityRef | None = None,
        b: EntityRef | None = None,
        *,
        env: EnvSelection = None,
    ) -> NoReturn:
        del a, b, env
        self._raise("contacts.manifold")

    def jacobian(
        self,
        a: EntityRef,
        b: EntityRef | None = None,
        *,
        point: int = 0,
        env: EnvSelection = None,
        frame: str = "world",
    ) -> NoReturn:
        del a, b, point, env, frame
        self._raise("contacts.jacobian")

    def _raise(self, op: str) -> NoReturn:
        self._query._require_open()
        self._query.capabilities.require(op)
        raise AssertionError(f"{op} is declared unsupported but require() returned")


class _UnsupportedActuators:
    def __init__(self, query: MjFamilyQuery) -> None:
        self._query = query

    def facts(self, name: str, *, articulation: str | None = None) -> NoReturn:
        del name, articulation
        self._raise("actuators.facts")

    def state(
        self,
        name: str,
        *,
        articulation: str | None = None,
        env: EnvSelection = None,
    ) -> NoReturn:
        del name, articulation, env
        self._raise("actuators.state")

    def _raise(self, op: str) -> NoReturn:
        self._query._require_open()
        self._query.capabilities.require(op)
        raise AssertionError(f"{op} is declared unsupported but require() returned")


class _ScratchContext:
    def __init__(self, query: MjFamilyQuery, env_id: int) -> None:
        self._query = query
        self._env_id = env_id
        self._world: _ScratchWorld | None = None

    def __enter__(self) -> _ScratchWorld:
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
            self._world = _ScratchWorld(self._query, self._env_id)
        except Exception:
            self._query._active_scratch.discard(self._env_id)
            raise
        return self._world

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc, traceback
        if self._world is not None:
            self._world.close()
        self._query._active_scratch.discard(self._env_id)
        return False


class _ScratchWorld:
    def __init__(self, query: MjFamilyQuery, env_id: int) -> None:
        self._query = query
        self._env = query._env
        self._env_id = env_id
        self._closed = False
        self._data = mujoco.MjData(self._env.mj_model)
        mujoco.mj_copyData(self._data, self._env.mj_model, self._env.mj_datas[env_id])
        self._state = _StateQuery(query, data=self._data, env_id=env_id, scratch=self)
        self._kinematics = _KinematicsQuery(query, data=self._data, env_id=env_id, scratch=self)
        self._geometry = _GeometryQuery(query, data=self._data, env_id=env_id, scratch=self)

    def set_root_state(
        self,
        name: str,
        *,
        pos_w: torch.Tensor,
        quat_wxyz: torch.Tensor,
        lin_vel_w: torch.Tensor | None = None,
        ang_vel_w: torch.Tensor | None = None,
    ) -> None:
        self._open()
        pos = _scratch_vector(pos_w, "pos_w", 3)
        quaternion = _scratch_vector(quat_wxyz, "quat_wxyz", 4)
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise QueryShapeError("quat_wxyz", "a non-zero quaternion", str(quaternion.tolist()))
        quaternion = quaternion / norm
        linear = None if lin_vel_w is None else _scratch_vector(lin_vel_w, "lin_vel_w", 3)
        angular = None if ang_vel_w is None else _scratch_vector(ang_vel_w, "ang_vel_w", 3)
        qpos_adr, dof_adr = self._root_addresses(name)
        self._data.qpos[qpos_adr : qpos_adr + 3] = pos
        self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = quaternion
        if linear is not None:
            self._data.qvel[dof_adr : dof_adr + 3] = linear
        if angular is not None:
            rotation = np.asarray(quat_math.to_rotmat(quaternion), dtype=np.float64)
            self._data.qvel[dof_adr + 3 : dof_adr + 6] = rotation.T @ angular

    def set_joint_state(
        self,
        articulation: str,
        joint_names: Sequence[str],
        position: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ) -> None:
        self._open()
        art = self._articulation(articulation)
        if isinstance(joint_names, (str, bytes)):
            raise QueryShapeError("joint_names", "a sequence of joint names", "a string")
        names = tuple(str(name) for name in joint_names)
        if not names:
            raise QueryShapeError("joint_names", "a non-empty sequence", "[]")
        if len(set(names)) != len(names):
            raise QueryShapeError("joint_names", "unique names", str(names))
        index = {name: position_index for position_index, name in enumerate(art.joint_names)}
        missing = sorted(set(names) - set(index))
        if missing:
            raise QueryReferenceError("scratch.set_joint_state", missing[0], art.joint_names)
        for name in names:
            row = self._query.model.joint(name, articulation=articulation)
            if row.type not in ("hinge", "slide"):
                raise QueryCapabilityError(
                    self._query.backend,
                    "scratch.set_joint_state",
                    "unsupported",
                    f"joint {name!r} has type {row.type!r}; only 1-DOF joints are supported",
                )
        positions = _scratch_vector(position, "position", len(names))
        velocities = None if velocity is None else _scratch_vector(velocity, "velocity", len(names))
        if art.jadr_art is None or art.jdr_art is None:
            raise QueryShapeError(
                "jadr_art/jdr_art",
                "compiled joint address tables",
                "None",
            )
        qpos_adr = np.asarray(art.jadr_art, dtype=np.int64).reshape(-1)[
            [index[name] for name in names]
        ]
        dof_adr = np.asarray(art.jdr_art, dtype=np.int64).reshape(-1)[
            [index[name] for name in names]
        ]
        self._data.qpos[qpos_adr] = positions
        if velocities is not None:
            self._data.qvel[dof_adr] = velocities

    def forward(self) -> None:
        self._open()
        mujoco.mj_forward(self._env.mj_model, self._data)

    def body_pose(self, name: str, *, articulation: str | None = None) -> Pose:
        self._open()
        return self._state.body_pose(name, articulation=articulation)

    def site_pose(self, name: str, *, articulation: str | None = None) -> Pose:
        self._open()
        return self._state.site_pose(name, articulation=articulation)

    def com_pose(self, ref: ComRef, *, frame: str = "world") -> Pose:
        self._open()
        return self._state.com_pose(ref, frame=frame)

    def geometry(self) -> _GeometryQuery:
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
        self._open()
        return self._kinematics.jacobian(
            target,
            articulation=articulation,
            frame=frame,
            reference=reference,
            rotational=rotational,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._query._active_scratch.discard(self._env_id)

    def _open(self) -> None:
        if self._closed:
            raise QueryStaleError("scratch world is closed")
        self._query._require_open()

    def _articulation(self, name: str) -> ArticulationRuntime:
        art = self._env.articulations.get(name)
        if art is None:
            raise QueryReferenceError("scratch", name, sorted(self._env.articulations))
        return art

    def _root_addresses(self, name: str) -> tuple[int, int]:
        art = self._env.articulations.get(name)
        if art is not None:
            if art.free_qpos_adr is None or art.free_dof_adr is None:
                if art.root == "free":
                    raise QueryCapabilityError(
                        self._query.backend,
                        "scratch.set_root_state",
                        "unsupported",
                        f"articulation {name!r} declares a free root but the "
                        f"compiled free joint is unresolved",
                    )
                raise QueryReferenceError(
                    "scratch.set_root_state",
                    name,
                    (),
                    "the articulation has a fixed root",
                )
            return int(art.free_qpos_adr), int(art.free_dof_adr)
        prop_backend = self._env.prop_backend
        if prop_backend is not None and name in prop_backend.free_prop_names():
            _, qpos_adr, dof_adr = prop_backend.free_addresses(name)
            return int(qpos_adr), int(dof_adr)
        free_props = [] if prop_backend is None else prop_backend.free_prop_names()
        raise QueryReferenceError(
            "scratch.set_root_state",
            name,
            (*sorted(self._env.articulations), *free_props),
        )


def _site_quat(data: mujoco.MjData, site_id: int) -> np.ndarray:
    """Return a site's world wxyz quaternion from ``site_xmat``."""
    value = np.zeros(4, dtype=np.float64)
    matrix = np.ascontiguousarray(data.site_xmat[site_id], dtype=np.float64)
    mujoco.mju_mat2Quat(value, matrix)
    return value
