"""Articulation observation + record publication (MuJoCo family).

Every declared observation robot publishes its full typed state into
``env.obs`` (the config-declared :class:`~unigym.env.obs.ObservationRegistry`)
under ``articulation/<name>/...``.  The CPU ``MujocoEnv`` and the ``MjwarpEnv``
share these helpers; every function takes the live env and only touches
data-shaped attributes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from unigym.core import quat
from unigym.scenes.articulation import resolve_spawn_quat

from .model.ir_access import body_id, site_id

if TYPE_CHECKING:
    from unigym.sites import SiteEntry
    from unigym.robots.base import RobotProfile

    from .name_map import MjNameMap

__all__ = [
    "MjSiteHandle",
    "read_articulation_state",
    "refresh_articulation_obs",
    "refresh_obs",
    "refresh_obs_sources",
    "resolve_mj_sites",
]


@dataclass(frozen=True)
class MjSiteHandle:
    """One resolved MuJoCo-family site observation handle.

    ``native`` reads ``site_xpos``/``site_xmat``; ``body`` composes the pose
    from the live body pose and the asset-local offset.  Exactly one id is
    present for each kind.
    """

    name: str
    kind: Literal["native", "body"]
    native_id: int | None = None
    body_id: int | None = None
    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.kind == "native" and self.native_id is None:
            raise ValueError(f"native site handle {self.name!r} requires a native_id")
        if self.kind == "body" and self.body_id is None:
            raise ValueError(f"body site handle {self.name!r} requires a body_id")
        if self.kind not in ("native", "body"):
            raise ValueError(f"site handle {self.name!r} has unknown kind {self.kind!r}")


def _native_site_pose(d, native_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one native MJ site pose as world ``(pos, quat_wxyz)``."""
    from .cameras.helpers import rotmat_to_quat

    pos = np.asarray(d.site_xpos[native_id], dtype=np.float64)
    rot = np.asarray(d.site_xmat[native_id], dtype=np.float64).reshape(3, 3)
    return pos, rotmat_to_quat(rot)


def _body_site_pose(
    d, body_id_value: int, local_pos: Sequence[float], local_quat: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Compose one body-local site pose from the live world body pose."""
    body_pos = np.asarray(d.xpos[body_id_value], dtype=np.float64)
    body_quat = np.asarray(d.xquat[body_id_value], dtype=np.float64)
    offset = np.asarray(local_pos, dtype=np.float64)
    local = np.asarray(local_quat, dtype=np.float64)
    pos = body_pos + quat.to_rotmat(body_quat) @ offset
    quat_w = quat.normalize(quat.mul(body_quat, local))
    return np.asarray(pos, dtype=np.float64), np.asarray(quat_w, dtype=np.float64)


def _site_pose(env, handle: MjSiteHandle, e: int) -> tuple[np.ndarray, np.ndarray]:
    d = env._datas[e]
    if handle.kind == "native":
        assert handle.native_id is not None
        return _native_site_pose(d, handle.native_id)
    assert handle.body_id is not None
    return _body_site_pose(d, handle.body_id, handle.local_pos, handle.local_quat)


def _require_native_match(backend: str, entry: SiteEntry, row, expected_body: str) -> None:
    """Fail when a native site disagrees with its asset-level declaration."""
    spec = entry.spec
    mismatches: list[str] = []
    if row.body != expected_body:
        mismatches.append(f"body {row.body!r} != {expected_body!r}")
    if not np.allclose(row.pos, spec.pos, rtol=0.0, atol=1e-6):
        mismatches.append(f"pos {tuple(row.pos)} != {tuple(spec.pos)}")
    native_quat = np.asarray(row.quat, dtype=np.float64)
    norm = float(np.linalg.norm(native_quat))
    if norm == 0.0:
        mismatches.append("native quat is zero")
    else:
        dot = abs(float(np.dot(native_quat / norm, np.asarray(spec.quat_wxyz))))
        if dot < 1.0 - 1e-6:
            mismatches.append(f"quat {tuple(row.quat)} != {tuple(spec.quat_wxyz)}")
    if mismatches:
        raise ValueError(
            f"{backend}: observations.site {spec.name!r} native site disagrees with "
            f"{entry.kind} {entry.owner!r} declaration: " + "; ".join(mismatches)
        )


def resolve_mj_sites(
    model_ir,
    profiles: Mapping[str, RobotProfile],
    inventory: Mapping[str, SiteEntry],
    declared: Sequence[str],
    *,
    backend: str,
    name_map: MjNameMap | None = None,
) -> list[MjSiteHandle]:
    """Resolve selected ``observations.site`` names for MuJoCo/mjWarp.

    MuJoCo keeps native sites as the fast path.  A robot-profile site without
    a same-named native site uses its canonical body + local pose; a generated
    prop site has no body fallback in this slice.  ``name_map`` applies the
    backend-owned articulation prefix used by composed multi-asset models.
    """
    from unigym.robots.base import normalize_canonical_body_id

    native = set(model_ir.sites.names())
    handles: list[MjSiteHandle] = []
    for name in declared:
        entry = inventory.get(name)
        if entry is not None:
            spec = entry.spec
            if entry.kind == "articulation":
                profile = profiles.get(entry.owner)
                if profile is None:
                    raise ValueError(
                        f"{backend}: observations.site {name!r} is owned by "
                        f"articulation {entry.owner!r} but no profile was supplied"
                    )
                canonical_body = normalize_canonical_body_id(
                    spec.body, f"observations.site {name!r} body"
                )
                binding = profile.body_bindings.get(canonical_body)
                if binding is None or "mjcf" not in binding.names:
                    raise ValueError(
                        f"{backend}: observations.site {name!r} body {spec.body!r} "
                        f"has no mjcf binding in profile {entry.owner!r}"
                    )
                body_name = binding.names["mjcf"]
                if name_map is not None:
                    body_name = name_map.backend_body(entry.owner, body_name)
                body_index = body_id(model_ir, body_name)
                if body_index < 0:
                    raise ValueError(
                        f"{backend}: observations.site {name!r} native body "
                        f"{body_name!r} is not in the MJ model"
                    )
                native_name = name
                prefix = None if name_map is None else name_map.prefix(entry.owner)
                if prefix and site_id(model_ir, f"{prefix}{name}") >= 0:
                    native_name = f"{prefix}{name}"
                native_id = site_id(model_ir, native_name)
                if native_id >= 0:
                    _require_native_match(backend, entry, model_ir.sites[native_name], body_name)
                    handles.append(MjSiteHandle(name=name, kind="native", native_id=native_id))
                else:
                    handles.append(
                        MjSiteHandle(
                            name=name,
                            kind="body",
                            body_id=body_index,
                            local_pos=spec.pos,
                            local_quat=spec.quat_wxyz,
                        )
                    )
                continue

            native_name = f"Prop_{entry.owner}_{spec.name}"
            native_id = site_id(model_ir, native_name)
            if native_id < 0:
                raise ValueError(
                    f"{backend}: observations.site {name!r} generated prop site "
                    f"{native_name!r} is not in the MJ model; native sites: "
                    f"{sorted(native)}; asset sites: {sorted(inventory)}"
                )
            expected_body = f"Prop_{entry.owner}_{spec.body}"
            _require_native_match(backend, entry, model_ir.sites[native_name], expected_body)
            handles.append(MjSiteHandle(name=name, kind="native", native_id=native_id))
            continue

        native_id = site_id(model_ir, name)
        if native_id < 0:
            raise ValueError(
                f"{backend}: observations.site {name!r} not found; native sites: "
                f"{sorted(native)}; asset sites: {sorted(inventory)}"
            )
        handles.append(MjSiteHandle(name=name, kind="native", native_id=native_id))
    return handles


def refresh_obs_sources(env) -> None:
    """Publish camera/contact/object/site sources once per step.

    Contacts are read exactly once per refresh; only the groups selected by
    ``observations.contact`` enter the registry.  Cameras declared only for
    recording never render here (the manager's obs-rig split owns that), and
    object poses come from the prop backend's typed world-pose face.
    """
    registry = env.obs
    schema = registry.schema
    if env.camera_manager is not None:
        for key, value in env.camera_manager.tick_and_read().items():
            if key in schema:
                registry.set_camera(key, value)
    if env.contact_manager is not None:
        contacts = env.contact_manager.read()
        for key, value in contacts.items():
            if key in schema:
                registry.set_contact(key[len("contact/") :], value)
        # episode facts: force/active-body facts are record columns, not
        # schema observations (the one raw read already produced them).
        env._last_contact_facts = {
            key: value
            for key, value in contacts.items()
            if key.endswith("/force") or key.endswith("/active_bodies")
        }
        for name in registry.config.contact:
            if env.contact_manager.has_detail(name):
                registry.set_contact_detail_provider(
                    name, lambda name=name: env.contact_manager.read_detail(name)
                )
    if env.prop_backend is not None:
        poses = env.prop_backend.get_world_poses(env)
        for name in registry.config.object:
            if name in poses:
                pos, quat_w = poses[name]
                registry.set_object(name, "pos", torch.as_tensor(pos, dtype=torch.float32))
                registry.set_object(name, "quat", torch.as_tensor(quat_w, dtype=torch.float32))
        joints = env.prop_backend.get_joint_states(env)
        for name in registry.config.object:
            for joint_name, value in (joints.get(name) or {}).items():
                key = f"object/{name}/joints/{joint_name}"
                if key in schema:
                    registry.set_object(
                        name, f"joints/{joint_name}", torch.as_tensor(value, dtype=torch.float32)
                    )
    for handle in getattr(env, "_obs_sites", ()):
        poses = [_site_pose(env, handle, e) for e in range(env.num_envs)]
        pos = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p, _ in poses], dim=0)
        quat_w = torch.stack([torch.as_tensor(q, dtype=torch.float32) for _, q in poses], dim=0)
        registry.set_site(handle.name, pos, quat_w)


def read_articulation_state(env, art, e: int = 0) -> dict:
    """Snapshot of ONE articulation's post-step state (numpy)."""
    d = env._datas[e]
    q_pol = d.qpos[art.jadr_pol]
    v_pol = d.qvel[art.jdr_pol]
    if art.free_dof_adr is not None:
        a = art.free_qpos_adr
        pos_w = d.qpos[a : a + 3].copy()
        quat_w = d.qpos[a + 3 : a + 7].copy()
        lin_vel_w = d.qvel[art.free_dof_adr : art.free_dof_adr + 3].copy()
        # MuJoCo freejoint angular qvel is BODY-frame; world = R @ local.
        ang_vel_b = d.qvel[art.free_dof_adr + 3 : art.free_dof_adr + 6].copy()
        ang_vel_w = quat.to_rotmat(quat_w) @ ang_vel_b
    else:
        spawn = art.spawn or {}
        if spawn:
            pos = list(spawn["pos"])
        else:
            pos = [float(env._spawn_xy[0]), float(env._spawn_xy[1]), float(env._spawn_height)]
        pos_w = np.asarray(pos[:3], dtype=np.float64)
        quat_w = np.asarray(
            resolve_spawn_quat(spawn, default_yaw=float(env._spawn_yaw)), dtype=np.float64
        )
        lin_vel_w = np.zeros(3)
        ang_vel_b = np.zeros(3)
        ang_vel_w = np.zeros(3)
    rot = quat.to_rotmat(quat_w)
    gravity_b = rot.T @ np.asarray([0.0, 0.0, -1.0])
    return {
        "q_pol": q_pol,
        "v_pol": v_pol,
        "pos_w": pos_w,
        "quat_w": quat_w,
        "rot": rot,
        "gravity_b": gravity_b,
        "lin_vel_w": lin_vel_w,
        "ang_vel_w": ang_vel_w,
        "lin_vel_b": rot.T @ lin_vel_w,
        "ang_vel_b": ang_vel_b,
    }


def _stack(values, dtype=torch.float32) -> torch.Tensor:
    return torch.stack([torch.as_tensor(v, dtype=dtype) for v in values], dim=0)


def refresh_articulation_obs(env) -> None:
    """Publish every declared observation robot under ``articulation/<name>``."""
    registry = env.obs
    f32 = torch.float32
    for name in registry.config.robot:
        art = env.articulations[name]
        states = [read_articulation_state(env, art, e) for e in range(env.num_envs)]
        registry.set_robot(
            name,
            "joint_pos",
            _stack([s["q_pol"] for s in states], dtype=f32),
        )
        registry.set_robot(
            name,
            "joint_pos_rel",
            _stack(
                [env._datas[e].qpos[art.jadr_art] - art.q0_mj for e in range(env.num_envs)],
                dtype=f32,
            ),
        )
        registry.set_robot(name, "joint_vel", _stack([s["v_pol"] for s in states], dtype=f32))
        registry.set_robot(
            name,
            "joint_vel_full",
            _stack(
                [env._datas[e].qvel[art.jdr_art] for e in range(env.num_envs)],
                dtype=f32,
            ),
        )
        registry.set_robot(
            name,
            "applied_torque",
            _stack(
                [env._datas[e].qfrc_actuator[art.jdr_pol] for e in range(env.num_envs)],
                dtype=f32,
            ),
        )
        for source, target in (
            ("gravity_b", "gravity_b"),
            ("pos_w", "base_pos_w"),
            ("quat_w", "base_quat_w"),
            ("lin_vel_w", "base_lin_vel_w"),
            ("ang_vel_w", "base_ang_vel_w"),
            ("lin_vel_b", "base_lin_vel_b"),
            ("ang_vel_b", "base_ang_vel_b"),
        ):
            registry.set_robot(name, target, _stack([s[source] for s in states], dtype=f32))
        registry.set_robot(
            name,
            "base_yaw",
            _stack(
                [np.asarray([float(np.arctan2(s["rot"][1, 0], s["rot"][0, 0]))]) for s in states],
                dtype=f32,
            ),
        )


def refresh_obs(env) -> None:
    """The single per-step registry refresh (sources + robot state)."""
    refresh_obs_sources(env)
    refresh_articulation_obs(env)
