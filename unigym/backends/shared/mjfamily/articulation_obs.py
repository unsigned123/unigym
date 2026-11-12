"""Articulation observation + record + termination keys.

Every articulation publishes its robot state under ``articulation/<name>/``.
These helpers are shared by the CPU ``MujocoEnv`` and the ``MjwarpEnv`` (same
``MjModel``/``MjData`` family); every function takes the live env and only
touches data-shaped attributes.
"""

from __future__ import annotations

import numpy as np
import torch

from unigym.core import quat
from unigym.env.obs_catalog import ObsCatalog, build_obs_catalog


def refresh_obs_sources(env) -> None:
    """Inject camera/contact/prop-served keys into ``env.obs_cache``."""
    if env.camera_manager is not None:
        for k, v in env.camera_manager.tick_and_read().items():
            env.obs_cache[k] = v
    if env.contact_manager is not None and env.contact_manager.obs_keys():
        for k, v in env.contact_manager.read().items():
            if k in env.contact_manager.obs_keys():
                env.obs_cache[k] = v
    if env.contact_manager is not None:
        for req in env._obs_requests:
            if req.key.startswith("contact_ops/"):
                name = req.key.split("/", 1)[1]
                if name in {s.name for s in env.contact_manager.specs}:
                    env.obs_cache[req.key] = env.contact_manager.read_detail(name)
    if env.prop_backend is not None:
        requested = {r.key for r in env._obs_requests}
        for k, v in env.prop_backend.get_state(env).items():
            if k in requested:
                env.obs_cache[k] = v


def articulation_catalog(articulation) -> ObsCatalog:
    """The per-articulation obs catalog, keys under ``articulation/<name>``."""
    return build_obs_catalog(
        articulation.profile,
        articulation_joint_names=articulation.robot_joint_names,
        articulation=articulation.name,
    )


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
        ang_vel_w = d.qvel[art.free_dof_adr + 3 : art.free_dof_adr + 6].copy()
    else:
        pos = list(art.spawn.get("pos") or [0.0, 0.0, env._spawn_height])
        yaw = float(art.spawn.get("yaw") or 0.0)
        pos_w = np.asarray(pos[:3], dtype=np.float64)
        quat_w = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)
        lin_vel_w = np.zeros(3)
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
        "ang_vel_b": rot.T @ ang_vel_w,
        "foot_force_w": foot_forces(env, art, d),
    }


def foot_forces(env, art, d) -> np.ndarray:
    """Per-foot world-frame net contact force for ONE articulation."""
    import mujoco

    out = np.zeros((len(art.foot_geom_sets), 3))
    frc6 = np.zeros(6)
    ncon = int(d.ncon)
    # array views: the cached contact[i]/geom1 aliases go stale after a
    # backend contact-buffer realloc (mjwarp get_data_into)
    geoms = np.asarray(d.contact.geom)[:ncon]
    frames = np.asarray(d.contact.frame)[:ncon].reshape(-1, 9)
    for i in range(ncon):
        g1, g2 = int(geoms[i, 0]), int(geoms[i, 1])
        for f, foot_geoms in enumerate(art.foot_geom_sets):
            if g1 in foot_geoms or g2 in foot_geoms:
                mujoco.mj_contactForce(env._model, d, i, frc6)
                frame = frames[i].reshape(3, 3)
                out[f] += frame.T @ frc6[:3]
                break
    return out


def articulation_extra(env, art) -> dict:
    """Namespaced state keys for ONE articulation (termination/record face)."""
    f32 = torch.float32
    states = [read_articulation_state(env, art, ee) for ee in range(env.num_envs)]

    def stack(values, dtype=f32):
        return torch.stack([torch.as_tensor(v, dtype=dtype) for v in values], dim=0)

    prefix = f"articulation/{art.name}"
    out = {
        f"{prefix}/joint_pos": stack([s["q_pol"] for s in states]),
        f"{prefix}/joint_vel": stack([s["v_pol"] for s in states]),
        f"{prefix}/applied_torque": stack(
            [env._datas[ee].qfrc_actuator[art.jdr_pol] for ee in range(env.num_envs)]
        ),
        f"{prefix}/gravity_b": stack([s["gravity_b"] for s in states]),
        f"{prefix}/base_pos_w": stack([s["pos_w"] for s in states]),
        f"{prefix}/base_quat_w": stack([s["quat_w"] for s in states]),
        f"{prefix}/base_lin_vel_w": stack([s["lin_vel_w"] for s in states]),
        f"{prefix}/base_ang_vel_w": stack([s["ang_vel_w"] for s in states]),
        f"{prefix}/base_lin_vel_b": stack([s["lin_vel_b"] for s in states]),
        f"{prefix}/base_ang_vel_b": stack([s["ang_vel_b"] for s in states]),
    }
    fz = [s["foot_force_w"][:, 2] for s in states]
    out[f"{prefix}/foot_contact_force_z"] = stack(fz)
    out[f"{prefix}/foot_contact"] = stack(
        [f > env._contact_force_threshold for f in fz], dtype=torch.bool
    )
    out[f"{prefix}/foot_pos_z"] = stack(
        [[env._datas[ee].xpos[b][2] for b in art.foot_body_ids] for ee in range(env.num_envs)]
    )
    return out


def refresh_articulation_obs(env) -> None:
    """Publish every articulation under ``articulation/<name>/...``."""
    f32 = torch.float32
    for art in env._articulation_order:
        prefix = f"articulation/{art.name}"
        states = [read_articulation_state(env, art, e) for e in range(env.num_envs)]
        q = torch.stack(
            [
                torch.as_tensor(env._datas[e].qpos[art.jadr_art] - art.q0_mj, dtype=f32)
                for e in range(env.num_envs)
            ],
            dim=0,
        )
        v = torch.stack(
            [
                torch.as_tensor(env._datas[e].qvel[art.jdr_art], dtype=f32)
                for e in range(env.num_envs)
            ],
            dim=0,
        )
        tau = torch.stack(
            [
                torch.as_tensor(env._datas[e].qfrc_actuator[art.jdr_art], dtype=f32)
                for e in range(env.num_envs)
            ],
            dim=0,
        )
        env.obs_cache[f"{prefix}/joint_pos_rel"] = q
        env.obs_cache[f"{prefix}/joint_vel"] = v
        env.obs_cache[f"{prefix}/applied_torque"] = tau
        for source, target in (
            ("gravity_b", "gravity_b"),
            ("pos_w", "base_pos_w"),
            ("quat_w", "base_quat_w"),
            ("ang_vel_b", "base_ang_vel_b"),
        ):
            env.obs_cache[f"{prefix}/{target}"] = torch.stack(
                [torch.as_tensor(s[source], dtype=f32) for s in states], dim=0
            )
        foot = [np.linalg.norm(s["foot_force_w"], axis=-1) for s in states]
        env.obs_cache[f"{prefix}/foot_contact_force"] = torch.stack(
            [torch.as_tensor(f, dtype=f32) for f in foot], dim=0
        )
        env.obs_cache[f"{prefix}/base_yaw"] = torch.stack(
            [
                torch.as_tensor([float(np.arctan2(s["rot"][1, 0], s["rot"][0, 0]))], dtype=f32)
                for s in states
            ],
            dim=0,
        )


def refresh_obs_cache(env, raw_action=None, cmd=None) -> None:
    """The namespaced observation refresh (bound at construction)."""
    del raw_action, cmd
    refresh_obs_sources(env)
    refresh_articulation_obs(env)
