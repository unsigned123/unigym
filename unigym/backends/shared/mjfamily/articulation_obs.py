"""Articulation observation + record publication (MuJoCo family).

Every declared observation robot publishes its full typed state into
``env.obs`` (the config-declared :class:`~unigym.env.obs.ObservationRegistry`)
under ``articulation/<name>/...``.  The CPU ``MujocoEnv`` and the ``MjwarpEnv``
share these helpers; every function takes the live env and only touches
data-shaped attributes.
"""

from __future__ import annotations

import numpy as np
import torch

from unigym.core import quat
from unigym.scenes.articulation import resolve_spawn_quat


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
        # benchmark v2: force/active-body facts are record columns, not
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
    for site_name, site_id in getattr(env, "_obs_sites", ()):
        if env.num_envs == 1:
            value = torch.tensor(env._data.site_xpos[site_id], dtype=torch.float32).unsqueeze(0)
        else:
            value = torch.stack(
                [torch.tensor(d.site_xpos[site_id], dtype=torch.float32) for d in env._datas],
                dim=0,
            )
        registry.set_site(site_name, value)


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
