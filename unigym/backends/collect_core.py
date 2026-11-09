"""Engine-agnostic env bookkeeping shared by every backend.

These functions operate on any env exposing the documented attribute set;
engine actions belong to the backend.  They take the live env as their first
argument but only read/write data-shaped attributes (``_datas``, joint
tables, action resolver, prop/contact managers) -- no engine calls.  The one
engine-shaped read, foot contact force, is a duck-typed method each backend
provides (``env._foot_contact_forces(d)``).

Imports here are deliberately restricted to numpy / torch / ``unigym.core``
(plus lazy in-function imports of data-only helpers): no ``mujoco``,
``mujoco_warp`` or ``warp``, and no ``unigym.backends.*`` -- a second backend
must be able to reuse this module without importing the mujoco backend.
"""

from __future__ import annotations

import numpy as np
import torch

from unigym.core import quat


def fetch_table(env, cmd, raw_action) -> dict:
    """The post-step obs fetch table (N=1 legacy expressions / N>1 stacks)."""
    if env.num_envs == 1:
        state = env._read_state()
        return {
            # joint_pos_rel / joint_vel are served per-request by
            # _joint_obs (name gather over the full 43-joint articulation)
            "gravity_b": lambda: torch.tensor(state["gravity_b"], dtype=torch.float32).unsqueeze(0),
            "base_ang_vel_b": lambda: torch.tensor(
                state["ang_vel_b"], dtype=torch.float32
            ).unsqueeze(0),
            "base_quat_w": lambda: torch.tensor(state["quat_w"], dtype=torch.float32).unsqueeze(0),
            "base_pos_w": lambda: torch.tensor(state["pos_w"], dtype=torch.float32).unsqueeze(0),
            "base_yaw": lambda: torch.tensor(
                [float(np.arctan2(state["rot"][1, 0], state["rot"][0, 0]))], dtype=torch.float32
            ).unsqueeze(0),
            "foot_contact_force": lambda: torch.tensor(
                [np.linalg.norm(state["foot_force_w"], axis=-1)], dtype=torch.float32
            ),
            "command": lambda: cmd,
            "prev_action": lambda: raw_action,
        }
    states = [env._read_state(e) for e in range(env.num_envs)]
    return {
        "gravity_b": lambda: torch.stack(
            [torch.tensor(s["gravity_b"], dtype=torch.float32) for s in states]
        ),
        "base_ang_vel_b": lambda: torch.stack(
            [torch.tensor(s["ang_vel_b"], dtype=torch.float32) for s in states]
        ),
        "base_quat_w": lambda: torch.stack(
            [torch.tensor(s["quat_w"], dtype=torch.float32) for s in states]
        ),
        "base_pos_w": lambda: torch.stack(
            [torch.tensor(s["pos_w"], dtype=torch.float32) for s in states]
        ),
        "base_yaw": lambda: torch.stack(
            [
                torch.tensor(
                    [float(np.arctan2(s["rot"][1, 0], s["rot"][0, 0]))], dtype=torch.float32
                )
                for s in states
            ]
        ),
        "foot_contact_force": lambda: torch.stack(
            [
                torch.tensor([np.linalg.norm(s["foot_force_w"], axis=-1)], dtype=torch.float32)
                for s in states
            ]
        ),
        "command": lambda: cmd,
        "prev_action": lambda: raw_action,
    }


def build_extra(
    env, state: dict | None, e: int = 0, articulation: str | None = None, include_state: bool = True
) -> dict:
    """Recorded extra keys (cpu tensors), per-articulation state namespaced.

    With *include_state* the 13 per-articulation state keys are emitted under
    ``articulation/<articulation>/...`` (single and multi use the same
    rule; there is no bare projection).  *include_state=False* emits only the
    global bookkeeping keys (time/episode/contact/object/site) -- the multi
    path publishes each binding through
    :func:`...multi_obs.binding_extra`.
    """
    f32 = torch.float32
    if include_state and not articulation:
        raise ValueError(
            "build_extra: per-articulation state keys require an "
            "articulation name (v103.5; no bare projection)"
        )

    def core(s, ee):
        d = env._datas[ee]
        fz = s["foot_force_w"][:, 2]
        torque = d.qfrc_actuator[env._jdr_pol]
        return {
            "joint_pos": torch.tensor(s["q_pol"], dtype=f32).unsqueeze(0),
            "joint_vel": torch.tensor(s["v_pol"], dtype=f32).unsqueeze(0),
            "applied_torque": torch.tensor(torque, dtype=f32).unsqueeze(0),
            "gravity_b": torch.tensor(s["gravity_b"], dtype=f32).unsqueeze(0),
            "base_pos_w": torch.tensor(s["pos_w"], dtype=f32).unsqueeze(0),
            "base_quat_w": torch.tensor(s["quat_w"], dtype=f32).unsqueeze(0),
            "base_lin_vel_w": torch.tensor(s["lin_vel_w"], dtype=f32).unsqueeze(0),
            "base_ang_vel_w": torch.tensor(s["ang_vel_w"], dtype=f32).unsqueeze(0),
            "base_lin_vel_b": torch.tensor(s["lin_vel_b"], dtype=f32).unsqueeze(0),
            "base_ang_vel_b": torch.tensor(s["ang_vel_b"], dtype=f32).unsqueeze(0),
            "foot_contact": torch.tensor(fz > env._contact_force_threshold).unsqueeze(0),
            "foot_contact_force_z": torch.tensor(fz, dtype=f32).unsqueeze(0),
            "foot_pos_z": torch.tensor(
                [d.xpos[b][2] for b in env._foot_body_ids], dtype=f32
            ).unsqueeze(0),
        }

    if include_state:
        if env.num_envs == 1:
            extra = core(state, 0)
        else:
            parts = [core(env._read_state(ee), ee) for ee in range(env.num_envs)]
            extra = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
        prefix = f"articulation/{articulation}"
        extra = {f"{prefix}/{k}": v for k, v in extra.items()}
    else:
        extra = {}
    extra["time_in_episode"] = env._time_in_episode.clone()
    extra["episode_id"] = env._episode_id.clone()
    # feat/contact-manager: contact/<name> record keys (additive)
    contact_values = env.contact_manager.read() if env.contact_manager is not None else None
    if contact_values is not None:
        extra.update(contact_values)
    # feat/object-state (slimmed): object/<name>/
    # {pos,quat} record keys (props only; no props section -> zero new
    # keys).  Contact criteria are declared contact groups -- no held
    # generation here any more.
    if env.prop_backend is not None:
        from unigym.props.object_state import build_object_extra

        extra.update(
            build_object_extra(
                env.prop_manager,
                env.prop_backend.get_world_poses(),
                joint_states=env.prop_backend.get_joint_states(),
            )
        )
    # EEF site record family: site/<name>/pos (world frame,
    # (N, 3) f32) -- additive; empty declaration emits nothing.
    for sname, sid in env._record_sites:
        if env.num_envs == 1:
            extra[f"site/{sname}/pos"] = torch.tensor(
                env._data.site_xpos[sid], dtype=f32
            ).unsqueeze(0)
        else:
            extra[f"site/{sname}/pos"] = torch.stack(
                [torch.tensor(d.site_xpos[sid], dtype=f32) for d in env._datas], dim=0
            )
    return extra


def read_state(env, e: int = 0) -> dict:
    """Snapshot of the post-step sim state (numpy, policy order)."""
    d = env._datas[e]
    q_pol = d.qpos[env._jadr_pol]
    v_pol = d.qvel[env._jdr_pol]
    if env._has_freejoint:
        pos_w = d.qpos[0:3].copy()
        quat_w = d.qpos[3:7].copy()  # freejoint: wxyz
        lin_vel_w = d.qvel[0:3].copy()  # world (freejoint)
        ang_vel_w = d.qvel[3:6].copy()  # world (measured)
    else:
        # fixed base -- qpos[0:7]/qvel[0:6] belong to the
        # robot's own hinges here (GR1 waist/arms), NOT a base frame.
        # The base frame is the spawn pose and it never moves.
        pos_w = env._fixed_base_pos_w.copy()
        quat_w = env._fixed_base_quat_w.copy()
        lin_vel_w = np.zeros(3)
        ang_vel_w = np.zeros(3)
    rot = quat.to_rotmat(quat_w)  # world <- body
    gravity_b = rot.T @ np.asarray([0.0, 0.0, -1.0])
    foot_force_w = env._foot_contact_forces(d)  # (F, 3) world
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
        "foot_force_w": foot_force_w,
    }
