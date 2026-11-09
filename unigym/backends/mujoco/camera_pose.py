"""World-pose read-back for the mujoco camera rigs.

Kept in its own small module so the backend's main file stays above the
complexity gate's per-file MI floor.  The rig passes itself; every engine
import stays lazy (the module is import-safe without mujoco).
"""
from __future__ import annotations

import numpy as np


def read_world_pose(rig):
    """``(pos(3,), quat_wxyz(4,))`` world convention for one bound rig.

    ``body`` mounts derive the pose from ``data.cam_xpos/cam_xmat`` (FK at
    the live state); ``world``/``chase`` read the single-world state-table
    row (C2 -- the authoritative pose), falling back to the model slots
    before the first row write.
    """
    from .cameras import (bridge_quat, _quat_conj, _quat_mul_np,
                          rig_type, rotmat_to_quat)

    if rig._env is None or rig._cam_id is None:
        raise RuntimeError(
            f"camera '{rig.name}': not bound (bind_env)")
    if rig_type(rig.spec) == "body":
        d = rig._env.mj_data
        pos = np.asarray(d.cam_xpos[rig._cam_id], dtype=np.float64).copy()
        R = np.asarray(d.cam_xmat[rig._cam_id],
                       dtype=np.float64).reshape(3, 3)
        q_isaac = _quat_mul_np(rotmat_to_quat(R),
                               _quat_conj(bridge_quat()))
        return pos, q_isaac
    row = rig.state_table.get(rig.name)
    if row is not None:
        return (np.asarray(row.pos, dtype=np.float64).copy(),
                np.asarray(row.quat, dtype=np.float64).copy())
    m = rig._env.mj_model
    pos = m.cam_pos[rig._cam_id].copy()
    q_mj = np.asarray(m.cam_quat[rig._cam_id], dtype=np.float64)
    return pos, _quat_mul_np(q_mj, _quat_conj(bridge_quat()))
