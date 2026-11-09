"""MuJoCo virtual gripper: spec-side injection + the mocap runtime hand.

The director's manipulable hand (contract §4).  Declarative
opt-in in the run config::

    director:
      gripper: {name: vgrip, size: [...], pos: [...], quat: [...], ...}

A mocap body (``mocap="true"``: force-free, teleportable) named ``<name>``
carries a non-colliding box geom (``contype=0 conaffinity=0`` -- the hand
never disturbs scene physics), the grasp ``<name>_grasp`` site (weld anchor
target) and the ``<name>_<camera>`` camera element.  Runtime pose lives in
``data.mocap_pos/mocap_quat``.  Spec-side also pre-allocates one DISABLED
weld equality per rigid prop (``director_weld_<prop>``, ``eq_active0=false``)
that Director.attach enables at runtime.

Naming collisions with existing model/scene names fail fast at build time.
"""
from __future__ import annotations

import numpy as np

from unigym.backends.shared.mjfamily.director.gripper import (
    add_director_bodies)

__all__ = ["MujocoGripper", "add_director_bodies"]


class MujocoGripper:
    """Runtime face of the mujoco mocap hand (N=1)."""

    def __init__(self, env, name: str):
        import mujoco

        m = env.mj_model
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"director gripper body {name!r} not in the "
                             f"MJ model (was the run started with --direct "
                             f"and a director.gripper section?)")
        self.name = name
        self.bid = bid
        self.mocapid = int(m.body_mocapid[bid])
        if self.mocapid < 0:
            raise ValueError(f"body {name!r} is not a mocap body")
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE,
                                 f"{name}_grasp")
        self.grasp_site = int(site)
        self._mujoco = mujoco

    # ---- pose face ------------------------------------------------------- #
    def get_pose(self, env) -> tuple[np.ndarray, np.ndarray]:
        """Current mocap world pose ``(pos (3,), quat (4,) wxyz)``."""
        d = env.mj_data
        return (np.array(d.mocap_pos[self.mocapid]),
                np.array(d.mocap_quat[self.mocapid]))

    def set_pose(self, env, pos=None, quat=None) -> None:
        """Write the mocap world pose (``None`` keeps that component)."""
        d = env.mj_data
        if pos is not None:
            d.mocap_pos[self.mocapid] = np.asarray(pos, dtype=np.float64)
        if quat is not None:
            d.mocap_quat[self.mocapid] = np.asarray(quat, dtype=np.float64)
