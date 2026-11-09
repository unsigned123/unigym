"""Mjwarp director mechanics: mujoco edits + an explicit device push.

The director's op semantics are identical to :class:`MujocoDirectorOps`
(qpos/qvel writes on the CPU mirror, model-level re-anchors / weld slots,
mocap pose writes).  The difference is the engine: a warp step reads the
DEVICE ``Data``/``Model``, never the mirror, so every mutating op must end
with :meth:`MjwarpEnv.sync_writes` (mirror state + director-mutable model
fields -> the batched device).  Without it a teleport/setvel/gripper/weld
write would sit on the mirror and the next step would silently undo it.

N=1 scope: the trait table scopes these ops to ``max_envs=1`` (one MjModel /
one mocap namespace); construction at N>1 is refused by
``reject_model_global_unsupported`` before this ops object exists.
"""

from __future__ import annotations

from unigym.backends.shared.mjfamily.director.ops import MujocoDirectorOps


class MjwarpDirectorOps(MujocoDirectorOps):
    """Mujoco director mechanics + a mirror->warp-device push per write."""

    backend = "mjwarp"

    def _sync(self) -> None:
        """Push the CPU mirror (and model edits) to the warp device."""
        self.env.sync_writes(0)

    def teleport(self, name, pos, quat, reset_vel, joints) -> dict:
        """Teleport a prop, then push the write to the device."""
        out = super().teleport(name, pos, quat, reset_vel, joints)
        self._sync()
        return out

    def setvel(self, name, lin, ang) -> dict:
        """Overwrite a prop root velocity, then push to the device."""
        out = super().setvel(name, lin, ang)
        self._sync()
        return out

    def attach_weld(self, obj: str, host) -> dict:
        """Activate a weld slot, then push eq state/model to the device."""
        out = super().attach_weld(obj, host)
        self._sync()
        return out

    def detach(self, att: dict) -> None:
        """Deactivate a weld, then push eq_active to the device."""
        super().detach(att)
        self._sync()

    def write_slave_pose(self, obj: str, pos, quat) -> None:
        """Write a kinematic follower pose, then push qpos to the device."""
        super().write_slave_pose(obj, pos, quat)
        self._sync()

    def halt(self, obj: str) -> None:
        """Zero a prop root velocity, then push qvel to the device."""
        super().halt(obj)
        self._sync()

    def push_root(self, lin, ang, mode: str) -> dict:
        """Write the robot root velocity, then push qvel to the device."""
        out = super().push_root(lin, ang, mode)
        self._sync()
        return out

    def write_joints(self, joints, zero_vel: bool) -> dict:
        """Write robot joints, then push qpos/qvel to the device."""
        out = super().write_joints(joints, zero_vel)
        self._sync()
        return out

    def after_gripper_move(self) -> None:
        """Refresh derived quantities, then push mocap/state to the device."""
        super().after_gripper_move()
        self._sync()
