"""IsaacLab director mechanics (the ``_il_*`` half of the old Director).

Every method here is a byte-preserving port of the corresponding
``Director._il_*`` / inline isaaclab branch (the pre-extraction monolith); the Director now
dispatches through :class:`unigym.director.ops.DirectorPropOps`.  Attach
semantics: the kinematic face is UNIVERSAL (per-tick follow via
``write_root_pose_to_sim`` is the default grasp mode); weld is a mujoco
trait (``Trait.DIRECTOR_ATTACH_WELD``).  isaaclab/torch stay lazy
function-body imports (agnostic zone convention).
"""

from __future__ import annotations

import numpy as np

from unigym.director.ops import HAND_ALIAS_CANDIDATES, _OpsBase, strip_prop_prefix


class IsaacLabDirectorOps(_OpsBase):
    """write_root_pose mechanics + RigidObject asset table + cuboid gripper."""

    backend = "isaaclab"

    def __init__(self, env):
        self.env = env
        # prop name -> spawned asset handle (RigidObject / Articulation)
        self._assets: dict = {}
        # prop name -> PropHandler (joints writes / asset metadata)
        self._handlers_by_prop: dict = {}
        self.gripper = None

    # ------------------------------------------------------------------ #
    # binding
    # ------------------------------------------------------------------ #
    def bind(self, gripper_cfg: dict | None):
        """Grab the prop handler/asset tables and mount the virtual gripper.

        Args:
            gripper_cfg: optional yaml ``director.gripper`` dict
                (director_contract §4.1); ``None`` mounts nothing.

        Returns:
            The mounted gripper (or ``None``).
        """
        pm = getattr(self.env, "prop_manager", None)
        if pm is not None:
            self._handlers_by_prop = {h.spec.name: h for h in pm._handlers}
            self._assets = {
                n: getattr(h, "_asset", None) for n, h in self._handlers_by_prop.items()
            }
            self._assets = {n: a for n, a in self._assets.items() if a is not None}
        if gripper_cfg is not None:
            from unigym.backends.registry import backend_info
            from unigym.director.gripper import build_gripper

            self.gripper = build_gripper(
                self.env, gripper_cfg, gripper_backend=backend_info(self.backend).gripper_backend
            )
        return self.gripper

    # ------------------------------------------------------------------ #
    # resolution
    # ------------------------------------------------------------------ #
    def prop_names(self) -> list[str]:
        """Sorted declared prop names (director target-addressing domain)."""
        return sorted(self._handlers_by_prop)

    def resolve_obj(self, name: str):
        """Return the spawned asset handle for a prop name (fail-fast)."""
        from unigym.director import _fail

        asset = self._assets.get(name)
        if asset is None:
            raise _fail(f"unknown prop {name!r}", f"declared props: {sorted(self.prop_names())}")
        return asset

    def resolve_weld_host(self, host: str) -> object:  # pragma: no cover
        """Unreachable on isaaclab (trait-gated; kept for the ops surface).

        The Director's trait gate rejects ``weld`` before this could be
        called (DIRECTOR_ATTACH_WELD is not in the isaaclab2 trait table).
        """
        raise NotImplementedError("this backend declares no weld hosts")

    # ------------------------------------------------------------------ #
    # ops
    # ------------------------------------------------------------------ #
    def teleport(self, name, pos, quat, reset_vel, joints) -> dict:
        """Write a prop's root pose (director_contract §3.1).

        Args:
            name: prop name (director target addressing).
            pos: world position (m; ``(x, y, z)``) or ``None`` to keep.
            quat: world orientation (wxyz) or ``None`` to keep.
            reset_vel: when true, zero the root linear+angular velocity.
            joints: optional ``{joint name: rad}`` writes for articulated
                props (rigid props fail fast).

        Returns:
            Echo dict of the applied writes.
        """
        import torch

        from unigym.director import _fail

        asset = self.resolve_obj(name)
        if joints and not hasattr(asset, "write_joint_state_to_sim"):
            raise _fail(f"prop {name!r} is rigid -- no joints to write")
        pose = torch.zeros(1, 7, device=self.env.device)
        cur = asset.data.root_pos_w[0]
        pose[0, 0:3] = cur if pos is None else torch.tensor(pos, device=self.env.device)
        pose[0, 3:7] = (
            asset.data.root_quat_w[0]
            if quat is None
            else torch.tensor(quat, device=self.env.device)
        )
        asset.write_root_pose_to_sim(pose)
        if reset_vel:
            asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device))
        if joints:
            handler = self._handlers_by_prop.get(name)
            jidx = (
                {n: i for i, n in enumerate(handler._asset.joint_names)}
                if handler is not None and hasattr(handler._asset, "joint_names")
                else {}
            )
            unknown = sorted(set(joints) - set(jidx))
            if unknown:
                raise _fail(
                    f"prop {name!r}: unknown joint(s) {unknown}", f"asset joints: {sorted(jidx)}"
                )
            # write ONLY the named subset (default joint_ids = ALL joints;
            # the others would be zeroed)
            ids = [jidx[n] for n in joints]
            pos_t = torch.zeros(1, len(ids), device=self.env.device)
            vel_t = torch.zeros(1, len(ids), device=self.env.device)
            for jname, val in joints.items():
                pos_t[0, ids.index(jidx[jname])] = float(val)
            asset.write_joint_state_to_sim(pos_t, vel_t, joint_ids=ids)
        return {
            "teleported": name,
            "pos": pos,
            "quat": quat,
            "joints": {k: float(v) for k, v in joints.items()},
        }

    def setvel(self, name, lin, ang) -> dict:
        """Write a prop's root velocity (director_contract §3.2; m/s, rad/s).

        Args:
            name: prop name.
            lin: world linear velocity (m/s; 3-vector).
            ang: world angular velocity (rad/s; 3-vector) or ``None`` to
                leave unchanged.
        """
        import torch

        asset = self.resolve_obj(name)
        vel = torch.zeros(1, 6, device=self.env.device)
        vel[0, 0:3] = torch.tensor(lin, device=self.env.device)
        if ang is not None:
            vel[0, 3:6] = torch.tensor(ang, device=self.env.device)
        asset.write_root_velocity_to_sim(vel)
        return {"setvel": name, "lin": list(lin), "ang": list(ang) if ang else None}

    def attach_weld(self, obj: str, host) -> dict:  # pragma: no cover
        """Unreachable on isaaclab (trait-gated; kept for the ops surface).

        The Director's trait gate rejects ``weld`` before this could be
        called (DIRECTOR_ATTACH_WELD is not in the isaaclab2 trait table).
        """
        raise NotImplementedError("this backend declares no weld attach")

    def detach(self, att: dict) -> None:
        """Detach a kinematic attachment (no backend-side state to clear)."""
        pass  # slave attachments carry no backend-side state

    def host_pose(self, host: str) -> tuple:
        """Read a host's world pose for pin/follow (director_contract §3.7/3.8).

        Args:
            host: ``"gripper"``, ``"hand"``, a prop name, or a robot body
                link name.

        Returns:
            ``(pos, quat)`` world pose (m / wxyz quat; numpy).

        Raises:
            _fail-style error for an unknown host (fail-fast with hint).
        """
        from unigym.director import _fail

        host = strip_prop_prefix(host)
        if self.gripper is not None and (host == self.gripper.name or host == "gripper"):
            return self.gripper.get_pose(self.env)
        asset = self._assets.get(host)
        if asset is not None:
            return self._asset_root_pose(asset)
        robot = getattr(self.env, "robot", None)
        if host == "hand" or robot is not None:
            idx = self._robot_body_idx(robot, host)
            if idx is not None:
                pos = robot.data.body_pos_w[0, idx].detach().cpu().numpy()
                quat = robot.data.body_quat_w[0, idx].detach().cpu().numpy()
                return pos, quat
        raise _fail(
            f"unknown host {host!r}",
            "'gripper', 'hand', a prop name, or a robot body "
            f"link name (declared props: {sorted(self.prop_names())})",
        )

    def write_slave_pose(self, obj: str, pos, quat) -> None:
        """Per-tick slave follow write (director_contract §3.8).

        Args:
            obj: prop name (unknown names are silently ignored -- the
                attachment table already vanished with the env).
            pos: world position (m; ``(x, y, z)``).
            quat: world orientation (wxyz).
        """
        import torch

        asset = self._assets.get(obj)
        if asset is None:
            return
        pose = torch.zeros(1, 7, device=self.env.device)
        pose[0, 0:3] = torch.tensor(np.asarray(pos, dtype=np.float32), device=self.env.device)
        pose[0, 3:7] = torch.tensor(np.asarray(quat, dtype=np.float32), device=self.env.device)
        asset.write_root_pose_to_sim(pose)

    def obj_pose(self, obj: str) -> tuple:
        """Read a rigid prop's world pose (m / wxyz quat; fail-fast).

        Articulation props are rejected (pin/follow is rigid-only;
        director_contract §3.7).
        """
        from unigym.director import _fail

        self.resolve_obj(obj)
        handler = self._handlers_by_prop.get(obj)
        if handler is not None and getattr(handler.spec, "type", "") in (
            "articulation",
            "usd_articulation",
        ):
            raise _fail(
                f"prop {obj!r} is an articulation: pin/follow " f"applies to rigid props",
                "articulation props: use teleport joints writes; " "rigid props: pin/follow",
            )
        return self._asset_root_pose(self._assets[obj])

    def halt(self, obj: str) -> None:
        """Zero a prop's root velocity (director_contract §3.7 pin)."""
        import torch

        asset = self._assets.get(obj)
        if asset is None:
            return
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device))

    def push_root(self, lin, ang, mode: str) -> dict:
        """Write the robot root velocity (director_contract §3.10).

        Args:
            lin: world linear velocity (m/s; 3-vector) or ``None``.
            ang: world angular velocity (rad/s; 3-vector) or ``None``.
            mode: ``"add"`` (delta on the current velocity) or ``"set"``
                (absolute write).
        """
        import torch

        robot = getattr(self.env, "robot", None)
        if robot is None:
            from unigym.director import _fail

            raise _fail(
                "no robot articulation in this scene", "push writes the robot root velocity"
            )
        cur_l = robot.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
        cur_a = robot.data.root_ang_vel_w[0].detach().cpu().numpy().astype(np.float64)
        if mode == "add":
            out_l = cur_l if lin is None else cur_l + np.asarray(lin)
            out_a = cur_a if ang is None else cur_a + np.asarray(ang)
        else:
            out_l = cur_l if lin is None else np.asarray(lin, dtype=np.float64)
            out_a = cur_a if ang is None else np.asarray(ang, dtype=np.float64)
        vel = torch.zeros(1, 6, device=self.env.device)
        vel[0, 0:3] = torch.tensor(np.asarray(out_l, dtype=np.float32))
        vel[0, 3:6] = torch.tensor(np.asarray(out_a, dtype=np.float32))
        robot.write_root_velocity_to_sim(vel)
        return {
            "pushed": "robot",
            "lin": None if lin is None else [float(v) for v in lin],
            "ang": None if ang is None else [float(v) for v in ang],
            "mode": mode,
        }

    def write_joints(self, joints: dict[str, float], zero_vel: bool) -> dict:
        """Robot-joint position write (director_contract §3.11).

        Pose-teleport family: ``write_joint_position_to_sim`` on the named
        subset, range-gated against the sim model limits; ``zero_vel``
        adds a zero ``write_joint_velocity_to_sim``.

        Args:
            joints: ``{IL joint name: rad}`` writes (unknown names fail
                fast; out-of-range values fail fast).
            zero_vel: also zero the written joints' velocity.
        """
        import torch

        from unigym.director import _fail

        robot = getattr(self.env, "robot", None)
        if robot is None:
            raise _fail(
                "no robot articulation in this scene", "joints writes the robot's joint angles"
            )
        from unigym.runtime.controllers import single_articulation_runtime

        names = [str(n) for n in single_articulation_runtime(self.env).joint_names]
        unknown = sorted(set(joints) - set(names))
        if unknown:
            raise _fail(f"unknown robot joint(s) {unknown}", f"robot joints: {names}")
        # range gate (contract §3.11): the scene-face sim limits
        # (same source chain); a sourceless face keeps no gate (honest).
        from unigym.backends.isaaclab2.scene import sim_joint_limits

        limits = sim_joint_limits(robot.data, names) or {}
        for name, val in joints.items():
            rng = limits.get(name)
            if rng is not None and not rng[0] <= float(val) <= rng[1]:
                raise _fail(
                    f"joint {name!r} value {float(val)} outside model "
                    f"limits [{rng[0]}, {rng[1]}]",
                    f"valid range for {name!r}: [{rng[0]}, {rng[1]}]",
                )
        jidx = {n: i for i, n in enumerate(names)}
        ids = [jidx[n] for n in joints]
        pos = torch.zeros(1, len(ids), device=self.env.device)
        for i, val in enumerate(joints.values()):
            pos[0, i] = float(val)
        robot.write_joint_position_to_sim(pos, joint_ids=ids)
        if zero_vel:
            robot.write_joint_velocity_to_sim(
                torch.zeros(1, len(ids), device=self.env.device), joint_ids=ids
            )
        return {"joints": {k: float(v) for k, v in joints.items()}, "zero_vel": bool(zero_vel)}

    def default_attach_mode(self) -> str:
        """Isaaclab attach default: per-tick kinematic follow (``"kin"``)."""
        return "kin"

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _asset_root_pose(asset) -> tuple:
        return (
            asset.data.root_pos_w[0].detach().cpu().numpy(),
            asset.data.root_quat_w[0].detach().cpu().numpy(),
        )

    def _robot_body_idx(self, robot, host: str) -> int | None:
        """Robot body index for a link name (or the ``hand`` alias).

        Match order: exact name -> qualified tail (``right_hand/`` scopes
        on the localized g1.usd) -> the alias candidates.  ``host="hand"``
        resolving to nothing fails fast.
        """
        from unigym.director import _fail

        if robot is None:
            if host == "hand":
                raise _fail(
                    'host alias "hand": no robot articulation in ' "this scene",
                    "declare a robot first",
                )
            return None
        names = [str(n) for n in robot.data.body_names]
        want = host if host != "hand" else None
        if want is not None:
            for i, n in enumerate(names):
                if n == want or n.split("/")[-1] == want:
                    return i
            return None
        for cand in HAND_ALIAS_CANDIDATES:
            for i, n in enumerate(names):
                if n == cand or n.split("/")[-1] == cand:
                    return i
        raise _fail(
            'host alias "hand" resolved to no robot hand body',
            "candidates: " f"{list(HAND_ALIAS_CANDIDATES)}; use an explicit body " "link name",
        )
