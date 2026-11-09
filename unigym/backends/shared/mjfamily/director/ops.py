"""Mujoco director mechanics (the ``_mj_*`` half of the old Director).

Every method here is a byte-preserving port of the corresponding
``Director._mj_*`` / inline mujoco branch (the pre-extraction monolith); the Director now
dispatches through :class:`unigym.director.ops.DirectorPropOps`.  mujoco
stays a lazy function-body import (agnostic zone convention).
the kinematic primitives land on the same qpos/qvel addressing (pin/
follow = per-tick ``write_slave_pose`` + ``halt``; push = the robot-base
freejoint qvel).
"""
from __future__ import annotations

import numpy as np

from unigym.director.ops import (HAND_ALIAS_CANDIDATES, _OpsBase,
                                 WeldHost, strip_prop_prefix)


def _weld_bid(host: WeldHost) -> int:
    """Read the MJ body id out of an opaque weld host handle."""
    bid = host.payload
    if not isinstance(bid, int):
        raise TypeError(
            "mujoco director: weld host payload must be a body id, got "
            f"{type(bid).__name__}")
    return bid


class MujocoDirectorOps(_OpsBase):
    """Weld-slots + qpos/ctrl addressing + mocap gripper mechanics.

    Every op here is an EXPLICIT scene EDIT on the live ``env.mj_data``
    (qpos/qvel writes) followed by ``mj_forward`` to refresh the derived
    quantities.  Live-edit perturbs physics by design -- the forward
    rewrite of ``qacc``/``qacc_warmstart`` feeds the next step.  That is
    the operator's intent for these write ops; passive observers
    (camera rendering) must never take this path -- they solve on a
    private mjData copy (cameras.py ``_render_data``).
    """

    backend = "mujoco"

    def __init__(self, env):
        self.env = env
        # free props: name -> (body_id, qpos addr, qvel addr)
        self._free: dict[str, tuple[int, int, int]] = {}
        # articulation props: name -> handle (joints/qpos/dof/ctrl addrs)
        self._arts: dict = {}
        # the virtual gripper (constructed in bind; None when undeclared)
        self.gripper = None

    # ------------------------------------------------------------------ #
    # binding
    # ------------------------------------------------------------------ #
    def bind(self, gripper_cfg: dict | None):
        """Load prop address tables and mount the virtual gripper.

        Args:
            gripper_cfg: optional yaml ``director.gripper`` dict
                (director_contract §4.1); ``None`` mounts nothing.

        Returns:
            The mounted gripper (or ``None``).
        """
        pb = getattr(self.env, "prop_backend", None)
        if pb is not None:
            for name in pb.free_prop_names():
                self._free[name] = pb.free_addresses(name)
            self._arts = dict(pb.articulations())
        if gripper_cfg is not None:
            from unigym.backends.registry import backend_info
            from unigym.director.gripper import build_gripper
            self.gripper = build_gripper(
                self.env, gripper_cfg,
                gripper_backend=backend_info(self.backend).gripper_backend)
        return self.gripper

    # ------------------------------------------------------------------ #
    # resolution
    # ------------------------------------------------------------------ #
    def prop_names(self) -> list[str]:
        """Sorted declared prop names, free rigid + articulation."""
        return sorted(self._free) + sorted(self._arts)

    def resolve_obj(self, name: str):
        """Validate a prop name (fail-fast) and return it unchanged.

        mujoco addressing is name-based (qpos/qvel addr lookup happens
        in the ops); the returned value is the prop name itself.
        """
        if name not in self._free and name not in self._arts:
            from unigym.director import _fail
            raise _fail(f"unknown prop {name!r}",
                        f"declared props: {sorted(self.prop_names())}")
        return name

    def resolve_weld_host(self, host: str) -> WeldHost:
        """Resolve a weld/pin host to an opaque MJ body handle.

        Resolution order:
        ``gripper`` -> ``hand`` alias -> ``prop:name`` -> prop name ->
        any MJ body name (robot links included).  The payload is the MJ
        body id, read only by this backend's ``attach_weld``/``host_pose``.
        """
        import mujoco

        from unigym.director import _fail

        host = strip_prop_prefix(host)
        m = self.env.mj_model
        if host == "gripper" or (
                self.gripper is not None and host == self.gripper.name):
            if self.gripper is None:
                raise _fail("no gripper on this run (declare "
                            "director.gripper)", "host: gripper needs the "
                            "director.gripper yaml section")
            grip_bid = getattr(self.gripper, "bid", None)
            if not isinstance(grip_bid, int):
                raise _fail("mujoco gripper exposes no body id",
                            "the mujoco gripper backend must publish `bid`")
            return WeldHost(grip_bid)
        if host == "hand":
            for cand in HAND_ALIAS_CANDIDATES:
                for backend in self._backend_bodies(cand):
                    bid = mujoco.mj_name2id(
                        m, mujoco.mjtObj.mjOBJ_BODY, backend)
                    if bid >= 0:
                        return WeldHost(bid)
            raise _fail('host alias "hand" resolved to no robot hand body',
                        "candidates: "
                        f"{list(HAND_ALIAS_CANDIDATES)}; use an explicit "
                        "unigym body name")
        pb = getattr(self.env, "prop_backend", None)
        if pb is not None:
            bid = pb.body_id(host)
            if bid is not None:
                return WeldHost(bid)
        for backend in self._backend_bodies(host):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, backend)
            if bid >= 0:
                return WeldHost(bid)
        raise _fail(f"unknown host {host!r}",
                    "'gripper', 'hand', a prop name, or a unigym body name "
                    f"(declared props: {sorted(self.prop_names())})")

    def _backend_bodies(self, name: str) -> list[str]:
        """Neutral body name -> candidate compiled names (identity first)."""
        candidates = [name]
        name_map = getattr(self.env, "_mj_name_map", None)
        bindings = getattr(self.env, "_mj_articulation_bindings", {}) or {}
        if name_map is not None:
            for articulation in bindings:
                backend = name_map.backend_body(articulation, name)
                if backend not in candidates:
                    candidates.append(backend)
        return candidates

    # ------------------------------------------------------------------ #
    # ops
    # ------------------------------------------------------------------ #
    def teleport(self, name, pos, quat, reset_vel, joints) -> dict:
        """Write a prop's qpos/anchor (director_contract §3.1).

        Args:
            name: prop name (free rigid: qpos write; articulation:
                model-level root re-anchor + per-joint qpos writes).
            pos: world position (m; ``(x, y, z)``) or ``None`` to keep.
            quat: world orientation (wxyz) or ``None`` to keep.
            reset_vel: when true, zero the affected velocities.
            joints: optional ``{joint name: rad}`` writes for
                articulation props (rigid props have none).

        Returns:
            Echo dict of the applied writes.
        """
        import mujoco

        from unigym.director import _fail

        if name in self._free:
            if pos is None and quat is None:
                raise _fail(f"teleport {name!r} (rigid) needs pos and/or "
                            f"quat")
            _, qadr, vadr = self._free[name]
            m, d = self.env.mj_model, self.env.mj_data
            if pos is not None:
                d.qpos[qadr:qadr + 3] = pos
            if quat is not None:
                d.qpos[qadr + 3:qadr + 7] = quat
            if reset_vel:
                d.qvel[vadr:vadr + 6] = 0.0
            mujoco.mj_forward(m, d)
            return {"teleported": name, "pos": pos, "quat": quat}
        art = self._arts.get(name)
        if art is not None:
            unknown = sorted(set(joints) - set(art.joints))
            if unknown:
                raise _fail(f"prop {name!r}: unknown joint(s) {unknown}",
                            f"declared joints: {sorted(art.joints)}")
            m, d = self.env.mj_model, self.env.mj_data
            # anchored root: model-level re-anchor (on_reset pattern);
            # pose keys optional (joints-only writes keep the anchor)
            if pos is not None:
                m.body_pos[art.root_bid] = pos
            if quat is not None:
                m.body_quat[art.root_bid] = quat
            for jname, jn in art.joints.items():
                if jname in joints:
                    d.qpos[jn["qadr"]] = float(joints[jname])
                if reset_vel:
                    d.qvel[jn["dadr"]] = 0.0
            mujoco.mj_forward(m, d)
            return {"teleported": name, "pos": pos, "quat": quat,
                    "joints": {k: float(v) for k, v in joints.items()}}
        raise _fail(f"unknown prop {name!r}",
                    f"declared props: {sorted(self.prop_names())}")

    def setvel(self, name, lin, ang) -> dict:
        """Write a free rigid prop's root qvel (director_contract §3.2).

        Args:
            name: prop name (articulation roots are model-anchored ->
                fail-fast with a pointer to teleport joints writes).
            lin: world linear velocity (m/s; 3-vector).
            ang: world angular velocity (rad/s; 3-vector) or ``None``.
        """
        import mujoco

        from unigym.director import _fail

        if name not in self._free:
            art = self._arts.get(name)
            if art is not None:
                raise _fail(
                    f"setvel on articulation prop {name!r}: the anchor root "
                    f"is model-level on mujoco",
                    "use teleport joints writes for per-joint motion")
            raise _fail(f"unknown prop {name!r}",
                        f"declared props: {sorted(self.prop_names())}")
        _, _, vadr = self._free[name]
        d = self.env.mj_data
        d.qvel[vadr:vadr + 3] = lin
        if ang is not None:
            d.qvel[vadr + 3:vadr + 6] = ang
        mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
        return {"setvel": name, "lin": list(lin),
                "ang": list(ang) if ang else None}

    def attach_weld(self, obj: str, host: WeldHost) -> dict:
        """Activate a pre-allocated weld equality slot (contract §3.4).

        The weld freezes the CURRENT relative pose: relpose of the host
        in the object frame is captured at attach time.  Requires
        ``--direct`` (slots are spec-injected, fail-fast otherwise).
        """
        import mujoco

        host_bid = _weld_bid(host)
        m, d = self.env.mj_model, self.env.mj_data
        eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY,
                               f"director_weld_{obj}")
        if eq < 0:
            from unigym.director import _fail
            raise _fail(
                f"no weld slot for prop {obj!r} (spec injection missing)",
                "start with --direct so director weld slots are allocated")
        bid, _, _ = self._free[obj]
        if bool(d.eq_active[eq]):
            d.eq_active[eq] = 0
        # relpose of the host in the object frame, captured NOW (contract
        # §3.4: the weld freezes the current relative pose)
        r1 = np.zeros(9)
        mujoco.mju_quat2Mat(r1, d.xquat[bid])
        r1 = r1.reshape(3, 3)
        relpos = d.xpos[host_bid] - d.xpos[bid]
        relq = np.zeros(4)
        mujoco.mju_mat2Quat(relq, (r1.T @ self._host_rotmat(host_bid))
                            .reshape(9))
        data = np.zeros(11)
        data[0:3] = 0.0                    # anchor at the object origin
        data[3:6] = relpos
        data[6:10] = relq
        data[10] = 1.0                     # torquescale
        m.eq_data[eq] = data
        m.eq_obj2id[eq] = host_bid
        d.eq_active[eq] = 1
        return {"host": self._host_name(host_bid), "mode": "weld", "eq": eq}

    def detach(self, att: dict) -> None:
        """Deactivate the weld equality (kin attachments carry no state)."""
        if att.get("mode") == "weld":
            self.env.mj_data.eq_active[att["eq"]] = 0

    def host_pose(self, host: str) -> tuple[np.ndarray, np.ndarray]:
        """Read a host's world pose (m / wxyz quat; numpy copies)."""
        bid = _weld_bid(self.resolve_weld_host(host))
        d = self.env.mj_data
        return (np.array(d.xpos[bid]), np.array(d.xquat[bid]))

    def write_slave_pose(self, obj: str, pos, quat) -> None:
        """Per-tick kinematic follow write (director_contract §3.8).

        Args:
            obj: free rigid prop name (qpos addresses from the bind
                table; unknown names fail fast via ``_free_prop``).
            pos: world position (m; ``(x, y, z)``).
            quat: world orientation (wxyz).
        """
        _, qadr, _ = self._free[obj]
        self.env.mj_data.qpos[qadr:qadr + 3] = pos
        self.env.mj_data.qpos[qadr + 3:qadr + 7] = quat

    def obj_pose(self, obj: str) -> tuple[np.ndarray, np.ndarray]:
        """Read a free rigid prop's qpos pose (m / wxyz quat)."""
        _, qadr, _ = self._free_prop(obj)
        d = self.env.mj_data
        return (np.array(d.qpos[qadr:qadr + 3]),
                np.array(d.qpos[qadr + 3:qadr + 7]))

    def halt(self, obj: str) -> None:
        """Zero a free rigid prop's qvel (director_contract §3.7 pin)."""
        _, _, vadr = self._free_prop(obj)
        self.env.mj_data.qvel[vadr:vadr + 6] = 0.0

    def push_root(self, lin, ang, mode: str) -> dict:
        """Robot-base freejoint qvel write, world axes (set | add).

        A
        None component keeps its current value (the teleport
        default-dimension convention: unset stays).
        """
        import mujoco

        from unigym.director import _fail

        m, d = self.env.mj_model, self.env.mj_data
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if bid < 0:
            raise _fail("no robot base body 'pelvis' in the MJ model",
                        "push needs a free-root robot model")
        jadr = int(m.body_jntadr[bid])
        if (jadr < 0
                or int(m.jnt_type[jadr]) != int(mujoco.mjtJoint.mjJNT_FREE)):
            raise _fail("robot base 'pelvis' has no freejoint",
                        "push writes the root freejoint qvel")
        vadr = int(m.jnt_dofadr[jadr])
        if mode == "set":
            if lin is not None:
                d.qvel[vadr:vadr + 3] = lin
            if ang is not None:
                d.qvel[vadr + 3:vadr + 6] = ang
        else:                                  # add (push = impulse)
            if lin is not None:
                d.qvel[vadr:vadr + 3] += lin
            if ang is not None:
                d.qvel[vadr + 3:vadr + 6] += ang
        mujoco.mj_forward(m, d)
        return {"pushed": "robot", "lin": lin, "ang": ang, "mode": mode}

    def write_joints(self, joints: dict[str, float],
                     zero_vel: bool) -> dict:
        """Robot-joint qpos write by name (director_contract §3.11).

        Pose
        teleport -- PAUSED = final pose; RUNNING = the position
        controller pulls toward the action targets.  Robot joints =
        the policy controller's ``asset_joint_names`` minus the injected
        ``Prop_*`` freejoints (the SAME table the scene face serves).  Range-gated against
        the model limits (out-of-range fails fast).
        """
        import mujoco

        from ..model.joint_limits import model_joint_limits
        from unigym.director import _fail

        from unigym.runtime.controllers import single_articulation_runtime

        have = [j for j in
                single_articulation_runtime(self.env).joint_names
                if not j.startswith("Prop_")]
        unknown = sorted(set(joints) - set(have))
        if unknown:
            raise _fail(f"unknown robot joint(s) {unknown}",
                        f"robot joints: {have}")
        m, d = self.env.mj_model, self.env.mj_data
        for name, val in joints.items():
            jid = m.joint(name).id
            # range gate (contract §3.11): the scene-face model
            # limits -- limited joints reject out-of-range writes with
            # the range hint; unlimited joints keep no gate.
            rng = model_joint_limits(m, [name])[name]
            if rng is not None and not rng[0] <= float(val) <= rng[1]:
                raise _fail(
                    f"joint {name!r} value {float(val)} outside model "
                    f"limits [{rng[0]}, {rng[1]}]",
                    f"valid range for {name!r}: [{rng[0]}, {rng[1]}]")
            d.qpos[int(m.jnt_qposadr[jid])] = float(val)
            if zero_vel:
                d.qvel[int(m.jnt_dofadr[jid])] = 0.0
        mujoco.mj_forward(m, d)
        return {"joints": {k: float(v) for k, v in joints.items()},
                "zero_vel": bool(zero_vel)}

    def default_attach_mode(self) -> str:
        """Mujoco attach default: ``"kin"`` (weld is the explicit slot form)."""
        return "kin"

    def after_gripper_move(self) -> None:
        """Post-move hook: refresh derived quantities after gripper writes."""
        import mujoco

        mujoco.mj_forward(self.env.mj_model, self.env.mj_data)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _free_prop(self, obj: str) -> tuple[int, int, int]:
        """Free-rigid prop addresses -- the pin/follow feasibility gate.

        Articulation roots are model-anchored -- the honest fail-fast.
        """
        from unigym.director import _fail

        free = self._free.get(obj)
        if free is None:
            if obj in self._arts:
                raise _fail(f"prop {obj!r} is an articulation: its root is "
                            f"model-anchored -- pin/follow applies to "
                            f"rigid props",
                            "articulation props: use teleport joints "
                            "writes; rigid props: pin/follow")
            raise _fail(f"unknown prop {obj!r}",
                        f"declared props: {sorted(self.prop_names())}")
        return free

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _host_rotmat(self, bid: int) -> np.ndarray:
        import mujoco

        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, self.env.mj_data.xquat[bid])
        return r.reshape(3, 3)

    def _host_name(self, bid: int) -> str:
        import mujoco

        return mujoco.mj_id2name(self.env.mj_model,
                                 mujoco.mjtObj.mjOBJ_BODY, bid)
