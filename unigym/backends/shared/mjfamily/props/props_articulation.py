"""mujoco articulation props: anchored build + runtime view .

The mujoco half of :mod:`unigym.props.articulation` (this package's
mujoco-facing modules are lazily imported by the env backend only).
Split from ``mujoco_backend.py`` by the file-size discipline -- together
the two modules form the props mujoco backend.

BUILD (pre-compile, ``add_articulation_body``): the asset (URDF or MJCF
fragment, joints defined inside) is loaded as a child MjSpec and ATTACHED
into the main spec with the ``Prop_<name>_`` prefix at a worldbody frame.
Anchored semantics fail fast: exactly one top-level body and NO freejoint
anywhere in the subtree.  Declared joints get their config damping (and
optional range) pre-attach; actuated joints additionally get a position
servo actuator on the main spec post-attach (gainprm[0]=stiffness,
biasprm=[0, stiffness, -damping] -- ``ctrl`` IS the position target,
force = kp*(ctrl - q) - kd*dq, ctrlrange from the declared or asset
range).  Internal joint/actuator names carry the ``Prop_<name>_`` prefix,
so they never collide with robot joints and are excluded from the env's
hand-DOF table by the existing ``startswith("Prop_")`` rule.

RUNTIME (``MujocoPropArticulation``, N=1): re-anchors the fixed root body
at the sampled placement (post-compile ``body_pos``/``body_quat`` mutation
+ ``mj_forward``), zeroes the subtree joints, exposes the anchor pose for
the ``object/<name>/{pos,quat}`` record layer and per-joint ``qpos`` for
the ``object/<name>/joints/<jname>`` keys.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only (mujoco imported lazily)
    import mujoco

import numpy as np
import torch

from ..scene.mjcf_io import load_spec


# --------------------------------------------------------------------------- #
# build side (pre-compile)
# --------------------------------------------------------------------------- #
def _iter_bodies(body):
    """Yield an MjsBody subtree depth-first."""
    yield body
    for child in body.bodies:
        yield from _iter_bodies(child)


def add_articulation_body(mjspec, handler) -> None:
    """Attach the anchored articulation asset into the main MjSpec.

    *handler* is the wired :class:`props.articulation.PropArticulation`
    (its ``.joints`` table is the VALIDATED declaration).
    """
    import mujoco

    spec = handler.spec
    declared = handler.joints
    if getattr(handler, "latch_enabled", False):
        # the runtime drive-switch latch is isaaclab-only (mujoco
        # would need actuator force-range surgery); fail-fast instead of
        # silently ignoring the declaration
        raise ValueError(
            f"articulation prop {spec.name!r}: latch declaration is "
            f"isaaclab-only ; remove spec.latch on the mujoco "
            f"backend")
    child = load_spec(spec.asset)
    tops = list(child.worldbody.bodies)
    if len(tops) != 1:
        raise ValueError(
            f"articulation prop {spec.name!r}: asset must declare exactly "
            f"one top-level body (found: {[b.name for b in tops]})")
    subtree = list(_iter_bodies(tops[0]))
    for b in subtree:
        for j in b.joints:
            if int(j.type) == int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(
                    f"articulation prop {spec.name!r}: asset joint "
                    f"{j.name!r} is a freejoint -- anchored semantics "
                    f"forbid free joints in the fragment")
    asset_joints: dict[str, mujoco.MjsJoint] = {
        j.name: j for b in subtree for j in b.joints}
    missing = [n for n in declared if n not in asset_joints]
    if missing:
        raise ValueError(
            f"articulation prop {spec.name!r}: declared joints {missing} "
            f"not in the asset joints {sorted(asset_joints)}")
    undeclared = [n for n in asset_joints if n not in declared]
    if undeclared:
        raise ValueError(
            f"articulation prop {spec.name!r}: asset joints {undeclared} "
            f"are not declared in spec.joints (every asset joint must be "
            f"declared)")
    # pre-attach: config damping / optional declared range onto the joints
    # (a declared range implies limited under the default autolimits rule;
    # asset-declared limits pass through untouched)
    ranges: dict[str, list] = {}
    for jname, jobj in asset_joints.items():
        jobj.damping = np.asarray(
            [declared[jname]["damping"], 0.0, 0.0], dtype=np.float64)
        if declared[jname]["range"] is not None:
            lo, hi = declared[jname]["range"]
            jobj.range = np.asarray([lo, hi], dtype=np.float64)
        ranges[jname] = [float(v) for v in np.asarray(jobj.range)]
    # the asset-side joint ranges, for the bind-time init re-check
    # (declaration-level validation only covers a declared range)
    handler._asset_ranges = ranges
    frame = mjspec.worldbody.add_frame(name=f"Prop_{spec.name}_attach")
    mjspec.attach(child, prefix=f"Prop_{spec.name}_", suffix="", frame=frame)
    # post-attach: position servo per actuated joint (ctrl = position target)
    for jname, p in declared.items():
        if p["mode"] != "actuated":
            continue
        kp, kd = p["stiffness"], p["damping"]
        lo, hi = p["range"] if p["range"] is not None else ranges[jname]
        limited = p["range"] is not None or lo != hi or lo != 0.0
        mjspec.add_actuator(
            name=f"Prop_{spec.name}_{jname}_act",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=f"Prop_{spec.name}_{jname}",
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=np.array([kp] + [0.0] * 9),
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            # position servo: force = kp*ctrl - kp*q - kd*dq  (biasprm[1]
            # carries the NEGATED stiffness -- the mjcf <position> form)
            biasprm=np.array([0.0, -kp, -kd] + [0.0] * 7),
            ctrllimited=limited, ctrlrange=[lo, hi])


def add_free_articulation_body(mjspec, handler) -> None:
    """Attach the FREE-FLOATING articulation asset into the main MjSpec.

    T-B3 GAP-A (the laptop family): same attach face as
    :func:`add_articulation_body` (``Prop_<name>_`` prefix at a
    worldbody frame) with the polarity of the freejoint gate REVERSED --
    the ROOT body must carry EXACTLY ONE freejoint and no other joint
    (that freejoint is the prop's live dof; the declared table covers
    the INTERNAL joints only, and the freejoint name is not declarable).
    Declared-narrowing v1: passive-only (an ``actuated`` internal joint
    fails fast -- the free-root servo/action face is unexercised) and
    no latch (same isaaclab-only rejection as the anchored build).
    """
    import mujoco

    spec = handler.spec
    declared = handler.joints
    if getattr(handler, "latch_enabled", False):
        raise ValueError(
            f"free_articulation prop {spec.name!r}: latch declaration is "
            f"isaaclab-only ; remove spec.latch on the mujoco backend")
    child = load_spec(spec.asset)
    tops = list(child.worldbody.bodies)
    if len(tops) != 1:
        raise ValueError(
            f"free_articulation prop {spec.name!r}: asset must declare "
            f"exactly one top-level body (found: {[b.name for b in tops]})")
    root = tops[0]
    root_joints = list(root.joints)
    free = [j for j in root_joints
            if int(j.type) == int(mujoco.mjtJoint.mjJNT_FREE)]
    if len(free) != 1 or len(root_joints) != 1:
        raise ValueError(
            f"free_articulation prop {spec.name!r}: the asset root body "
            f"{root.name!r} must carry exactly one freejoint and no other "
            f"joint (found: {[(j.name, int(j.type)) for j in root_joints]})")
    if free[0].name in declared:
        raise ValueError(
            f"free_articulation prop {spec.name!r}: the freejoint "
            f"{free[0].name!r} is the prop's live dof -- it is not a "
            f"declarable joint (declare the INTERNAL joints only)")
    subtree = list(_iter_bodies(root))
    asset_joints: dict[str, mujoco.MjsJoint] = {
        j.name: j for b in subtree
        for j in b.joints
        if int(j.type) != int(mujoco.mjtJoint.mjJNT_FREE)}
    missing = [n for n in declared if n not in asset_joints]
    if missing:
        raise ValueError(
            f"free_articulation prop {spec.name!r}: declared joints "
            f"{missing} not in the asset joints {sorted(asset_joints)}")
    undeclared = [n for n in asset_joints if n not in declared]
    if undeclared:
        raise ValueError(
            f"free_articulation prop {spec.name!r}: asset joints "
            f"{undeclared} are not declared in spec.joints (every internal "
            f"asset joint must be declared)")
    actuated = [n for n, p in declared.items() if p["mode"] != "passive"]
    if actuated:
        raise ValueError(
            f"free_articulation prop {spec.name!r}: passive-only in v1 "
            f"(got actuated {actuated} -- the free-root servo/action face "
            f"is unexercised scope)")
    # pre-attach: config damping / optional declared range onto the
    # internal joints (the freejoint passes through untouched -- the
    # asset's free damping is part of the dump-verbatim contract)
    ranges: dict[str, list] = {}
    for jname, jobj in asset_joints.items():
        jobj.damping = np.asarray(
            [declared[jname]["damping"], 0.0, 0.0], dtype=np.float64)
        if declared[jname]["range"] is not None:
            lo, hi = declared[jname]["range"]
            jobj.range = np.asarray([lo, hi], dtype=np.float64)
        ranges[jname] = [float(v) for v in np.asarray(jobj.range)]
    handler._asset_ranges = ranges
    frame = mjspec.worldbody.add_frame(name=f"Prop_{spec.name}_attach")
    mjspec.attach(child, prefix=f"Prop_{spec.name}_", suffix="", frame=frame)


# --------------------------------------------------------------------------- #
# runtime side (N=1)
# --------------------------------------------------------------------------- #
class MujocoPropArticulation:
    """Runtime view of one anchored articulation prop (mujoco, N=1)."""

    def __init__(self, handler):
        self.spec = handler.spec
        self._placement = handler.placement
        # the VALIDATED declaration (test fakes may omit it)
        self._joints_decl = getattr(handler, "joints", None) or {}
        self.name = self.spec.name
        self.root_bid = -1
        self._root_name = ""
        self.joints: dict[str, dict] = {}
        self._inits: dict[str, float] = {}
        self._asset_ranges: dict[str, list] = getattr(
            handler, "_asset_ranges", {})
        self._env = None

    # ------------------------------------------------------------------ #
    def bind_env(self, env) -> None:
        """Resolve the root body + per-joint qpos/dof/ctrl addresses."""
        from ..model.ir_access import model_ir_of

        ir = model_ir_of(env)
        self._env = env
        prefix = f"Prop_{self.name}_"
        cands = [b for b in ir.bodies if b.name.startswith(prefix)]
        # the subtree root is the candidate whose parent lies OUTSIDE the
        # prefixed set (world or an unrelated body)
        roots = [b for b in cands
                 if not (b.parent and b.parent.startswith(prefix))]
        if len(roots) != 1:
            raise ValueError(
                f"articulation prop {self.name!r}: expected exactly one "
                f"root body under prefix {prefix!r}, found "
                f"{[b.body_id for b in roots]}")
        self.root_bid = int(roots[0].body_id)
        self._root_name = roots[0].name
        # compiled IR rows are emitted in native model order, so the row
        # index IS the native joint id (Joint carries addresses, not an id)
        joint_ids = {j.name: i for i, j in enumerate(ir.joints)}
        self._inits: dict[str, float] = {}
        for jname in (self.spec.joints or {}):
            internal = f"{prefix}{jname}"
            if internal not in ir.joints:
                raise ValueError(
                    f"articulation prop {self.name!r}: joint {jname!r} "
                    f"({internal!r}) not in the MJ model")
            jid = joint_ids[internal]
            # the VALIDATED table when the handler carries one; the raw
            # spec declaration otherwise (test stubs bind pre-validation)
            decl = (self._joints_decl.get(jname)
                    or (self.spec.joints or {}).get(jname) or {})
            init = float(decl.get("init", 0.0))
            if decl.get("range") is None \
                    and jname in self._asset_ranges:
                # no declared range: re-check the init against the ASSET
                # range the build recorded (the declaration-level check
                # only covers an explicitly declared range)
                lo, hi = self._asset_ranges[jname]
                if not (lo <= init <= hi):
                    raise ValueError(
                        f"articulation prop {self.name!r}: joint {jname!r} "
                        f"init {init!r} outside the asset range "
                        f"[{lo!r}, {hi!r}]")
            self._inits[jname] = init
            act_name = f"{internal}_act"
            act_row = ir.actuators[act_name] \
                if act_name in ir.actuators else None
            adr = -1 if act_row is None or act_row.actuator_id is None \
                else int(act_row.actuator_id)
            joint = ir.joints[internal]
            qpos_adr = joint.qpos_adr
            dof_adr = joint.dof_adr
            if qpos_adr is None or dof_adr is None:
                raise ValueError(
                    f"articulation prop {self.name!r}: joint {internal!r} "
                    f"is not compiled (missing qpos/dof address)")
            self.joints[jname] = {
                "jid": jid,
                "qadr": int(qpos_adr),
                "dadr": int(dof_adr),
                "act_adr": adr,   # -1 for passive joints
            }

    def actuated_addresses(self) -> dict[str, int]:
        """``{jname: ctrl address}`` for the actuated joints (resolver)."""
        return {n: j["act_adr"] for n, j in self.joints.items()
                if j["act_adr"] >= 0}

    # ------------------------------------------------------------------ #
    def on_reset(self, m, d, env_ids=None) -> None:
        """Re-anchor at the sampled placement + write the declared state.

        Per-joint reset position = the declared ``init`` (T-B1, default
        0.0 -- the robocasa fixture semantics write their open fraction
        straight into qpos; a fixture must be able to START open and
        stay freely movable, which the passive declaration guarantees:
        no actuator, hence no soft-PD, ever pulls it back).  Velocity is
        always zeroed.
        """
        import mujoco

        x, y, z, yaw = self._placement.sample(1, "cpu")[0].tolist()
        m.body_pos[self.root_bid] = [x, y, z]
        m.body_quat[self.root_bid] = [np.cos(yaw / 2), 0.0, 0.0,
                                      np.sin(yaw / 2)]
        for jname, j in self.joints.items():
            d.qpos[j["qadr"]] = self._inits[jname]
            d.qvel[j["dadr"]] = 0.0
        mujoco.mj_forward(m, d)

    # ------------------------------------------------------------------ #
    def _require_env(self):
        """The bound env (fail-fast when the runtime is unbound/closed)."""
        if self._env is None:
            raise RuntimeError(
                f"articulation prop {self.name!r} is not bound to an env")
        return self._env

    def get_world_pose(self, e: int = 0):
        """``(pos (1,3) f32, quat (1,4) wxyz f32)`` anchor pose (torch)."""
        env = self._require_env()
        d = env.mj_datas[e]
        pos = np.asarray(d.xpos[self.root_bid], dtype=np.float32)
        quat = np.asarray(d.xquat[self.root_bid], dtype=np.float32)
        return (torch.from_numpy(pos).unsqueeze(0),
                torch.from_numpy(quat).unsqueeze(0))

    def get_joint_states(self, e: int = 0) -> dict[str, torch.Tensor]:
        """``{jname: (1,1) f32}`` joint positions (record/state face)."""
        env = self._require_env()
        d = env.mj_datas[e]
        return {n: torch.tensor([[float(d.qpos[j["qadr"]])]],
                                dtype=torch.float32)
                for n, j in self.joints.items()}

    def close(self) -> None:
        """Release the joint table (PropArticulation lifecycle)."""
        self.joints.clear()
        self._env = None


class MujocoPropFreeArticulation(MujocoPropArticulation):
    """Runtime view of one FREE-FLOATING articulation prop (mujoco, N=1).

    Same read faces as the anchored view (root pose + per-joint qpos
    record keys); the reset face differs structurally: the root is a
    LIVE freejoint, so on_reset writes the 7-dof qpos (pos + yaw
    quaternion, from the sampled placement) and zeroes the 6-dof
    velocity instead of re-anchoring body_pos/body_quat (which for a
    freejoint body only feed qpos0).  The root freejoint never enters
    the declared joint table and carries no actuator, so it is doubly
    outside the env's soft-PD set (``Prop_`` prefix + no actuator slot).
    """

    def __init__(self, handler):
        super().__init__(handler)
        self.free_qadr = -1
        """qpos address of the root freejoint (-1 pre-bind)."""
        self.free_vadr = -1
        """dof address of the root freejoint (-1 pre-bind)."""

    def bind_env(self, env) -> None:
        """Resolve the root freejoint addresses (super() does the rest)."""
        from ..model.ir_access import model_ir_of

        super().bind_env(env)
        ir = model_ir_of(env)
        joints = [j for j in ir.joints if j.body == self._root_name]
        if len(joints) != 1 or joints[0].type != "free":
            raise ValueError(
                f"free_articulation prop {self.name!r}: the attached root "
                f"body lost its freejoint (compiled model drift)")
        joint = joints[0]
        qpos_adr = joint.qpos_adr
        dof_adr = joint.dof_adr
        if qpos_adr is None or dof_adr is None:
            raise ValueError(
                f"free_articulation prop {self.name!r}: the attached root "
                f"body lost its freejoint (compiled model drift)")
        self.free_qadr = int(qpos_adr)
        self.free_vadr = int(dof_adr)

    def on_reset(self, m, d, env_ids=None) -> None:
        """Write the freejoint qpos + the declared ``init`` state.

        Velocity is always zeroed (both the freejoint's 6 dofs and the
        internal joints').
        """
        import mujoco

        x, y, z, yaw = self._placement.sample(1, "cpu")[0].tolist()
        d.qpos[self.free_qadr:self.free_qadr + 3] = [x, y, z]
        d.qpos[self.free_qadr + 3:self.free_qadr + 7] = [
            np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[self.free_vadr:self.free_vadr + 6] = 0.0
        for jname, j in self.joints.items():
            d.qpos[j["qadr"]] = self._inits[jname]
            d.qvel[j["dadr"]] = 0.0
        mujoco.mj_forward(m, d)
