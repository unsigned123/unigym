"""Unitree G1 (29-DoF) robot profile with SONIC deployment parameters.

Asset: ``G1_29DOF_CFG`` (the 29-DoF ``g1.usd`` under ISAAC_NUCLEUS_DIR,
isaaclab_assets/robots/unitree.py:388-536) — NOT ``G1_CFG`` /
``G1_MINIMAL_CFG`` (those carry the old 37-DoF torso/elbow_roll/finger-joint
asset whose joint set cannot match the SONIC policy).

Patches applied via ``.replace()`` (the asset ships contact sensors OFF,
a hardcoded ``/World/envs/env_.*/Robot`` prim path and a yaw+90° spawn):
  * contact sensors ON            — foot_contact / duty data source
  * prim_path per framework conv  — ``{ENV_REGEX_NS}/Robot``
  * upright spawn (identity yaw)  — SONIC anchors assume yaw 0
  * SONIC standing q0             — spawn pose AND the controller q0 (obs
    ``joint_pos_rel = q - q0`` == deployment ``body_q`` semantics)
  * implicit-PD actuators in 5 groups with per-regex effort limits from
    the deployment motor specs (policy_parameters.hpp); the 5th group
    ("hands") merely holds the 14 non-policy finger DOFs of the actual
    ``g1_29dof_with_hand_rev_1_0`` sub-asset at their rest pose — those
    DOFs are excluded from obs/action by IsaaclabEnv's name permutation

isaaclab decoupling (M2a): all isaaclab imports live INSIDE
:func:`make_g1_profile` behind the ``with_isaaclab`` flag, so the mujoco
backend can build the profile (pure constants) without isaaclab installed.
With ``with_isaaclab=False`` (default) ``articulation_cfg`` is ``None`` and
every other field is identical.

Pure-data constants live in :mod:`unigym.robots.g1_constants`
(isaaclab-free; the single source of truth for tests + contract doc).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

from ..core import quat
from .base import RobotRecipe
from .g1_constants import (
    G1_BODY_PARTS,
    G1_DEFAULT_Q,
    G1_EFFORT_ARMS,
    G1_EFFORT_FEET,
    G1_EFFORT_HANDS,
    G1_EFFORT_LEGS,
    G1_EFFORT_WAIST,
    G1_HAND_EXPRS,
    G1_HAND_JOINTS,
    G1_JOINT_ORDER_IL,
    G1_KD_ARMS,
    G1_KD_FEET,
    G1_KD_HANDS,
    G1_KD_LEGS,
    G1_KD_WAIST,
    G1_KP_ARMS,
    G1_KP_FEET,
    G1_KP_HANDS,
    G1_KP_LEGS,
    G1_KP_WAIST,
    G1_SPAWN_HEIGHT,
)

# NOTE: two Nucleus asset generations exist for G1 (both S3-listed
# 2026-09-02): Isaac 4.5 ships ``G1/g1.usd`` as a single self-contained
# 29-DoF file (22,155,396 bytes) while Isaac 5.0 ships a 6.9 KB stub plus a
# ``configuration/`` folder whose sub-assets are the 43-DoF
# ``g1_29dof_with_hand_rev_1_0`` build.  The ``hands`` actuator group below
# is therefore OPT-IN (``actuate_hands=True``): a hands regex against the
# 29-DoF asset would match nothing, and unactuated DOFs on the 43-DoF
# variant would dangle.  With ``actuate_hands=True`` the 14 finger DOFs are
# held at their USD rest pose (0) and stay EXCLUDED from obs/action via
# IsaaclabEnv's name-based permutation (extra asset DOFs are allowed).


def _override_gains(base: Dict[str, float],
                    override: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Merge a yaml gain override into one actuator group's dict.

    Only regex keys that already exist in the group dict are applied (per
    group); unknown keys are the escape hatch's problem — the yaml ``actuator``
    section replaces whole groups (collect.py build_actuator).
    """
    out = dict(base)
    if override:
        out.update({k: v for k, v in override.items() if k in out})
    return out


def _spawn_rot_wxyz(yaw_deg: float) -> tuple:
    """Spawn quaternion (w, x, y, z) for a yaw of *yaw_deg* degrees.

    (DELEGATED to unigym.core.quat.about_z, M1 convergence; the il3
    boundary conversion happens at the cfg-construction site.)
    """
    yaw = math.radians(float(yaw_deg))
    return tuple(float(v) for v in quat.about_z(yaw))


def make_g1_profile(spawn_height: float = G1_SPAWN_HEIGHT,
                    stiffness: Optional[Dict] = None,
                    damping: Optional[Dict] = None,
                    effort: Optional[Dict] = None,
                    hands_kp: Optional[float] = None,
                    hands_kd: Optional[float] = None,
                    prim_path: str = "{ENV_REGEX_NS}/Robot",
                    usd_path: Optional[str] = None,
                    actuate_hands: bool = False,
                    body_parts: Optional[Dict] = None,
                    default_q: Optional[Dict] = None,
                    spawn_x: float = 0.0,
                    spawn_y: float = 0.0,
                    spawn_yaw_deg: float = 0.0,
                    with_isaaclab: bool = False) -> RobotRecipe:
    """Build the G1 29-DoF RobotRecipe.

    Args:
        spawn_height: spawn base z (m).  SONIC standing reference ~0.74;
            default 0.78 gives a small settle drop.
        stiffness/damping/effort: optional per-regex overrides merged into
            the playground-PD defaults (keys must match existing group
            regexes).  ``effort`` + hands gains were added so a foreign
            task's actuator table (e.g. HumanoidArena) is expressible in
            config without touching the constants.
        hands_kp/hands_kd: the (non-policy) finger group gains; defaults
            keep the upstream G1_29DOF_CFG values.
        prim_path: framework-standard clone regex (asset hardcodes another).
        usd_path: local USD override (P3-A4 Nucleus localization); ``None``
            keeps the asset's official Nucleus path.
        actuate_hands: include the "hands" actuator group.  Set ``True``
            ONLY for the 43-DoF ``g1_29dof_with_hand_rev_1_0`` asset
            (Isaac 5.0 ``configuration/`` build / ``G1_with_hand``); the
            default 29-DoF Isaac 4.5 ``g1.usd`` has no finger DOFs.
        body_parts: optional partial override of :data:`G1_BODY_PARTS`.
        default_q: optional per-joint override of the SONIC standby q0
            (same channel as the built-in profiles; lets a non-SONIC policy spawn at its
            own q0 — drives spawn pose AND the controller ``q0``).
        spawn_x/spawn_y: initial root world xy (m), applied on
            ``init_state.pos`` together with ``spawn_height`` (isaaclab
            spawn-pose parity of the mujoco ``spawn_xy`` planar spawn:
            the pose is owned by the Isaac Lab init system, not written
            per-reset).  Defaults 0/0 keep the historical identity spawn.
        spawn_yaw_deg: initial root yaw in degrees (world frame, applied as
            ``(cos, 0, 0, sin)`` wxyz on ``init_state.rot``).  Default 0 keeps
            the historical identity spawn.  Foreign scenes that place the goal
            on the +y axis (e.g. HumanoidArena open_door, whose upstream
            ``ROBOT_INIT_ROT`` is ``(0.7071, 0, 0, 0.7071)`` = yaw +90°) set
            90.0 — the HA G1 USD is authored facing +x (verified 2026-09-26:
            identity spawn measures heading ≈ 0°), so the old "asset ships
            yaw+90°" comment was a SONIC-asset assumption and is removed.
        with_isaaclab: build ``articulation_cfg`` (requires isaaclab +
            isaaclab_assets importable, i.e. after AppLauncher).  Default
            ``False``: ``articulation_cfg`` is ``None`` and NO isaaclab
            import happens (mujoco-backend path); all other fields are
            identical.
    """
    bp = dict(G1_BODY_PARTS)
    if body_parts:
        bp.update(body_parts)
    # default_joint_pos covers the FULL 43-joint articulation —
    # the 29 SONIC policy joints + the 14 finger joints at their rest
    # pose (0).  Drives spawn pose, the controller q0 (policy joints)
    # and the full-articulation obs q0 (fingers: relative == absolute).
    joint_pos = dict(G1_DEFAULT_Q)
    joint_pos.update({n: 0.0 for n in G1_HAND_JOINTS})
    if default_q:
        # regex keys (IsaacLab patterns like ".*_hip_pitch_joint") are
        # EXPANDED to literal joint names before the merge: IsaacLab's
        # init_state matches patterns against the asset, and a dict holding
        # both the literal and a pattern matching the same joint raises a
        # multiple-matches error
        import re
        from .g1_constants import G1_JOINT_ORDER_FULL as _ALL_JOINTS
        expanded: Dict[str, float] = {}
        for key, val in default_q.items():
            if re.search(r"[.*]", key):
                pat = re.compile(key)
                hits = [n for n in _ALL_JOINTS if pat.fullmatch(n)]
                if not hits:
                    raise ValueError(
                        f"default_q pattern {key!r} matches no G1 joint")
                for n in hits:
                    expanded[n] = float(val)
            else:
                expanded[key] = float(val)
        joint_pos.update(expanded)

    cfg = None
    if with_isaaclab:
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab_assets.robots.unitree import G1_29DOF_CFG

        spawn_kwargs: dict[str, Any] = {
            "activate_contact_sensors": True}  # asset ships OFF
        if usd_path:
            spawn_kwargs["usd_path"] = usd_path
        cfg = G1_29DOF_CFG.replace(
            prim_path=prim_path,
            spawn=G1_29DOF_CFG.spawn.replace(**spawn_kwargs),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(spawn_x, spawn_y, spawn_height),
                rot=_spawn_rot_wxyz(spawn_yaw_deg),
                joint_pos=joint_pos,
                joint_vel={".*": 0.0},
            ),
            actuators={
                "legs": ImplicitActuatorCfg(
                    joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint",
                                      ".*_hip_yaw_joint", ".*_knee_joint"],
                    stiffness=_override_gains(G1_KP_LEGS, stiffness),
                    damping=_override_gains(G1_KD_LEGS, damping),
                    effort_limit_sim=_override_gains(G1_EFFORT_LEGS, effort)),
                "feet": ImplicitActuatorCfg(
                    joint_names_expr=[".*_ankle_pitch_joint",
                                      ".*_ankle_roll_joint"],
                    stiffness=_override_gains(G1_KP_FEET, stiffness),
                    damping=_override_gains(G1_KD_FEET, damping),
                    effort_limit_sim=_override_gains(G1_EFFORT_FEET, effort)),
                "waist": ImplicitActuatorCfg(
                    joint_names_expr=["waist_yaw_joint", "waist_roll_joint",
                                      "waist_pitch_joint"],
                    stiffness=_override_gains(G1_KP_WAIST, stiffness),
                    damping=_override_gains(G1_KD_WAIST, damping),
                    effort_limit_sim=_override_gains(G1_EFFORT_WAIST, effort)),
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=[".*_shoulder_pitch_joint",
                                      ".*_shoulder_roll_joint",
                                      ".*_shoulder_yaw_joint",
                                      ".*_elbow_joint",
                                      ".*_wrist_.*_joint"],
                    stiffness=_override_gains(G1_KP_ARMS, stiffness),
                    damping=_override_gains(G1_KD_ARMS, damping),
                    effort_limit_sim=_override_gains(G1_EFFORT_ARMS, effort)),
                **({"hands": ImplicitActuatorCfg(  # non-policy DOFs, held at rest
                    joint_names_expr=list(G1_HAND_EXPRS),
                    stiffness=G1_KP_HANDS if hands_kp is None else hands_kp,
                    damping=G1_KD_HANDS if hands_kd is None else hands_kd,
                    effort_limit_sim=G1_EFFORT_HANDS)} if actuate_hands else {}),
            },
        )
    return RobotRecipe(
        articulation_cfg=cfg,
        # v105: the named joint vocabulary.  ``hand`` is always declared (the
        # mujoco 43-DoF asset carries the fingers even when the isaaclab
        # actuator gate is off); the isaaclab *action term* for it stays
        # gated by actuate_hands below.
        joint_sets={"body": tuple(G1_JOINT_ORDER_IL),
                    "hand": tuple(G1_HAND_JOINTS)},
        default_joint_pos=joint_pos,
        foot_bodies=list(bp["feet"]),
        body_parts=bp,
        # with the hands actuator group, the finger DOFs become a
        # driveable NamedAction face ("hand", dex3 open/close targets) --
        # term input order == G1_HAND_JOINTS (the hand-face order external
        # drivers emit).  Without actuate_hands the 29-DoF single-face
        # surface is unchanged.  This is an isaaclab action-term capability
        # gate, not a joint-resolution source (v105).
        _extra_action_terms=({"hand": list(G1_HAND_JOINTS)}
                            if actuate_hands else None),
    )
