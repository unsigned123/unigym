"""Unitree G1 (29-DoF) neutral robot profile with SONIC deployment parameters.

The profile carries authoring data only (joint sets, standby pose, feet, body
parts, declared assets and action faces).  The isaaclab ``ArticulationCfg``
realization -- asset choice, contact-sensor patch, prim path, upright spawn,
actuator groups and gain/effort tables -- lives in
:mod:`unigym.backends.isaaclab2.robots.g1`; the mujoco family consumes this
profile as pure constants.

Asset: ``G1_29DOF_CFG`` (the 29-DoF ``g1.usd`` under ISAAC_NUCLEUS_DIR,
isaaclab_assets/robots/unitree.py:388-536) — NOT ``G1_CFG`` /
``G1_MINIMAL_CFG`` (those carry the old 37-DoF torso/elbow_roll/finger-joint
asset whose joint set cannot match the SONIC policy).

Pure-data constants live in :mod:`unigym.robots.g1_constants`
(isaaclab-free; the single source of truth for tests + contract doc).
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from .base import RobotAsset, RobotProfile
from .g1_constants import (
    G1_BODY_PARTS,
    G1_DEFAULT_Q,
    G1_HAND_JOINTS,
    G1_JOINT_ORDER_FULL,
    G1_JOINT_ORDER_IL,
    G1_SPAWN_HEIGHT,
)

# NOTE: two Nucleus asset generations exist for G1 (both S3-listed
# 2026-09-02): Isaac 4.5 ships ``G1/g1.usd`` as a single self-contained
# 29-DoF file (22,155,396 bytes) while Isaac 5.0 ships a 6.9 KB stub plus a
# ``configuration/`` folder whose sub-assets are the 43-DoF
# ``g1_29dof_with_hand_rev_1_0`` build.  The ``hands`` actuator group in the
# isaaclab realizer is therefore OPT-IN (``actuate_hands=True``): a hands
# regex against the 29-DoF asset would match nothing, and unactuated DOFs on
# the 43-DoF variant would dangle.  With ``actuate_hands=True`` the 14
# finger DOFs are held at their USD rest pose (0) and stay EXCLUDED from
# obs/action via IsaaclabEnv's name-based permutation (extra asset DOFs are
# allowed).


def _expand_default_q(default_q: Dict[str, float]) -> Dict[str, float]:
    """Expand regex override keys to literal G1 joint names.

    IsaacLab's init_state matches patterns against the asset, and a dict
    holding both the literal and a pattern matching the same joint raises a
    multiple-matches error; the neutral profile therefore stores literals
    only.
    """
    expanded: Dict[str, float] = {}
    for key, val in default_q.items():
        if re.search(r"[.*]", key):
            pat = re.compile(key)
            hits = [n for n in G1_JOINT_ORDER_FULL if pat.fullmatch(n)]
            if not hits:
                raise ValueError(f"default_q pattern {key!r} matches no G1 joint")
            for n in hits:
                expanded[n] = float(val)
        else:
            expanded[key] = float(val)
    return expanded


def make_g1_profile(
    spawn_height: float = G1_SPAWN_HEIGHT,
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
) -> RobotProfile:
    """Build the neutral G1 29-DoF :class:`RobotProfile`.

    The gain/spawn/prim-path/asset parameters are accepted here because the
    run config declares them on the robot section; they are consumed by the
    backend realizer, never stored on the neutral profile (only ``usd_path``
    and ``actuate_hands`` change profile data: the declared asset and the
    hand action face).

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
        usd_path: local USD override (Nucleus localization); declared as the
            profile's ``usd`` asset when present, otherwise the backend's
            official asset stays in effect.
        actuate_hands: include the "hands" actuator group and the ``hand``
            action face.  Set ``True`` ONLY for the 43-DoF
            ``g1_29dof_with_hand_rev_1_0`` asset (Isaac 5.0
            ``configuration/`` build / ``G1_with_hand``); the default
            29-DoF Isaac 4.5 ``g1.usd`` has no finger DOFs.
        body_parts: optional partial override of :data:`G1_BODY_PARTS`.
        default_q: optional per-joint override of the SONIC standby q0
            (same channel as the built-in profiles; lets a non-SONIC policy
            spawn at its own q0 — drives spawn pose AND the controller q0).
        spawn_x/spawn_y: initial root world xy (m), applied on
            ``init_state.pos`` together with ``spawn_height`` (isaaclab
            spawn-pose parity of the mujoco ``spawn_xy`` planar spawn:
            the pose is owned by the Isaac Lab init system, not written
            per-reset).  Defaults 0/0 keep the historical identity spawn.
        spawn_yaw_deg: initial root yaw in degrees (world frame, applied as
            ``(cos, 0, 0, sin)`` wxyz on ``init_state.rot``).  Default 0
            keeps the historical identity spawn.  Foreign scenes that place
            the goal on the +y axis (e.g. HumanoidArena open_door, whose
            upstream ``ROBOT_INIT_ROT`` is ``(0.7071, 0, 0, 0.7071)`` =
            yaw +90°) set 90.0 — the HA G1 USD is authored facing +x
            (verified 2026-09-26: identity spawn measures heading ≈ 0°), so
            the old "asset ships yaw+90°" comment was a SONIC-asset
            assumption and is removed.
    """
    # Backend-owned construction params: accepted for config compatibility
    # and consumed by the backend realizer, never stored on the profile.
    del spawn_height, stiffness, damping, effort, hands_kp, hands_kd, spawn_x, spawn_y
    del spawn_yaw_deg
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
        joint_pos.update(_expand_default_q(default_q))
    assets = (RobotAsset(format="usd", ref=usd_path),) if usd_path else ()
    return RobotProfile(
        name="g1",
        joint_sets={"body": tuple(G1_JOINT_ORDER_IL), "hand": tuple(G1_HAND_JOINTS)},
        default_joint_pos=joint_pos,
        foot_bodies=tuple(bp["feet"]),
        body_parts=bp,
        assets=assets,
        # with the hands actuator group, the finger DOFs become a driveable
        # NamedAction face ("hand", dex3 open/close targets) -- term input
        # order == G1_HAND_JOINTS (the hand-face order external drivers
        # emit).  Without actuate_hands the 29-DoF single-face surface is
        # unchanged.
        action_faces={"hand": tuple(G1_HAND_JOINTS)} if actuate_hands else {},
    )
