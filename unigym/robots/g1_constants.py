"""Unitree G1 (29-DoF) deployment constants — pure data, NO isaaclab imports.

Single source of truth for the G1 policy-facing contract (joint order, standby
pose, canonical body facts/bindings).  Kept isaaclab-free so the unit tests
(and the interface-contract doc) can import it without Isaac Sim.

Provenance of every number:
  * SONIC release constants (the GR00T-WholeBodyControl deployment
    checkout, byte-verified 2026-09-02):
      - gear_sonic_deploy/.../policy_parameters.hpp  (q0,
        upper/lower-body IL index tables)
      - gear_sonic/scripts/play/sonic_decoder.py     (IL 29-joint order,
        DEFAULT_ANGLES)
  * IL 29 type-major joint order: memory doc
    ``unigym-G1 humanoid P3P4 design & implementation plan`` §0.4 (three-source
    cross-check: MJ_TO_IL values + policy_parameters.hpp index tables +
    sonic_decoder.py constants).  Probed against the actual USD order by
    probe_g1_profile.py (retired in git history, v063 cleanup) — the
    name-based permutation in
    IsaaclabEnv makes order differences harmless, only the name SET matters.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# SONIC policy joint order (IsaacLab type-major, 29) — §0.4, index i is the
# IL/policy index.  NOT necessarily the USD spawn order (probed separately).
# ---------------------------------------------------------------------------
G1_JOINT_ORDER_IL: List[str] = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# upper/lower-body partition in IL indices (policy_parameters.hpp:151-157
# upper_body_joint_isaaclab_order_in_isaaclab_index / lower_body_...).
G1_UPPER_BODY_IDX = (
    2,
    5,
    8,
    11,
    12,
    15,
    16,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
)  # 17 joints (waist + arms)
G1_LOWER_BODY_IDX = (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18)  # 12 joints (legs)

# Canonical body facts: semantic ids only, never backend paths.  The native
# names live in G1_BODY_BINDINGS below (one entry per canonical id).
G1_FOOT_BODIES: List[str] = ["left_ankle_roll", "right_ankle_roll"]
G1_BASE_BODY = "base"

# SONIC standing q0 ``default_angles`` (policy_parameters.hpp:269-299,
# sonic_decoder.py DEFAULT_ANGLES; MJ order mapped to IL by joint name).
# Drives both the spawn pose and the controller q0, so obs
# ``joint_pos_rel = q - q0`` matches the deployment ``body_q`` bit-for-bit.
G1_DEFAULT_Q: Dict[str, float] = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


def _g1_hand_finger_ids(side: str) -> List[str]:
    """The 7 canonical finger body ids of one G1 hand.

    The canonical suffix is the same 0-based suffix as the native links, so
    no offset is implicit: ``left_hand.index_0`` <-> ``left_hand_index_0_link``.
    """
    return [
        f"{side}_hand.{part}_{k}"
        for part, n in (("index", 2), ("middle", 2), ("thumb", 3))
        for k in range(n)
    ]


def _g1_hand_finger_native(side: str) -> List[str]:
    """The 7 MJCF finger link names of one G1 hand (native data only)."""
    return [
        f"{side}_hand_{part}_{k}_link"
        for part, n in (("index", 2), ("middle", 2), ("thumb", 3))
        for k in range(n)
    ]


#: Named canonical body sets.  ``grasp.left`` / ``grasp.right`` carry the
#: 14 canonical finger ids via the per-hand groups below; ``grasp.both`` is
#: the EITHER-hand group used by ``sensor_bodies: {contact_group: ...}``.
G1_BODY_SETS: Dict[str, List[str]] = {
    "grasp.left": _g1_hand_finger_ids("left"),
    "grasp.right": _g1_hand_finger_ids("right"),
    "grasp.both": _g1_hand_finger_ids("left") + _g1_hand_finger_ids("right"),
}


def _g1_body_bindings() -> Dict[str, Dict[str, str]]:
    """Canonical id -> per-format native name/path.

    ``mjcf`` uses the MJCF body names; ``usd`` uses the isaaclab prim
    sub-paths (the localized G1 asset keeps the finger links under the
    ``{side}_hand/`` scope).  The palm links are deliberately absent: they
    are fixed-joint-merged into the wrist bodies at PhysX build time.
    """
    bindings: Dict[str, Dict[str, str]] = {
        G1_BASE_BODY: {"mjcf": "pelvis", "usd": "pelvis"},
        "left_ankle_roll": {
            "mjcf": "left_ankle_roll_link",
            "usd": "left_ankle_roll_link",
        },
        "right_ankle_roll": {
            "mjcf": "right_ankle_roll_link",
            "usd": "right_ankle_roll_link",
        },
    }
    for side in ("left", "right"):
        for canonical, native in zip(_g1_hand_finger_ids(side), _g1_hand_finger_native(side)):
            bindings[canonical] = {"mjcf": native, "usd": f"{side}_hand/{native}"}
    return bindings


G1_BODY_BINDINGS: Dict[str, Dict[str, str]] = _g1_body_bindings()

# ---------------------------------------------------------------------------
# Hand DOFs (NOT part of the 29-joint policy contract).  The actual Nucleus
# ``g1.usd`` (IsaacSim asset tree ``Robots/Unitree/G1/``, sub-asset
# ``g1_29dof_with_hand_rev_1_0``) carries 14 finger joints
# (``{left,right}_hand_{thumb_0,thumb_1,thumb_2,middle_0,middle_1,
# index_0,index_1}_joint``) beyond the 29 body joints — probed 2026-09-02.
# They are actuated (held at USD rest pose 0, matching the Unitree URDF rest
# configuration) but excluded from obs/action by the name-based permutation.
# ---------------------------------------------------------------------------

# The 14 finger JOINT names of the with-hand asset (psi/Unitree naming:
# per hand thumb_0/1/2, middle_0/1, index_0/1, L then R).  Not part of the
# 29-joint policy contract; rest pose = 0.
G1_HAND_JOINTS: List[str] = [
    f"{side}_hand_{part}_{k}_joint"
    for side in ("left", "right")
    for part, n in (("thumb", 3), ("middle", 2), ("index", 2))
    for k in range(n)
]

# Full articulation set (43): the 29 policy joints + the finger joints.
# Offline canonical order for the obs-catalog publication ; a run
# catalog uses the backend's own resolved order instead (isaaclab asset
# order / MJ hinge order) — safe because publication is name-explicit.
G1_JOINT_ORDER_FULL: List[str] = list(G1_JOINT_ORDER_IL) + list(G1_HAND_JOINTS)

#: Curated joint groups folded into the profile vocabulary (the scene face
#: reads these; ``body``/``hand`` remain the action-face sources).
G1_JOINT_SETS: Dict[str, List[str]] = {
    "body": list(G1_JOINT_ORDER_IL),
    "hand": list(G1_HAND_JOINTS),
    "hands": list(G1_HAND_JOINTS),
    "hips": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    "legs": [j for j in G1_JOINT_ORDER_IL if "hip" in j or "knee" in j or "ankle" in j],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "arms": [j for j in G1_JOINT_ORDER_IL if "shoulder" in j or "elbow" in j or "wrist" in j],
    "upper_body": [G1_JOINT_ORDER_IL[i] for i in G1_UPPER_BODY_IDX],
    "lower_body": [G1_JOINT_ORDER_IL[i] for i in G1_LOWER_BODY_IDX],
}
