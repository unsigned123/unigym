"""Unitree G1 (29-DoF) deployment constants — pure data, NO isaaclab imports.

Single source of truth for the G1 policy-facing contract (joint order, standby
pose, body parts, effort limits, PD gains).  Kept isaaclab-free so the unit
tests (and the interface-contract doc) can import it without Isaac Sim.

Provenance of every number:
  * SONIC release constants (the GR00T-WholeBodyControl deployment
    checkout, byte-verified 2026-09-02):
      - gear_sonic_deploy/.../policy_parameters.hpp  (q0, effort limits,
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
  * PD gains: SONIC playground MuJoCo sim values
    (playground_official_sonic.py:96-104), sim-to-sim reference; the hardware
    deployment gains (policy_parameters.hpp:141-210) remain the documented
    A/B alternative via the yaml ``actuator`` override.
"""
from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# SONIC policy joint order (IsaacLab type-major, 29) — §0.4, index i is the
# IL/policy index.  NOT necessarily the USD spawn order (probed separately).
# ---------------------------------------------------------------------------
G1_JOINT_ORDER_IL: List[str] = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# upper/lower-body partition in IL indices (policy_parameters.hpp:151-157
# upper_body_joint_isaaclab_order_in_isaaclab_index / lower_body_...).
G1_UPPER_BODY_IDX = (2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26,
                     27, 28)  # 17 joints (waist + arms)
G1_LOWER_BODY_IDX = (0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18)  # 12 joints (legs)

# foot bodies == contact-sensor bodies (MJ XML + IL asset: the ankle-roll
# links are the last links of each leg)
G1_FOOT_BODIES: List[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
G1_BASE_BODY = "pelvis"

# SONIC standing q0 ``default_angles`` (policy_parameters.hpp:269-299,
# sonic_decoder.py DEFAULT_ANGLES; MJ order mapped to IL by joint name).
# Drives both the spawn pose and the controller q0, so obs
# ``joint_pos_rel = q - q0`` matches the deployment ``body_q`` bit-for-bit.
G1_DEFAULT_Q: Dict[str, float] = {
    "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669, "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6, "right_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0, "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0, "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}


def _g1_hand_finger_links(side: str) -> List[str]:
    """The 7 finger link names of one G1 hand.

    MJ body names (g1_29dof_with_hand.xml: exactly these 14 hand bodies,
    no palm body); the localized g1.usd prims additionally live under the
    ``{side}_hand/`` scope, hence the per-backend forms below.
    """
    return [f"{side}_hand_{part}_{k}_link"
            for part, n in (("index", 2), ("middle", 2), ("thumb", 3))
            for k in range(n)]


# Named grasp groups (declarative): the robot-level
# default for ``body_parts["grasp"]`` -- one group per hand.  The props layer
# no longer consumes it directly (the auto held-chain was retired): contact
# group declarations address these groups by NAME
# (``contacts: [..., sensor_bodies: {body_parts: grasp.left}]``),
# contacts/base.resolve_body_part_group resolves the per-backend sensor
# specs.  Each group carries per-backend sensor-body specs (the two engines
# name robot bodies differently):
#   mujoco   : explicit MJ body names -- the 7 finger links of that hand
#              (more precise than the former ``.*_hand_.*`` scan default:
#              exact names cannot over-match, e.g. a future palm body)
#   isaaclab : Robot-relative prim sub-paths (finger prims nest under
#              ``{side}_hand/`` and ContactSensor prim-path matching is
#              segment-wise; USD audit scripts/usd_hand_audit.py, runtime
#              probe probe_hand_contacts.py -- retired in git history,
#              v063 cleanup)
# The palm links are deliberately NOT part of any group: they are
# fixed-joint-merged into the wrist bodies at PhysX build time (reference
# frames; palm-contact investigation) -- their contacts belong to the wrist.
G1_GRASP_GROUPS: Dict[str, Dict[str, List[str]]] = {
    side: {
        "mujoco": _g1_hand_finger_links(side),
        "isaaclab": [f"{side}_hand/{link}"
                     for link in _g1_hand_finger_links(side)],
    }
    for side in ("left", "right")
}
# (g1pick L3+): the EITHER-hand group -- one declared contact group
# with ``sensor_bodies: {body_parts: grasp.both}`` reads "either hand touches
# the filtered prop", which keeps per-object held criteria single-metric
# (a per-hand pair would need a third combinator nesting level in formula
# success trees, over the depth-2 whitelist).  Additive: left/right keep
# their exact bodies, so every existing declaration resolves unchanged.
G1_GRASP_GROUPS["both"] = {
    "mujoco": (_g1_hand_finger_links("left")
               + _g1_hand_finger_links("right")),
    "isaaclab": ([f"left_hand/{link}"
                  for link in _g1_hand_finger_links("left")]
                 + [f"right_hand/{link}"
                    for link in _g1_hand_finger_links("right")]),
}


# body_parts: base keys (feet/base/joints/hips — IsaaclabEnv reads
# "feet"; adapters .get() what they know) + humanoid extensions (P4
# (upper/lower/legs/waist/arms; no framework code reads those yet)
# + "grasp" (named per-hand groups addressed by DECLARED contact
# groups via {body_parts: grasp.<group>}; see G1_GRASP_GROUPS).
G1_BODY_PARTS: Dict[str, Dict | List | str] = {
    "feet": list(G1_FOOT_BODIES),
    "base": G1_BASE_BODY,
    "joints": list(G1_JOINT_ORDER_IL),            # full policy order (29)
    # dominant proximal joints
    "hips": ["left_hip_pitch_joint", "right_hip_pitch_joint"],
    # humanoid extensions
    "legs": [j for j in G1_JOINT_ORDER_IL
             if "hip" in j or "knee" in j or "ankle" in j],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "arms": [j for j in G1_JOINT_ORDER_IL
             if "shoulder" in j or "elbow" in j or "wrist" in j],
    "upper_body": [G1_JOINT_ORDER_IL[i] for i in G1_UPPER_BODY_IDX],
    "lower_body": [G1_JOINT_ORDER_IL[i] for i in G1_LOWER_BODY_IDX],
    "grasp": {side: {be: list(bodies) for be, bodies in m.items()}
              for side, m in G1_GRASP_GROUPS.items()},
}

# ---------------------------------------------------------------------------
# Effort limits (N·m) from policy_parameters.hpp EFFORT_LIMIT_* (motor spec
# 7520_22=139, 7520_14=88, 5020=25, 4010=5).  Keys are joint-name regexes as
# consumed by ImplicitActuatorCfg.  NB: deliberately NOT the G1_29DOF_CFG
# asset defaults (which use ankle 50 / waist_yaw 88) — the deployment motor
# ratings are the contract.
# ---------------------------------------------------------------------------
G1_EFFORT_LEGS: Dict[str, float] = {
    ".*_hip_pitch_joint": 139.0, ".*_hip_roll_joint": 139.0,
    ".*_knee_joint": 139.0, ".*_hip_yaw_joint": 88.0,
}
G1_EFFORT_FEET: Dict[str, float] = {
    ".*_ankle_pitch_joint": 25.0, ".*_ankle_roll_joint": 25.0,
}
G1_EFFORT_WAIST: Dict[str, float] = {
    "waist_yaw_joint": 50.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0,
}
G1_EFFORT_ARMS: Dict[str, float] = {
    ".*_shoulder_pitch_joint": 25.0, ".*_shoulder_roll_joint": 25.0,
    ".*_shoulder_yaw_joint": 25.0, ".*_elbow_joint": 25.0,
    ".*_wrist_roll_joint": 25.0,
    ".*_wrist_pitch_joint": 5.0, ".*_wrist_yaw_joint": 5.0,
}

# ---------------------------------------------------------------------------
# PD gains (playground MuJoCo sim values, playground_official_sonic.py:96-104;
# keys are joint-name regexes).  Deployment gains stay an A/B alternative via
# yaml ``actuator`` whole-group replacement (collect.py build_actuator).
# ---------------------------------------------------------------------------
G1_KP_LEGS: Dict[str, float] = {
    ".*_hip_pitch_joint": 150.0, ".*_hip_roll_joint": 150.0,
    ".*_hip_yaw_joint": 150.0, ".*_knee_joint": 200.0,
}
G1_KD_LEGS: Dict[str, float] = {
    ".*_hip_pitch_joint": 2.0, ".*_hip_roll_joint": 2.0,
    ".*_hip_yaw_joint": 2.0, ".*_knee_joint": 4.0,
}
G1_KP_FEET: Dict[str, float] = {
    ".*_ankle_pitch_joint": 40.0, ".*_ankle_roll_joint": 40.0,
}
G1_KD_FEET: Dict[str, float] = {
    ".*_ankle_pitch_joint": 2.0, ".*_ankle_roll_joint": 2.0,
}
G1_KP_WAIST: Dict[str, float] = {
    "waist_yaw_joint": 250.0, "waist_roll_joint": 250.0,
    "waist_pitch_joint": 250.0,
}
G1_KD_WAIST: Dict[str, float] = {
    "waist_yaw_joint": 5.0, "waist_roll_joint": 5.0, "waist_pitch_joint": 5.0,
}
G1_KP_ARMS: Dict[str, float] = {
    ".*_shoulder_pitch_joint": 100.0, ".*_shoulder_roll_joint": 100.0,
    ".*_shoulder_yaw_joint": 40.0, ".*_elbow_joint": 40.0,
    ".*_wrist_.*_joint": 20.0,
}
G1_KD_ARMS: Dict[str, float] = {
    ".*_shoulder_pitch_joint": 5.0, ".*_shoulder_roll_joint": 5.0,
    ".*_shoulder_yaw_joint": 2.0, ".*_elbow_joint": 2.0,
    ".*_wrist_.*_joint": 2.0,
}

# ---------------------------------------------------------------------------
# Hand DOFs (NOT part of the 29-joint policy contract).  The actual Nucleus
# ``g1.usd`` (IsaacSim asset tree ``Robots/Unitree/G1/``, sub-asset
# ``g1_29dof_with_hand_rev_1_0``) carries 14 finger joints
# (``{left,right}_hand_{thumb_0,thumb_1,thumb_2,middle_0,middle_1,
# index_0,index_1}_joint``) beyond the 29 body joints — probed 2026-09-02.
# They are actuated (held at USD rest pose 0, matching the Unitree URDF rest
# configuration) but excluded from obs/action by the name-based permutation.
# Gains/limits verbatim from the upstream ``G1_29DOF_CFG`` "hands" group
# (container isaaclab_assets/robots/unitree.py, Isaac Lab 2.3.2).
# ---------------------------------------------------------------------------
G1_HAND_EXPRS: List[str] = [".*_index_.*", ".*_middle_.*", ".*_thumb_.*"]
G1_KP_HANDS = 20.0
G1_KD_HANDS = 2.0
G1_EFFORT_HANDS = 300.0

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

# spawn height for the standing q0 (SONIC clip reference z ~0.74; small
# clearance so the robot settles onto its feet)
G1_SPAWN_HEIGHT = 0.78
