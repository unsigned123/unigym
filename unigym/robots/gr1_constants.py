"""Fourier GR1 (GR1ArmsAndWaistFourierHands) constants — pure data, NO deps.

Single source of truth for the GR1 tabletop campaign's robot-side contract
(joint order, reset poses, hand drive mapping, computed-torque gains).
Kept dependency-free so the unit tests import it on a plain python.

Every number is transcribed verbatim from the exhaustive control-chain
audit (docs/audits/v063_gr1_control_chain_audit.md -- T0.2a, runtime
model.xml line-verified):

* §6  joint naming/order  -> :data:`GR1_JOINT_ORDER` (right 7 | left 7 |
  waist 3, the audited env 29-dim body layout; per-part order == the MJ
  document order) and :data:`GR1_JOINT_ORDER_MJ` (the full 39-robot-joint
  MJ document order: waist, right arm, right hand, left arm, left hand).
* §7  reset poses         -> :data:`GR1_INIT_QPOS` (the nominal 17-dim
  init_qpos: waist [0,0,0], right [0,-0.1,0,-1.57,0,0,0], left mirrored)
  and :data:`GR1_COTRAIN_POSE` (the 14 arm-joint cotrain real-robot pose;
  the waist is deliberately not in the table).  The audited reset is
  four-stage (init_qpos -> 250-substep settle -> elbows forced to -2.1 ->
  cotrain + U(-0.2, 0.2) per arm joint); the pose tables here are the
  data half, the stage logic belongs to the reset task (T1.4).
* §5  finger drive mapping -> :data:`GR1_HAND_EXPANSION` and the per-hand
  joint tables.  The 6-dim hand action (semantic order
  pinky/ring/middle/index/thumb_curl/thumb_rot) is expanded to the 11
  native affine servo actuators by index repetition
  ``[0,0,1,1,2,2,3,3,4,4,5]``; :data:`GR1_{RIGHT,LEFT}_HAND_DRIVE_ORDER`
  lists the driven joints in ACTUATOR document order so
  ``action6[GR1_HAND_EXPANSION]`` is name-aligned 1:1 with the drive
  order (the doc-order tables -- MJ joint order, reversed wrt the drives
  -- are kept alongside for obs-side consumers, audit §5 asymmetry note).
* §2  body gains          -> kp 1000 / kd 200 uniform on all 17 body
  joints (controllers/config/robots/default_gr1.json, software
  computed-torque layer; NOT MJCF gainprm).
"""
from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Body joints (audit §6: env layout right[0:7] | left[7:14] | torso[14:17];
# within each part the MJ document order -- RUNTIME lines in the audit table)
# ---------------------------------------------------------------------------
GR1_RIGHT_ARM_JOINTS: List[str] = [
    "robot0_r_shoulder_pitch", "robot0_r_shoulder_roll",
    "robot0_r_shoulder_yaw", "robot0_r_elbow_pitch",
    "robot0_r_wrist_yaw", "robot0_r_wrist_roll", "robot0_r_wrist_pitch",
]
GR1_LEFT_ARM_JOINTS: List[str] = [
    "robot0_l_shoulder_pitch", "robot0_l_shoulder_roll",
    "robot0_l_shoulder_yaw", "robot0_l_elbow_pitch",
    "robot0_l_wrist_yaw", "robot0_l_wrist_roll", "robot0_l_wrist_pitch",
]
GR1_WAIST_JOINTS: List[str] = [
    "robot0_torso_waist_yaw", "robot0_torso_waist_pitch",
    "robot0_torso_waist_roll",
]

#: the 17 policy-facing body joints (NamedAction primary face order).
GR1_JOINT_ORDER: List[str] = (GR1_RIGHT_ARM_JOINTS + GR1_LEFT_ARM_JOINTS
                              + GR1_WAIST_JOINTS)

# ---------------------------------------------------------------------------
# Hand joints (audit §5; joint names carry the *_joint suffix, the DRIVING
# actuators are the parallel *_joint_drive entries -- 1:1, kp=2 gainprm /
# thumb-yaw kp=4, ctrlrange [0, ~1.74])
# ---------------------------------------------------------------------------
#: 6-dim hand action semantic order (the audited GR00T/robosuite layout).
GR1_HAND_ACTION_ORDER: List[str] = [
    "pinky", "ring", "middle", "index", "thumb_curl", "thumb_rot"]
#: 6 -> 11 index repetition onto the DRIVE order (fourier_hands.py
#: format_action, audit §5 verbatim: thumb rot a5 is 1:1, the other five
#: dims each drive an intermediate+proximal pair).
GR1_HAND_EXPANSION: List[int] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]

#: right-hand driven joints in ACTUATOR document order (RUNTIME drives
#: 1178-1188): ``action6[GR1_HAND_EXPANSION][i]`` targets element i.
GR1_RIGHT_HAND_DRIVE_ORDER: List[str] = [
    "gripper0_right_R_pinky_intermediate_joint",
    "gripper0_right_R_pinky_proximal_joint",
    "gripper0_right_R_ring_intermediate_joint",
    "gripper0_right_R_ring_proximal_joint",
    "gripper0_right_R_middle_intermediate_joint",
    "gripper0_right_R_middle_proximal_joint",
    "gripper0_right_R_index_intermediate_joint",
    "gripper0_right_R_index_proximal_joint",
    "gripper0_right_R_thumb_distal_joint",
    "gripper0_right_R_thumb_proximal_pitch_joint",
    "gripper0_right_R_thumb_proximal_yaw_joint",
]
#: left-hand twin (RUNTIME drives 1189-1199).
GR1_LEFT_HAND_DRIVE_ORDER: List[str] = [
    name.replace("gripper0_right_R_", "gripper0_left_L_")
    for name in GR1_RIGHT_HAND_DRIVE_ORDER
]


def _hand_doc_order(prefix: str) -> List[str]:
    """One hand's 11 finger joints in MJ DOCUMENT order.

    Audit §5 table, RUNTIME:370-436 right / 514-580 left -- the
    reverse-symmetric twin of the drive order (obs-side consumers; the
    audit's asymmetry note).
    """
    parts = ["thumb_proximal_yaw", "thumb_proximal_pitch", "thumb_distal",
             "index_proximal", "index_intermediate", "middle_proximal",
             "middle_intermediate", "ring_proximal", "ring_intermediate",
             "pinky_proximal", "pinky_intermediate"]
    return [f"{prefix}_{joint}_joint" for joint in parts]


#: right-hand finger joints in MJ document order (model joints 10-20).
GR1_RIGHT_HAND_JOINTS: List[str] = _hand_doc_order("gripper0_right_R")
#: left-hand twin (model joints 28-38).
GR1_LEFT_HAND_JOINTS: List[str] = _hand_doc_order("gripper0_left_L")

#: the full 39-joint robot articulation in MJ document order (waist 3 +
#: right arm 7 + right hand 11 + left arm 7 + left hand 11; no freejoint:
#: the pelvis is welded to the world through mobilebase0_support).
GR1_JOINT_ORDER_MJ: List[str] = (
    GR1_WAIST_JOINTS + GR1_RIGHT_ARM_JOINTS + GR1_RIGHT_HAND_JOINTS
    + GR1_LEFT_ARM_JOINTS + GR1_LEFT_HAND_JOINTS)

# ---------------------------------------------------------------------------
# Reset poses (audit §7)
# ---------------------------------------------------------------------------
#: nominal init_qpos by joint name (gr1_robot.py:47-54): waist [0,0,0],
#: right [0,-0.1,0,-1.57,0,0,0], left [0,0.1,0,-1.57,0,0,0].
GR1_INIT_QPOS: Dict[str, float] = {
    "robot0_torso_waist_yaw": 0.0,
    "robot0_torso_waist_pitch": 0.0,
    "robot0_torso_waist_roll": 0.0,
    "robot0_r_shoulder_pitch": 0.0,
    "robot0_r_shoulder_roll": -0.1,
    "robot0_r_shoulder_yaw": 0.0,
    "robot0_r_elbow_pitch": -1.57,
    "robot0_r_wrist_yaw": 0.0,
    "robot0_r_wrist_roll": 0.0,
    "robot0_r_wrist_pitch": 0.0,
    "robot0_l_shoulder_pitch": 0.0,
    "robot0_l_shoulder_roll": 0.1,
    "robot0_l_shoulder_yaw": 0.0,
    "robot0_l_elbow_pitch": -1.57,
    "robot0_l_wrist_yaw": 0.0,
    "robot0_l_wrist_roll": 0.0,
    "robot0_l_wrist_pitch": 0.0,
}

#: cotrain real-robot arm pose (cotrain_utils.py:7-23; 14 arm joints, the
#: waist is NOT in the table -- audit §7 stage 4 applies it with per-joint
#: U(-0.2, 0.2) noise on top).
GR1_COTRAIN_POSE: Dict[str, float] = {
    "robot0_r_shoulder_pitch": -0.22963779,
    "robot0_r_shoulder_roll": -0.38363408,
    "robot0_r_shoulder_yaw": 0.14360377,
    "robot0_r_elbow_pitch": -1.5289252,
    "robot0_r_wrist_yaw": -0.2897802,
    "robot0_r_wrist_roll": -0.07134621,
    "robot0_r_wrist_pitch": -0.04550289,
    "robot0_l_shoulder_pitch": -0.10933163,
    "robot0_l_shoulder_roll": 0.43292055,
    "robot0_l_shoulder_yaw": -0.15983289,
    "robot0_l_elbow_pitch": -1.48233023,
    "robot0_l_wrist_yaw": 0.2359135,
    "robot0_l_wrist_roll": 0.26184522,
    "robot0_l_wrist_pitch": 0.00830735,
}

# ---------------------------------------------------------------------------
# Body PD gains (audit §2: default_gr1.json, software computed-torque
# layer -- uniform on all 17 body joints; the hand servo gain lives in the
# MJCF gainprm and is deliberately NOT mirrored here)
# ---------------------------------------------------------------------------
GR1_KP_BODY = 1000.0
GR1_KD_BODY = 200.0

# ---------------------------------------------------------------------------
# body_parts (profile cross-robot configurability channel; feet is EMPTY --
# the base is world-fixed, there is no locomotion contact surface)
# ---------------------------------------------------------------------------
GR1_BASE_BODY = "robot0_whole_body"

GR1_BODY_PARTS: Dict[str, Dict | List] = {
    "feet": [],
    "base": GR1_BASE_BODY,
    "joints": list(GR1_JOINT_ORDER),
    "waist": list(GR1_WAIST_JOINTS),
    "arms": GR1_RIGHT_ARM_JOINTS + GR1_LEFT_ARM_JOINTS,
    "right_hand": list(GR1_RIGHT_HAND_DRIVE_ORDER),
    "left_hand": list(GR1_LEFT_HAND_DRIVE_ORDER),
}
