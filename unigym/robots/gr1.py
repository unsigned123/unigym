"""Fourier GR1 robot profile (GR1ArmsAndWaistFourierHands, tabletop).

Asset: the T1.1 bitwise-reconstructed gr1 tabletop world -- since the
v075 P1 retirement it is assembled from the robot world-part XML
(``assets/gr1_tabletop/gr1_tabletop_robot.xml``) plus the registered
``gr1_tabletop`` scene (the retired live scene XMLs live in
the anchors machine domain, set gr00t-r1 ``scene_retirement/``); 39
actuators over 39 robot hinges, NO freejoint -- the
pelvis is welded to the world.  The profile is consumed by the mujoco
backend as pure constants (``articulation_cfg=None``): joint order, reset
pose tables, hand drive faces and body parts all come from
:mod:`unigym.robots.gr1_constants` (audit-transcribed, provenance there).

Action face (docs/designs/v063-gr00t-adapter.md D2, T1.2b): the 5-group
29-dim GR00T layout maps onto TWO NamedAction faces:

* body 17 -- ``right_arm 7 | left_arm 7 | waist 3`` absolute position
  targets over :data:`GR1_JOINT_ORDER` (audit §6 order), driven by the
  ``computed_torque_pd`` actuator law (per-part blocks torso/right/left,
  kp 1000 / kd 200 -- audit §2/§3);
* hands 2x6 -- the GR00T 6-dim hand action expands to 11 native affine
  servo actuators per hand by index repetition (audit §5,
  ``[0,0,1,1,2,2,3,3,4,4,5]``).  The NamedAction contract is
  name-explicit with ``dim == len(joint_names)``, so the profile declares
  each hand face over the 11 DRIVE-order joint names and the 6 -> 11
  repetition arithmetic rides the adapter side (T3.2): emit
  ``action6[GR1_HAND_EXPANSION]`` as the 11-value vector.  The ctrl law
  is ``native_servo`` (ctrl = clip(target, ctrlrange); the servo gain
  lives in the MJCF gainprm -- software gains are rejected there).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import RobotRecipe
from .gr1_constants import (
    GR1_BODY_PARTS,
    GR1_COTRAIN_POSE,
    GR1_INIT_QPOS,
    GR1_JOINT_ORDER,
    GR1_JOINT_ORDER_MJ,
    GR1_LEFT_HAND_DRIVE_ORDER,
    GR1_RIGHT_HAND_DRIVE_ORDER,
)


def make_gr1_profile(default_q: Optional[Dict[str, float]] = None,
                     body_parts: Optional[Dict] = None,
                     with_isaaclab: bool = False) -> RobotRecipe:
    """Build the GR1 (arms+waist+Fourier hands) RobotRecipe.

    Args:
        default_q: optional per-joint override of the nominal reset pose
            (literal joint names over the 39-joint articulation; unknown
            names fail fast).  The audited cotrain stage applies
            :data:`GR1_COTRAIN_POSE` + per-joint noise at reset time
            (T1.4) -- this channel is for the profile-level pose only.
        body_parts: optional partial override of :data:`GR1_BODY_PARTS`.
        with_isaaclab: NOT supported -- there is no isaaclab GR1 asset in
            this campaign and the isaaclab-side computed-torque
            ActuatorModel is a later task (design D2); the mujoco backend
            consumes this profile as pure constants.
    """
    if with_isaaclab:
        raise NotImplementedError(
            "make_gr1_profile: no isaaclab ArticulationCfg exists for the "
            "GR1 tabletop asset; the isaaclab backend cannot carry the "
            "computed-torque laws yet (design D2 defers the ActuatorModel "
            "mapping) | hint: run the GR1 campaign on the mujoco backend")
    joint_pos = dict(GR1_INIT_QPOS)
    # fingers rest at 0 (audit §7: gripper.init_qpos = [0]*11) -- same
    # convention as the G1 profile's hand block
    hand_joints = [n for n in GR1_JOINT_ORDER_MJ if n not in GR1_JOINT_ORDER]
    joint_pos.update({n: 0.0 for n in hand_joints})
    if default_q:
        unknown = sorted(set(default_q) - set(GR1_JOINT_ORDER_MJ))
        if unknown:
            raise ValueError(
                f"make_gr1_profile: default_q names {unknown} are not GR1 "
                f"joints | hint: keys are literal names over the 39-joint "
                f"articulation (17 body + 11 + 11 finger joints)")
        joint_pos.update({k: float(v) for k, v in default_q.items()})
    bp = dict(GR1_BODY_PARTS)
    if body_parts:
        bp.update(body_parts)
    return RobotRecipe(
        articulation_cfg=None,
        # v105: the named joint vocabulary.  ``body`` is the 17 policy
        # joints; the two hand faces keep their exact drive order.
        joint_sets={"body": tuple(GR1_JOINT_ORDER),
                    "right_hand": tuple(GR1_RIGHT_HAND_DRIVE_ORDER),
                    "left_hand": tuple(GR1_LEFT_HAND_DRIVE_ORDER)},
        default_joint_pos=joint_pos,
        foot_bodies=[],                      # world-fixed base: no feet
        body_parts=bp,
        # document-order preservation for the full-articulation obs keys;
        # the set union is body+right_hand+left_hand, not MJ document order
        _full_joint_names=list(GR1_JOINT_ORDER_MJ),
        # the two hand NamedAction faces (11 drive-order names each; the
        # 6 -> 11 repetition is adapter-side arithmetic over
        # GR1_HAND_EXPANSION -- module docstring)
        _extra_action_terms={"right_hand": list(GR1_RIGHT_HAND_DRIVE_ORDER),
                             "left_hand": list(GR1_LEFT_HAND_DRIVE_ORDER)},
    )


def make_gr1_cotrain_targets() -> List[float]:
    """The 14 arm-joint cotrain pose in :data:`GR1_JOINT_ORDER` subset order.

    The waist entries keep the nominal 0.0 (audit §7: the waist is not
    in the cotrain table).  Convenience for the reset task (T1.4) and
    the probes; NOT applied by the profile itself.
    """
    return [GR1_COTRAIN_POSE.get(n, 0.0) for n in GR1_JOINT_ORDER]
