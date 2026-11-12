"""IsaacLab 2.x realization of the Unitree G1 (29-DoF) neutral profile.

The neutral module (:mod:`unigym.robots.g1`) owns the joint vocabulary, q0,
feet, body parts, declared assets and action faces.  This module owns the
engine-specific construction: the ``G1_29DOF_CFG`` asset, the contact-sensor
patch, the clone prim path, the upright spawn conversion and the implicit-PD
actuator table (5 groups with per-regex effort limits from the deployment
motor specs).

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
  DOFs are excluded from obs/action by IsaaclabEnv's name permutation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from unigym._typing import JsonValue
from unigym.core import quat
from unigym.env.naming import ENV_REGEX_NS
from unigym.robots.base import RobotProfile
from unigym.robots.g1_constants import (
    G1_EFFORT_ARMS,
    G1_EFFORT_FEET,
    G1_EFFORT_HANDS,
    G1_EFFORT_LEGS,
    G1_EFFORT_WAIST,
    G1_HAND_EXPRS,
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


def _number(params: Mapping[str, JsonValue], key: str, default: float) -> float:
    """Read one numeric realizer param (fail-fast on a non-number)."""
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"G1 realizer param {key!r} must be numeric, got {value!r}")
    return float(value)


def _optional_number(params: Mapping[str, JsonValue], key: str) -> float | None:
    """Read one optional numeric realizer param (``None`` when absent)."""
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"G1 realizer param {key!r} must be numeric, got {value!r}")
    return float(value)


def _gain_override(params: Mapping[str, JsonValue], key: str) -> dict[str, float] | None:
    """Read one optional per-regex gain override mapping."""
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"G1 realizer param {key!r} must be a mapping, got {value!r}")
    out: dict[str, float] = {}
    for name, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"G1 realizer param {key}[{name!r}] must be numeric, got {raw!r}")
        out[str(name)] = float(raw)
    return out


def _override_gains(base: dict[str, float], override: dict[str, float] | None) -> dict[str, float]:
    """Merge a yaml gain override into one actuator group's dict.

    Only regex keys that already exist in the group dict are applied (per
    group); unknown keys are the escape hatch's problem — the yaml ``actuator``
    section replaces whole groups (collect.py build_actuator).
    """
    out = dict(base)
    if override:
        out.update({k: v for k, v in override.items() if k in out})
    return out


def _spawn_rot_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    """Spawn quaternion (w, x, y, z) for a yaw of *yaw_deg* degrees."""
    yaw = math.radians(yaw_deg)
    w, x, y, z = quat.about_z(yaw)
    return (float(w), float(x), float(y), float(z))


def _usd_path(profile: RobotProfile) -> str | None:
    """The profile's declared USD asset ref, if any."""
    for asset in profile.assets:
        if asset.format == "usd":
            return asset.ref
    return None


def realize_g1(
    profile: RobotProfile,
    *,
    params: Mapping[str, JsonValue],
) -> ArticulationCfg:
    """Build the isaaclab G1 ``ArticulationCfg`` from the neutral profile.

    Args:
        profile: the neutral G1 profile (q0/action faces come from here).
        params: the robot section's factory params that stay backend-owned
            (spawn pose, gain/effort overrides, prim path).
    """
    if profile.name != "g1":
        raise ValueError(f"realize_g1 requires a 'g1' profile, got {profile.name!r}")
    spawn_height = _number(params, "spawn_height", G1_SPAWN_HEIGHT)
    spawn_x = _number(params, "spawn_x", 0.0)
    spawn_y = _number(params, "spawn_y", 0.0)
    spawn_yaw_deg = _number(params, "spawn_yaw_deg", 0.0)
    prim_path = params.get("prim_path", f"{ENV_REGEX_NS}/Robot")
    if not isinstance(prim_path, str):
        raise TypeError(f"G1 realizer param 'prim_path' must be a string, got {prim_path!r}")
    stiffness = _gain_override(params, "stiffness")
    damping = _gain_override(params, "damping")
    effort = _gain_override(params, "effort")
    hands_kp = _optional_number(params, "hands_kp")
    hands_kd = _optional_number(params, "hands_kd")

    spawn_kwargs: dict[str, str | bool] = {"activate_contact_sensors": True}
    usd_path = _usd_path(profile)
    if usd_path:
        spawn_kwargs["usd_path"] = usd_path
    return G1_29DOF_CFG.replace(
        prim_path=prim_path,
        spawn=G1_29DOF_CFG.spawn.replace(**spawn_kwargs),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(spawn_x, spawn_y, spawn_height),
            rot=_spawn_rot_wxyz(spawn_yaw_deg),
            joint_pos=dict(profile.default_joint_pos),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                    ".*_knee_joint",
                ],
                stiffness=_override_gains(G1_KP_LEGS, stiffness),
                damping=_override_gains(G1_KD_LEGS, damping),
                effort_limit_sim=_override_gains(G1_EFFORT_LEGS, effort),
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness=_override_gains(G1_KP_FEET, stiffness),
                damping=_override_gains(G1_KD_FEET, damping),
                effort_limit_sim=_override_gains(G1_EFFORT_FEET, effort),
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                stiffness=_override_gains(G1_KP_WAIST, stiffness),
                damping=_override_gains(G1_KD_WAIST, damping),
                effort_limit_sim=_override_gains(G1_EFFORT_WAIST, effort),
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
                stiffness=_override_gains(G1_KP_ARMS, stiffness),
                damping=_override_gains(G1_KD_ARMS, damping),
                effort_limit_sim=_override_gains(G1_EFFORT_ARMS, effort),
            ),
            **(
                {
                    "hands": ImplicitActuatorCfg(  # non-policy DOFs, held at rest
                        joint_names_expr=list(G1_HAND_EXPRS),
                        stiffness=G1_KP_HANDS if hands_kp is None else hands_kp,
                        damping=G1_KD_HANDS if hands_kd is None else hands_kd,
                        effort_limit_sim=G1_EFFORT_HANDS,
                    )
                }
                if "hand" in profile.action_faces
                else {}
            ),
        },
    )
