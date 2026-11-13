"""Unitree G1 (29-DoF) neutral robot profile with SONIC deployment parameters.

The profile carries authoring data only (joint sets, standby pose, canonical
body facts and bindings, declared assets and action faces).  The isaaclab
``ArticulationCfg`` realization is built through the generic backend builder
from this profile, the run-level ``spec.model`` USD asset and the run-level
``ArticulationSpec.actuation`` groups; the mujoco family consumes this profile
as pure constants.

Asset: the isaaclab path takes the 43-DoF with-hand ``g1.usd`` from the
run-level model reference.  The older 29-DoF single-file ``g1.usd`` (Isaac 4.5)
and the 43-DoF ``configuration/`` build (Isaac 5.0) are both supported by the
neutral profile; ``actuate_hands`` opts into the finger action face.  ``G1_CFG``
/ ``G1_MINIMAL_CFG`` are not used (their old 37-DoF joint set cannot match the
SONIC policy).

Pure-data constants live in :mod:`unigym.robots.g1_constants`
(isaaclab-free; the single source of truth for tests + contract doc).
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from unigym.assets.asset_ref import AssetRef

from .base import BodyBinding, RobotProfile
from .g1_constants import (
    G1_BASE_BODY,
    G1_BODY_BINDINGS,
    G1_BODY_SETS,
    G1_DEFAULT_Q,
    G1_FOOT_BODIES,
    G1_HAND_JOINTS,
    G1_JOINT_ORDER_FULL,
    G1_JOINT_SETS,
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
    usd_path: Optional[str] = None,
    actuate_hands: bool = False,
    default_q: Optional[Dict] = None,
) -> RobotProfile:
    """Build the neutral G1 29-DoF :class:`RobotProfile`.

    Only neutral profile data is accepted here.  Backend construction values
    (gains, effort limits, prim path, spawn options, collision group, soft
    joint-limit factor) are declared at the run level and consumed by the
    backend realizer.

    Args:
        usd_path: local USD override (Nucleus localization); declared as the
            profile's ``usd`` asset when present.  A run-level ``spec.model``
            still wins over this declared asset.
        actuate_hands: include the ``hand`` action face.  Set ``True`` ONLY for
            the 43-DoF ``g1_29dof_with_hand_rev_1_0`` asset (Isaac 5.0
            ``configuration/`` build / ``G1_with_hand``); the default 29-DoF
            Isaac 4.5 ``g1.usd`` has no finger DOFs.
        default_q: optional per-joint override of the SONIC standby q0
            (same channel as the built-in profiles; lets a non-SONIC policy
            spawn at its own q0 — drives spawn pose AND the controller q0).
    """
    # default_joint_pos covers the FULL 43-joint articulation —
    # the 29 SONIC policy joints + the 14 finger joints at their rest
    # pose (0).  Drives spawn pose, the controller q0 (policy joints)
    # and the full-articulation obs q0 (fingers: relative == absolute).
    joint_pos = dict(G1_DEFAULT_Q)
    joint_pos.update({n: 0.0 for n in G1_HAND_JOINTS})
    if default_q:
        joint_pos.update(_expand_default_q(default_q))
    return RobotProfile(
        name="g1",
        joint_sets={name: tuple(joints) for name, joints in G1_JOINT_SETS.items()},
        default_joint_pos=joint_pos,
        assets=AssetRef(usd=usd_path) if usd_path else None,
        # with the hands actuator group, the finger DOFs become a driveable
        # NamedAction face ("hand", dex3 open/close targets) -- term input
        # order == G1_HAND_JOINTS (the hand-face order external drivers
        # emit).  Without actuate_hands the 29-DoF single-face surface is
        # unchanged.
        action_faces={"hand": tuple(G1_HAND_JOINTS)} if actuate_hands else {},
        base_body=G1_BASE_BODY,
        foot_bodies=tuple(G1_FOOT_BODIES),
        body_sets={name: tuple(bodies) for name, bodies in G1_BODY_SETS.items()},
        body_bindings={
            canonical: BodyBinding(canonical, dict(names))
            for canonical, names in G1_BODY_BINDINGS.items()
        },
    )
