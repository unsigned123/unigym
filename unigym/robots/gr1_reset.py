"""GR1 four-stage reset semantics (robot-body stages; T1.4 audit chain).

The audited GR1 reset is NOT a single qpos0 -- it is a four-stage chain
(docs/audits/v063_gr1_control_chain_audit.md §7, robocasa stack):

  1. ``init_qpos`` (17 body joints from the neutral profile) + zero noise --
     the profile q0 the mujoco backend writes on every reset (B7 face).
  2. 250 substeps of zero action settle with the PD goal pinned to the
     stage-1 qpos (``composite_controller.reset()`` == our actuator-seam
     targets held at q0); objects re-placed by the placement stage settle
     in the same substeps.
  3. both elbows forced to ``range[0]*0.7 + range[1]*0.3`` = -2.1 rad
     (``r/l_elbow_pitch``; XML range [-3, 0]).
  4. the 14 arm joints overwritten with the cotrain real-robot pose
     (`GR1_COTRAIN_POSE`) + per-joint ``U(-noise, +noise)`` noise (the
     waist is NOT in the cotrain table and keeps its settle value), then
     a single ``mj_forward``.

This module is the robot-body half ONLY (stage constants + the validated
stage spec + the stage runners).  The declarative SplitA placement half
lives in ``unigym/initializers/splita_tabletop.py`` (the scene-bundle
initializer) -- placement sampling and the stage-4 noise share
ONE numpy stream (the initializer's ``rng``), consumed in the fixed
documented order: placement draws first, then the 14 cotrain noise
draws (model document order, stage 4).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

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
# stage constants (audit §7)
# ---------------------------------------------------------------------------
GR1_SETTLE_SUBSTEPS = 250  # 10 * (0.05 control / 0.002 model)
GR1_ELBOW_JOINTS = ["robot0_l_elbow_pitch", "robot0_r_elbow_pitch"]
GR1_ELBOW_MIX = (0.7, 0.3)  # range[0]*0.7 + range[1]*0.3 -> -2.1
GR1_COTRAIN_NOISE = 0.2  # per-joint U(-0.2, 0.2)
GR1_RESET_PROTOCOL = "gr1_cotrain"  # the only protocol name (fail-fast)


# ---------------------------------------------------------------------------
# declarative spec validation (the robot-stage keys)
# ---------------------------------------------------------------------------
class Gr1ResetSpec:
    """Validated robot-stage spec of the GR1 four-stage reset chain.

    Attributes mirror the yaml keys; ``rng`` is the protocol-local numpy
    stream (created here for face parity -- the LIVE stream in the
    scene-bundle wiring is the initializer's, shared into
    :func:`apply_gr1_reset_stages` by the backend, so placement draws
    precede the stage-4 noise draws on one stream).  Construction fails
    fast with teaching errors.
    """

    def __init__(self, cfg: dict):
        if not isinstance(cfg, dict):
            raise ValueError(f"reset stage spec must be a mapping, got " f"{type(cfg).__name__}")
        protocol = str(cfg.get("protocol", ""))
        if protocol != GR1_RESET_PROTOCOL:
            raise ValueError(
                f"reset.protocol {protocol!r} unknown | legal: "
                f"[{GR1_RESET_PROTOCOL!r}] (the audited GR1 four-stage "
                f"chain, audit v063 §7)"
            )
        self.protocol = protocol
        self.seed = int(cfg.get("seed", 0))
        self.settle_substeps = int(cfg.get("settle_substeps", GR1_SETTLE_SUBSTEPS))
        if self.settle_substeps < 0:
            raise ValueError(f"reset.settle_substeps must be >= 0, got " f"{self.settle_substeps}")
        self.cotrain_noise = float(cfg.get("cotrain_noise", GR1_COTRAIN_NOISE))
        if self.cotrain_noise < 0:
            raise ValueError(f"reset.cotrain_noise must be >= 0, got " f"{self.cotrain_noise}")
        self.elbow_joints = list(cfg.get("elbow_joints", GR1_ELBOW_JOINTS))
        mix = list(cfg.get("elbow_mix", GR1_ELBOW_MIX))
        if len(mix) != 2 or abs(sum(mix) - 1.0) > 1e-9:
            raise ValueError(f"reset.elbow_mix must be a 2-vector summing to " f"1, got {mix}")
        self.elbow_mix = (float(mix[0]), float(mix[1]))
        # the protocol-local stream: same seed -> bit-identical resets
        self.rng = np.random.default_rng(self.seed)


# ---------------------------------------------------------------------------
# reset stages 2-4 (stage 1 is the backend's B7 q0 write)
# ---------------------------------------------------------------------------
def stage_elbow_release(
    model, data, joint_names=None, mix: tuple = GR1_ELBOW_MIX
) -> Dict[str, float]:
    """Stage 3: force the declared elbow joints to their mixed range pose.

    ``range[0]*mix[0] + range[1]*mix[1]`` (tabletop.py:1080-1091).
    Returns ``{joint: written_value}`` (the audit's -2.1 with the stock
    XML range [-3, 0]).
    """
    import mujoco

    written = {}
    names = GR1_ELBOW_JOINTS if joint_names is None else list(joint_names)
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"reset stage 3: elbow joint {name!r} not in the MJ model")
        lo, hi = (float(v) for v in model.jnt_range[jid])
        val = mix[0] * lo + mix[1] * hi
        data.qpos[model.jnt_qposadr[jid]] = val
        written[name] = val
    return written


def stage_cotrain_noise(
    model, data, rng, noise: float = GR1_COTRAIN_NOISE, pose: Optional[Dict[str, float]] = None
) -> int:
    """Stage 4: overwrite the cotrain-table joints with the noisy pose.

    Model document order, ``pose + U(-noise, +noise)``; every other
    joint (the waist, the fingers) keeps its current qpos
    (tabletop_24dc.py:667-696).  Returns the number of overwritten
    joints (14 expected).
    """
    pose = GR1_COTRAIN_POSE if pose is None else pose
    hits = 0
    for j in range(model.njnt):
        name = model.joint(j).name
        if name in pose:
            data.qpos[model.jnt_qposadr[j]] = pose[name] + rng.uniform(-noise, noise)
            hits += 1
    return hits


def apply_gr1_reset_stages(
    model,
    data,
    rng,
    spec: Gr1ResetSpec,
    settle_fn: Optional[Callable[[int], None]] = None,
    forward: bool = True,
) -> dict:
    """Stages 2-4 in the audited order: settle -> elbow -> cotrain -> forward.

    A single ``mj_forward`` closes the chain; ``settle_fn`` is the
    backend's actuator-aware zero-action substep loop (goal pinned to
    the stage-1 q0).  ``rng`` is the shared reset stream (the
    initializer's, so the placement draws precede the noise draws);
    returns the stage info dict (elbow writes, cotrain hit count).
    """
    if settle_fn is not None and spec.settle_substeps > 0:
        settle_fn(spec.settle_substeps)
    elbows = stage_elbow_release(model, data, spec.elbow_joints, spec.elbow_mix)
    hits = stage_cotrain_noise(model, data, rng, spec.cotrain_noise)
    if forward:
        import mujoco

        mujoco.mj_forward(model, data)
    return {"elbows": elbows, "cotrain_joints": hits}
