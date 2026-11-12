"""MuJoCo-specific multi-env helpers (N independent MjData, one MjModel).

Only the pieces that genuinely need the MuJoCo engine live here; the
engine-agnostic bookkeeping moved to ``unigym.backends.collect_core``.
Every function takes the live env as its first argument and reads the
same private tables the inline bodies used; N=1 output is unchanged.
"""

from __future__ import annotations


def reject_model_global_unsupported(
    num_envs: int, prop_manager, camera_manager, director_cfg
) -> None:
    """Fail fast for model-global writes that have no per-env face.

    Cameras re-pose one rig state, the director edits one body/eq table,
    and an ANCHORED articulation prop re-anchors ``model.body_pos``/
    ``body_quat`` (shared by every MjData).  At N>1 these would silently
    apply env 0's declaration to all worlds.
    """
    if num_envs <= 1:
        return
    anchored = [
        h.spec.name
        for h in getattr(prop_manager, "_handlers", ())
        if getattr(h.spec, "type", None) == "articulation"
    ]
    unsupported = []
    if camera_manager is not None:
        unsupported.append("camera_manager")
    if director_cfg is not None:
        unsupported.append("director_cfg")
    if anchored:
        unsupported.append(f"anchored articulation props {sorted(anchored)}")
    if unsupported:
        raise NotImplementedError(
            f"MujocoEnv num_envs={num_envs}>1 does not support "
            f"model-global state: {', '.join(unsupported)} | each env "
            f"shares one MjModel, so camera rigs / director ops / anchored "
            f"articulation anchors have no per-env write face; run these "
            f"features with num_envs=1 or switch to a free_articulation prop"
        )
