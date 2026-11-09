"""Pre-compile MjSpec rewrites for the declarative world faces.

Two seams of :class:`~unigym.backends.mujoco.env.MujocoEnv` live
here so the constructor stays a thin dispatcher:

* :func:`compose_declared_scene` -- the ``env.params.mj_scene`` slot
  names a REGISTERED scene definition; the provider dispatch (with the
  ``mj_scene_params`` slot dict passed through VERBATIM) is composed and
  attached into the base spec before the camera/prop/director
  injections.  Slots are the provider's own fail-closed gates; this
  helper only hands them over.
* :func:`apply_spawn_base_pose` -- the ``env.params.spawn_base_pose``
  fixed-base spawn face.  A world-fixed robot model welds its base into
  the XML (no freejoint: ``spawn_xy``/``spawn_yaw`` have no qpos to land
  in), so a world that is not the XML's authority declares its own base
  pose here: the declared body's SPEC pose is rewritten BEFORE compile
  and the declared yaw IS the base's world yaw (the baked quat is
  replaced).

Both helpers are pure spec rewrites (no engine state), fail closed on
bad declarations and are no-ops when the caller does not invoke them --
the historical model-load path stays byte-identical.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def needs_spec(
    camera_manager=None, prop_manager=None, director_cfg=None, mj_scene=None, spawn_base=None
) -> bool:
    """True when the base model must be loaded as an editable MjSpec.

    The pre-compile injections (camera rigs, props, director bodies) and
    the two declarative rewrites (composed scene, fixed-base spawn) all
    need a spec; without any of them the model loads straight from the
    XML path and the historical code path is byte-identical.
    """
    return any(
        item is not None
        for item in (camera_manager, prop_manager, director_cfg, mj_scene, spawn_base)
    )


def compose_declared_scene(mjspec, name: str, slots) -> Mapping[str, object]:
    """Compose + attach one registered scene definition into *mjspec*.

    The provider dispatch is the registry's closed face
    (:func:`unigym.scenes.registry.build_scene_definition`); ``slots`` is
    the run's ``mj_scene_params`` mapping, passed through verbatim
    (``None`` -> ``{}`` so the provider's own required-slot gates stay
    the single truth).  The scene engine is resolved through the backend
    registry, then composes and attaches at the conventional frame.
    Returns the composed placement-region table.
    """
    from unigym.scenes.registry import build_scene_definition
    from .scene_engine import compose_and_attach

    return compose_and_attach(mjspec, build_scene_definition(name, slots or {}))


def apply_spawn_base_pose(mjspec, declaration) -> None:
    """Rewrite the declared body's spec pose (the fixed-base spawn face).

    ``declaration`` is ``{body: <name>, pos: [x, y, z], yaw: rad}``;
    ``None`` is a no-op (the XML keeps its baked pose).  Fail-closed: the
    body must exist, ``pos`` a finite 3-vector and ``yaw`` a finite
    radians scalar.  The pose is written on the MjSpec BEFORE compile,
    so the compiled model carries it in every frame.
    """
    if declaration is None:
        return
    body_name = str(declaration.get("body") or "robot0_base")
    body = mjspec.body(body_name)
    if body is None:
        bodies = [candidate.name for candidate in mjspec.bodies if candidate.name]
        raise ValueError(
            f"mujoco backend: spawn_base_pose body {body_name!r} not "
            f"found -- available bodies: {bodies}"
        )
    pos = [float(value) for value in (declaration.get("pos") or [])]
    yaw = float(declaration.get("yaw") or 0.0)
    if len(pos) != 3 or not np.isfinite(pos).all() or not np.isfinite(yaw):
        raise ValueError(
            "mujoco backend: spawn_base_pose.pos must be a finite "
            "3-vector and spawn_base_pose.yaw a finite radians scalar, "
            f"got pos={declaration.get('pos')!r} "
            f"yaw={declaration.get('yaw')!r}"
        )
    body.pos = pos
    body.quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
