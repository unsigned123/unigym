"""Kinematic-obstacle URDF spawner (DEPRECATED).

A URDF obstacle spawned as a *free* rigid body in a multi-env cloned scene can
be knocked around / launched by the robot.  ``fix_base`` would anchor
it, but the importer's world-fixed root joint is not cloned with a valid body
rel by Isaac Lab's GridCloner ("Cloning joints ... root_joint without a body
rel ... localPose wont be updated"), which corrupts the per-env anchoring and
accumulates into batch-wide solver explosions right after the first episode
(reset) wave.

Marking the body *kinematic* instead (``RigidBodyPropertiesCfg.kinematic_enabled``)
makes it an immovable rigid body: immune to gravity/forces (it can never fly),
moved only through user poses, while still colliding solidly with dynamic
actors (the robot) — the same behaviour as the built-in obstacle course's
terrain-embedded obstacles, but through Isaac Lab's standard RigidObject
machinery so root-pose views, reset teleports and monitoring keep working.

.. warning::

    DEPRECATED (2026-08): in practice a kinematic URDF body generates **no
    contact** in GridCloner-cloned multi-env scenes — walls and the whole
    course were walked straight through (see ``obstacles/border.py`` header).
    The working anchor is ``ObstacleSpec.heavy_dynamic``: a normal dynamic
    rigid body (real collision) with a 2000 kg mass + strong damping.  This
    module is kept only as a fallback for isolated / non-cloned scenes.

Only the modules' stdlib is imported at module import time: isaaclab/pxr
imports happen lazily inside the spawn function (keeps the obstacle package
importable in pure-torch unit tests).
"""
from __future__ import annotations


def spawn_kinematic_urdf(
    prim_path: str,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Import a URDF (cloned per env) and set every rigid body kinematic.

    ``prim_path`` may be an ``{ENV_REGEX_NS}/...`` regex; the underlying
    ``spawn_from_urdf`` is ``@clone``-decorated and creates one instance per
    matching environment.  ``kinematic_enabled=True`` is then applied to the
    base of *every* clone (clones are plain copies, so the attribute must be
    set instance-by-instance, not once on the source).
    """
    from isaaclab.sim.schemas.schemas import modify_rigid_body_properties
    from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
    from isaaclab.sim.spawners.from_files import spawn_from_urdf
    from isaaclab.sim.utils.queries import find_matching_prim_paths

    prim = spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
    props = RigidBodyPropertiesCfg(kinematic_enabled=True)
    for path in find_matching_prim_paths(prim_path):
        try:
            modify_rigid_body_properties(path, props)
        except Exception as _exc:  # pragma: no cover - defensive
            print(f"[kinematic] failed to freeze '{path}': "
                  f"{type(_exc).__name__}: {_exc}", flush=True)
    return prim
