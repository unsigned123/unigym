"""Mujoco scene-bundle reset face: base semantics + initializer wiring.

The v067 scene-bundle consumption helpers for the mujoco backend
(docs/designs/v067-scene-bundle-and-initializers.md §2/§3).  An
initializer never touches engine handles directly: sampling reads the
model and every mutation goes through a :class:`ResetPrimitives`
primitive built here (``write_freejoint`` is the existing prop
freejoint path verbatim -- ``prop_backend.free_addresses`` -> posed
qpos + zeroed qvel; the settle loop is the actuator-aware zero-action
face the reset chain already used).  ``joint_group``/``set_targets``
are the robot-side seams for future initializer types (interface stubs
until the robot-body reset migrates, §5 route).

The GR1 robot-body stages keep their engine-side call site
(``MujocoEnv._apply_reset_protocol``); :func:`wire_reset_face`
only hands the stage spec through, so the execution order and the RNG
consumption order stay bitwise (determinism red line).

``_probe_region_placement`` is the region-bound candidate-probe face
(the backend's collision-query seam the placement sampler reaches
through ``ResetPrimitives.probe_candidate``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from unigym.initializers import InitializerBase, ResetPrimitives

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.robots.gr1_reset import Gr1ResetSpec


def _probe_region_placement(env, rule, position, yaw, clearance):
    """Probe a synthetic freejoint pose through a private MuJoCo data copy."""
    from ..scene.placement_query import CollisionPair, probe_freejoint_candidate
    import mujoco
    from unigym.initializers.placement import PlacementCapabilityError

    _bid, _qadr, _vadr = env.prop_backend.free_addresses(rule.object_name)
    region = env._placement_regions[rule.region_key]
    support_body = mujoco.mj_name2id(
        env._model, mujoco.mjtObj.mjOBJ_BODY, region.owner)
    joint_name = mujoco.mj_id2name(
        env._model, mujoco.mjtObj.mjOBJ_JOINT,
        int(env._model.body_jntadr[_bid]))
    candidate = [
        mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        for geom in range(env._model.body_geomadr[_bid],
                          env._model.body_geomadr[_bid] + env._model.body_geomnum[_bid])
    ]
    if not candidate:
        raise PlacementCapabilityError(
            f"prop {rule.object_name!r} has no collision geoms")
    # the narrow-phase collision set is the model's COLLISION truth
    # (contype/conaffinity nonzero): region markers and visual twins
    # are compiled contype=0/conaffinity=0 and are never collision
    # evidence (docs/designs/v076-kitchen-construction-and-
    # initialization.md §6.2)
    blocking_geoms = [
        CollisionPair(
            geom,
            mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_GEOM, other),
            allow_support_contact=(
                rule.relation == "on"
                and int(env._model.geom_bodyid[other]) == support_body))
        for geom in candidate
        for other in range(env._model.ngeom)
        if (int(env._model.geom_bodyid[other]) != _bid
            and (int(env._model.geom_contype[other])
                 or int(env._model.geom_conaffinity[other])))
    ]
    try:
        return probe_freejoint_candidate(
            env._model, env._active_data, joint_name, position,
            (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)),
            blocking_geoms, min_clearance_m=clearance)
    except (AttributeError, ValueError) as exc:
        raise PlacementCapabilityError(str(exc)) from exc


def resolve_base_height(params: dict) -> float:
    """The base declaration's spawn height (scene-bundle §3 semantics).

    ``base.joint: fixed`` absorbs the legacy spawn_height param: a
    world-fixed base has no freejoint and no spawn drop -> height 0.0
    (the GR1 tabletop scenes' declared behavior, bitwise).  A free-base
    declaration fails fast until the base migration lands.  No
    declaration -> the legacy spawn_height param (G1 scenes, default
    profile height).
    """
    base_decl = params.get("base")
    if base_decl is None:
        from unigym.robots.g1_constants import G1_SPAWN_HEIGHT
        return float(params.get("spawn_height", G1_SPAWN_HEIGHT))
    if not isinstance(base_decl, dict):
        raise ValueError(
            f"mujoco backend: env.params.base must be a mapping "
            f"({{joint: fixed}}), got {type(base_decl).__name__}")
    base_joint = str(base_decl.get("joint", "fixed"))
    if base_joint != "fixed":
        raise NotImplementedError(
            f"mujoco backend: env.params.base.joint {base_joint!r} is "
            f"not implemented yet (free-base spawning arrives with the "
            f"scene-bundle base migration) | hint: declare joint: fixed")
    return 0.0


def build_reset_primitives(env) -> ResetPrimitives:
    """The :class:`ResetPrimitives` face for one constructed backend env."""
    m = env.mj_model

    def write_freejoint(name: str, pos, yaw: float) -> None:
        _bid, qadr, vadr = env.prop_backend.free_addresses(name)
        d = env._active_data
        d.qpos[qadr:qadr + 3] = pos
        d.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0.0, 0.0,
                                     np.sin(yaw / 2)]
        d.qvel[vadr:vadr + 6] = 0.0

    def joint_group(name: str):
        raise NotImplementedError(
            "ResetPrimitives.joint_group: the robot-side joint seam "
            "arrives with the robot-body reset migration "
            "(docs/designs/v067-scene-bundle-and-initializers.md §5)")

    def set_targets(idx, vals) -> None:
        raise NotImplementedError(
            "ResetPrimitives.set_targets: the robot-side joint seam "
            "arrives with the robot-body reset migration "
            "(docs/designs/v067-scene-bundle-and-initializers.md §5)")

    def forward() -> None:
        import mujoco
        mujoco.mj_forward(m, env._active_data)

    def resolve_region(name: str):
        try:
            return env._placement_regions[name]
        except KeyError as exc:
            raise ValueError(f"placement region {name!r} is not declared") from exc

    def object_collision_bounds(name: str):
        """The prop's exact object-origin-local AABB (per-axis footprint).

        The placement-sampler bounds contract (``sample_placements``:
        "bounds are object-origin-local min/max XYZ tuples" -- the
        envelope math is exact only over honest per-axis inputs).  The
        footprint is the box geoms' body-frame AABB: per-axis
        ``|pos| + |R| @ half-sizes``.  The SplitA ``prop_extents`` face
        keeps its own square-envelope contract; the placement chain no
        longer borrows it (the F4 counter strips are official-thin --
        layout001's sink-counter ``top_front`` is the clip-saturated
        0.02 m pad face -- and a square-stamped footprint makes every
        non-square prop infeasible on them, which the sink regions'
        large extents had masked).
        """
        import mujoco

        bid, _, _ = env.prop_backend.free_addresses(name)
        half_x, half_y, height = 0.0, 0.0, 0.0
        for g in range(m.body_geomadr[bid],
                       m.body_geomadr[bid] + m.body_geomnum[bid]):
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            size = m.geom_size[g]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, np.asarray(m.geom_quat[g], dtype=float))
            # |R| @ half-sizes: the geom's axis-aligned half extents in
            # the body frame (the exact AABB image under the geom quat)
            extents = (
                abs(rot[0]) * size[0] + abs(rot[1]) * size[1] + abs(rot[2]) * size[2],
                abs(rot[3]) * size[0] + abs(rot[4]) * size[1] + abs(rot[5]) * size[2],
            )
            pos = m.geom_pos[g]
            half_x = max(half_x, abs(float(pos[0])) + float(extents[0]))
            half_y = max(half_y, abs(float(pos[1])) + float(extents[1]))
            height = max(height, float(size[2]) * 2)
        return ((-half_x, -half_y, 0.0), (half_x, half_y, height))

    def probe_candidate(rule, position, yaw, clearance):
        return _probe_region_placement(env, rule, position, yaw, clearance)

    return ResetPrimitives(
        model=m, settle_fn=env._settle_substeps,
        write_freejoint=write_freejoint, joint_group=joint_group,
        set_targets=set_targets, forward=forward,
        prop_extents=env._prop_extents,
        prop_names=frozenset(s.name for s in env.prop_manager.specs)
        if env.prop_manager is not None else frozenset(),
        resolve_region=resolve_region,
        object_collision_bounds=object_collision_bounds,
        probe_candidate=probe_candidate)


def wire_reset_face(env, params: dict, profile
                    ) -> Tuple[Optional[InitializerBase],
                               Optional[Gr1ResetSpec]]:
    """Dispatch the declared reset face (the six-step contract).

    read the segment (``initializer`` or the legacy ``reset`` sugar in
    :func:`unigym.initializers.normalize_initializer_spec`) -> registry
    lookup -> validate (inside build) -> resolve against the compiled
    model.  The per-episode apply + the name-neutrality check run in
    the backend's reset chain.  Returns ``(initializer, stage_spec)``
    (``(None, None)`` when no reset face is declared -- the legacy
    reset, bit-identical); the stage spec comes from the initializer's
    own ``stage_spec`` face and validates through the legacy
    ``Gr1ResetSpec`` stage face.
    """
    from unigym.initializers import build_initializer, \
        normalize_initializer_spec
    init_cfg = normalize_initializer_spec(params)
    if init_cfg is None:
        return None, None
    initializer = build_initializer(
        init_cfg, profile, build_reset_primitives(env))
    if initializer.places_props and env.prop_backend is None:
        raise ValueError(
            "env.params.initializer declares object placement but no "
            "props are wired (yaml `props:`) -- the placement writes "
            "need the injected prop bodies")
    initializer.resolve()
    stage = initializer.stage_spec()
    if stage is None:
        return initializer, None
    from unigym.robots.gr1_reset import Gr1ResetSpec
    return initializer, Gr1ResetSpec(stage)
