"""Mujoco scene-preset reset face: base semantics + initializer wiring.

The scene-preset consumption helpers for the mujoco backend.  An
initializer never touches engine handles directly: sampling reads the
model and every mutation goes through a :class:`ResetPrimitives`
primitive built here.  ``write_freejoint`` is the prop freejoint path --
``prop_backend.free_addresses`` -> world pose + optional world velocity,
default zeroed qvel.  ``set_root_state`` / ``set_joint_state`` are the
robot-side writes: they bind to the backend's own state-write face
(``env.set_root_state([active_id], ...)`` /
``env.set_joint_state([active_id], ...)``), so the semantics match the
public contract and only the CURRENT reset env is touched.  The settle
loop is the actuator-aware zero-action face the reset chain already
used.

The GR1 robot-body stages keep their engine-side call site
(``MujocoEnv._apply_reset_protocol``); :func:`wire_reset_face`
only hands the stage spec through, so the execution order and the RNG
consumption order stay bitwise (determinism red line).

``_probe_region_placement`` is the region-bound candidate-probe face
(the backend's collision-query seam the placement sampler reaches
through ``ResetPrimitives.probe_candidate``).

Scope: only the mujoco and mjwarp reset chains build this face.  The
isaaclab2 backend does not construct ``ResetPrimitives`` and is not
wired to the initializer reset face (its reset path stays its own); a
future initializer-capable isaaclab2 reset would bind the same two
callbacks to its state-write methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from unigym.core import quat as quat_math
from unigym.initializers import InitializerBase, ResetPrimitives

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.robots.gr1_reset import Gr1ResetSpec

_ZERO_TOL = 1e-12


def _freejoint_vector(value, *, name: str, field: str, size: int) -> np.ndarray:
    """Validate one finite ``(size,)`` freejoint input (prop-name bearing)."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ResetPrimitives.write_freejoint({name!r}): {field} must be numeric, "
            f"got {type(value).__name__}"
        ) from exc
    if array.shape != (size,):
        raise ValueError(
            f"ResetPrimitives.write_freejoint({name!r}): {field} must have shape "
            f"({size},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"ResetPrimitives.write_freejoint({name!r}): {field} must be finite")
    return array


def _freejoint_yaw(value, *, name: str) -> None:
    """Validate a yaw input without changing its arithmetic (bit-exactness)."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ResetPrimitives.write_freejoint({name!r}): yaw must be numeric, "
            f"got {type(value).__name__}"
        ) from exc
    if array.shape != ():
        raise ValueError(
            f"ResetPrimitives.write_freejoint({name!r}): yaw must be a scalar, "
            f"got shape {array.shape}"
        )
    if not np.isfinite(array):
        raise ValueError(f"ResetPrimitives.write_freejoint({name!r}): yaw must be finite")


def _freejoint_unit_quat(value, *, name: str) -> np.ndarray:
    """Validate + normalize a world-frame ``(w, x, y, z)`` quaternion."""
    quat = _freejoint_vector(value, name=name, field="quat_wxyz", size=4)
    norm = float(np.linalg.norm(quat))
    if norm <= _ZERO_TOL:
        raise ValueError(f"ResetPrimitives.write_freejoint({name!r}): quat_wxyz has zero norm")
    return quat / norm


def _active_reset_env_id(env, *, method: str) -> int:
    """The reset env the initializer is writing (set around ``apply``)."""
    env_id = getattr(env, "_active_reset_env_id", None)
    if env_id is None:
        raise RuntimeError(
            f"ResetPrimitives.{method}: robot writes are only valid while "
            "the reset chain runs initializer.apply() (no active reset env)"
        )
    return int(env_id)


def _reset_state_vector(value, *, method: str, field: str, size: int) -> np.ndarray:
    """Validate one finite ``(size,)`` robot-state input (state-write shape)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ResetPrimitives.{method}: {field} must be numeric, " f"got {type(value).__name__}"
        ) from exc
    if array.shape != (size,):
        raise ValueError(
            f"ResetPrimitives.{method}: {field} must have shape ({size},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"ResetPrimitives.{method}: {field} must be finite")
    return array


def _state_write_row(env, array: np.ndarray):
    """One row on the env device for the backend's state-write face."""
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float64))
    return tensor.reshape(1, -1).to(env.device)


def _probe_region_placement(env, rule, position, yaw, clearance):
    """Probe a synthetic freejoint pose through a private MuJoCo data copy."""
    from ..scene.placement_query import CollisionPair, probe_freejoint_candidate
    import mujoco
    from unigym.initializers.placement import PlacementCapabilityError

    _bid, _qadr, _vadr = env.prop_backend.free_addresses(rule.object_name)
    region = env._placement_regions[rule.region_key]
    support_body = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_BODY, region.owner)
    joint_name = mujoco.mj_id2name(
        env._model, mujoco.mjtObj.mjOBJ_JOINT, int(env._model.body_jntadr[_bid])
    )
    candidate = [
        mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        for geom in range(
            env._model.body_geomadr[_bid],
            env._model.body_geomadr[_bid] + env._model.body_geomnum[_bid],
        )
    ]
    if not candidate:
        raise PlacementCapabilityError(f"prop {rule.object_name!r} has no collision geoms")
    # the narrow-phase collision set is the model's COLLISION truth
    # (contype/conaffinity nonzero): region markers and visual twins
    # are compiled contype=0/conaffinity=0 and are never collision
    # evidence
    blocking_geoms = [
        CollisionPair(
            geom,
            mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_GEOM, other),
            allow_support_contact=(
                rule.relation == "on" and int(env._model.geom_bodyid[other]) == support_body
            ),
        )
        for geom in candidate
        for other in range(env._model.ngeom)
        if (
            int(env._model.geom_bodyid[other]) != _bid
            and (int(env._model.geom_contype[other]) or int(env._model.geom_conaffinity[other]))
        )
    ]
    try:
        return probe_freejoint_candidate(
            env._model,
            env._active_data,
            joint_name,
            position,
            (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)),
            blocking_geoms,
            min_clearance_m=clearance,
        )
    except (AttributeError, ValueError) as exc:
        raise PlacementCapabilityError(str(exc)) from exc


def resolve_base_height(params: dict, profile) -> float:
    """The free-base spawn height in metres (scene-preset §3 semantics).

    Precedence: an explicit ``env.params.spawn_height`` wins over the
    effective profile's ``base_height``; with neither the fallback is
    ``0.0``.  ``base.joint: fixed`` absorbs the legacy spawn_height param:
    a world-fixed base has no freejoint and no spawn drop -> height 0.0
    (the GR1 tabletop scenes' declared behavior, bitwise).  A free-base
    declaration fails fast until the base migration lands.
    """
    base_decl = params.get("base")
    if base_decl is None:
        explicit = params.get("spawn_height")
        if explicit is not None:
            return float(explicit)
        base_height = getattr(profile, "base_height", None)
        return float(base_height) if base_height is not None else 0.0
    if not isinstance(base_decl, dict):
        raise ValueError(
            f"mujoco backend: env.params.base must be a mapping "
            f"({{joint: fixed}}), got {type(base_decl).__name__}"
        )
    unknown = sorted(str(key) for key in base_decl if key != "joint")
    if unknown:
        raise ValueError(
            f"mujoco backend: env.params.base declares unsupported key(s) {unknown}; "
            "the only supported key is 'joint' (currently only 'joint: fixed')"
        )
    base_joint = str(base_decl.get("joint", "fixed"))
    if base_joint != "fixed":
        raise NotImplementedError(
            f"mujoco backend: env.params.base.joint {base_joint!r} is "
            f"not implemented yet (free-base spawning arrives with the "
            f"scene-preset base migration) | hint: declare joint: fixed"
        )
    return 0.0


def build_reset_primitives(env) -> ResetPrimitives:
    """The :class:`ResetPrimitives` face for one constructed backend env."""
    m = env.mj_model

    def write_freejoint(
        name: str,
        pos,
        yaw: Optional[float] = None,
        *,
        quat_wxyz=None,
        lin_vel_w=None,
        ang_vel_w=None,
    ) -> None:
        """Write a prop freejoint's world pose (+ optional world velocity).

        Exactly one orientation source:

        * ``yaw`` -- pure +Z rotation (the placement sampler's call shape,
          unchanged and bit-exact);
        * ``quat_wxyz`` -- full world-frame ``(w, x, y, z)``, normalized
          here; zero-norm / non-finite / wrong-shape inputs fail fast.

        ``lin_vel_w`` is the world-frame linear velocity at the body origin;
        ``ang_vel_w`` is the world-frame angular velocity, rotated by
        ``R.T`` into the freejoint's body-frame qvel slot (MuJoCo's
        convention).  Each velocity is optional; absent writes zero.
        """
        if (yaw is None) == (quat_wxyz is None):
            raise ValueError(
                f"ResetPrimitives.write_freejoint({name!r}): supply exactly one "
                f"of yaw / quat_wxyz (got yaw={yaw is not None}, "
                f"quat_wxyz={quat_wxyz is not None})"
            )
        position = _freejoint_vector(pos, name=name, field="pos", size=3)
        if yaw is not None:
            _freejoint_yaw(yaw, name=name)
            orientation = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=float)
        else:
            orientation = _freejoint_unit_quat(quat_wxyz, name=name)
        lin = (
            np.zeros(3, dtype=float)
            if lin_vel_w is None
            else _freejoint_vector(lin_vel_w, name=name, field="lin_vel_w", size=3)
        )
        if ang_vel_w is None:
            ang_body = np.zeros(3, dtype=float)
        else:
            ang_world = _freejoint_vector(ang_vel_w, name=name, field="ang_vel_w", size=3)
            ang_body = quat_math.to_rotmat(orientation).T @ ang_world
        _bid, qadr, vadr = env.prop_backend.free_addresses(name)
        d = env._active_data
        d.qpos[qadr : qadr + 3] = position
        d.qpos[qadr + 3 : qadr + 7] = orientation
        d.qvel[vadr : vadr + 3] = lin
        d.qvel[vadr + 3 : vadr + 6] = ang_body

    def set_root_state(
        name: str,
        pos_w,
        quat_wxyz,
        lin_vel_w=None,
        ang_vel_w=None,
    ) -> None:
        """Write a free root's world pose/velocity in the CURRENT reset env.

        ``name`` addresses a declared free-root articulation or a free rigid
        prop.  ``pos_w``/``lin_vel_w``/``ang_vel_w`` are world frame; linear
        velocity is at the root body origin.  The wxyz quaternion is
        normalized by the backend state-write face (zero-norm fails).  Both
        velocities are optional; absent writes zero.  Does not settle.
        """
        env_id = _active_reset_env_id(env, method="set_root_state")
        pos = _reset_state_vector(pos_w, method="set_root_state", field="pos_w", size=3)
        quat = _reset_state_vector(quat_wxyz, method="set_root_state", field="quat_wxyz", size=4)
        lin = np.zeros(3, dtype=float)
        if lin_vel_w is not None:
            lin = _reset_state_vector(lin_vel_w, method="set_root_state", field="lin_vel_w", size=3)
        ang = np.zeros(3, dtype=float)
        if ang_vel_w is not None:
            ang = _reset_state_vector(ang_vel_w, method="set_root_state", field="ang_vel_w", size=3)
        env.set_root_state(
            [env_id],
            name=str(name),
            pos_w=_state_write_row(env, pos),
            quat_wxyz=_state_write_row(env, quat),
            lin_vel_w=_state_write_row(env, lin),
            ang_vel_w=_state_write_row(env, ang),
        )

    def set_joint_state(
        articulation: str,
        joint_names,
        position,
        velocity=None,
    ) -> None:
        """Write absolute 1-DOF joint positions in the CURRENT reset env.

        ``joint_names`` is an explicit, duplicate-free name sequence; it
        fixes the ``position``/``velocity`` columns.  ``velocity=None``
        leaves every selected joint's velocity unchanged.  Does not settle.
        """
        env_id = _active_reset_env_id(env, method="set_joint_state")
        if isinstance(joint_names, (str, bytes)):
            raise TypeError(
                "ResetPrimitives.set_joint_state: joint_names must be a "
                "sequence of joint names, not a string"
            )
        names = tuple(str(joint_name) for joint_name in joint_names)
        if not names:
            raise ValueError("ResetPrimitives.set_joint_state: joint_names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError(
                f"ResetPrimitives.set_joint_state: joint_names carries duplicate names: {names}"
            )
        pos = _reset_state_vector(
            position, method="set_joint_state", field="position", size=len(names)
        )
        vel = None
        if velocity is not None:
            vel = _reset_state_vector(
                velocity, method="set_joint_state", field="velocity", size=len(names)
            )
        env.set_joint_state(
            [env_id],
            articulation=str(articulation),
            joint_names=names,
            position=_state_write_row(env, pos),
            velocity=None if vel is None else _state_write_row(env, vel),
        )

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
        for g in range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid]):
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
        model=m,
        settle_fn=env._settle_substeps,
        write_freejoint=write_freejoint,
        set_root_state=set_root_state,
        set_joint_state=set_joint_state,
        forward=forward,
        prop_extents=env._prop_extents,
        prop_names=frozenset(s.name for s in env.prop_manager.specs)
        if env.prop_manager is not None
        else frozenset(),
        resolve_region=resolve_region,
        object_collision_bounds=object_collision_bounds,
        probe_candidate=probe_candidate,
    )


def wire_reset_face(
    env, params: dict, profile
) -> Tuple[Optional[InitializerBase], Optional[Gr1ResetSpec]]:
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
    from unigym.initializers import build_initializer, normalize_initializer_spec

    init_cfg = normalize_initializer_spec(params)
    if init_cfg is None:
        return None, None
    initializer = build_initializer(init_cfg, profile, build_reset_primitives(env))
    if initializer.places_props and env.prop_backend is None:
        raise ValueError(
            "env.params.initializer declares object placement but no "
            "props are wired (yaml `props:`) -- the placement writes "
            "need the injected prop bodies"
        )
    initializer.resolve()
    stage = initializer.stage_spec()
    if stage is None:
        return initializer, None
    from unigym.robots.gr1_reset import Gr1ResetSpec

    return initializer, Gr1ResetSpec(stage)
