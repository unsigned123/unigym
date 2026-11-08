"""Articulation + control runtime tables (v104.2, MuJoCo family).

Every engine iterates ``env.articulations`` and ``env.controls``; there is no
single/multi mode switch.  This module owns the shared, engine-agnostic half
(per-articulation name/address tables, the drive strategies, action routing,
reset, observation and termination keys) for the CPU ``MujocoEnv`` and the
``MjwarpEnv`` (both speak the same ``MjModel``/``MjData`` family).  Each
backend keeps only its engine-shaped step loop.
"""
from __future__ import annotations

import numpy as np
import torch

from unigym.actuators import StandardPdActuator
from unigym.backends.shared.mjfamily.articulation_obs import (
    articulation_catalog,
)
from unigym.backends.shared.mjfamily.model.ir_access import (
    actuator_by_joint,
    body_id,
    dof_addresses,
    qpos_addresses,
)
from unigym.backends.shared.mjfamily.name_map import (
    MjArticulationBinding,
    MjControlBinding,
)
from unigym.env.obs_catalog import ObsCatalog
from unigym.robots.g1_constants import G1_KD_HANDS, G1_KP_HANDS
from unigym.runtime.controllers import (
    ArticulationRuntime,
    ControlRuntime,
)
from unigym.runtime.controls import ControlError

__all__ = [
    "apply_control_substep",
    "free_joint_adr",
    "init_tables",
    "reset_articulations",
    "route_actions",
    "validate_drive_support",
]


def _label(env) -> str:
    """Backend label for teaching errors (defaults to the CPU mujoco lane)."""
    return getattr(env, "_backend_label", "mujoco backend")


def free_joint_adr(model, body_bid):
    """``(qpos_adr, dof_adr)`` of the FREE joint owned by *body_bid*."""
    import mujoco

    if body_bid < 0:
        return None, None
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        if int(model.jnt_bodyid[jid]) == int(body_bid):
            return (int(model.jnt_qposadr[jid]),
                    int(model.jnt_dofadr[jid]))
    return None, None


# --------------------------------------------------------------------------- #
# drive support (honest capability, fail-fast at construction)
# --------------------------------------------------------------------------- #
def _actuator_law(model, slot: int) -> str:
    """Classify a compiled actuator's honest ctrl law.

    ``motor``           -- gain fixed / bias none (direct torque/effort ctrl)
    ``position_servo``  -- affine bias with a stiffness term
    ``velocity_servo``  -- affine bias with a damping term only
    ``general``         -- anything else (no honest generic drive)
    """
    import mujoco

    gain = int(model.actuator_gaintype[slot])
    bias = int(model.actuator_biastype[slot])
    if bias == int(mujoco.mjtBias.mjBIAS_NONE) \
            and gain == int(mujoco.mjtGain.mjGAIN_FIXED):
        return "motor"
    if bias == int(mujoco.mjtBias.mjBIAS_AFFINE) \
            and gain == int(mujoco.mjtGain.mjGAIN_FIXED):
        biasprm = model.actuator_biasprm[slot]
        if float(biasprm[1]) != 0.0:
            return "position_servo"
        if float(biasprm[2]) != 0.0:
            return "velocity_servo"
    return "general"


_LAW_FOR_DRIVE = {
    "position_servo": "position_servo",
    "torque": "motor",
    "effort": "motor",
    "velocity": "velocity_servo",
}


def validate_drive_support(env, control: ControlRuntime, label: str) -> None:
    """Fail fast when the compiled actuators cannot honour *control.drive*.

    ``position_pd`` is software PD (always available); ``position_servo`` /
    ``torque`` / ``effort`` / ``velocity`` require a matching compiled
    actuator law.  ``current`` has no MuJoCo actuator representation and is
    always refused here.
    """
    if control.prop:
        return
    if control.drive == "position_pd":
        return
    if control.drive == "current":
        raise NotImplementedError(
            f"{label}: articulation {control.articulation!r} control "
            f"{control.name!r} requests drive 'current', which the "
            f"MuJoCo actuator model does not represent | hint: use "
            f"'torque'/'effort' with a motor actuator, or 'position_pd'")
    wanted = _LAW_FOR_DRIVE.get(control.drive)
    if wanted is None:
        raise NotImplementedError(
            f"{label}: articulation {control.articulation!r} control "
            f"{control.name!r} requests unsupported drive "
            f"{control.drive!r}")
    slots = np.asarray(control.addresses.get("act"), dtype=np.int64)
    bad = []
    for slot in slots.tolist():
        if _actuator_law(env._model, int(slot)) != wanted:
            bad.append(int(slot))
    if bad:
        raise NotImplementedError(
            f"{label}: articulation {control.articulation!r} control "
            f"{control.name!r} requests drive {control.drive!r} but its "
            f"compiled actuators {bad} are not {wanted!r} actuators | hint: "
            f"compile a matching actuator (motor for torque/effort, affine "
            f"velocity actuator for velocity) or use drive 'position_pd'")


# --------------------------------------------------------------------------- #
# gains
# --------------------------------------------------------------------------- #
def _resolve_named_gains(gains, count, what):
    kp = gains.get("kp")
    kd = gains.get("kd")
    out = {}
    for key, value in (("kp", kp), ("kd", kd)):
        if value is None:
            continue
        if isinstance(value, (int, float)):
            out[key] = np.full(count, float(value), dtype=np.float64)
        else:
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != (count,):
                raise ControlError(
                    f"{what} gain {key!r} must be a scalar or length-"
                    f"{count} vector")
            out[key] = arr
    if kp is not None and kd is not None:
        return out["kp"], out["kd"]
    return None


def _default_gains(env, control, bare, params, gains_fn):
    if control.action_face == control.articulation:
        return gains_fn(bare, params.get("kp"), params.get("kd"))
    kp = (float(params["hand_kp"]) if params.get("hand_kp") is not None
          else float(G1_KP_HANDS))
    kd = (float(params["hand_kd"]) if params.get("hand_kd") is not None
          else float(G1_KD_HANDS))
    return (np.full(len(bare), kp, dtype=np.float64),
            np.full(len(bare), kd, dtype=np.float64))


def _build_drive(env, control, bare, backend_bare, params, gains_fn, label):
    """Attach the executable strategy state to one control runtime."""
    control.addresses["bare_joint_names"] = tuple(bare)
    if control.drive == "position_pd":
        resolved = _resolve_named_gains(
            control.gains, len(bare), f"control {control.name!r}")
        if resolved is None:
            kp, kd = _default_gains(env, control, bare, params, gains_fn)
        else:
            kp, kd = resolved
        control.addresses["kp"] = kp
        control.addresses["kd"] = kd
        control._model = env._model
        control._act_of_joint = env._act_of_joint
        control._damping_stability_scale = getattr(
            env, "_damping_stability_scale", None)
        pd = StandardPdActuator(
            live_ctrlrange=(control.action_face != control.articulation))
        control._compiled_joint_names = tuple(backend_bare)
        pd.bind(control, {"joint_names": list(backend_bare),
                          "kp": kp, "kd": kd})
        control.pd = pd
    elif control.prop:
        # prop servos keep their persistent target and ctrl write face
        control.addresses.setdefault(
            "targets",
            np.zeros((env.num_envs, len(control.joint_names)),
                     dtype=np.float64))
    validate_drive_support(env, control, label)


def _control_addresses(env, control, art):
    """Resolve the joint/actuator address tables of one control runtime.

    The neutral ``control.joint_names`` are mapped to the compiled backend
    names through ``env._mj_name_map``; the resulting binding lives on the
    backend-private ``env._mj_bindings`` table.
    """
    ir = env._model_ir
    name_map = env._mj_name_map
    names = tuple(control.joint_names)
    missing = sorted(set(names) - set(art.joint_names))
    if missing:
        raise ValueError(
            f"{_label(env)}: control {control.name!r} names joints {missing} "
            f"that are not asset joints of articulation {art.name!r} | "
            f"asset joints: {list(art.joint_names)}")
    backend_names = tuple(name_map.backend_name(art.name, name)
                          for name in names)
    control.asset_joint_names = tuple(art.joint_names)
    control.feet = tuple(art.feet)
    jadr = np.asarray(qpos_addresses(ir, backend_names), dtype=np.int64)
    jdr = np.asarray(dof_addresses(ir, backend_names), dtype=np.int64)
    control.addresses.update({"jadr": jadr, "jdr": jdr})
    if control.prop:
        return
    act_of_joint = env._act_of_joint
    missing_act = [n for n in backend_names if n not in act_of_joint]
    if missing_act:
        raise ValueError(
            f"{_label(env)}: control {control.name!r} has no actuator for "
            f"joints {missing_act}")
    act = np.asarray([act_of_joint[n] for n in backend_names],
                     dtype=np.int64)
    control.addresses["act"] = act
    index = {n: i for i, n in enumerate(art.joint_names)}
    perm = np.asarray([index[n] for n in names], dtype=np.int64)
    control.addresses["perm_il2pol"] = perm
    env._mj_bindings[control.name] = MjControlBinding(
        unigym_joint_names=names,
        backend_joint_names=backend_names,
        jadr=tuple(int(v) for v in jadr),
        jdr=tuple(int(v) for v in jdr),
        act=tuple(int(v) for v in act),
        perm_il2pol=tuple(int(v) for v in perm),
    )


def _asset_robot_names(env, spec, name_map) -> tuple[str, ...]:
    """The articulation's neutral hinge names in compiled asset order."""
    ir = env._model_ir
    declared = name_map.unigym_joint_names(spec.name)
    if declared:
        names = []
        for name in declared:
            backend = name_map.backend_name(spec.name, name)
            row = ir.joints[backend] if backend in ir.joints else None
            if row is None or row.type in ("free", "ball"):
                continue
            names.append(name)
        return tuple(names)
    # fused / identity path: scan the compiled IR (no composition map).
    names = []
    for row in ir.joints:
        if row.type in ("free", "ball"):
            continue
        if not row.name or row.name.startswith("Prop_"):
            continue
        names.append(row.name)
    return tuple(names)


def _build_articulation(env, spec, recipe, name_map, params, label):
    """Build one :class:`ArticulationRuntime` for a declared articulation.

    ``art.joint_names`` / ``art.feet`` / ``art.base_body`` are neutral
    ``unigym_name`` values; the compiled names live on the backend-private
    ``env._mj_articulation_bindings`` table.
    """
    ir = env._model_ir
    model = env._model
    art = ArticulationRuntime(
        name=spec.name,
        model=getattr(spec, "model", None),
        root=getattr(spec, "root", "free"),
        body_parts=dict(getattr(spec, "body_parts", {}) or {}),
        spawn=dict(getattr(spec, "spawn", {}) or {}),
        recipe=recipe,
    )
    if recipe is None:
        joints = tuple(str(j.name) for j in (getattr(spec, "joints", ()) or ()))
        init = {str(j.name): float(getattr(j, "init", 0.0))
                for j in (getattr(spec, "joints", ()) or ())}
        backend_joints = tuple(name_map.backend_name(spec.name, name)
                               for name in joints)
        art.joint_names = joints
        art.q0 = torch.tensor([init.get(j, 0.0) for j in joints],
                              dtype=torch.float32)
        art.robot_joint_names = joints
        art.feet = tuple((art.body_parts or {}).get("feet") or ())
        art.jadr_pol = np.asarray(qpos_addresses(ir, backend_joints),
                                  dtype=np.int64)
        art.jdr_pol = np.asarray(dof_addresses(ir, backend_joints),
                                 dtype=np.int64)
        art.jadr_art = art.jadr_pol
        art.jdr_art = art.jdr_pol
        art.q0_mj = np.asarray(art.q0, dtype=np.float64)
        art.free_qpos_adr = art.free_dof_adr = None
        art.foot_body_ids, art.foot_geom_sets = [], []
        env._mj_articulation_bindings[art.name] = MjArticulationBinding(
            unigym_joint_names=joints,
            backend_joint_names=backend_joints,
            unigym_base_body=str((art.body_parts or {}).get("base") or ""),
            backend_base_body=str(
                name_map.backend_body(
                    art.name, str((art.body_parts or {}).get("base") or ""))),
            unigym_feet=art.feet,
            backend_feet=tuple(
                name_map.backend_body(art.name, n) for n in art.feet),
        )
        return art

    robot_names = _asset_robot_names(env, spec, name_map)
    backend_names = tuple(name_map.backend_name(spec.name, name)
                          for name in robot_names)
    defaults = dict(getattr(recipe, "default_joint_pos", {}) or {})
    q0_mj = np.asarray([float(defaults.get(n, 0.0)) for n in robot_names],
                       dtype=np.float64)
    parts = dict(getattr(recipe, "body_parts", None) or {})
    base_body = str(parts.get("base") or "pelvis")
    backend_base = name_map.backend_body(spec.name, base_body)
    foot_names = [str(n) for n in (getattr(recipe, "foot_bodies", None) or ())]
    backend_feet = [name_map.backend_body(spec.name, n)
                    for n in foot_names]
    art.joint_names = tuple(robot_names)
    art.q0 = torch.tensor(q0_mj, dtype=torch.float32)
    art.robot_joint_names = tuple(robot_names)
    # v103 fingerprint parity: the reset write uses the float32-rounded q0
    # (the historical binding.q0), not the raw float64 recipe defaults.
    art.q0_mj = np.asarray(art.q0, dtype=np.float64)
    art.base_body = base_body
    art.feet = tuple(foot_names)
    art.jadr_pol = np.asarray(qpos_addresses(ir, backend_names),
                              dtype=np.int64)
    art.jdr_pol = np.asarray(dof_addresses(ir, backend_names),
                             dtype=np.int64)
    art.jadr_art = art.jadr_pol
    art.jdr_art = art.jdr_pol
    base_bid = body_id(ir, backend_base)
    if base_bid < 0:
        raise ValueError(
            f"{label}: articulation {spec.name!r} base body {backend_base!r} "
            f"not in the MJ model")
    art.addresses["base_body_id"] = base_bid
    art.free_qpos_adr, art.free_dof_adr = free_joint_adr(model, base_bid)
    art.foot_body_ids, art.foot_geom_sets = [], []
    for fname in backend_feet:
        bid = body_id(ir, fname)
        if bid < 0:
            raise ValueError(
                f"{label}: foot body {fname!r} of articulation "
                f"{spec.name!r} not in the MJ model")
        art.foot_body_ids.append(bid)
        art.foot_geom_sets.append(set(range(
            model.body_geomadr[bid],
            model.body_geomadr[bid] + model.body_geomnum[bid])))
    art.addresses["foot_body_ids"] = list(art.foot_body_ids)
    env._mj_articulation_bindings[art.name] = MjArticulationBinding(
        unigym_joint_names=tuple(robot_names),
        backend_joint_names=backend_names,
        unigym_base_body=base_body,
        backend_base_body=backend_base,
        unigym_feet=tuple(foot_names),
        backend_feet=tuple(backend_feet),
    )
    # spawn defaults mirror the historical single-robot reset write
    spawn = dict(art.spawn or {})
    if spawn.get("pos") is None:
        spawn["pos"] = [float(env._spawn_xy[0]), float(env._spawn_xy[1]),
                        float(env._spawn_height)]
    spawn.setdefault("yaw", float(env._spawn_yaw))
    art.spawn = spawn
    return art


def _add_prop_controls(env, controls: dict) -> None:
    """Represent articulated prop action faces as ControlRuntime entries."""
    if env.prop_backend is None:
        return
    for name, table in (env._prop_actuators or {}).items():
        if not table:
            # passive articulated props (all joints declared passive) expose
            # no actuated table: they are NOT control groups (v104: undriven
            # joints live in passive_joints, never a servo face)
            continue
        if name in controls:
            raise ValueError(
                f"{_label(env)}: prop {name!r} control face collides with a "
                f"declared control group")
        joints = tuple(str(j) for j in table)
        controls[name] = ControlRuntime(
            name=name,
            action_face=name,
            adapter="prop",
            articulation=next(iter(env.articulations), name),
            joint_names=joints,
            q0=tuple(0.0 for _ in joints),
            drive="position_servo",
            target="position",
            required=False,
            prop=True,
        )


def _body_control(env, art_name):
    for control in env._control_order:
        if (not control.prop and control.articulation == art_name
                and control.action_face == art_name):
            return control
    for control in env._control_order:
        if not control.prop and control.articulation == art_name:
            return control
    return None


def _bind_articulation_limits(env, art_order, model) -> None:
    """Per-articulation limits (asset order) + the record policy-order view."""
    import mujoco

    env._joint_limits_by_articulation = {}
    if not art_order:
        env._joint_limits = None
        return
    for art in art_order:
        body = _body_control(env, art.name)
        # the record/extra policy-order view is the body control's order (the
        # v103 ``jadr_pol`` semantics); the full asset order stays on
        # ``jadr_art`` / ``robot_joint_names`` for the obs catalog.
        if body is not None:
            art.jadr_pol = np.asarray(body.addresses["jadr"], dtype=np.int64)
            art.jdr_pol = np.asarray(body.addresses["jdr"], dtype=np.int64)
        if body is not None:
            binding = env._mj_bindings.get(body.name)
            names = (list(binding.backend_joint_names)
                     if binding is not None else list(body.joint_names))
        else:
            art_binding = env._mj_articulation_bindings.get(art.name)
            names = (list(art_binding.backend_joint_names)
                     if art_binding is not None else list(art.joint_names))
        jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                   for n in names]
        ranges = model.jnt_range[np.asarray(jnt_ids, dtype=np.int64)]
        env._joint_limits_by_articulation[art.name] = (
            torch.tensor(ranges[:, 0], dtype=torch.float32)
            .unsqueeze(0).expand(env.num_envs, -1),
            torch.tensor(ranges[:, 1], dtype=torch.float32)
            .unsqueeze(0).expand(env.num_envs, -1))
    first_limits = env._joint_limits_by_articulation.get(art_order[0].name)
    env._joint_limits = first_limits


def init_tables(env, model, params: dict, gains_fn, *, label=None) -> None:
    """Build ``env.articulations`` / ``env.controls`` + shared address tables."""
    label = label or _label(env)
    act_of_joint = actuator_by_joint(env._model_ir)
    env._act_of_joint = act_of_joint
    specs = tuple(getattr(env, "_articulation_specs", ()) or ())
    recipes = dict(getattr(env, "_recipes", {}) or {})
    name_map = env._mj_name_map
    env._mj_bindings = {}
    env._mj_articulation_bindings = {}
    art_order = []
    arts = {}
    for spec in specs:
        art = _build_articulation(env, spec, recipes.get(spec.name),
                                  name_map, params, label)
        arts[art.name] = art
        art_order.append(art)
    env.articulations = arts
    env._articulation_order = art_order

    controls = {}
    for control in tuple(getattr(env, "_planned_controls", ()) or ()):
        controls[control.name] = control
    # prop actuator bookkeeping stays in the prop backend (asset machinery)
    env._prop_actuators = (env.prop_backend.actuated_addresses()
                           if env.prop_backend else {})
    _add_prop_controls(env, controls)
    env._prop_ctrl_mask = np.zeros(model.nu, dtype=bool)
    for table in env._prop_actuators.values():
        env._prop_ctrl_mask[np.fromiter(table.values(), dtype=np.int64)] = True
    env._prop_targets = np.zeros((env.num_envs, model.nu), dtype=np.float64)
    for control in controls.values():
        if control.prop:
            table = env._prop_actuators.get(control.name) or {}
            control.addresses.update({
                "ctrl": np.asarray(list(table.values()), dtype=np.int64),
                "targets": env._prop_targets[:, list(table.values())],
            })
            continue
        art = arts.get(control.articulation)
        if art is None:
            raise ValueError(
                f"{label}: control {control.name!r} names articulation "
                f"{control.articulation!r}, which has no runtime")
        _control_addresses(env, control, art)
        bare = list(control.joint_names)
        binding = env._mj_bindings[control.name]
        _build_drive(env, control, bare, binding.backend_joint_names,
                     params, gains_fn, label)
    env.controls = controls
    env._control_order = [controls[g.name] for g in
                          (getattr(env, "_planned_controls", ()) or ())]
    env._control_order += [c for c in controls.values() if c.prop]

    _bind_articulation_limits(env, art_order, model)

    entries: dict = {}
    requestable: set = set()
    env.obs_catalogs = {}
    for art in art_order:
        art.catalog = articulation_catalog(art)
        env.obs_catalogs[art.name] = art.catalog
        entries.update(art.catalog.entries)
        requestable |= art.catalog.requestable
    env.obs_catalog = ObsCatalog(entries, requestable)


# --------------------------------------------------------------------------- #
# action routing + physics
# --------------------------------------------------------------------------- #

# v104.4: action routing/physics split into a sibling module (MI floor);
# re-exported here so every existing import face stays stable.
from .articulation_actions import (  # noqa: E402,F401
    apply_control_substep,
    reset_articulations,
    route_actions,
)
