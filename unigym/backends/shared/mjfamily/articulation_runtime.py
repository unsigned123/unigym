"""Articulation + control runtime tables (MuJoCo family).

Every engine iterates ``env.articulations`` and ``env.controls``; there is no
single/multi mode switch.  This module owns the shared, engine-agnostic half
(per-articulation name/address tables, the drive strategies, action routing,
reset, observation and termination keys) for the CPU ``MujocoEnv`` and the
``MjwarpEnv`` (both speak the same ``MjModel``/``MjData`` family).  Each
backend keeps only its engine-shaped step loop.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import torch

from unigym.actuators import ActuatorBase, build_actuator
from unigym.actuators.ir import (
    ResolvedActuationGroup,
    validate_actuation_group,
    validate_drive_law_matrix,
)
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
from unigym.runtime.controllers import (
    ArticulationRuntime,
    ControlRuntime,
)

__all__ = [
    "build_actuator_seam",
    "free_joint_adr",
    "init_tables",
    "reset_articulations",
    "route_actions",
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
            return (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
    return None, None


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
            f"asset joints: {list(art.joint_names)}"
        )
    backend_names = tuple(name_map.backend_name(art.name, name) for name in names)
    control.asset_joint_names = tuple(art.joint_names)
    jadr = np.asarray(qpos_addresses(ir, backend_names), dtype=np.int64)
    jdr = np.asarray(dof_addresses(ir, backend_names), dtype=np.int64)
    control.addresses.update({"jadr": jadr, "jdr": jdr})
    if control.prop:
        return
    act_of_joint = env._act_of_joint
    missing_act = [n for n in backend_names if n not in act_of_joint]
    if missing_act:
        raise ValueError(
            f"{_label(env)}: control {control.name!r} has no actuator for " f"joints {missing_act}"
        )
    act = np.asarray([act_of_joint[n] for n in backend_names], dtype=np.int64)
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


def _build_articulation(env, spec, profile, name_map, label):
    """Build one :class:`ArticulationRuntime` for a declared articulation.

    ``art.joint_names`` / ``art.foot_bodies`` / ``art.base_body_id`` are
    neutral ``unigym_name`` values; the compiled names live on the
    backend-private ``env._mj_articulation_bindings`` table.
    """
    ir = env._model_ir
    model = env._model
    art = ArticulationRuntime(
        name=spec.name,
        model=getattr(spec, "model", None),
        root=getattr(spec, "root", "free"),
        spawn=dict(getattr(spec, "spawn", {}) or {}),
        contacts=getattr(spec, "contacts", None),
        profile=profile,
    )
    if profile is None:
        joints = tuple(str(j.name) for j in (getattr(spec, "joints", ()) or ()))
        init = {
            str(j.name): float(getattr(j, "init", 0.0)) for j in (getattr(spec, "joints", ()) or ())
        }
        backend_joints = tuple(name_map.backend_name(spec.name, name) for name in joints)
        art.joint_names = joints
        art.q0 = torch.tensor([init.get(j, 0.0) for j in joints], dtype=torch.float32)
        art.robot_joint_names = joints
        base_body = str(getattr(spec, "base_body", None) or "")
        backend_base = str(name_map.backend_body(art.name, base_body)) if base_body else ""
        foot_names = tuple(str(n) for n in (getattr(spec, "foot_bodies", None) or ()))
        backend_feet = tuple(name_map.backend_body(art.name, n) for n in foot_names)
        art.base_body = base_body
        art.base_body_id = getattr(spec, "base_body", None)
        art.foot_bodies = foot_names
        art.body_sets = dict(getattr(spec, "body_sets", None) or {})
        art.jadr_pol = np.asarray(qpos_addresses(ir, backend_joints), dtype=np.int64)
        art.jdr_pol = np.asarray(dof_addresses(ir, backend_joints), dtype=np.int64)
        art.jadr_art = art.jadr_pol
        art.jdr_art = art.jdr_pol
        art.q0_mj = np.asarray(art.q0, dtype=np.float64)
        art.free_qpos_adr = art.free_dof_adr = None
        env._mj_articulation_bindings[art.name] = MjArticulationBinding(
            unigym_joint_names=joints,
            backend_joint_names=backend_joints,
            unigym_base_body=base_body,
            backend_base_body=backend_base,
            unigym_feet=foot_names,
            backend_feet=backend_feet,
        )
        return art

    robot_names = _asset_robot_names(env, spec, name_map)
    backend_names = tuple(name_map.backend_name(spec.name, name) for name in robot_names)
    defaults = dict(getattr(profile, "default_joint_pos", {}) or {})
    q0_mj = np.asarray([float(defaults.get(n, 0.0)) for n in robot_names], dtype=np.float64)
    base_body_id = getattr(profile, "base_body", None)
    if base_body_id is None:
        base_body = ""
        backend_base = ""
    else:
        base_body = str(base_body_id)
        native_base = str(profile.body_bindings[base_body_id].names["mjcf"])
        backend_base = name_map.backend_body(spec.name, native_base)
    foot_names = tuple(str(n) for n in (getattr(profile, "foot_bodies", None) or ()))
    backend_feet = []
    for name in foot_names:
        native = str(profile.body_bindings[name].names["mjcf"])
        backend_feet.append(name_map.backend_body(spec.name, native))
    art.joint_names = tuple(robot_names)
    art.q0 = torch.tensor(q0_mj, dtype=torch.float32)
    art.robot_joint_names = tuple(robot_names)
    # Fingerprint parity: the reset write uses the float32-rounded q0
    # (the historical binding.q0), not the raw float64 profile defaults.
    art.q0_mj = np.asarray(art.q0, dtype=np.float64)
    art.base_body = backend_base
    art.base_body_id = base_body_id
    art.foot_bodies = foot_names
    art.body_sets = dict(getattr(profile, "body_sets", {}) or {})
    art.jadr_pol = np.asarray(qpos_addresses(ir, backend_names), dtype=np.int64)
    art.jdr_pol = np.asarray(dof_addresses(ir, backend_names), dtype=np.int64)
    art.jadr_art = art.jadr_pol
    art.jdr_art = art.jdr_pol
    base_bid = body_id(ir, backend_base) if backend_base else -1
    if backend_base and base_bid < 0:
        raise ValueError(
            f"{label}: articulation {spec.name!r} base body {backend_base!r} "
            f"not in the MJ model"
        )
    art.addresses["base_body_id"] = base_bid
    art.free_qpos_adr, art.free_dof_adr = free_joint_adr(model, base_bid)
    env._mj_articulation_bindings[art.name] = MjArticulationBinding(
        unigym_joint_names=tuple(robot_names),
        backend_joint_names=backend_names,
        unigym_base_body=base_body,
        backend_base_body=backend_base,
        unigym_feet=foot_names,
        backend_feet=tuple(backend_feet),
    )
    # the declared spawn mapping is stored verbatim (empty when absent);
    # env.params backend defaults are applied by the reset path only.
    art.spawn = dict(getattr(spec, "spawn", {}) or {})
    return art


def _add_prop_controls(env, controls: dict) -> None:
    """Represent articulated prop action faces as ControlRuntime entries."""
    if env.prop_backend is None:
        return
    for name, table in (env._prop_actuators or {}).items():
        if not table:
            # passive articulated props (all joints declared passive) expose
            # no actuated table: they are NOT control groups (undriven
            # joints live in passive_joints, never a servo face)
            continue
        if name in controls:
            raise ValueError(
                f"{_label(env)}: prop {name!r} control face collides with a "
                f"declared control group"
            )
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
        if (
            not control.prop
            and control.articulation == art_name
            and control.action_face == art_name
        ):
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
        # ``jadr_pol`` semantics); the full asset order stays on
        # ``jadr_art`` / ``robot_joint_names`` for the obs catalog.
        if body is not None:
            art.jadr_pol = np.asarray(body.addresses["jadr"], dtype=np.int64)
            art.jdr_pol = np.asarray(body.addresses["jdr"], dtype=np.int64)
        if body is not None:
            binding = env._mj_bindings.get(body.name)
            names = (
                list(binding.backend_joint_names) if binding is not None else list(body.joint_names)
            )
        else:
            art_binding = env._mj_articulation_bindings.get(art.name)
            names = (
                list(art_binding.backend_joint_names)
                if art_binding is not None
                else list(art.joint_names)
            )
        jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        ranges = model.jnt_range[np.asarray(jnt_ids, dtype=np.int64)]
        env._joint_limits_by_articulation[art.name] = (
            torch.tensor(ranges[:, 0], dtype=torch.float32).unsqueeze(0).expand(env.num_envs, -1),
            torch.tensor(ranges[:, 1], dtype=torch.float32).unsqueeze(0).expand(env.num_envs, -1),
        )
    first_limits = env._joint_limits_by_articulation.get(art_order[0].name)
    env._joint_limits = first_limits


def init_tables(env, model, *, label=None) -> None:
    """Build ``env.articulations`` / ``env.controls`` + shared address tables."""
    label = label or _label(env)
    act_of_joint = actuator_by_joint(env._model_ir)
    env._act_of_joint = act_of_joint
    specs = tuple(getattr(env, "_articulation_specs", ()) or ())
    profiles = dict(getattr(env, "_profiles", {}) or {})
    name_map = env._mj_name_map
    env._mj_bindings = {}
    env._mj_articulation_bindings = {}
    art_order = []
    arts = {}
    for spec in specs:
        art = _build_articulation(env, spec, profiles.get(spec.name), name_map, label)
        arts[art.name] = art
        art_order.append(art)
    env.articulations = arts
    env._articulation_order = art_order

    controls = {}
    for control in tuple(getattr(env, "_planned_controls", ()) or ()):
        controls[control.name] = control
    # prop actuator bookkeeping stays in the prop backend (asset machinery)
    env._prop_actuators = env.prop_backend.actuated_addresses() if env.prop_backend else {}
    _add_prop_controls(env, controls)
    env._prop_ctrl_mask = np.zeros(model.nu, dtype=bool)
    for table in env._prop_actuators.values():
        env._prop_ctrl_mask[np.fromiter(table.values(), dtype=np.int64)] = True
    env._prop_targets = np.zeros((env.num_envs, model.nu), dtype=np.float64)
    for control in controls.values():
        if control.prop:
            table = env._prop_actuators.get(control.name) or {}
            control.addresses.update(
                {
                    "ctrl": np.asarray(list(table.values()), dtype=np.int64),
                    "targets": env._prop_targets[:, list(table.values())],
                }
            )
            continue
        art = arts.get(control.articulation)
        if art is None:
            raise ValueError(
                f"{label}: control {control.name!r} names articulation "
                f"{control.articulation!r}, which has no runtime"
            )
        _control_addresses(env, control, art)
    env.controls = controls
    env._control_order = [controls[g.name] for g in (getattr(env, "_planned_controls", ()) or ())]
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
# actuator seam
# --------------------------------------------------------------------------- #
class _ActuatorTargetFeed:
    """Compose one actuator's joint-ordered target vector from controls."""

    def __init__(self, act: ActuatorBase, sources) -> None:
        self.act = act
        self._sources = list(sources)

    def __call__(self, targets: dict, e: int) -> np.ndarray:
        out = np.zeros(len(self._sources), dtype=np.float64)
        for index, (control_name, column) in enumerate(self._sources):
            out[index] = targets[control_name][e, column]
        return out


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
    if bias == int(mujoco.mjtBias.mjBIAS_NONE) and gain == int(mujoco.mjtGain.mjGAIN_FIXED):
        return "motor"
    if bias == int(mujoco.mjtBias.mjBIAS_AFFINE) and gain == int(mujoco.mjtGain.mjGAIN_FIXED):
        biasprm = model.actuator_biasprm[slot]
        if float(biasprm[1]) != 0.0:
            return "position_servo"
        if float(biasprm[2]) != 0.0:
            return "velocity_servo"
    return "general"


def _translate_gain_value(
    value,
    neutral_names: Sequence[str],
    compiled_names: Sequence[str],
    what: str,
):
    """Translate a scalar or joint-keyed gain into compiled-name form."""
    if not isinstance(value, Mapping):
        return float(value)
    out: dict[str, float] = {}
    for key, gain in value.items():
        name = str(key)
        if name in neutral_names:
            out[compiled_names[neutral_names.index(name)]] = float(gain)
            continue
        matches = [i for i, joint in enumerate(neutral_names) if re.fullmatch(name, joint)]
        if not matches:
            raise ValueError(
                f"{what} key {name!r} matches none of the group joints "
                f"{list(neutral_names)} | hint: keys are joint names or regexes "
                f"fullmatching them"
            )
        for index in matches:
            out[compiled_names[index]] = float(gain)
    return out


def _translate_parts(
    parts: Sequence[str],
    neutral_names: Sequence[str],
    compiled_names: Sequence[str],
    what: str,
) -> list[str]:
    """Turn each parts regex into one alternation regex over compiled names."""
    translated: list[str] = []
    for expr in parts:
        matched = [
            compiled_names[index]
            for index, joint in enumerate(neutral_names)
            if re.fullmatch(str(expr), joint)
        ]
        if not matched:
            raise ValueError(
                f"{what} part {expr!r} matches none of the group joints " f"{list(neutral_names)}"
            )
        translated.append("(?:" + "|".join(re.escape(name) for name in matched) + ")")
    return translated


def _law_config(
    resolved: ResolvedActuationGroup,
    compiled_names: Sequence[str],
) -> dict:
    """Build the law config dict consumed by :func:`build_actuator`."""
    group = resolved.group
    cfg: dict = {"type": group.law, "joint_names": list(compiled_names)}
    neutral_names = resolved.joint_names
    for key in ("kp", "kd"):
        if key in group.gains:
            cfg[key] = _translate_gain_value(
                group.gains[key],
                neutral_names,
                compiled_names,
                f"actuation group {group.name!r} gain {key!r}",
            )
    if group.parts:
        cfg["parts"] = _translate_parts(
            group.parts,
            neutral_names,
            compiled_names,
            f"actuation group {group.name!r}",
        )
    return cfg


def build_actuator_seam(
    env,
    plans: Mapping[str, Sequence[ResolvedActuationGroup]] | None,
    *,
    backend: str,
    label: str | None = None,
) -> tuple[list[ActuatorBase], list[_ActuatorTargetFeed], bool]:
    """Build the always-active ctrl seam from the resolved actuation plans.

    The run layer owns exactly-once coverage; this seam owns name translation
    (neutral -> compiled), law payload validation, slot uniqueness and the
    native_servo compiled-plant check.  A direct env with driven controls but
    no plan fails fast instead of silently leaving the world unactuated.
    """
    label = label or _label(env)
    plans = dict(plans or {})
    # The env is a duck-typed backend; this boundary cast states the table
    # type its init_tables owner already established.
    art_order = cast(
        "tuple[ArticulationRuntime, ...]",
        tuple(getattr(env, "_articulation_order", ()) or ()),
    )
    controls: list[ControlRuntime] = [c for c in getattr(env, "_control_order", ()) if not c.prop]
    by_articulation: dict[str, list[ControlRuntime]] = {}
    for control in controls:
        by_articulation.setdefault(control.articulation, []).append(control)

    resolved_groups: list[ResolvedActuationGroup] = []
    group_rows: list[tuple[ResolvedActuationGroup, Sequence[str]]] = []
    for art in art_order:
        art_controls = by_articulation.get(art.name, [])
        if not art_controls:
            continue
        groups = tuple(plans.get(art.name, ()) or ())
        if not groups:
            raise ValueError(
                f"{label}: articulation {art.name!r} has driven joints but no "
                f"actuation plan | hint: declare "
                f"ArticulationSpec.actuation groups"
            )
        for resolved in groups:
            validate_actuation_group(resolved.group, backend=backend)
            compiled = tuple(
                env._mj_name_map.backend_name(art.name, joint) for joint in resolved.joint_names
            )
            resolved_groups.append(resolved)
            group_rows.append((resolved, compiled))
    validate_drive_law_matrix(resolved_groups, controls)

    compiled_universe: list[str] = []
    for art in art_order:
        binding = env._mj_articulation_bindings.get(art.name)
        if binding is not None:
            compiled_universe.extend(str(name) for name in binding.backend_joint_names)
            continue
        neutral_joints: tuple[str, ...] = art.joint_names
        for joint in neutral_joints:
            compiled_universe.append(str(env._mj_name_map.backend_name(art.name, joint)))
    env._compiled_joint_names = tuple(dict.fromkeys(compiled_universe))

    owner: dict[tuple[str, str], tuple[str, int]] = {}
    for control in controls:
        for index, joint in enumerate(control.joint_names):
            owner[(str(control.articulation), str(joint))] = (control.name, index)

    acts: list[ActuatorBase] = []
    feeds: list[_ActuatorTargetFeed] = []
    claimed: dict[int, str] = {}
    prop_slots = set(np.nonzero(env._prop_ctrl_mask)[0].tolist())
    for resolved, compiled in group_rows:
        group = resolved.group
        cfg = _law_config(resolved, compiled)
        act = build_actuator(cfg)
        act.bind(env, cfg)
        slots = act.bound_slots.tolist()
        duplicate = sorted(set(slots) & set(claimed))
        if duplicate:
            owners = sorted({claimed[slot] for slot in duplicate})
            raise ValueError(
                f"{label}: actuation group {group.name!r} re-addresses ctrl "
                f"slots {duplicate} already owned by group(s) {owners} (each "
                f"actuator slot must have exactly one ctrl law)"
            )
        prop_hit = sorted(set(slots) & prop_slots)
        if prop_hit:
            raise ValueError(
                f"{label}: actuation group {group.name!r} claims prop-servo "
                f"ctrl slots {prop_hit} (props own their position-hold write "
                f"face; actuation groups address ROBOT joints only)"
            )
        for slot in slots:
            claimed[slot] = group.name
        if group.law == "native_servo":
            for joint, slot in zip(act.joint_names, slots):
                compiled_law = _actuator_law(env._model, int(slot))
                if compiled_law != "position_servo":
                    raise ValueError(
                        f"{label}: articulation {resolved.articulation!r} "
                        f"actuation group {group.name!r} joint {joint!r} law "
                        f"'native_servo' requires a compiled position_servo; "
                        f"compiled law is {compiled_law!r}"
                    )
        sources = [owner[(resolved.articulation, str(joint))] for joint in resolved.joint_names]
        acts.append(act)
        feeds.append(_ActuatorTargetFeed(act, sources))
    phase_split = any(resolved.group.law != "standard_pd" for resolved in resolved_groups)
    return acts, feeds, phase_split


# --------------------------------------------------------------------------- #
# action routing + physics
# --------------------------------------------------------------------------- #

# action routing/physics split into a sibling module (MI floor);
# re-exported here so every existing import face stays stable.
from .articulation_actions import (  # noqa: E402,F401
    reset_articulations,
    route_actions,
)
