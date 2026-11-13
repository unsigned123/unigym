"""IsaacLab 2.x articulation/control runtime.

Every engine iterates ``env.articulations`` and ``env.controls``.  This
module owns the IsaacLab-side runtime tables: one scene field per
articulation (``robot`` / ``robot_<name>``), one action term per control
face, the per-articulation address/catalog tables, action routing, reset and
the namespaced observation/termination records.

There is no policy runtime type: a policy is a ``position_pd`` control plus
an adapter.
"""

from __future__ import annotations

import math

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions import JointPositionActionCfg

from unigym.backends.isaaclab2.naming import (
    articulation_prim_leaf,
    articulation_scene_field,
)
from unigym.backends.isaaclab2.robots import articulation_cfg
from unigym.env.naming import ENV_REGEX_NS
from unigym.env.obs_catalog import ObsCatalog, ObsSpec, build_obs_catalog
from unigym.runtime.controllers import ArticulationRuntime, ControlRuntime

__all__ = [
    "bind_articulations",
    "build_extra",
    "configure_scene",
    "refresh_obs_cache",
    "reset_articulations",
    "route_actions",
    "sync_articulation_controls",
    "validate_options",
]

_LABEL = "isaaclab2 backend"


def validate_options(
    adapter,
    *,
    events_cfg,
    recorder,
    prop_manager,
    camera_manager,
    contact_manager,
    director_cfg,
    init_pose_cfg,
) -> None:
    """Reject the partition options this MVP does not implement.

    Contact groups ARE supported with multiple articulations (each group
    owns one articulation); the remaining options below stay rejected.
    """
    requests = adapter.get_obs_requests()
    if requests:
        raise NotImplementedError(
            f"{_LABEL}: adapter obs requests are not supported with "
            f"multiple articulations; every articulation is published "
            f"under articulation/<name>/... (use an external/manual driver) "
            f"| requested: {sorted({r.key for r in requests})}"
        )
    for option, value in (
        ("prop_manager", prop_manager),
        ("camera_manager", camera_manager),
        ("director_cfg", director_cfg),
        ("events_cfg", events_cfg),
        ("init_pose", init_pose_cfg),
    ):
        if value is not None:
            raise NotImplementedError(
                f"{_LABEL}: {option} is not supported with multiple "
                f"articulations (MVP: one env, external/manual driver)"
            )
    del recorder


def _initial_state(init, pos, rot):
    """Clone an articulation ``InitialStateCfg`` with a new root pose."""
    joint_pos = dict(getattr(init, "joint_pos", None) or {})
    joint_vel = dict(getattr(init, "joint_vel", None) or {})
    return type(init)(
        pos=tuple(float(v) for v in pos),
        rot=tuple(float(v) for v in rot),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )


def configure_scene(cfg, controls, articulations, profiles, robot_realizations) -> dict:
    """Add per-articulation scene fields + one action term per control."""
    if len(articulations) < 2:
        raise ValueError(
            f"{_LABEL}: partitioned scene needs >= 2 articulations, got " f"{len(articulations)}"
        )
    controls = tuple(controls or ())
    fields: dict[str, str] = {}
    for index, spec in enumerate(articulations):
        if profiles.get(spec.name) is None:
            raise ValueError(f"{_LABEL}: articulation {spec.name!r} has no profile")
        asset_cfg = articulation_cfg(robot_realizations, spec.name)
        field = articulation_scene_field(spec.name, index)
        leaf = articulation_prim_leaf(spec.name, index)
        prim_path = f"{ENV_REGEX_NS}/{leaf}"
        init = asset_cfg.init_state
        base_pos = tuple(float(v) for v in (getattr(init, "pos", None) or (0.0, 0.0, 0.0)))
        spawn = dict(getattr(spec, "spawn", {}) or {})
        spawn_pos = spawn.get("pos")
        if spawn_pos is not None:
            pos = tuple(float(v) for v in spawn_pos)
            if len(pos) != 3:
                raise ValueError(
                    f"{_LABEL}: articulation {spec.name!r} spawn.pos "
                    f"must be a 3-vector, got {spawn_pos!r}"
                )
        else:
            # spawn is the sole robot-root authority: absent -> the
            # realizer/cfg init state, never an invented index offset.
            pos = base_pos
        rot = tuple(getattr(init, "rot", (1.0, 0.0, 0.0, 0.0)))
        if spawn.get("yaw") is not None:
            yaw = float(spawn["yaw"])
            rot = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        new_cfg = asset_cfg.replace(
            prim_path=prim_path,
            init_state=_initial_state(init, pos, rot),
        )
        setattr(cfg.scene, field, new_cfg)
        fields[spec.name] = field

    for control in controls:
        if control.prop:
            continue
        if control.articulation not in fields:
            continue
        face = control.action_face
        if not str(face).isidentifier():
            raise ValueError(
                f"{_LABEL}: action face {face!r} of articulation "
                f"{control.articulation!r} is not a valid action-term name"
            )
        if hasattr(cfg.actions, face):
            raise ValueError(
                f"{_LABEL}: action entry {face!r} collides with an " f"existing action term name"
            )
        setattr(
            cfg.actions,
            face,
            JointPositionActionCfg(
                asset_name=fields[control.articulation],
                joint_names=list(control.joint_names),
                scale=1.0,
                use_default_offset=False,
                preserve_order=True,
            ),
        )
    return fields


def _articulation_runtime(env, spec, profile, asset) -> ArticulationRuntime:
    asset_names = tuple(asset.joint_names)
    defaults = dict(getattr(profile, "default_joint_pos", {}) or {})
    q0_art = torch.tensor(
        [float(defaults.get(n, 0.0)) for n in asset_names], dtype=torch.float32, device=env.device
    )
    base_body_id = getattr(profile, "base_body", None)
    base_body = ""
    if base_body_id is not None:
        base_body = str(profile.body_bindings[base_body_id].names["usd"])
    art = ArticulationRuntime(
        name=spec.name,
        model=getattr(spec, "model", None),
        root=getattr(spec, "root", "free"),
        joint_names=asset_names,
        q0=q0_art.detach().cpu().numpy(),
        foot_bodies=tuple(getattr(profile, "foot_bodies", None) or ()),
        spawn=dict(getattr(spec, "spawn", {}) or {}),
        base_body=base_body,
        base_body_id=base_body_id,
        body_sets=dict(getattr(profile, "body_sets", {}) or {}),
        contacts=getattr(spec, "contacts", None),
        profile=profile,
    )
    art.asset = asset
    art.q0_art = q0_art
    art.catalog = build_obs_catalog(
        profile, articulation_joint_names=list(asset_names), articulation=spec.name
    )
    return art


def _bind_scene_fields(env, specs, fields, partitioned) -> str:
    env.robots = {}
    for index, spec in enumerate(specs):
        field = fields.get(spec.name, "robot" if index == 0 else None)
        if field is None:
            raise ValueError(f"{_LABEL}: articulation {spec.name!r} has no scene field")
        env.robots[spec.name] = env.scene[field]
    first = specs[0].name
    env.robot = env.robots[first]
    return first


def _bind_term_resolvers(env) -> None:
    from unigym.runtime.actions import IsaaclabJointResolver

    env.action_entry_names = list(env.action_manager.active_terms)
    env._term_resolvers = {}
    for name in env.action_entry_names:
        term = env.action_manager.get_term(name)
        joint_names = getattr(getattr(term, "cfg", None), "joint_names", None)
        if not joint_names:
            raise ValueError(
                f"action term {name!r} has no explicit joint_names; the "
                f"NamedAction contract requires preserve_order=True terms "
                f"with explicit name lists"
            )
        env._term_resolvers[name] = IsaaclabJointResolver(joint_names)


def _bind_control_runtimes(env, specs, profiles, partitioned, first) -> dict:
    env.articulations = {}
    for spec in specs:
        env.articulations[spec.name] = _articulation_runtime(
            env, spec, profiles[spec.name], env.robots[spec.name]
        )
    controls: dict[str, ControlRuntime] = {}
    for control in tuple(env._planned_controls or ()):
        if not control.prop and control.drive not in ("position_pd", "position_servo"):
            raise NotImplementedError(
                f"{_LABEL}: articulation {control.articulation!r} control "
                f"{control.name!r} requests drive {control.drive!r}; the "
                f"isaaclab backend has no native direct-drive actuator "
                f"mapping and implements only position joint targets "
                f"(position_pd / position_servo) | hint: use a backend that "
                f"supports direct drive"
            )
        art = env.articulations.get(control.articulation)
        if art is None:
            raise ValueError(
                f"{_LABEL}: control {control.name!r} names articulation "
                f"{control.articulation!r}, which has no runtime"
            )
        control.asset_joint_names = tuple(art.joint_names)
        control.addresses["term"] = control.action_face
        index = {n: i for i, n in enumerate(art.joint_names)}
        control.addresses["perm_il2pol"] = [index[n] for n in control.joint_names if n in index]
        controls[control.name] = control
    # every remaining action term (articulated props) becomes a servo control
    for term in env.action_entry_names:
        if term in controls:
            continue
        joint_names = tuple(getattr(env.action_manager.get_term(term).cfg, "joint_names", ()))
        controls[term] = ControlRuntime(
            name=term,
            action_face=term,
            adapter="prop",
            articulation=first,
            joint_names=joint_names,
            q0=tuple(0.0 for _ in joint_names),
            drive="position_servo",
            target="position",
            required=True,
            prop=True,
        )
    return controls


def _bind_control_order(env, controls) -> None:
    env.controls = controls
    env._control_order = [controls[n] for n in env.action_entry_names if n in controls]
    for art in env.articulations.values():
        body = next(
            (
                c
                for c in env._control_order
                if c.articulation == art.name and c.action_face == art.name
            ),
            None,
        )
        perm = (
            body.addresses["perm_il2pol"] if body is not None else list(range(len(art.joint_names)))
        )
        art.addresses["perm_il2pol"] = list(perm)
        art.perm_il2pol = torch.tensor(list(perm), dtype=torch.int64, device=env.device)
    env._joint_limits_by_articulation = {}
    for art in env.articulations.values():
        lim = getattr(art.asset.data, "joint_pos_limits", None)
        if lim is None:
            env._joint_limits_by_articulation[art.name] = None
            continue
        lim = torch.as_tensor(lim)
        perm = art.perm_il2pol
        env._joint_limits_by_articulation[art.name] = (lim[:, perm, 0], lim[:, perm, 1])


def _bind_obs_catalogs(env) -> None:
    entries: dict[str, ObsSpec] = {}
    requestable: set[str] = set()
    env.obs_catalogs = {}
    for art in env.articulations.values():
        entries.update(art.catalog.entries)
        requestable |= art.catalog.requestable
        env.obs_catalogs[art.name] = art.catalog
    env.obs_catalog = ObsCatalog(entries, requestable)


def bind_articulations(env) -> None:
    """Bind every articulation/control runtime after ``super().__init__``."""
    specs = tuple(env._articulation_specs)
    profiles = dict(env._profiles)
    fields = dict(env._articulation_fields or {})
    partitioned = len(specs) > 1
    first = _bind_scene_fields(env, specs, fields, partitioned)
    _bind_term_resolvers(env)
    controls = _bind_control_runtimes(env, specs, profiles, partitioned, first)
    _bind_control_order(env, controls)

    first_art = env.articulations[first]
    env._perm_il2pol = first_art.perm_il2pol
    env._perm_pol2il = torch.argsort(env._perm_il2pol)
    env._q0_art = first_art.q0_art
    env._joint_gathers = {}
    _bind_obs_catalogs(env)


def sync_articulation_controls(env, articulation: str) -> None:
    """Fill the first control's address table for the flat compatibility face."""
    art = env.articulations[articulation]
    index = {n: i for i, n in enumerate(art.joint_names)}
    art.addresses["perm_il2pol"] = [index[n] for n in art.joint_names if n in index]


def route_actions(env, actions: dict) -> torch.Tensor:
    """Resolve every control's action face onto the flat action-term input."""
    if not isinstance(actions, dict):
        raise TypeError(
            f"{_LABEL} step expects dict[str, NamedAction], got " f"{type(actions).__name__}"
        )
    accepted = [c.action_face for c in env._control_order]
    unknown = sorted(set(actions) - set(accepted))
    if unknown:
        raise ValueError(
            f"{_LABEL}: action entries {unknown} match no control face; "
            f"accepted action faces: {accepted}"
        )
    missing = [
        c.action_face for c in env._control_order if c.required and c.action_face not in actions
    ]
    if missing:
        raise ValueError(
            f"{_LABEL}: action entries leave controls unaddressed: "
            f"{missing}; accepted action faces: {accepted}"
        )
    n = env.num_envs
    chunks = []
    for control in env._control_order:
        face = control.action_face
        term = env.action_manager.get_term(face)
        dim = int(term.action_dim)
        na = actions.get(face)
        if na is None:
            buf = torch.zeros(n, dim, dtype=torch.float32, device=env.device)
            chunks.append(buf)
            continue
        if na.kind != control.target:
            raise ValueError(
                f"{_LABEL}: action entry {face!r} has kind {na.kind!r} but "
                f"control {control.name!r} expects target {control.target!r}"
            )
        if na.num_envs != n:
            raise ValueError(f"action entry {face!r} has {na.num_envs} env rows, " f"expected {n}")
        if na.dim != dim:
            raise ValueError(
                f"action entry {face!r} addresses {na.dim} joints but "
                f"term {face!r} covers {dim}"
            )
        gather = env._term_resolvers[face].gather(na.joint_names).to(env.device)
        buf = torch.zeros(n, dim, dtype=torch.float32, device=env.device)
        buf[:, gather] = na.values.to(device=env.device, dtype=torch.float32)
        chunks.append(buf)
    return torch.cat(chunks, dim=-1)


def _articulation_extra(env, art) -> dict:
    d = art.asset.data
    prefix = f"articulation/{art.name}"
    return {
        f"{prefix}/joint_pos": d.joint_pos[:, art.perm_il2pol].clone(),
        f"{prefix}/joint_vel": d.joint_vel[:, art.perm_il2pol].clone(),
        f"{prefix}/applied_torque": d.applied_torque[:, art.perm_il2pol].clone(),
        f"{prefix}/gravity_b": d.projected_gravity_b.clone(),
        f"{prefix}/base_pos_w": d.root_pos_w.clone(),
        f"{prefix}/base_quat_w": d.root_quat_w.clone(),
        f"{prefix}/base_lin_vel_w": d.root_lin_vel_w.clone(),
        f"{prefix}/base_ang_vel_w": d.root_ang_vel_w.clone(),
        f"{prefix}/base_lin_vel_b": d.root_lin_vel_b.clone(),
        f"{prefix}/base_ang_vel_b": d.root_ang_vel_b.clone(),
    }


def build_extra(env) -> dict:
    """Global bookkeeping + every articulation's namespaced state."""
    extra = {
        "time_in_episode": env._time_in_episode.clone(),
        "episode_id": env._episode_id.clone(),
    }
    for art in env.articulations.values():
        extra.update(_articulation_extra(env, art))
    return extra


def refresh_obs_cache(env, raw_action=None, cmd=None) -> None:
    """Publish every articulation under ``articulation/<name>/...``."""
    del raw_action, cmd
    for art in env.articulations.values():
        d = art.asset.data
        prefix = f"articulation/{art.name}"
        env.obs_cache[f"{prefix}/joint_pos_rel"] = d.joint_pos - art.q0_art
        env.obs_cache[f"{prefix}/joint_vel"] = d.joint_vel.clone()
        env.obs_cache[f"{prefix}/applied_torque"] = d.applied_torque.clone()
        env.obs_cache[f"{prefix}/gravity_b"] = d.projected_gravity_b.clone()
        env.obs_cache[f"{prefix}/base_pos_w"] = d.root_pos_w.clone()
        env.obs_cache[f"{prefix}/base_quat_w"] = d.root_quat_w.clone()
        env.obs_cache[f"{prefix}/base_ang_vel_b"] = d.root_ang_vel_b.clone()
        env.obs_cache[f"{prefix}/base_yaw"] = math_utils.euler_xyz_from_quat(d.root_quat_w)[
            2
        ].unsqueeze(-1)


def reset_articulations(env, env_ids) -> None:
    """Write every articulation's spawn root pose + q0 (declaration order)."""
    for art in env.articulations.values():
        robot = art.asset
        n = env_ids.numel()
        dr = robot.data.default_root_state[env_ids].clone()
        spawn = dict(art.spawn or {})
        if spawn.get("pos") is not None:
            dr[:, :3] = torch.tensor(spawn["pos"], dtype=dr.dtype, device=dr.device)
        if spawn.get("yaw") is not None:
            yaw = float(spawn["yaw"])
            dr[:, 3:7] = torch.tensor(
                [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                dtype=dr.dtype,
                device=dr.device,
            )
        dj = art.q0_art.unsqueeze(0).expand(n, -1).clone()
        robot.write_root_pose_to_sim(root_pose=dr[:, :7], env_ids=env_ids)
        robot.write_root_velocity_to_sim(root_velocity=dr[:, 7:13], env_ids=env_ids)
        robot.write_joint_state_to_sim(position=dj, velocity=torch.zeros_like(dj), env_ids=env_ids)
        robot.data.root_state_w[env_ids] = dr
