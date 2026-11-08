"""Action routing + physics application for the MuJoCo-family backends.

Split out of :mod:`articulation_runtime` (v104.4) to keep each module small
and maintainable.  Every engine still iterates the neutral
``env._control_order`` tables built there.
"""
from __future__ import annotations

import numpy as np
import torch


def _label(env) -> str:
    """Backend label for teaching errors (defaults to the CPU mujoco lane)."""
    return getattr(env, "_backend_label", "mujoco backend")


def _initial_targets(env, order) -> dict:
    targets = {}
    for control in order:
        if control.prop:
            slots = control.addresses.get("ctrl")
            targets[control.name] = (
                np.zeros((env.num_envs, len(control.joint_names)),
                         dtype=np.float64) if slots is None
                else env._prop_targets[:, slots].copy())
        else:
            targets[control.name] = np.zeros(
                (env.num_envs, len(control.joint_names)), dtype=np.float64)
    return targets


def _locate_control(order, by_face, by_name, key, entry):
    """Resolve one action entry to its control (face, name, joint-set)."""
    control = by_face.get(key) or by_name.get(key)
    if control is not None:
        return control
    for candidate in order:
        if tuple(entry.joint_names) == candidate.joint_names:
            return candidate
    return None


def _apply_entry(env, control, key, entry, targets, covered, label) -> None:
    """Scatter one NamedAction entry into its control target row."""
    names = list(entry.joint_names or ())
    index = {n: i for i, n in enumerate(control.joint_names)}
    unknown = sorted(set(names) - set(index))
    if unknown:
        raise ValueError(
            f"{label}: action entry {key!r} names joints {unknown} that "
            f"control {control.name!r} does not own | owns: "
            f"{list(control.joint_names)}")
    row = entry.values.detach().to("cpu").to(torch.float32).numpy()
    if row.shape[-1] != len(names):
        raise ValueError(
            f"{label}: action entry {key!r} carries {row.shape[-1]} "
            f"values for {len(names)} joints")
    if control.prop:
        resolved = env._action_resolver.resolve(names, prop=control.name)
        env._prop_targets[:, resolved.prop_adr] = row
        targets[control.name] = env._prop_targets[
            :, control.addresses["ctrl"]].copy()
        return
    cols = np.asarray([index[n] for n in names], dtype=np.int64)
    if covered[control.name][cols].any():
        raise ValueError(
            f"{label}: action entry {key!r} re-addresses joints of "
            f"control {control.name!r}")
    targets[control.name][:, cols] = row
    covered[control.name][cols] = True


def route_actions(env, actions: dict, label=None) -> dict:
    """Resolve every action face onto its control's target array."""
    label = label or _label(env)
    if not isinstance(actions, dict):
        raise TypeError(
            f"{label} step expects dict[str, NamedAction], got "
            f"{type(actions).__name__}")
    order = list(getattr(env, "_control_order", env.controls.values()))
    targets = _initial_targets(env, order)
    covered = {c.name: np.zeros(len(c.joint_names), dtype=bool)
               for c in order if not c.prop}
    by_face = {c.action_face: c for c in order}
    by_name = {c.name: c for c in order}
    seen = set()
    accepted = sorted(by_face)
    for key, entry in actions.items():
        control = _locate_control(order, by_face, by_name, key, entry)
        if control is None:
            raise ValueError(
                f"{label}: action entry {key!r} is not a declared control "
                f"face; accepted faces: {accepted}")
        if control.name in seen:
            raise ValueError(
                f"{label}: control {control.name!r} is addressed by more than "
                f"one action entry")
        seen.add(control.name)
        _apply_entry(env, control, key, entry, targets, covered, label)
    for control in env._control_order:
        if control.prop or not control.required:
            continue
        if not bool(covered[control.name].all()):
            left = [n for n, hit in zip(control.joint_names,
                                        covered[control.name]) if not hit]
            raise ValueError(
                f"{label}: action entry for {control.action_face!r} leaves "
                f"joints unaddressed: {left}")
    return targets


def apply_control_substep(env, targets: dict, e: int, d) -> None:
    """Apply every control strategy once for one sim substep on env *e*."""
    for control in env._control_order:
        target = targets[control.name][e]
        if control.drive == "position_pd":
            control.pd.set_targets(target)
            control.pd.step(env._model, d)
        elif control.drive in ("position_servo", "torque", "effort",
                               "velocity"):
            slots = control.addresses.get("ctrl")
            if slots is None:
                slots = control.addresses["act"]
            d.ctrl[slots] = target
        else:  # pragma: no cover - construction rejects unsupported drives
            raise NotImplementedError(
                f"{_label(env)}: control {control.name!r} drive "
                f"{control.drive!r} has no runtime strategy")


def reset_articulations(env, d) -> None:
    """Write every articulation's root spawn + q0 (declaration order)."""
    for art in env._articulation_order:
        spawn = art.spawn or {}
        pos = list(spawn.get("pos") or [0.0, 0.0, env._spawn_height])
        if len(pos) != 3:
            raise ValueError(
                f"{_label(env)}: articulation {art.name!r} spawn.pos must be "
                f"a 3-vector, got {spawn.get('pos')!r}")
        yaw = float(spawn.get("yaw", env._spawn_yaw))
        if art.free_qpos_adr is not None:
            a = art.free_qpos_adr
            d.qpos[a:a + 3] = [float(v) for v in pos]
            d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0,
                                   np.sin(yaw / 2)]
            d.qpos[art.jadr_art] = art.q0_mj
        else:
            # fixed base: write q0 only at actuated joints (device hinges /
            # baked scene freejoints keep their XML qpos0)
            for control in env._control_order:
                if control.prop or control.articulation != art.name:
                    continue
                d.qpos[control.addresses["jadr"]] = \
                    control.q0.detach().cpu().numpy()


__all__ = ["apply_control_substep", "reset_articulations", "route_actions"]
