"""Action routing + physics application for the MuJoCo-family backends.

Split out of :mod:`articulation_runtime` to keep each module small
and maintainable.  Every engine still iterates the neutral
``env._control_order`` tables built there.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from unigym.runtime.actions import NamedAction


def _label(env) -> str:
    """Backend label for teaching errors (defaults to the CPU mujoco lane)."""
    return getattr(env, "_backend_label", "mujoco backend")


def _initial_targets(env, order) -> dict:
    targets = {}
    for control in order:
        key = (control.articulation, control.name)
        if control.prop:
            slots = control.addresses.get("ctrl")
            targets[key] = (
                np.zeros((env.num_envs, len(control.joint_names)), dtype=np.float64)
                if slots is None
                else env._prop_targets[:, slots].copy()
            )
        else:
            targets[key] = np.zeros((env.num_envs, len(control.joint_names)), dtype=np.float64)
    return targets


def _apply_entry(env, control, identity, entry, targets, covered, label) -> None:
    """Scatter one NamedAction entry into its control target row."""
    if entry.kind != control.target:
        raise ValueError(
            f"{label}: action group {identity!r} has kind {entry.kind!r} but "
            f"control {control.name!r} expects target {control.target!r}"
        )
    names = list(entry.joint_names)
    index = {n: i for i, n in enumerate(control.joint_names)}
    unknown = sorted(set(names) - set(index))
    if unknown:
        raise ValueError(
            f"{label}: action group {identity!r} names joints {unknown} that "
            f"control {control.name!r} does not own | owns: "
            f"{list(control.joint_names)}"
        )
    row = entry.values.detach().to("cpu").to(torch.float32).numpy()
    if entry.num_envs != env.num_envs:
        raise ValueError(
            f"{label}: action group {identity!r} has {entry.num_envs} env "
            f"rows, expected {env.num_envs}"
        )
    if row.shape[-1] != len(names):
        raise ValueError(
            f"{label}: action group {identity!r} carries {row.shape[-1]} "
            f"values for {len(names)} joints"
        )
    if control.prop:
        resolved = env._action_resolver.resolve(names, prop=control.name)
        env._prop_targets[:, resolved.prop_adr] = row
        key = (control.articulation, control.name)
        targets[key] = env._prop_targets[:, control.addresses["ctrl"]].copy()
        return
    keys = (control.articulation, control.name)
    cols = np.asarray([index[n] for n in names], dtype=np.int64)
    if covered[keys][cols].any():
        raise ValueError(
            f"{label}: action group {identity!r} re-addresses joints of "
            f"control {control.name!r}"
        )
    targets[keys][:, cols] = row
    covered[keys][cols] = True


def route_actions(env, actions: Sequence[NamedAction], label=None) -> dict:
    """Resolve every self-describing action onto its control's target array.

    Entries are routed by ``(entry.articulation, entry.group)``; the
    sequence order is not a layout.  A control's routing identity is
    ``(control.articulation, control.name)``.
    """
    label = label or _label(env)
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise TypeError(
            f"{label} step expects a sequence of NamedAction, got " f"{type(actions).__name__}"
        )
    order = list(getattr(env, "_control_order", env.controls.values()))
    targets = _initial_targets(env, order)
    covered = {
        (c.articulation, c.name): np.zeros(len(c.joint_names), dtype=bool)
        for c in order
        if not c.prop
    }
    by_key: dict[tuple[str, str], object] = {}
    for control in order:
        identity = (control.articulation, control.name)
        if identity in by_key:
            raise ValueError(
                f"{label}: control group {identity!r} is declared by more "
                f"than one control runtime"
            )
        by_key[identity] = control
    accepted = sorted(by_key)
    seen: set[tuple[str, str]] = set()
    for entry in actions:
        if not isinstance(entry, NamedAction):
            raise TypeError(
                f"{label} step expects a sequence of NamedAction, got "
                f"{type(entry).__name__} entry"
            )
        identity = (entry.articulation, entry.group)
        control = by_key.get(identity)
        if control is None:
            raise ValueError(
                f"{label}: action group {identity!r} is not a declared "
                f"control group; accepted control groups: {accepted}"
            )
        if identity in seen:
            raise ValueError(
                f"{label}: control group {identity!r} is addressed by more "
                f"than one action entry"
            )
        seen.add(identity)
        _apply_entry(env, control, identity, entry, targets, covered, label)
    for control in order:
        if control.prop or not control.required:
            continue
        key = (control.articulation, control.name)
        if not bool(covered[key].all()):
            left = [n for n, hit in zip(control.joint_names, covered[key]) if not hit]
            identity = (control.articulation, control.name)
            raise ValueError(
                f"{label}: action group {identity!r} leaves joints " f"unaddressed: {left}"
            )
    return targets


def reset_articulations(env, d) -> None:
    """Write every articulation's root spawn + q0 (declaration order)."""
    for art in env._articulation_order:
        spawn = art.spawn or {}
        if spawn:
            pos = list(spawn["pos"])
            yaw = float(spawn.get("yaw", 0.0))
        else:
            pos = [float(env._spawn_xy[0]), float(env._spawn_xy[1]), float(env._spawn_height)]
            yaw = float(env._spawn_yaw)
        if art.free_qpos_adr is not None:
            a = art.free_qpos_adr
            d.qpos[a : a + 3] = [float(v) for v in pos]
            d.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            d.qpos[art.jadr_art] = art.q0_mj
        else:
            # fixed base: write q0 only at actuated joints (device hinges /
            # baked scene freejoints keep their XML qpos0)
            for control in env._control_order:
                if control.prop or control.articulation != art.name:
                    continue
                d.qpos[control.addresses["jadr"]] = control.q0.detach().cpu().numpy()


__all__ = ["reset_articulations", "route_actions"]
