"""Action routing + physics application for the MuJoCo-family backends.

Split out of :mod:`articulation_runtime` to keep each module small
and maintainable.  Every engine still iterates the neutral
``env._control_order`` tables built there.

Runtime action columns are routed by ``(articulation, joint)`` through the
shared :class:`~unigym.runtime.action_routing.JointActionIndex`; a control-row
label is a physical target-table key, never an action identity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from unigym.runtime.action_routing import JointActionIndex, RoutedColumn
from unigym.runtime.actions import NamedAction
from unigym.scenes.articulation import resolve_spawn_quat


def _label(env) -> str:
    """Backend label for teaching errors (defaults to the CPU mujoco lane)."""
    return getattr(env, "_backend_label", "mujoco backend")


def _control_key(control) -> tuple[str, str]:
    """The physical target-table key of one control row (not an action id)."""
    return (control.articulation, control.name)


def _initial_targets(env, order) -> dict:
    targets = {}
    for control in order:
        key = _control_key(control)
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


def _apply_column(env, routed: RoutedColumn, targets: dict) -> None:
    """Scatter one routed action column into its physical target row."""
    entry = routed.entry
    slot = routed.slot
    control = slot.owner
    row = entry.values.detach().to("cpu").to(torch.float32).numpy()
    column = row[:, routed.source_index]
    if slot.prop:
        resolved = env._action_resolver.resolve([slot.joint], prop=control.name)
        env._prop_targets[:, resolved.prop_adr] = column[:, None]
        key = _control_key(control)
        targets[key] = env._prop_targets[:, control.addresses["ctrl"]].copy()
        return
    key = _control_key(control)
    targets[key][:, slot.column] = column


def route_actions(env, actions: Sequence[NamedAction], label=None) -> dict:
    """Resolve every joint-addressed action column onto its control target array.

    Columns are routed by ``(articulation, joint)``; the sequence order is not
    a layout.  One entry may span several physical control rows of one
    articulation as long as every column's kind matches the owning control.
    Required coverage is checked per joint by the shared index.
    """
    label = label or _label(env)
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise TypeError(
            f"{label} step expects a sequence of NamedAction, got " f"{type(actions).__name__}"
        )
    order = list(getattr(env, "_control_order", env.controls.values()))
    targets = _initial_targets(env, order)
    index = getattr(env, "_action_joint_index", None)
    if index is None:
        index = JointActionIndex.from_controls(order, label=label)
        env._action_joint_index = index
    routed = index.route(actions, env.num_envs)
    for routed_column in routed:
        _apply_column(env, routed_column, targets)
    return targets


def reset_articulations(env, d) -> None:
    """Write every articulation's root spawn + q0 (declaration order)."""
    for art in env._articulation_order:
        spawn = art.spawn or {}
        if spawn:
            pos = list(spawn["pos"])
        else:
            pos = [float(env._spawn_xy[0]), float(env._spawn_xy[1]), float(env._spawn_height)]
        quat_w = resolve_spawn_quat(spawn, default_yaw=float(env._spawn_yaw))
        if art.free_qpos_adr is not None:
            a = art.free_qpos_adr
            d.qpos[a : a + 3] = [float(v) for v in pos]
            d.qpos[a + 3 : a + 7] = quat_w
            d.qpos[art.jadr_art] = art.q0_mj
        else:
            # fixed base: write q0 only at actuated joints (device hinges /
            # baked scene freejoints keep their XML qpos0)
            for control in env._control_order:
                if control.prop or control.articulation != art.name:
                    continue
                d.qpos[control.addresses["jadr"]] = control.q0.detach().cpu().numpy()


__all__ = ["reset_articulations", "route_actions"]
