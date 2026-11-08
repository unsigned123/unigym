"""Object-state record layer: ``object/<name>/{pos,quat,joints/*}`` extras.

Composes the per-step record keys for props, decoupled from the obs path
(prop body-frame state keeps flowing into ``obs_cache`` on request exactly
as before):

    object/<name>/pos                (N, 3) float32  world-frame root position
    object/<name>/quat               (N, 4) float32  world-frame root quat, wxyz
    object/<name>/joints/<jname>     (N, 1) float32  articulation joint state
                                                     (the ANCHOR root
                                                     pose is the articulation's
                                                     pos/quat contribution)

Domain-neutral: this layer knows ONLY geometry (root pose +
declared joint values).  There is deliberately no ``held`` key family any
more -- any hand-contact criterion is a DECLARED contact group (yaml
``contacts:`` referencing ``{body_parts: grasp.left}`` and
``{prop: <name>}``; contacts/base.py) that lands in npz as
``contact/<name>`` and is judged by the FormulaTask ``contact_any``
template (benchmarks/terms.py).

Zero-change guarantee: no props configured -> none of these keys exist and
no code path runs (the env backends gate on ``prop_manager is not None``).
The BenchmarkSink/CollectSink extra buffering is key-agnostic (passthrough),
so the keys reach npz shards and FormulaTask metrics without sink changes.
"""
from __future__ import annotations

import torch


def object_record_keys(prop_manager) -> list[str]:
    """Canonical record keys for a run's props (wiring-time declaration)."""
    keys: list[str] = []
    for h in prop_manager._handlers:
        name = h.spec.name
        keys += [f"object/{name}/pos", f"object/{name}/quat"]
        # articulation handlers carry the normalized declared joint table
        for jname in (getattr(h, "joints", None) or {}):
            keys.append(f"object/{name}/joints/{jname}")
    return keys


def build_object_extra(prop_manager, world_poses: dict,
                       joint_states: dict | None = None
                       ) -> dict[str, torch.Tensor]:
    """Assemble the ``object/*`` extras for one step.

    Args:
        prop_manager: the wired :class:`unigym.props.PropManager`.
        world_poses: ``{name: (pos (N,3), quat (N,4) wxyz)}`` world-frame
            root poses (isaaclab: :meth:`PropManager.get_world_poses`;
            mujoco: the props mujoco backend equivalent).  Articulation
            props contribute their ANCHOR pose.
        joint_states: optional ``{name: {jname: (N,1)}}`` articulation
            joint positions (isaaclab: :meth:`PropManager.
            get_articulation_joint_states`; mujoco: the props backend
            equivalent).

    Returns freshly-allocated tensors keyed ``object/<name>/...`` on the
    poses' device.
    """
    out: dict[str, torch.Tensor] = {}
    for name, (pos, quat) in world_poses.items():
        out[f"object/{name}/pos"] = torch.as_tensor(pos).clone()
        out[f"object/{name}/quat"] = torch.as_tensor(quat).clone()
    for name, joints in (joint_states or {}).items():
        for jname, val in joints.items():
            out[f"object/{name}/joints/{jname}"] = torch.as_tensor(val).clone()
    return out
