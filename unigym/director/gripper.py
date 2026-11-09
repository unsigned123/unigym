"""Virtual gripper: neutral config face + per-backend factory.

The director's manipulable hand (contract §4).  Declarative
opt-in in the run config::

    director:
      gripper: {name: vgrip, size: [...], pos: [...], quat: [...], ...}

The backend implementations live in the backend packages:

* ``unigym/backends/mujoco/director/gripper.py`` -- the MuJoCo mocap hand
  (non-colliding box + grasp site + camera) and its spec-side injection
  (``add_director_bodies``: the gripper body + one DISABLED weld equality
  per rigid prop).
* ``unigym/backends/isaaclab2/director/gripper.py`` -- the IsaacLab fixed-
  base KINEMATIC rigid cuboid (scene cfg field).

This module stays engine-free at module level: the runtime class is
INJECTED as the :attr:`unigym.backends.registry.BackendInfo.gripper_backend`
factory by the backend ops that already knows its identity --
no env attribute sniffing, no compatibility shim.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

GRIPPER_DEFAULTS = {
    "name": "vgrip",
    "size": [0.025, 0.025, 0.05],
    "pos": [0.5, 0.0, 1.0],
    "quat": [1.0, 0.0, 0.0, 0.0],
    "rgba": [0.9, 0.3, 0.2, 0.5],
    "camera": {"name": "eye", "fovy": 90.0},
}


def normalize_gripper_cfg(cfg: dict | None) -> dict | None:
    """`director.gripper` yaml -> normalized table (None when absent).

    Unknown keys fail fast (typo protection, same discipline as
    ContactGroupSpec.from_yaml).
    """
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(f"director.gripper must be a mapping, got {cfg!r}")
    out = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in GRIPPER_DEFAULTS.items()}
    known = set(GRIPPER_DEFAULTS)
    extra = sorted(set(cfg) - known)
    if extra:
        raise ValueError(f"director.gripper: unknown key(s) {extra} "
                         f"(known: {sorted(known)})")
    for key in ("name",):
        if key in cfg:
            out[key] = str(cfg[key])
    for key in ("size", "pos", "quat", "rgba"):
        if key in cfg:
            vals = [float(v) for v in cfg[key]]
            want = 3 if key in ("size", "pos") else 4
            if len(vals) != want:
                raise ValueError(f"director.gripper.{key} needs {want} "
                                 f"values, got {vals}")
            out[key] = vals
    cam = cfg.get("camera")
    if cam is not None:
        if not isinstance(cam, dict):
            raise ValueError("director.gripper.camera must be a mapping "
                             "{name, fovy}")
        extra = sorted(set(cam) - {"name", "fovy"})
        if extra:
            raise ValueError(f"director.gripper.camera: unknown key(s) "
                             f"{extra} (known: ['fovy', 'name'])")
        out["camera"].update(cam)
    return out


@runtime_checkable
class Gripper(Protocol):
    """Runtime pose face shared by both backend gripper implementations."""

    name: str

    def get_pose(self, env) -> tuple: ...

    def set_pose(self, env, pos=None, quat=None) -> None: ...


def build_gripper(env, cfg: dict | None, gripper_backend) -> Gripper | None:
    """Construct the backend gripper from a NORMALIZED cfg (or None).

    ``gripper_backend`` is the :attr:`BackendInfo.gripper_backend`
    zero-argument factory injected by the caller (the backend ops object
    that already knows its identity).  This module never sniffs ``env``.
    """
    if cfg is None:
        return None
    if gripper_backend is None:
        raise NotImplementedError(
            "this backend declares no director gripper face")
    grip_cls = gripper_backend()
    return grip_cls(env, cfg["name"])
