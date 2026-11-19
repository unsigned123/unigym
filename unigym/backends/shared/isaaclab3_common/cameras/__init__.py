"""IsaacLab 3 camera face: sensor construction, frame reads and rig classes.

This is the shared (engine-neutral) camera library for the
``isaaclab3_physx`` / ``isaaclab3_newton`` identities.  It never imports a
concrete backend; the only engine-divergent input is the explicit
``renderer_cfg`` (from :meth:`EngineProfile.renderer_cfg`) and the depth
background sentinel (from :attr:`EngineProfile.depth_background`).

The package map:

* :mod:`.cfg` -- ``build_sensor_cfg`` / ``read_frames`` /
  ``push_state_table_poses`` (lazy isaaclab imports inside the functions);
* :mod:`.rigs` -- ``BodyRig`` / ``WorldRig`` / ``ChaseRig`` and the
  ``build_isaaclab_rig`` registry face (torch-only at import time).

Every public name importable from
``unigym.backends.isaaclab2.cameras`` stays importable here.
"""

from __future__ import annotations

from .cfg import _read_depth as _read_depth
from .cfg import _read_rgb as _read_rgb
from .cfg import build_sensor_cfg, push_state_table_poses, read_frames
from .rigs import BodyRig, ChaseRig, WorldRig, build_isaaclab_rig

__all__ = [
    "BodyRig",
    "ChaseRig",
    "WorldRig",
    "build_isaaclab_rig",
    "build_sensor_cfg",
    "push_state_table_poses",
    "read_frames",
]
