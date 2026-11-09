"""IsaacLab camera face: sensor construction, frame reads and rig classes.

The package map (layout move):

* :mod:`.cfg` -- ``build_sensor_cfg`` / ``read_frames`` /
  ``push_state_table_poses`` (lazy isaaclab imports inside the functions);
* :mod:`.rigs` -- ``BodyRig`` / ``WorldRig`` / ``ChaseRig`` and the
  ``build_isaaclab_rig`` registry face (torch-only at import time).

Every public name previously importable from
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
