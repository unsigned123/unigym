"""cameras: request-driven multi-camera obs system (feat/camera-obs).

Public surface:

* :func:`build_camera_manager` — collect.py entry point; builds a
  :class:`CameraManager` from the yaml ``cameras:`` section + the adapter's
  obs requests (request-driven instantiation; the depth aliases are served
  only by an explicitly declared camera named ``depth``).
* :class:`CameraManager` — key space ``cam/<name>/<rgb|depth>``, interval
  sampling, placeholder fallback frames.
* :class:`cameras.base.CameraSpec` — yaml-declarable camera rig spec.

The package is numpy/torch-only at import time; every isaaclab import is
lazy (inside ``backends/isaaclab2/cameras`` / rig scene methods), mirroring the
``robots/`` convention.  Contract docs: ``docs/contracts/camera_contract.md``.
"""

from __future__ import annotations

from .base import CameraRig, CameraSpec, depth_key, rgb_key
from .manager import CameraManager, build_camera_manager

__all__ = [
    "CameraManager",
    "CameraRig",
    "CameraSpec",
    "build_camera_manager",
    "depth_key",
    "rgb_key",
]
