"""Central backend registry package (backend-traits architecture).

Layout (docs/designs/backend-traits.md §R.2):

* ``unigym/backends/registry.py`` — the Trait enum, ``BackendInfo``,
  the per-backend trait tables and canonical-only backend selection
  (``mujoco`` / ``mjwarp`` / ``isaaclab2`` / ``isaaclab3_physx`` /
  ``isaaclab3_newton`` — equal backends, no aliases).
* ``unigym/backends/<backend>/`` — that backend's implementation of the
  per-capability Protocols (which live on the capability side, e.g.
  ``unigym/env/backends/base.py``, ``unigym/cameras/base.py``).

Dependency direction is one-way: capability packages never import
``unigym.backends`` at module level (registry is stdlib-only and safe to
import anywhere, lazily), and everything under ``backends/<backend>/`` is
where module-level engine imports (mujoco/isaaclab/omni) are allowed —
nowhere else in ``unigym/`` (see ``tests/test_arch.py``).
"""

from __future__ import annotations

from .registry import (
    CANONICAL_BACKENDS,
    BackendInfo,
    BackendSelectionError,
    StackProbe,
    Trait,
    backend_info,
    build_name,
    isaaclab_version,
    known_backends,
    mujoco_version,
    mujoco_warp_version,
    newton_version,
    select_backend,
    selection_error,
    warp_version,
)

__all__ = [
    "CANONICAL_BACKENDS",
    "BackendInfo",
    "BackendSelectionError",
    "StackProbe",
    "Trait",
    "backend_info",
    "build_name",
    "isaaclab_version",
    "known_backends",
    "mujoco_version",
    "mujoco_warp_version",
    "newton_version",
    "select_backend",
    "selection_error",
    "warp_version",
]
