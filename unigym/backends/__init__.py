"""Central backend registry package (backend-traits architecture).

Layout (docs/designs/backend-traits.md §R.2):

* ``unigym/backends/registry.py`` — the Trait enum, ``BackendInfo``,
  the per-backend trait tables and the import-probe backend selection
  (``mujoco`` / ``isaaclab2`` / ``isaaclab3`` — three equal backends).
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
    BACKEND_ALIASES,
    CANONICAL_BACKENDS,
    BackendInfo,
    BackendSelectionError,
    Trait,
    backend_info,
    isaaclab_version,
    known_backends,
    mujoco_version,
    mujoco_warp_version,
    select_backend,
)

__all__ = [
    "BACKEND_ALIASES",
    "CANONICAL_BACKENDS",
    "BackendInfo",
    "BackendSelectionError",
    "Trait",
    "backend_info",
    "isaaclab_version",
    "known_backends",
    "mujoco_version",
    "mujoco_warp_version",
    "select_backend",
]
