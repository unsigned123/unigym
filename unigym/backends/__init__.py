"""Backend composition package.

Layout (docs/designs/backend-traits.md §R.2):

* :mod:`unigym.capabilities` — pure backend vocabulary (traits, specs,
  version probes, canonical selection); no concrete backend imports.
* ``unigym/backends/bindings.py`` — the flat ``BackendBindings`` table of
  lazy concrete construction callables (every engine import stays inside a
  factory).
* ``unigym/backends/compose.py`` — ``BackendRuntime`` + ``backend_runtime()``,
  the composition-facing value object (spec + bindings).
* ``unigym/backends/<backend>/`` — that backend's implementation of the
  per-capability Protocols (which live on the capability side, e.g.
  ``unigym/env/backends/base.py``, ``unigym/cameras/base.py``).

Neutral layers import :mod:`unigym.capabilities` only; composition code
imports :mod:`unigym.backends.compose` / :mod:`unigym.backends.bindings`.
Everything under ``backends/<backend>/`` is where module-level engine imports
(mujoco/isaaclab/omni) are allowed — nowhere else in ``unigym/``.
"""

from __future__ import annotations

from unigym.backends.bindings import BackendBindings, ModelEditProvider
from unigym.backends.compose import (
    BackendRuntime,
    backend_runtime,
    validate_backend_runtime,
    validate_backend_runtimes,
)
from unigym.capabilities.selection import (
    BackendSelectionError,
    StackProbe,
    select_backend,
    selection_error,
)
from unigym.capabilities.specs import (
    BackendSpec,
    backend_spec,
    validate_backend_options,
    validate_trait_specs,
)
from unigym.capabilities.traits import CANONICAL_BACKENDS, Trait, build_name, known_backends
from unigym.capabilities.versions import (
    isaaclab_version,
    mujoco_version,
    mujoco_warp_version,
    newton_version,
    warp_version,
)

__all__ = [
    "CANONICAL_BACKENDS",
    "BackendBindings",
    "BackendRuntime",
    "BackendSelectionError",
    "BackendSpec",
    "ModelEditProvider",
    "StackProbe",
    "Trait",
    "backend_runtime",
    "backend_spec",
    "build_name",
    "isaaclab_version",
    "known_backends",
    "mujoco_version",
    "mujoco_warp_version",
    "newton_version",
    "select_backend",
    "selection_error",
    "validate_backend_options",
    "validate_backend_runtime",
    "validate_backend_runtimes",
    "validate_trait_specs",
    "warp_version",
]
