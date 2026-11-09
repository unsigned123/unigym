"""The closed scene-provider registry.

The SIXTH closed registry (isomorphic to the task
providers and the prop-template providers):

* ``@register_scene`` -- closed name list, lowercase snake_case (the
  same charset rule that rejects path-like spellings by construction),
  duplicate rejection, docstring MANDATORY (a scene provider is a pure
  scene-builder returning the closed :class:`~unigym.scenes.scene_ir.Scene`;
  IO / runtime / backend imports are out of its axis and pinned by the
  import-linter contract);
* ``build_scene_definition(name, slots)`` -- the pure dispatch: every
  provider output passes the SAME :func:`~unigym.scenes.scene_ir.validate_scene`
  gate (one output type, one gate).

Declarative scene templates cover the routine needs; a scene provider
exists for DEEP customization only -- it may call ASSET providers (the
closed prop-template registry) to assemble subtrees, but it mints
no scene vocabulary of its own.  The registry fills from the scenes
package's own provider modules; core stays empty so it imports with no
content pack mounted.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .scene_ir import Scene, validate_scene

#: a provider: pure (slots, context) -> Scene (no IO, no runtime, no
#: backend -- the import-linter contract pins the zone).
SceneProvider = Callable[[Mapping[str, Any], "SceneContext"], Scene]

_SCENES: dict[str, SceneProvider] = {}

_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class SceneContext:
    """The provider's resolution context.

    ``name`` is the scene being built (the provider's error-message
    face); further resolution facts join here as the consumer face
    grows -- the context is data, never a runtime handle.
    """

    name: str


def register_scene(name: str) -> Callable[[SceneProvider], SceneProvider]:
    """Register a closed scene provider under a snake_case name.

    The registry discipline: closed list, path-like spellings
    rejected by the charset rule, duplicates rejected, docstring forced
    (config citations name scenes, never provider paths).
    """
    if (
        not isinstance(name, str)
        or not name
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)
    ):
        raise ValueError(
            f"scene name {name!r} must be lowercase snake_case "
            f"(path-like spellings are rejected)"
        )
    if name in _SCENES:
        raise ValueError(f"scene {name!r} registered twice")

    def decorate(fn: SceneProvider) -> SceneProvider:
        if not callable(fn):
            raise ValueError(f"scene {name!r} must be callable")
        if not (getattr(fn, "__doc__", "") or "").strip():
            raise ValueError(f"scene {name!r} requires a docstring")
        _SCENES[name] = fn
        return fn

    return decorate


def registered_scenes() -> list[str]:
    """The closed scene list (sorted; the registry's citation face)."""
    return sorted(_SCENES)


def build_scene_definition(name: str, slots: Mapping[str, Any]) -> Scene:
    """Dispatch a registered scene provider to a validated Scene.

    The provider is a pure function of ``(slots, context)``; its output
    must be a :class:`~unigym.scenes.scene_ir.Scene` and passes the
    :func:`~unigym.scenes.scene_ir.validate_scene` gate.
    """
    if not isinstance(slots, Mapping):
        raise ValueError(f"scene {name!r}: slots must be a mapping")
    if name not in _SCENES:
        raise ValueError(f"scene: unknown scene {name!r} (registered: {registered_scenes()})")
    scene = _SCENES[name](dict(slots), SceneContext(name=name))
    if not isinstance(scene, Scene):
        raise ValueError(
            f"scene {name!r}: provider must return a Scene, " f"got {type(scene).__name__}"
        )
    validate_scene(scene)
    return scene


__all__ = [
    "SceneContext",
    "SceneProvider",
    "build_scene_definition",
    "register_scene",
    "registered_scenes",
]
