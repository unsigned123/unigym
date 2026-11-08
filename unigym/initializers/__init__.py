"""Reset initializers (scene-bundle face): registry + generic primitives.

The registry, the three-phase contract and the generic placement
primitives (docs/designs/v067-scene-bundle-and-initializers.md §2).
The robocasa builtin types now ship with the robocasa pack's code face
(v081 P5b); importing this package never needs a mounted pack.

Regular subpackage marker: import-linter's graph builder (grimp) skips
namespace subpackages, which would leave every module here out of the
contract graphs (same pattern as unigym/scripts/__init__.py).
"""
from unigym.initializers.base import (
    INITIALIZER_REGISTRY,
    InitializerBase,
    ResetPrimitives,
    build_initializer,
    normalize_initializer_spec,
    register,
)

__all__ = [
    "INITIALIZER_REGISTRY",
    "InitializerBase",
    "ResetPrimitives",
    "build_initializer",
    "normalize_initializer_spec",
    "register",
]
