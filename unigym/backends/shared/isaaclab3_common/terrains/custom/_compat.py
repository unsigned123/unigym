"""Lazy access to isaaclab's ``height_field_to_mesh`` decorator.

Locally (no isaaclab installed) the decorator degrades to the identity, so the
custom terrain functions remain *plain height-field producers* returning a
``np.ndarray[int16]`` — directly callable and testable without isaaclab or
``pxr``. On the server (isaaclab present, after AppLauncher) the real decorator
wraps them into mesh-returning functions exactly as Isaac Lab's
``TerrainGenerator`` expects.

The isaaclab import is deliberately deferred to the first call of a decorated
function: importing this module (and every terrain generator that uses the
decorator) must never import isaaclab at module scope.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

__all__ = ["height_field_to_mesh"]


def _load_real_decorator() -> Callable[[Any], Any] | None:
    """Return isaaclab's decorator, or ``None`` when isaaclab is absent."""
    try:  # pragma: no cover - exercised only on the server (isaaclab present)
        from isaaclab.terrains.height_field.utils import (
            height_field_to_mesh as decorator,
        )
    except Exception:  # identity fallback keeps modules importable locally
        return None
    return decorator


def height_field_to_mesh(func):
    """Defer isaaclab's decorator until the wrapped generator is first called.

    Args:
        func: A height-field producer ``(difficulty, cfg) -> np.ndarray``.

    Returns:
        A callable with the same contract: on the server it resolves to the
        real isaaclab decorator's result, locally it stays the plain
        height-field producer.
    """
    resolved: list[Any] = []

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not resolved:
            real = _load_real_decorator()
            resolved.append(real(func) if real is not None else func)
        return resolved[0](*args, **kwargs)

    return wrapper
