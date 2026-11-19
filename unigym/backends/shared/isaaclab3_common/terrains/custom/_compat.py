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

Fallback policy (no silent degradation): when isaaclab is *absent* the first
call warns once -- ``"isaaclab terrain decorator unavailable; using identity
fallback"`` -- and keeps the plain height-field producer.  When isaaclab is
*installed* but the decorator import fails, the call raises ``ImportError``
instead of masquerading a broken server install as the local identity path.
"""

from __future__ import annotations

import functools
import importlib.util
import warnings
from typing import Any, Callable

__all__ = ["height_field_to_mesh"]

_warned_identity_fallback = False


def _isaaclab_installed() -> bool:
    """Best-effort check that the ``isaaclab`` package is discoverable."""
    try:
        return importlib.util.find_spec("isaaclab") is not None
    except (ImportError, ValueError):
        return False


def _warn_identity_fallback() -> None:
    """Warn exactly once that the identity fallback is in use."""
    global _warned_identity_fallback
    if not _warned_identity_fallback:
        _warned_identity_fallback = True
        warnings.warn(
            "isaaclab terrain decorator unavailable; using identity fallback "
            "(height-field generators remain plain callables)",
            RuntimeWarning,
            stacklevel=3,
        )


def _load_real_decorator() -> Callable[[Any], Any] | None:
    """Return isaaclab's decorator, or ``None`` when isaaclab is absent.

    Raises:
        ImportError: if isaaclab is installed but the decorator cannot be
            imported (a broken install must not silently degrade).
    """
    try:  # pragma: no cover - exercised only on the server (isaaclab present)
        from isaaclab.terrains.height_field.utils import (
            height_field_to_mesh as decorator,
        )
    except Exception as exc:  # noqa: BLE001 - distinguish absent vs broken install
        if _isaaclab_installed():
            raise ImportError(
                "isaaclab is installed but "
                "isaaclab.terrains.height_field.utils.height_field_to_mesh "
                "failed to import; refusing to fall back to the identity "
                "decorator on a broken install"
            ) from exc
        _warn_identity_fallback()
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
