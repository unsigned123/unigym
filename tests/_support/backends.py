"""Test helpers for injecting composition faces into neutral builders."""

from __future__ import annotations

from collections.abc import Callable

from unigym.backends.compose import backend_runtime


def contact_factory(backend: str) -> Callable[[], object]:
    """Return the injected contact-face factory for *backend*."""
    factory = backend_runtime(backend).bindings.contact_backend
    if factory is None:
        raise AssertionError(f"backend {backend!r} declares no contact face")
    return factory


def prop_faces(backend: str) -> tuple[Callable[[], object] | None, Callable[[], object] | None]:
    """Return ``(prop_face, mujoco_prop_face)`` for *backend*."""
    return (
        backend_runtime(backend).bindings.prop_backend,
        backend_runtime("mujoco").bindings.prop_backend,
    )
