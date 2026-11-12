"""The coarse scene-engine protocol and its closed registry.

Standard library plus the sibling neutral IR module only -- no engine
import, no content import.  The protocol is the whole scene-side contract
between the neutral world and a backend:

* :meth:`SceneEngine.compose` turns one :class:`~unigym.scenes.scene_ir.Scene`
  into the backend's composed product (an opaque object -- the core never
  inspects it);
* :meth:`SceneEngine.attach` joins an ALREADY composed product into a base
  world at one named frame and returns the composed region table;
* :meth:`SceneEngine.to_xml` serializes the composed product for artifact
  capture and A/B comparison;
* :meth:`SceneEngine.describe` returns the neutral structural description
  used by conformance.

The interface is deliberately COARSE: four calls, no element vocabulary,
no per-generator dispatch.  A backend self-registers through
:func:`register_scene_engine`; the registry is closed (duplicate or
illegal names are teaching errors) and the core never names a backend.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .scene_ir import Scene

if TYPE_CHECKING:
    from unigym.assets.plan import AssetPlan

_ENGINE_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


class SceneEngineError(ValueError):
    """A scene-engine declaration broke one closed-registry rule."""


@runtime_checkable
class SceneEngine(Protocol):
    """Compose, attach, serialize and describe one scene for a backend.

    Implementations live in the backend layer and self-register at import.
    The composed product is opaque to the core: only the engine that built
    it may read it back through :meth:`attach`, :meth:`to_xml` and
    :meth:`describe`.
    """

    name: str

    def compose(self, scene: Scene, asset_plan: AssetPlan) -> object:
        """Return the backend's composed product for *scene*.

        *asset_plan* is the composition boundary that resolves every asset
        reference before the backend builds engine elements.
        """
        ...

    def attach(self, base: object, composed: object, *, frame: str) -> Mapping[str, object]:
        """Join *composed* into *base* at *frame*; return the region table."""
        ...

    def to_xml(self, composed: object) -> str:
        """Serialize *composed* as text (artifact + A/B face)."""
        ...

    def describe(self, composed: object) -> Mapping[str, object]:
        """Return the neutral structural description of *composed*.

        This is the ONLY conformance face: the core digests the returned
        mapping and never reads another engine representation.  ``to_xml``
        stays available for artifact capture and A/B comparison but is not
        a conformance contract.  The mapping is engine-neutral -- it may
        not expose format vocabulary (element or serialization names), so
        a second engine can describe its own product without pretending
        to share a serialization format.
        """
        ...


_ENGINES: dict[str, Callable[[], object]] = {}


def register_scene_engine(
    name: str,
    factory: Callable[[], object],
) -> Callable[[], object]:
    """Register one scene-engine factory under a lowercase name.

    The core owns the protocol, never an implementation: a backend calls
    this at import time.  The factory is stored un-called so importing a
    backend does not construct an engine.

    Args:
        name: The engine name (lowercase snake_case, unique).
        factory: The zero-argument callable returning a
            :class:`SceneEngine` instance.

    Returns:
        The registered factory (decorator-friendly).

    Raises:
        SceneEngineError: The name is not snake_case, it is already
            registered, or the factory is not callable.
    """
    if not isinstance(name, str) or _ENGINE_NAME_RE.fullmatch(name) is None:
        raise SceneEngineError(
            f"scene engine name {name!r} must be lowercase snake_case "
            f"(path-like spellings are rejected)"
        )
    if not callable(factory):
        raise SceneEngineError(
            f"scene engine {name!r} factory must be callable, got " f"{type(factory).__name__}"
        )
    if name in _ENGINES:
        raise SceneEngineError(
            f"scene engine {name!r} is registered twice -- give each engine " f"a unique name"
        )
    _ENGINES[name] = factory
    return factory


def scene_engine(name: str) -> SceneEngine:
    """Return a new instance of the engine registered under *name*.

    Args:
        name: The engine name a caller resolved (e.g. a backend name).

    Returns:
        The factory's product, after the protocol shape is checked.

    Raises:
        SceneEngineError: The name is missing from the registry, or the
            factory product does not satisfy :class:`SceneEngine`.
    """
    if not isinstance(name, str) or not name:
        raise SceneEngineError(f"scene engine name must be a non-empty string, got {name!r}")
    try:
        factory = _ENGINES[name]
    except KeyError:
        raise SceneEngineError(
            f"scene engine {name!r} is not registered (registered: "
            f"{registered_scene_engines()}) -- import the backend that owns it"
        ) from None
    engine = factory()
    if isinstance(engine, type) or not isinstance(engine, SceneEngine):
        raise SceneEngineError(
            f"scene engine {name!r} factory returned "
            f"{type(engine).__name__} -- it must satisfy SceneEngine "
            f"(name, compose, attach, to_xml, describe)"
        )
    if engine.name != name:
        raise SceneEngineError(
            f"scene engine {name!r} factory returned name {engine.name!r} -- "
            f"the registered name and the instance name must agree"
        )
    return engine


def registered_scene_engines() -> list[str]:
    """Return the registered scene-engine names, sorted."""
    return sorted(_ENGINES)


__all__ = [
    "SceneEngine",
    "SceneEngineError",
    "register_scene_engine",
    "registered_scene_engines",
    "scene_engine",
]
