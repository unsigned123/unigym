"""The closed scene-provider registry.

The SIXTH closed registry (isomorphic to the task
providers and the prop-template providers):

* ``register_scene_provider(name, factory)`` -- closed name list, the
  adapter/robot dotted grammar (``acme.table_top``), duplicate rejection,
  docstring MANDATORY (a scene provider is a pure scene-builder returning the
  closed :class:`~unigym.scenes.scene_ir.Scene`; IO / runtime / backend
  imports are out of its axis and pinned by the import-linter contract);
* ``build_scene_from_provider(name, params, *, preset=None)`` -- the pure
  dispatch: every provider output passes the SAME
  :func:`~unigym.scenes.scene_ir.validate_scene` gate (one output type, one
  gate).  A registered provider parameter schema fills defaults and validates
  the supplied mapping BEFORE the provider is called.

Declarative scene templates cover the routine needs; a scene provider
exists for DEEP customization only -- it may call ASSET providers (the
closed prop-template registry) to assemble subtrees, but it mints
no scene vocabulary of its own.  The registry fills from the scenes
package's own provider modules; core stays empty so it imports with no
content pack mounted.

``World.inline`` registers through :func:`register_inline_scene_provider`,
which keys a process-local map: an inline provider never appears in
``registered_scene_providers()`` or the provider-info query, and its
:class:`ProviderOrigin` is ``kind="inline"``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from unigym._typing import JsonValue
from unigym.provenance import ProviderInfo as _ProviderInfo
from unigym.provenance import ProviderOrigin

from .provider_params import (
    ProviderParamSpec,
    fill_and_validate,
    normalize_code_params,
)
from .scene_ir import Scene, validate_scene

__all__ = [
    "ProviderInfo",
    "ProviderOrigin",
    "SceneProvider",
    "SceneProviderContext",
    "SceneProviderRegistrySnapshot",
    "attribute_scene_provider",
    "build_scene_from_provider",
    "register_inline_scene_provider",
    "register_scene_provider",
    "registered_scene_providers",
    "restore_scene_providers",
    "scene_provider_origin",
    "scene_provider_providers",
    "snapshot_scene_providers",
]

#: a provider: pure (params, context) -> Scene (no IO, no runtime, no
#: backend -- the import-linter contract pins the zone).
SceneProvider = Callable[[Mapping[str, JsonValue], "SceneProviderContext"], Scene]

_SCENE_PROVIDERS: dict[str, SceneProvider] = {}
_INLINE_SCENE_PROVIDERS: dict[str, SceneProvider] = {}
_SCENE_PROVIDER_ORIGINS: dict[str, "ProviderOrigin"] = {}
_SCENE_PROVIDER_PARAMS: dict[str, Mapping[str, ProviderParamSpec]] = {}

#: adapter/robot grammar: lowercase snake_case, optionally dot-namespaced.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


@dataclass(frozen=True)
class SceneProviderContext:
    """The provider's resolution context.

    ``name`` is the provider being dispatched; ``preset`` is the preset
    catalog id when the dispatch came through a preset (metadata for errors
    and artifacts -- it never changes dispatch).  The context is data, never a
    runtime handle.
    """

    name: str
    preset: str | None = None


@dataclass(frozen=True)
class ProviderInfo(_ProviderInfo):
    """One registered scene provider, with origin and parameter schema.

    ``ProviderOrigin`` and the live-record base class are owned by
    :mod:`unigym.provenance`; this specialization adds the scene provider's
    normalized parameter declaration.
    """

    params_schema: Mapping[str, ProviderParamSpec] | None = None


@dataclass(frozen=True)
class SceneProviderRegistrySnapshot:
    """A total snapshot of the scene-provider registries (restore is exact)."""

    providers: Mapping[str, SceneProvider]
    inline: Mapping[str, SceneProvider]
    origins: Mapping[str, ProviderOrigin]
    params: Mapping[str, Mapping[str, ProviderParamSpec]]


def _factory_label(factory: object) -> str:
    """Return a stable ``module.qualname`` label for an error message."""
    module = getattr(factory, "__module__", None)
    qualname = getattr(factory, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    return repr(factory)


def _check_factory(name: str, factory: object) -> SceneProvider:
    """Validate one provider callable (callable + non-empty docstring)."""
    if not callable(factory):
        raise ValueError(f"scene provider {name!r} must be callable, got {type(factory).__name__}")
    if not (getattr(factory, "__doc__", "") or "").strip():
        raise ValueError(f"scene provider {name!r} requires a docstring")
    return cast(SceneProvider, factory)


def _check_alias(name: str, factory: object, registry: Mapping[str, SceneProvider]) -> None:
    """Reject one factory registered under two names."""
    alias = next((key for key, value in registry.items() if value is factory), None)
    if alias is not None:
        raise ValueError(
            f"scene provider factory {_factory_label(factory)} is already registered "
            f"as {alias!r}; one factory owns exactly one name"
        )


def register_scene_provider(
    name: str,
    factory: SceneProvider,
    *,
    params: Mapping[str, object] | None = None,
    origin: ProviderOrigin | None = None,
) -> SceneProvider:
    """Register a closed scene provider under a dotted snake_case name.

    The registry discipline: closed list, path-like spellings rejected by the
    grammar, duplicates rejected, docstring forced (config citations name
    providers, never provider paths).  ``params`` is the optional inline
    parameter schema mirror; it is normalized here for dispatch-time default
    filling and validation.

    Args:
        name: The provider id (adapter/robot grammar, e.g. ``acme.table_top``).
        factory: The pure ``(params, context) -> Scene`` callable.
        params: Optional compact/full parameter declarations.
        origin: Internal/runtime-only provenance; pack authors never pass it.

    Returns:
        ``factory`` (the validated callable), so callers can keep the
        registered reference.

    Raises:
        ValueError: The name, factory or parameter schema violates the
            registration contract.
    """
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        raise ValueError(
            f"scene provider name {name!r} must be lowercase snake_case, "
            f"optionally dot-namespaced (e.g. 'acme.table_top')"
        )
    if name in _SCENE_PROVIDERS:
        raise ValueError(f"scene provider {name!r} registered twice")
    checked = _check_factory(name, factory)
    _check_alias(name, checked, _SCENE_PROVIDERS)
    _check_alias(name, checked, _INLINE_SCENE_PROVIDERS)
    if origin is not None and not isinstance(origin, ProviderOrigin):
        raise ValueError(
            f"scene provider {name!r}: origin must be a ProviderOrigin, "
            f"got {type(origin).__name__}"
        )
    schema = normalize_code_params(params) if params is not None else None
    _SCENE_PROVIDERS[name] = checked
    if schema is not None:
        _SCENE_PROVIDER_PARAMS[name] = MappingProxyType(schema)
    if origin is not None:
        _SCENE_PROVIDER_ORIGINS[name] = origin
    return checked


def attribute_scene_provider(
    name: str,
    origin: ProviderOrigin,
    *,
    params: Mapping[str, ProviderParamSpec] | None = None,
) -> None:
    """Attach runtime-owned provenance (and an authoritative schema) to a provider.

    This is the pack runtime's attribution boundary, not pack-author API: pack
    code registers through :func:`register_scene_provider` and never passes an
    origin.  It may only label an already-registered public provider whose
    current origin is absent, ``unknown`` or ``pack-code``; a ``manifest`` or
    ``inline`` origin is never relabeled.  ``params`` is an already-normalized
    manifest schema; the manifest is authoritative, so it replaces an absent or
    matching code mirror.

    Raises:
        ValueError: The provider is unknown, *origin* is not a
            :class:`ProviderOrigin`, or the current origin must not be relabeled.
    """
    if name not in _SCENE_PROVIDERS:
        raise ValueError(
            f"scene provider {name!r} is not registered "
            f"(registered: {registered_scene_providers()})"
        )
    if not isinstance(origin, ProviderOrigin):
        raise ValueError(
            f"scene provider {name!r}: origin must be a ProviderOrigin, "
            f"got {type(origin).__name__}"
        )
    existing = _SCENE_PROVIDER_ORIGINS.get(name)
    if existing is not None and existing.kind not in ("unknown", "pack-code"):
        raise ValueError(
            f"scene provider {name!r} already has a {existing.kind!r} origin; "
            "only an unattributed or pack-code registration may be labeled"
        )
    _SCENE_PROVIDER_ORIGINS[name] = origin
    if params is not None:
        _SCENE_PROVIDER_PARAMS[name] = MappingProxyType(dict(params))


def register_inline_scene_provider(name: str, factory: SceneProvider) -> SceneProvider:
    """Register a process-local ``World.inline`` provider.

    Inline providers live in a private map: they are absent from
    :func:`registered_scene_providers` and :func:`scene_provider_providers`,
    while their origin (``kind="inline"``) is inspectable through
    :func:`scene_provider_origin` and captured by the registry snapshot.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"inline scene provider name must be a non-empty string, got {name!r}")
    if name in _SCENE_PROVIDERS or name in _INLINE_SCENE_PROVIDERS:
        raise ValueError(f"inline scene provider {name!r} registered twice")
    checked = _check_factory(name, factory)
    _check_alias(name, checked, _SCENE_PROVIDERS)
    _check_alias(name, checked, _INLINE_SCENE_PROVIDERS)
    _INLINE_SCENE_PROVIDERS[name] = checked
    _SCENE_PROVIDER_ORIGINS[name] = ProviderOrigin.inline()
    return checked


def registered_scene_providers() -> list[str]:
    """The public scene-provider list (sorted; inline providers excluded)."""
    return sorted(_SCENE_PROVIDERS)


def _lookup_scene_provider(name: str) -> SceneProvider | None:
    """Return the public or private inline provider registered under *name*."""
    if name in _SCENE_PROVIDERS:
        return _SCENE_PROVIDERS[name]
    return _INLINE_SCENE_PROVIDERS.get(name)


def build_scene_from_provider(
    name: str,
    params: Mapping[str, JsonValue],
    *,
    preset: str | None = None,
) -> Scene:
    """Dispatch a registered scene provider to a validated Scene.

    The provider is a pure function of ``(params, context)``; its output must
    be a :class:`~unigym.scenes.scene_ir.Scene` and passes the
    :func:`~unigym.scenes.scene_ir.validate_scene` gate.  When the provider
    registered a parameter schema, defaults are filled and the supplied
    mapping is validated BEFORE the call.

    Args:
        name: The registered provider name.
        params: The provider parameter mapping.
        preset: The preset catalog id when the dispatch came from a preset.

    Returns:
        The validated scene.

    Raises:
        ValueError: The name is unknown (the error lists the public registry),
            the params are not a mapping, the parameter schema rejects them,
            or the provider does not return a Scene.
    """
    if not isinstance(params, Mapping):
        raise ValueError(
            f"scene provider {name!r}: params must be a mapping, got {type(params).__name__}"
        )
    factory = _lookup_scene_provider(name)
    if factory is None:
        raise ValueError(
            f"scene provider {name!r} is not registered "
            f"(registered: {registered_scene_providers()})"
        )
    schema = _SCENE_PROVIDER_PARAMS.get(name)
    values = (
        dict(params)
        if schema is None
        else fill_and_validate(schema, params, provider=name, preset=preset)
    )
    scene = factory(values, SceneProviderContext(name=name, preset=preset))
    if not isinstance(scene, Scene):
        raise ValueError(
            f"scene provider {name!r}: provider must return a Scene, " f"got {type(scene).__name__}"
        )
    validate_scene(scene)
    return scene


def scene_provider_origin(name: str) -> ProviderOrigin:
    """Return the origin of one registered provider (never ``None``).

    A registered provider with no recorded origin returns
    :meth:`ProviderOrigin.unknown`; an unknown name raises the registry error
    listing the public registered set.
    """
    if _lookup_scene_provider(name) is None:
        raise ValueError(
            f"scene provider {name!r} is not registered "
            f"(registered: {registered_scene_providers()})"
        )
    return _SCENE_PROVIDER_ORIGINS.get(name, ProviderOrigin.unknown())


def scene_provider_providers(kind: str | None = None) -> tuple[ProviderInfo, ...]:
    """Return the registered scene providers as :class:`ProviderInfo` records.

    Inline providers are excluded.  ``kind`` selects the provider kind; the
    scene registry owns only ``"scene"``, so any other kind returns an empty
    tuple (``None`` returns every record).
    """
    if kind is not None and kind != "scene":
        return ()
    return tuple(
        ProviderInfo(
            name=name,
            kind="scene",
            origin=_SCENE_PROVIDER_ORIGINS.get(name, ProviderOrigin.unknown()),
            params_schema=_SCENE_PROVIDER_PARAMS.get(name),
        )
        for name in sorted(_SCENE_PROVIDERS)
    )


def snapshot_scene_providers() -> SceneProviderRegistrySnapshot:
    """Return a total snapshot of the public, inline, origin and param maps."""
    return SceneProviderRegistrySnapshot(
        providers=dict(_SCENE_PROVIDERS),
        inline=dict(_INLINE_SCENE_PROVIDERS),
        origins=dict(_SCENE_PROVIDER_ORIGINS),
        params=dict(_SCENE_PROVIDER_PARAMS),
    )


def restore_scene_providers(snapshot: SceneProviderRegistrySnapshot) -> None:
    """Restore the scene-provider registries to exactly *snapshot*."""
    if not isinstance(snapshot, SceneProviderRegistrySnapshot):
        raise TypeError(
            f"restore_scene_providers expects a SceneProviderRegistrySnapshot, "
            f"got {type(snapshot).__name__}"
        )
    _SCENE_PROVIDERS.clear()
    _SCENE_PROVIDERS.update(snapshot.providers)
    _INLINE_SCENE_PROVIDERS.clear()
    _INLINE_SCENE_PROVIDERS.update(snapshot.inline)
    _SCENE_PROVIDER_ORIGINS.clear()
    _SCENE_PROVIDER_ORIGINS.update(snapshot.origins)
    _SCENE_PROVIDER_PARAMS.clear()
    _SCENE_PROVIDER_PARAMS.update(snapshot.params)
