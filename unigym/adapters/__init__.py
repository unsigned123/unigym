"""Adapter registry. Imports must stay torch-only for local unit tests."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from typing import cast

from unigym._typing import JsonValue
from unigym.provenance import (
    OriginTable,
    ProviderInfo,
    ProviderOrigin,
    RegistrySnapshot,
)

from .base import Adapter, RawObsRequest  # noqa: F401  (package-level re-export)
from .params import (  # noqa: F401  (package-level re-export)
    REQUIRED,
    adapter_params_registry,
    adapter_params_schema,
    declared_params,
    check_spawn_values,
    validate_effective_section,
)
from .manual import ManualAdapter

ADAPTER_REGISTRY: dict[str, type[Adapter]] = {
    # The zero-config / explicit-loop adapter: no policy forward; the
    # external driver feeds actions through ExternalRunner.step(actions).
    "manual": ManualAdapter,
}

_ADAPTER_ORIGINS = OriginTable(
    {"manual": ProviderOrigin(kind="core")},
)
"""Private provenance side table beside :data:`ADAPTER_REGISTRY`."""

AdapterRegistrySnapshot = RegistrySnapshot[type[Adapter]]
"""The origin-aware adapter registry snapshot type."""

_ADAPTER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


def register_adapter(
    name: str,
    cls: type[Adapter],
    *,
    origin: ProviderOrigin | None = None,
) -> type[Adapter]:
    """Register one adapter class in the process-global adapter registry.

    The registry is shared by dynamic pack code and manifest-declared
    providers, so a duplicate name always fails instead of silently replacing
    the existing entry.  ``name`` may be dot-namespaced (``acme.policy``) but
    every segment must be lowercase snake_case.

    Args:
        name: Adapter name the run config selects in ``adapter.name``.
        cls: An :class:`~unigym.adapters.base.Adapter` subclass (not an
            instance and not a factory).
        origin: Internal/runtime-only provenance; pack authors never pass it.
            A registration without an origin reports
            :meth:`ProviderOrigin.unknown`.

    Returns:
        *cls*, so the helper can be used as a decorator.

    Raises:
        ValueError: The name is malformed or already registered.
        TypeError: *cls* is not a class or does not subclass ``Adapter``.
    """
    if not isinstance(name, str) or not _ADAPTER_NAME_RE.fullmatch(name):
        raise ValueError(
            f"adapter name {name!r} must be lowercase snake_case, optionally "
            f"dot-namespaced (e.g. 'acme.policy')"
        )
    if not isinstance(cls, type):
        raise TypeError(f"adapter {name!r} target must be a class, got {type(cls).__name__}")
    if not issubclass(cls, Adapter):
        raise TypeError(f"adapter {name!r} target {cls!r} must subclass Adapter")
    if origin is not None and not isinstance(origin, ProviderOrigin):
        raise TypeError(
            f"adapter {name!r}: origin must be a ProviderOrigin, got {type(origin).__name__}"
        )
    existing = ADAPTER_REGISTRY.get(name)
    if existing is not None:
        raise ValueError(
            f"adapter name {name!r} is already registered to "
            f"{existing.__module__}.{existing.__qualname__}"
        )
    alias = next((key for key, value in ADAPTER_REGISTRY.items() if value is cls), None)
    if alias is not None:
        raise ValueError(
            f"adapter class {cls.__module__}.{cls.__qualname__} is already "
            f"registered as {alias!r}; one class owns exactly one name"
        )
    ADAPTER_REGISTRY[name] = cls
    if origin is None:
        _ADAPTER_ORIGINS.remove(name)
    else:
        _ADAPTER_ORIGINS.set(name, origin)
    return cls


def adapter_origin(name: str) -> ProviderOrigin:
    """Return the origin of one registered adapter (never ``None``).

    A registered adapter with no recorded origin returns
    :meth:`ProviderOrigin.unknown`; the built-in ``manual`` adapter reports
    ``kind="core"``.  An unknown name raises a teaching ``ValueError``.
    """
    if name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"adapter {name!r} is not registered (registered adapters: {sorted(ADAPTER_REGISTRY)})"
        )
    return _ADAPTER_ORIGINS.origin(name)


def adapter_providers(kind: str | None = None) -> tuple[ProviderInfo, ...]:
    """Return the registered adapters as live :class:`ProviderInfo` records.

    ``kind`` selects the provider kind; the adapter registry owns only
    ``"adapter"``, so any other kind returns an empty tuple (``None`` returns
    every record).  Records are ordered by name.
    """
    if kind is not None and kind != "adapter":
        return ()
    return tuple(
        ProviderInfo(name=name, kind="adapter", origin=_ADAPTER_ORIGINS.origin(name))
        for name in sorted(ADAPTER_REGISTRY)
    )


def attribute_adapter(name: str, origin: ProviderOrigin) -> None:
    """Attach runtime-owned provenance to an already-registered adapter.

    This is the pack runtime's activation-window attribution boundary, not
    pack-author API: pack code registers through :func:`register_adapter` and
    never passes an origin here.  It may only label an entry whose current
    origin is absent or ``unknown``; a ``manifest`` or ``core`` origin is never
    relabeled.

    Raises:
        ValueError: The adapter is unknown, or its current origin must not be
            relabeled.
        TypeError: *origin* is not a :class:`ProviderOrigin`.
    """
    if name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"adapter {name!r} is not registered (registered adapters: {sorted(ADAPTER_REGISTRY)})"
        )
    if not isinstance(origin, ProviderOrigin):
        raise TypeError(
            f"adapter {name!r}: origin must be a ProviderOrigin, got {type(origin).__name__}"
        )
    existing = _ADAPTER_ORIGINS.get(name)
    if existing is not None and existing.kind != "unknown":
        raise ValueError(
            f"adapter {name!r} already has a {existing.kind!r} origin; "
            "only an unattributed registration may be labeled"
        )
    _ADAPTER_ORIGINS.set(name, origin)


def snapshot_adapter_registry() -> AdapterRegistrySnapshot:
    """Return a total snapshot of the adapter classes and their origins."""
    return RegistrySnapshot.capture(ADAPTER_REGISTRY, _ADAPTER_ORIGINS)


def restore_adapter_registry(snapshot: AdapterRegistrySnapshot) -> None:
    """Restore the adapter classes and origins to exactly *snapshot*."""
    if not isinstance(snapshot, RegistrySnapshot):
        raise TypeError(
            f"restore_adapter_registry expects an AdapterRegistrySnapshot, "
            f"got {type(snapshot).__name__}"
        )
    snapshot.restore(ADAPTER_REGISTRY, _ADAPTER_ORIGINS)


class AdapterParamsError(ValueError):
    """Spawn-params channel contract violation (teaching payload).

    ``.doc`` carries the structured {error, legal, example, hint} dict the
    faces return verbatim -- the caller corrects itself without reading
    source (the teaching-error style).
    """

    def __init__(self, message: str, doc: dict[str, object]):
        super().__init__(message)
        self.doc = doc


def _teaching_doc(message: str, *, legal: object, example: object, hint: str) -> dict[str, object]:
    return {
        "ok": False,
        "error_code": "bad_adapter_params",
        "error": message,
        "legal": legal,
        "example": example,
        "hint": hint,
    }


#: the one scalar-type whitelist for spawn params (JSON-simple values)
_SCALARS = (bool, int, float, str)
_SPAWN_EXAMPLE = {"name": "manual", "control_dt": 0.02}
_SPAWN_HINT = (
    "spawn params overlay ONLY the yaml 'adapter:' section "
    "(never env/benchmark/cameras); values are scalars or "
    "scalar lists"
)


def validate_spawn_params(params: object) -> dict[str, JsonValue]:
    """Value-domain gate for the spawn-params channel (fail-fast).

    Legal: ``None`` / bool / int / float / str, or a flat list of those.
    Rejected: nested dicts and lists carrying dicts/lists -- the channel
    is a shallow overlay of scalar knobs, not a second config system.

    Returns *params* (as a plain dict) on success; raises
    :class:`AdapterParamsError` with the teaching payload otherwise.
    """
    if params is None:
        return {}
    if isinstance(params, str):
        raise AdapterParamsError(
            f"adapter_params must be an object, got a string: " f"{str(params)[:120]!r}",
            _teaching_doc(
                "send adapter_params as the structured JSON object, never " "as a stringified one",
                legal="object of scalar / scalar-list values",
                example=_SPAWN_EXAMPLE,
                hint=_SPAWN_HINT,
            ),
        )
    if not isinstance(params, dict):
        raise AdapterParamsError(
            f"adapter_params must be an object, got " f"{type(params).__name__}",
            _teaching_doc(
                "adapter_params must be a JSON object",
                legal="object of scalar / scalar-list values",
                example=_SPAWN_EXAMPLE,
                hint=_SPAWN_HINT,
            ),
        )
    for key, value in params.items():
        if value is None or isinstance(value, _SCALARS):
            continue
        if isinstance(value, (list, tuple)):
            bad = [v for v in value if not (v is None or isinstance(v, _SCALARS))]
            if bad:
                raise AdapterParamsError(
                    f"adapter_params[{key!r}]: lists must carry scalars "
                    f"only, found {type(bad[0]).__name__} entries",
                    _teaching_doc(
                        f"adapter_params[{key!r}] is not a scalar list",
                        legal="flat list of scalars (bool/int/float/str)",
                        example={key: [1, 2, 3]},
                        hint=_SPAWN_HINT,
                    ),
                )
            continue
        raise AdapterParamsError(
            f"adapter_params[{key!r}]: nested structures are not "
            f"accepted ({type(value).__name__})",
            _teaching_doc(
                f"adapter_params[{key!r}] must be a scalar or a flat "
                f"scalar list -- nested dicts are rejected (the channel "
                f"overlays the adapter section shallowly)",
                legal="scalar (bool/int/float/str/null) or flat scalar " "list",
                example=_SPAWN_EXAMPLE,
                hint=_SPAWN_HINT,
            ),
        )
    return dict(params)


def _legal_keys(name: str) -> set[str] | None:
    """Constructor keyword set of one registry adapter (None = **kwargs
    present, nothing to check against)."""
    sig = inspect.signature(ADAPTER_REGISTRY[name].__init__)
    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params} - {"self", "num_envs", "device"}


def apply_spawn_params(
    section: Mapping[str, JsonValue],
    params: object,
) -> tuple[dict[str, JsonValue], list[str]]:
    """Overlay spawn adapter_params onto the yaml adapter section (shallow).

    Priority chain: defaults < config yaml < spawn adapter_params (the
    most specific, per-spawn layer).  SCOPE LOCK: the overlay merges into
    the adapter SECTION ONLY -- env/benchmark/cameras/fuse-schedule are
    architecturally unreachable from here (the caller hands in one
    section, not the config tree).

    Semantics:
    * no ``name`` in *params* (or the config's own name) -> shallow merge:
      ``{**section, **params}``;
    * ``name`` OVERRIDDEN to a different adapter -> the section is
      REPLACED by the spawn params: the previous adapter's constructor
      keys would poison the new one (stale ``kinematics``/``control_port``-style
      keys are not the new adapter's business);
    * every *params* key must be a legal constructor keyword of the FINAL
      adapter (``name`` handled separately) -- unknown keys fail-fast with
      the teaching payload
      listing the legal set + an example;
    * DECLARATION-FIRST (the PARAMS convention): an adapter WITH a
      ``PARAMS`` table accepts only its declared keys (types gated too)
      -- constructor kwargs outside the declaration are config-yaml
      surface, not spawn surface; undeclared adapters keep the pure
      signature inference.

    Returns ``(effective_section, overridden_keys)``.
    """
    params = validate_spawn_params(params)
    if not params:
        return dict(section or {}), []
    base = dict(section or {})
    switching = "name" in params and params["name"] != base.get("name")
    if switching:
        merged = dict(params)
    else:
        merged = {**base, **params}
    final_name = merged.get("name")
    if final_name is None:
        raise AdapterParamsError(
            "adapter section has no 'name' (spawn params must carry it)",
            _teaching_doc(
                "the adapter section carries no 'name'",
                legal=f"one of {sorted(ADAPTER_REGISTRY)}",
                example=_SPAWN_EXAMPLE,
                hint="spawn params may set 'name' to switch the " "adapter",
            ),
        )
    if final_name not in ADAPTER_REGISTRY:
        raise AdapterParamsError(
            f"unknown adapter name {final_name!r}",
            _teaching_doc(
                f"unknown adapter name {final_name!r}",
                legal=f"one of {sorted(ADAPTER_REGISTRY)}",
                example=_SPAWN_EXAMPLE,
                hint="pick a registered adapter name",
            ),
        )
    declared = declared_params(ADAPTER_REGISTRY[final_name])
    if declared is not None:
        check_spawn_values(final_name, params)
        legal = set(declared)
    else:
        legal = _legal_keys(final_name)
    if legal is not None:
        unknown = set(params) - legal - {"name"}
        if unknown:
            raise AdapterParamsError(
                f"unknown adapter_params keys for '{final_name}': " f"{sorted(unknown)}",
                _teaching_doc(
                    f"unknown keys {sorted(unknown)} for adapter " f"'{final_name}'",
                    legal=sorted(legal | {"name"}),
                    example=_SPAWN_EXAMPLE,
                    hint="check spelling against the legal-key list; "
                    "config yaml keys stay legal too",
                ),
            )
    if switching:
        # the whole section was replaced: every spawn key is an override
        overridden = sorted(params)
    else:
        overridden = sorted(set(params) & set(base))
    return merged, overridden


def build_adapter(section: dict, num_envs: int, device: str) -> Adapter:
    """Instantiate an adapter from a config dict entry ``{name: ..., **params}``.

    DECLARED adapters (a ``PARAMS`` table on the class) are validated
    here on the EFFECTIVE section (yaml merged with spawn params):
    required present, declared values type-legal, every key either
    declared or a constructor keyword -- all teaching-style errors.
    Undeclared adapters keep the legacy behavior (the constructor's own
    TypeError is the unknown-key gate).
    """
    name = (section or {}).get("name")
    if name not in ADAPTER_REGISTRY:
        raise AdapterParamsError(
            f"unknown adapter name {name!r}",
            _teaching_doc(
                f"unknown adapter name {name!r}",
                legal=f"one of {sorted(ADAPTER_REGISTRY)}",
                example=_SPAWN_EXAMPLE,
                hint="pick a registered adapter name",
            ),
        )
    validate_effective_section(name, section)
    params = {k: v for k, v in section.items() if k != "name"}
    constructor = ADAPTER_REGISTRY[name]
    if not (isinstance(constructor, type) and issubclass(constructor, Adapter)):
        raise TypeError(f"adapter registry entry {name!r} must be an Adapter subclass class")
    # Each adapter class owns its constructor kwargs; ``type[Adapter]`` cannot
    # express that per-class signature, so the callable view is established by
    # the runtime class check above.
    adapter = cast(Callable[..., Adapter], constructor)(num_envs=num_envs, device=device, **params)
    # registry identity: the build-time controls[] partition and the
    # runner's group validation resolve ownership by this name (never by
    # joint-order / action-face inference).
    adapter._unigym_name = name
    return adapter
