"""Task providers: the fifth registry.

The scripted fallback seam for pack tasks -- a task provider is a PURE
FUNCTION from data to a FormulaTask spec dict.  It is the escape
hatch for pack tasks whose spec COMBINATION cannot be spelled as
a template x parameter row -- the same role the terms
whitelist plays for judge terms and the Python plugin seam plays for
whole Task classes, one axis per seam:

A task provider is a PURE FUNCTION from data to a FormulaTask spec dict:
``provider(slots, tables) -> spec``.  It is the escape
hatch for pack tasks whose spec COMBINATION cannot be spelled as a
template x parameter row -- the same role the terms whitelist
plays for judge terms and the Python plugin seam plays for whole Task
classes, one axis per seam:

    spec-combination oddity  -> task provider  (this module)
    runtime-judge oddity     -> term        (benchmarks/terms.py)
    reset-behavior oddity    -> initializer (unigym/initializers/)
    geometry oddity          -> pool        (future tree campaign)

Contract (mirrors the terms registry discipline):
    * PURE spec-builder -- no IO, no runtime env, no backend touch; the
      import-linter forbids any unigym import outright (the module must
      stay stdlib + this package's data faces).
    * The docstring must declare the OFFICIAL mechanism the provider
      mirrors (what the official stack does that this replaces).
    * The output crosses the SAME validation gate as every other pack
      task: rewrite_pack_references + validate_formula_spec, run by the
      pack loader (a provider cannot smuggle an invalid spec in).
    * Rows reference providers BY NAME, never by path (a path-shaped
      value fails the name gate at load).
    * One-way stream: providers BUILD specs at load time; nothing writes
      back at runtime.

YAGNI note: today the registry carries exactly one documented
example and zero real consumers; the first real task that cannot be a
template row registers here and sets the precedent.
"""

from __future__ import annotations

from typing import Callable

from unigym.provenance import OriginTable, ProviderOrigin, RegistrySnapshot

Provider = Callable[[dict, dict], dict]

PROVIDERS: dict[str, Provider] = {}
"""The closed provider registry: registered name -> pure spec-builder."""

_TASK_PROVIDER_ORIGINS = OriginTable()
"""Private provenance side table beside :data:`PROVIDERS`."""

TaskProviderRegistrySnapshot = RegistrySnapshot[Provider]
"""The origin-aware task-provider registry snapshot type."""

__all__ = [
    "PROVIDERS",
    "Provider",
    "TaskProviderRegistrySnapshot",
    "attribute_task_provider",
    "build_provider_spec",
    "register",
    "restore_task_providers",
    "snapshot_task_providers",
    "task_provider_origin",
]

_PROVIDER_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")


def register(name: str, *, origin: ProviderOrigin | None = None) -> Callable[[Provider], Provider]:
    """Register a provider under ``name`` (lowercase snake_case).

    Duplicate names and path-shaped values fail fast -- a provider row
    references the registry entry, never a file (data cites
    NAMES, never paths).
    """

    def deco(fn: Provider) -> Provider:
        from unigym.registration import PackFace, require_registration

        require_registration(PackFace.TASK_PROVIDER, name=name)
        if not callable(fn):
            raise ValueError(
                f"task provider {name!r} must be a callable, got " f"{type(fn).__name__}"
            )
        if not (isinstance(name, str) and name and set(name) <= _PROVIDER_NAME_CHARS):
            raise ValueError(
                f"task provider name {name!r} invalid (lowercase "
                "snake_case -- rows reference providers BY NAME, never "
                "by path; a path-shaped value is rejected here so it "
                "can never look like a legal reference)"
            )
        if name in PROVIDERS:
            raise ValueError(
                f"task provider {name!r} registered twice "
                f"({PROVIDERS[name]!r} and {fn!r}) -- rename one"
            )
        if not (getattr(fn, "__doc__", "") or "").strip():
            raise ValueError(
                f"task provider {name!r}: a docstring is required -- it "
                "must declare the official mechanism the provider "
                "mirrors and restate the pure-builder contract"
            )
        PROVIDERS[name] = fn
        if origin is None:
            _TASK_PROVIDER_ORIGINS.remove(name)
        else:
            if not isinstance(origin, ProviderOrigin):
                raise TypeError(
                    f"task provider {name!r}: origin must be a ProviderOrigin, got "
                    f"{type(origin).__name__}"
                )
            _TASK_PROVIDER_ORIGINS.set(name, origin)
        return fn

    return deco


def attribute_task_provider(name: str, origin: ProviderOrigin) -> None:
    """Label one already-registered task provider with *origin*."""
    if name in PROVIDERS:
        _TASK_PROVIDER_ORIGINS.set(name, origin)


def task_provider_origin(name: str) -> ProviderOrigin:
    """Return the origin of one registered task provider (never ``None``)."""
    return _TASK_PROVIDER_ORIGINS.origin(name)


def snapshot_task_providers() -> TaskProviderRegistrySnapshot:
    """Capture the task-provider registry and origins together."""
    return RegistrySnapshot.capture(PROVIDERS, _TASK_PROVIDER_ORIGINS)


def restore_task_providers(snapshot: TaskProviderRegistrySnapshot) -> None:
    """Restore a snapshot captured by :func:`snapshot_task_providers`."""
    snapshot.restore(PROVIDERS, _TASK_PROVIDER_ORIGINS)


def build_provider_spec(name: str, slots: dict, tables: dict) -> dict:
    """Dispatch one provider call; loud ValueError on unknown names.

    ``slots`` is the row's slot mapping, ``tables`` the pack's loaded
    table registry ({"task_rows": <the full rows table>}) -- providers
    may cross-reference rows by name, never paths.  The RETURNED dict is
    NOT validated here: the pack loader runs the same gate it runs for
    every task (rewrite + validate_formula_spec), so a broken provider
    output fails the pack load with the validator's own message.
    """
    if name not in PROVIDERS:
        raise ValueError(
            f"unknown task provider {name!r} (registered: "
            f"{sorted(PROVIDERS)}; rows reference providers by NAME -- "
            "a path-shaped value is rejected by the row schema)"
        )
    spec = PROVIDERS[name](slots, tables)
    if not isinstance(spec, dict):
        raise ValueError(
            f"task provider {name!r} must return a dict (the formula "
            f"task section), got {type(spec).__name__}"
        )
    return spec


@register("example_shape")
def _example_shape(slots: dict, tables: dict) -> dict:
    """Documented minimal example.

    A single-metric presence task built from scalar slots.

    Mirrors the official mechanism: robocasa's task-class composition
    (the tabletop task classes assemble per-task criteria in
    Python; this provider is the declarative-image of that seam).
    Pure spec-builder: no IO, no runtime env, no backend touch; the
    loader runs the returned dict through rewrite_pack_references +
    validate_formula_spec (the same gate as every pack task).
    ``slots``: description (str), key (str), threshold (float).
    """
    del tables  # the example needs no cross-refs; the face stays uniform
    return {
        "description": slots["description"],
        "metrics": [
            {"id": "presence", "term": "final", "key": slots["key"]},
        ],
        "success": {"metric": "presence", "gte": slots["threshold"]},
    }
