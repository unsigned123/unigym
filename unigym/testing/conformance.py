"""Provider conformance kit (v079 §5).

Seven mechanical checks over a provider manifest, shared by the core
synthetic providers and the source-side family sweep:

1. dispatch integrity -- the dispatch table keys and the generated data
   module set are the same set (no orphan row, no unbacked table);
2. build determinism -- the same variant built twice is bit-identical;
3. compile determinism -- the same variant compiled twice is
   field-identical;
4. structural self-check -- observed counts / reference existence /
   attribute round-trip / region parity (the ``verify_generated_data``
   face, expressed as data rather than an XML parser);
5. region binding namespace -- every region key is the fully-qualified
   ``<instance>.<region>`` pair, unique per provider;
6. scale face -- a registry ``size`` override resolves through the
   official scale arithmetic to the declared factor;
7. naming discipline -- lowercase snake_case, no generic path specials.

A manifest is a small declaration; this module never imports a provider
or a backend by itself.  The core tests build synthetic manifests; the
source tests map the robocasa family tables onto the same structure.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, is_dataclass
from typing import Any

from unigym.props.scaling import scale_from_registry_size

__all__ = [
    "ConformanceError",
    "ConformanceManifest",
    "StructureFacts",
    "check_build_determinism",
    "check_compile_determinism",
    "check_dispatch_integrity",
    "check_naming",
    "check_region_namespace",
    "check_scale_face",
    "canonical_digest",
    "check_structure",
    "run_conformance",
    "scene_digest",
]

_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_QUALIFIED_RE = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)+")

#: generic path/placeholder tokens the localization rule forbids in names
_GENERIC_TOKENS = ("default", "generic", "misc", "temp", "tmp")


class ConformanceError(ValueError):
    """A provider manifest violated one conformance contract."""


@dataclass(frozen=True)
class StructureFacts:
    """The structural self-check payload for one variant.

    ``counts`` are the observed element counts; ``expected_counts`` the
    source-of-truth counts (official model / declared table).  The
    reference face is ``(references, reference_exists)``; attribute and
    region faces compare observed vs expected mappings directly.
    """

    counts: Mapping[str, int]
    expected_counts: Mapping[str, int]
    references: tuple[str, ...] = ()
    reference_exists: Callable[[str], bool] | None = None
    attributes: Mapping[str, Any] | None = None
    expected_attributes: Mapping[str, Any] | None = None
    regions: Mapping[str, str] | None = None
    expected_regions: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ConformanceManifest:
    """One provider's conformance declaration.

    ``variants`` is the dispatch table (variant -> payload/declaration);
    ``data_variants`` is the generated data module set.  The callables
    are per-variant and optional -- a missing face is reported as
    ``skipped`` rather than silently passing.
    """

    variants: Mapping[str, Any]
    data_variants: frozenset[str]
    build: Callable[[str], Any] | None = None
    compile: Callable[[str], Any] | None = None
    structure: Callable[[str], StructureFacts] | None = None
    regions: Callable[[str], Iterable[Any]] | None = None
    scale: Callable[[str], tuple] | None = None
    names: Callable[[str], Iterable[str]] | None = None


def _freeze(value: Any) -> Any:
    """A deterministic, comparable projection of an arbitrary payload."""
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__name__,
            tuple((f.name, _freeze(getattr(value, f.name)))
                  for f in value.__dataclass_fields__.values()),
        )
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _eq(left: Any, right: Any, what: str) -> None:
    if _freeze(left) != _freeze(right):
        raise ConformanceError(f"{what}: determinism/parity mismatch")


def check_dispatch_integrity(variants: Mapping[str, Any],
                             data_variants: Iterable[str]) -> None:
    """The dispatch table and the generated data set are the same set."""
    table = set(variants)
    generated = set(data_variants)
    missing = sorted(generated - table)
    extra = sorted(table - generated)
    if missing or extra:
        raise ConformanceError(
            "dispatch integrity: unbacked table rows "
            f"{extra}, data modules with no dispatch {missing}")


def check_build_determinism(build: Callable[[str], Any],
                            variants: Iterable[str]) -> None:
    """Build every variant twice; the two payloads must be identical."""
    for variant in variants:
        _eq(build(variant), build(variant), f"build {variant!r}")


def check_compile_determinism(compile_fn: Callable[[str], Any],
                              variants: Iterable[str]) -> None:
    """Compile every variant twice; the two products must be identical."""
    for variant in variants:
        _eq(compile_fn(variant), compile_fn(variant), f"compile {variant!r}")


def check_structure(facts: StructureFacts) -> None:
    """Counts, references, attribute round-trip and region parity."""
    if facts.counts != facts.expected_counts:
        raise ConformanceError(
            f"structural counts: observed {dict(facts.counts)!r} != "
            f"expected {dict(facts.expected_counts)!r}")
    if (facts.references or facts.reference_exists) \
            and facts.reference_exists is None:
        raise ConformanceError(
            "structural references declared without an existence probe")
    for reference in facts.references:
        if not facts.reference_exists(reference):
            raise ConformanceError(
                f"structural reference missing: {reference!r}")
    if (facts.attributes or facts.expected_attributes) and (
            facts.attributes != facts.expected_attributes):
        raise ConformanceError(
            "structural attribute round-trip: "
            f"{facts.attributes!r} != {facts.expected_attributes!r}")
    if (facts.regions or facts.expected_regions) and (
            facts.regions != facts.expected_regions):
        raise ConformanceError(
            f"structural region parity: {facts.regions!r} != "
            f"{facts.expected_regions!r}")


def _region_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{value[0]}.{value[1]}"
    raise ConformanceError(f"region entry must be a string or pair: {value!r}")


def check_region_namespace(names: Iterable[Any]) -> None:
    """Every region key is a unique, fully-qualified ``instance.region``."""
    seen: set[str] = set()
    for raw in names:
        key = _region_key(raw)
        if not _QUALIFIED_RE.fullmatch(key):
            raise ConformanceError(
                f"region key {key!r} is not <instance>.<region>")
        if key in seen:
            raise ConformanceError(f"duplicate region key {key!r}")
        seen.add(key)


def check_scale_face(size: Any, native: tuple,
                     expected: tuple | None = None,
                     max_size: Any = None) -> None:
    """A registry ``size`` triple resolves through the official arithmetic."""
    got = scale_from_registry_size(size, native, max_size)
    if expected is not None:
        _eq(got, tuple(expected), "scale face")
    if any(not math.isfinite(v) or v <= 0 for v in got):
        raise ConformanceError(f"scale face produced non-positive {got!r}")


def check_naming(names: Iterable[str]) -> None:
    """Lowercase snake_case, localized, no generic path specials."""
    for name in names:
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ConformanceError(
                f"name {name!r} is not lowercase snake_case")
        if any(token in name for token in _GENERIC_TOKENS):
            raise ConformanceError(
                f"name {name!r} carries a generic token "
                f"{[t for t in _GENERIC_TOKENS if t in name]}")
        if "/" in name or "\\" in name:
            raise ConformanceError(f"name {name!r} carries a path separator")


_HEX_FLOAT_RE = re.compile(r"[-+]?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?p[-+]?\d+")
_DECIMAL_FLOAT_RE = re.compile(
    r"[-+]?(?:(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)")


def _canonical_json(value: Any) -> str:
    """A deterministic compact JSON projection of one normalised value."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _check_float(value: str) -> None:
    """Validate a hex float spelling and reject a decimal one."""
    if _HEX_FLOAT_RE.fullmatch(value):
        try:
            float.fromhex(value)
        except ValueError:
            raise ConformanceError(
                f"scene digest: malformed hex float {value!r}") from None
        return
    if _DECIMAL_FLOAT_RE.fullmatch(value):
        raise ConformanceError(
            f"scene digest: {value!r} is a decimal float -- describe() must "
            "emit float.hex() strings (no repr drift)")


def _canonicalise(value: Any) -> Any:
    """Recursively validate and order one neutral description node."""
    if isinstance(value, Mapping):
        return {str(key): _canonicalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_canonicalise(item) for item in value]
        if all(isinstance(item, Mapping) for item in items):
            return sorted(items, key=_canonical_json)
        return items
    if isinstance(value, float):
        raise ConformanceError(
            f"scene digest: bare float {value!r} -- describe() must emit "
            "float.hex() strings (no repr drift)")
    if isinstance(value, str):
        _check_float(value)
    return value


def canonical_digest(description: Mapping[str, object]) -> dict:
    """Canonicalise one engine-neutral structural description.

    The engine owns row extraction; the core only makes the description
    deterministic and rejects float drift.  Every list of mappings is
    sorted by its canonical JSON (so row order is engine-independent);
    lists of primitives keep their declared order (a pose triple and a
    texture binding list are ordered).  No element or format vocabulary is
    read here.

    Args:
        description: The value returned by ``SceneEngine.describe``.

    Returns:
        A JSON-ready, deterministically ordered mapping.

    Raises:
        ConformanceError: The description is not a mapping, carries a bare
            ``float``, or spells a float in decimal form.
    """
    if not isinstance(description, Mapping):
        raise ConformanceError(
            "scene digest: describe() must return a mapping, got "
            f"{type(description).__name__}")
    return _canonicalise(description)


def scene_digest(engine, scene) -> dict:
    """Describe one composed scene and return its structural digest.

    The digest crosses the engine's coarse ``compose`` -> ``describe``
    seam only, so it is genuinely engine-agnostic: the core never parses
    a serialization, and ``to_xml`` is not consulted.  The engine's
    mapping carries its own stable ``digest_version`` tag (the MuJoCo
    engine answers ``unigym.modelir/2``); floats arrive as
    ``float.hex()`` strings, enforced by :func:`canonical_digest`.

    Args:
        engine: A :class:`~unigym.scenes.engine.SceneEngine`.
        scene: The neutral scene to compose.

    Returns:
        A JSON-ready structural digest.

    Raises:
        ConformanceError: The description is malformed (see
            :func:`canonical_digest`).
    """
    return canonical_digest(engine.describe(engine.compose(scene)))


def run_conformance(provider_name: str,
                    manifest: ConformanceManifest) -> dict:
    """Run the seven faces and return a per-face report.

    Raises :class:`ConformanceError` on the first violation; a manifest
    face that is not declared is recorded as ``"skipped"`` (honest: the
    report never claims a check it did not run).
    """
    if not isinstance(manifest, ConformanceManifest):
        raise ConformanceError(
            f"{provider_name}: manifest must be a ConformanceManifest")
    variants = tuple(sorted(str(v) for v in manifest.variants))
    report = {
        "provider": provider_name,
        "variant_count": len(variants),
        "data_variant_count": len(manifest.data_variants),
        "checks": {},
    }

    check_dispatch_integrity(manifest.variants, manifest.data_variants)
    report["checks"]["dispatch"] = "ok"

    if manifest.build is not None:
        check_build_determinism(manifest.build, variants)
        report["checks"]["build"] = "ok"
    else:
        report["checks"]["build"] = "skipped"

    if manifest.compile is not None:
        check_compile_determinism(manifest.compile, variants)
        report["checks"]["compile"] = "ok"
    else:
        report["checks"]["compile"] = "skipped"

    if manifest.structure is not None:
        for variant in variants:
            check_structure(manifest.structure(variant))
        report["checks"]["structure"] = "ok"
    else:
        report["checks"]["structure"] = "skipped"

    if manifest.regions is not None:
        for variant in variants:
            check_region_namespace(manifest.regions(variant))
        report["checks"]["regions"] = "ok"
    else:
        report["checks"]["regions"] = "skipped"

    if manifest.scale is not None:
        for variant in variants:
            payload = manifest.scale(variant)
            if payload is not None:
                check_scale_face(*payload)
        report["checks"]["scale"] = "ok"
    else:
        report["checks"]["scale"] = "skipped"

    if manifest.names is not None:
        for variant in variants:
            check_naming(manifest.names(variant))
        report["checks"]["naming"] = "ok"
    else:
        report["checks"]["naming"] = "skipped"

    return report
