"""Closed registry for templates that construct standard PropSpec mappings.

Templates run once while PropManager is constructed. They do not create
handlers, backend objects, or run any per-step work; the returned spec
mapping passes through the ordinary PropSpec and path-validation path.

A provider MAY additionally return a generated asset payload --
:class:`PropTemplateResult` carrying a neutral
:class:`~unigym.props.asset_ir.AssetDefinition` -- as the stage-2 result
contract (design v072 §7/§12).  A plain mapping stays legal as the
legacy no-payload form; closed names, typed params, and fail-fast
validation are unchanged.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..assets.manifest import SCHEMA, validate_manifest
from .asset_ir import AssetDefinition, validate_asset_definition

PropTemplate = Callable[[str, Mapping[str, Any]],
                        "Mapping[str, Any] | PropTemplateResult"]
_TEMPLATES: dict[str, tuple[dict[str, dict], PropTemplate]] = {}
#: asset metadata per registered template (tags / uses / variants); the
#: registry stores it for the asset catalog to consume, nothing reads it yet
_TEMPLATE_METADATA: dict[str, dict[str, Any]] = {}


def _metadata_list(name: str, key: str, value: Any) -> list[str]:
    """Normalize one tag/uses list and reject a non-list spelling."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"prop template {name!r}: {key} must be a list of strings, "
            f"got {type(value).__name__}")
    return list(value)


def _validate_template_metadata(name: str, tags: Any, uses: Any,
                                variants: Any) -> dict[str, Any]:
    """Validate and normalize one template's optional asset metadata.

    Tags and uses follow the ``unigym.asset/1`` rules (v086 R4/R7); the
    synthetic manifest below reuses :func:`validate_manifest` so the
    registry cannot accept metadata the asset catalog could not index.
    Variants are the sorted index of generated variant names -- the facts
    stay in the ``_data`` tables, so the index is not a second fact sheet.
    """
    if variants is None:
        normalized_variants: list[str] = []
    elif isinstance(variants, (str, bytes)) or not isinstance(
            variants, (list, tuple)):
        raise ValueError(
            f"prop template {name!r}: variants must be a list of "
            f"variant-name strings, got {type(variants).__name__}")
    else:
        normalized_variants = list(variants)
    for variant in normalized_variants:
        if not isinstance(variant, str) or not variant:
            raise ValueError(
                f"prop template {name!r}: variant names must be non-empty "
                f"strings, got {variant!r}")
    metadata = {
        "tags": _metadata_list(name, "tags", tags),
        "uses": _metadata_list(name, "uses", uses),
        "variants": normalized_variants,
    }
    validate_manifest({
        "schema": SCHEMA,
        "id": f"pack:robocasa/generator/{name}",
        "kind": "model",
        "files": {"model": "model.xml"},
        "capabilities": {},
        "license": "unspecified",
        "provenance": {"origin": "generator"},
        "tags": metadata["tags"],
        "uses": metadata["uses"],
    })
    return metadata


@dataclass(frozen=True)
class PropTemplateResult:
    """Provider result: the PropSpec mapping + an optional generated asset.

    ``asset`` carries the neutral IR payload (:mod:`unigym.props.asset_ir`);
    ``spec`` stays a plain PropSpec-compatible mapping (the same keys as
    the legacy form).
    """

    spec: Mapping[str, Any]
    asset: AssetDefinition | None = None


def register_prop_template(name: str, *, params: dict[str, dict],
                           tags: list[str] | None = None,
                           uses: list[str] | None = None,
                           variants: list[str] | None = None):
    """Register a closed PropSpec builder, its params and asset metadata.

    ``tags`` are namespaced manifest tags (``pack:<ns>/tag=<name>``) and
    ``uses`` are asset ids; both follow ``unigym.asset/1`` (v086 R4/R7).
    ``variants`` is the sorted index of generated variant names (the facts
    stay in the ``_data`` tables).  The registry stores the three fields
    for the asset catalog; the build path never reads them.
    """
    if (not isinstance(name, str) or not name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                   for c in name)):
        raise ValueError(
            f"prop template name {name!r} must be lowercase snake_case")
    if name in _TEMPLATES:
        raise ValueError(f"prop template {name!r} registered twice")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"prop template {name!r} needs a non-empty params schema")
    metadata = _validate_template_metadata(name, tags, uses, variants)

    def decorate(fn: PropTemplate) -> PropTemplate:
        if not callable(fn):
            raise ValueError(f"prop template {name!r} must be callable")
        if not (getattr(fn, "__doc__", "") or "").strip():
            raise ValueError(f"prop template {name!r} requires a docstring")
        _TEMPLATES[name] = (params, fn)
        _TEMPLATE_METADATA[name] = metadata
        return fn

    return decorate


def prop_template_metadata(name: str) -> dict[str, Any]:
    """Return a copy of one registered template's asset metadata."""
    if name not in _TEMPLATE_METADATA:
        raise ValueError(
            f"unknown prop template {name!r} "
            f"(registered: {sorted(_TEMPLATE_METADATA)})")
    stored = _TEMPLATE_METADATA[name]
    return {
        "tags": list(stored["tags"]),
        "uses": list(stored["uses"]),
        "variants": list(stored["variants"]),
    }


#: the closed param-kind vocabulary of the registration schemas
_PARAM_KINDS = ("string", "enum", "number", "number_list", "size_spec",
                "half_size_spec", "mapping")


def _check_number_list(key: str, value: Any, errors: list) -> None:
    if (not isinstance(value, list)
            or any(isinstance(v, bool)
                   or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        errors.append(f"{key} must be a list of finite numbers")


def _check_size_spec(key: str, value: Any, errors: list) -> None:
    """The official registry size triple (v078 §4 scale face).

    Entries are None (native-fallback axis) or finite positive metres,
    per docs/designs/v078-variant-and-axis-completion.md §4.
    """
    if (not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(entry is not None
                   and (isinstance(entry, bool)
                        or not isinstance(entry, (int, float))
                        or not math.isfinite(entry)
                        or entry <= 0)
                   for entry in value)):
        errors.append(
            f"{key} must be a 3-entry list of None or finite positive numbers")


def _check_half_size_spec(key: str, value: Any, errors: list) -> None:
    """The registry ``handle_config.half_size`` pair (v079 card E).

    Entries are None (the model's raw ``reg_main`` axis) or finite
    positive metres; exactly two entries (the x/y override face of the
    visual-mesh elongated handles).
    """
    if (not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(entry is not None
                   and (isinstance(entry, bool)
                        or not isinstance(entry, (int, float))
                        or not math.isfinite(entry)
                        or entry <= 0)
                   for entry in value)):
        errors.append(
            f"{key} must be a 2-entry list of None or finite positive numbers")


def _check_param_value(key: str, rule: dict, value: Any, errors: list) -> None:
    """One param's kind check, appended to ``errors`` (the kind dispatch)."""
    kind = rule["type"]
    if kind == "string" and not isinstance(value, str):
        errors.append(f"{key} must be a string")
    elif kind == "enum" and value not in rule["values"]:
        errors.append(f"{key} must be one of {rule['values']}")
    elif kind == "number":
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            errors.append(f"{key} must be a finite number")
            return
        if "min" in rule and value < rule["min"]:
            errors.append(f"{key} must be >= {rule['min']}")
        if "max" in rule and value > rule["max"]:
            errors.append(f"{key} must be <= {rule['max']}")
    elif kind == "number_list":
        _check_number_list(key, value, errors)
    elif kind == "size_spec":
        _check_size_spec(key, value, errors)
    elif kind == "half_size_spec":
        _check_half_size_spec(key, value, errors)
    elif kind == "mapping" and not isinstance(value, Mapping):
        errors.append(f"{key} must be a mapping")
    elif kind not in _PARAM_KINDS:
        errors.append(f"template schema has unsupported type {kind!r} for {key}")


def _validate_params(name: str, params: Mapping[str, Any], schema: dict,
                     where: str) -> None:
    if not isinstance(params, Mapping):
        raise ValueError(f"{where}/params must be a mapping")
    # schema keys are required unless the rule declares ``required: False``
    # (v074: the robocasa_object scale variant carrier -- given when the
    # config pins a sampling variant, absent for the registry default).
    missing = sorted(k for k in set(schema) - set(params)
                     if schema[k].get("required", True))
    extra = sorted(set(params) - set(schema))
    errors = []
    if missing:
        errors.append(f"missing params {missing}")
    if extra:
        errors.append(f"unknown params {extra}")
    for key, rule in schema.items():
        if key in params:
            _check_param_value(key, rule, params[key], errors)
    if errors:
        raise ValueError(f"{where}/template {name!r}: " + "; ".join(errors))


def resolve_prop_template(name: str, prop_name: str,
                          params: Mapping[str, Any], *, where: str
                          ) -> tuple[dict, AssetDefinition | None]:
    """Resolve a registered template to (PropSpec mapping, generated asset).

    The generated asset is ``None`` for the legacy plain-mapping form.
    """
    if name not in _TEMPLATES:
        raise ValueError(
            f"{where}/template: unknown prop template {name!r} "
            f"(registered: {sorted(_TEMPLATES)})")
    schema, provider = _TEMPLATES[name]
    _validate_params(name, params, schema, where)
    result = provider(prop_name, dict(params))
    if isinstance(result, PropTemplateResult):
        entry, asset = dict(result.spec), result.asset
    elif isinstance(result, Mapping):
        entry, asset = dict(result), None
    else:
        raise ValueError(
            f"{where}/template {name!r}: provider must return a mapping "
            f"or PropTemplateResult, got {type(result).__name__}")
    if asset is not None:
        if entry.get("asset"):
            raise ValueError(
                f"{where}/template {name!r}: provider returned both an "
                f"asset path and an AssetDefinition (mutually exclusive)")
        validate_asset_definition(asset)
    if entry.get("name") != prop_name:
        raise ValueError(
            f"{where}/template {name!r}: provider changed prop identity "
            f"from {prop_name!r} to {entry.get('name')!r}")
    forbidden = {"template", "params"} & entry.keys()
    if forbidden:
        raise ValueError(
            f"{where}/template {name!r}: provider leaked template keys "
            f"{sorted(forbidden)}")
    return entry, asset


def resolve_prop_entries(config: Mapping[str, Any]
                         ) -> list[tuple[dict, AssetDefinition | None]]:
    """Resolve static and templated entries to ``(entry, asset)`` pairs.

    The provider output is placeholder-expanded through the sanctioned
    :func:`unigym.core.asset_root.expand_placeholders`: providers spell
    host paths with ``${UNIGYM_REPO}`` (the closed asset table), while
    the collect_cli order expands the config BEFORE PropManager runs --
    without this expansion the provider-emitted placeholder would never
    resolve.  Static entries keep their exact pass-through (and never
    carry a payload).
    """
    raw_types = config.get("types", [])
    if not isinstance(raw_types, list):
        raise ValueError("props/types must be a list")
    resolved = []
    names = set()
    for index, raw in enumerate(raw_types):
        where = f"props.types[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where} must be a mapping")
        if "template" not in raw:
            entry, asset = dict(raw), None
        else:
            allowed = {"name", "template", "params"}
            extra = sorted(set(raw) - allowed)
            missing = sorted({"name", "params"} - set(raw))
            if extra or missing:
                raise ValueError(
                    f"{where}: templated prop accepts only name/template/params; "
                    f"missing={missing}, unknown={extra}")
            entry, asset = resolve_prop_template(
                raw["template"], raw["name"], raw["params"], where=where)
            from ..core.asset_root import (canonical_asset_values,
                                           expand_placeholders)
            entry = canonical_asset_values(expand_placeholders(entry))
        prop_name = entry.get("name")
        if not isinstance(prop_name, str) or not prop_name:
            raise ValueError(f"{where}/name must be a non-empty string")
        if prop_name in names:
            raise ValueError(f"{where}/name: duplicate prop name {prop_name!r}")
        names.add(prop_name)
        resolved.append((entry, asset))
    return resolved


def resolve_prop_declarations(config: Mapping[str, Any]) -> list[dict]:
    """Static and templated entries as plain PropSpec mappings.

    Thin wrapper over :func:`resolve_prop_entries` (the stage-2 face);
    PropManager takes the pairs so generated payloads reach
    ``asset_definitions``.
    """
    return [entry for entry, _ in resolve_prop_entries(config)]
