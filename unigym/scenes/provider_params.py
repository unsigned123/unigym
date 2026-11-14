"""One parameter-schema truth for scene providers.

A provider parameter declaration carries its type, optional default and
optional choice set together; there is no separate defaults table.  This module
owns the ONE normalizer for the manifest form
(:func:`normalize_manifest_params`) and the code-registration form
(:func:`normalize_code_params`), the ONE value validator
(:func:`fill_and_validate`) and the ONE drift comparator
(:func:`schemas_equal`).

The JSON type names are ``int`` / ``float`` / ``str`` / ``bool`` / ``list`` /
``dict`` / ``null``.  Python ``bool`` is not an ``int``; ``float`` accepts JSON
integers and floats but not booleans.  A parameter without a default is
required; a parameter with a default is optional and filled at dispatch.
Choices are a non-empty list of distinct values compatible with the declared
type, and an explicit default must be one of them.

The normalizers are pure: they validate the declaration shape and return a
fresh ``name -> ProviderParamSpec`` mapping.  Values are validated here only;
the runtime fills defaults, rejects unknown supplied keys and checks
types/choices BEFORE calling the provider.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TypeGuard, cast

from unigym._typing import JsonValue

__all__ = [
    "JSON_TYPES",
    "ProviderParamSpec",
    "SceneProviderParamError",
    "fill_and_validate",
    "normalize_code_params",
    "normalize_manifest_params",
    "schemas_equal",
]

JSON_TYPES = ("int", "float", "str", "bool", "list", "dict", "null")

_PYTHON_TYPE_NAMES: dict[type, str] = {
    bool: "bool",
    dict: "dict",
    float: "float",
    int: "int",
    list: "list",
    str: "str",
    type(None): "null",
}

_DECLARATION_KEYS = frozenset({"type", "default", "choices"})


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    """Whether *value* is a mapping with opaque keys and values."""
    return isinstance(value, Mapping)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Whether *value* is a non-string sequence."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


class SceneProviderParamError(ValueError):
    """A provider parameter schema or supplied value failed validation."""


@dataclass(frozen=True)
class ProviderParamSpec:
    """One normalized provider parameter declaration.

    ``required`` is true when no default was declared; ``default`` is then
    ignored.  ``choices`` is ``None`` when the parameter is unconstrained.
    """

    type: str
    required: bool
    default: JsonValue = None
    choices: tuple[JsonValue, ...] | None = None


def _type_name(value: object, *, where: str) -> str:
    """Normalize one declared type (JSON name or Python scalar type)."""
    if isinstance(value, str):
        if value in JSON_TYPES:
            return value
    elif value is None:
        return "null"
    elif isinstance(value, type):
        name = _PYTHON_TYPE_NAMES.get(value)
        if name is not None:
            return name
    raise SceneProviderParamError(
        f"{where}: parameter type must be one of {JSON_TYPES} or a Python "
        f"scalar type, got {value!r}"
    )


def _matches(type_name: str, value: object) -> bool:
    """Whether *value* is a legal JSON value of *type_name*."""
    if type_name == "null":
        return value is None
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, float) or (isinstance(value, int) and not isinstance(value, bool))
    if type_name == "str":
        return isinstance(value, str)
    if type_name == "list":
        return isinstance(value, list)
    return isinstance(value, Mapping)


def _validate_value(type_name: str, parameter: str, value: object, *, where: str) -> JsonValue:
    """Validate one value against a declared type and return its JSON form."""
    if not _matches(type_name, value):
        raise SceneProviderParamError(
            f"{where}: parameter {parameter!r} must be {type_name}, got " f"{type(value).__name__}"
        )
    if type_name in ("dict", "list"):
        # ``_matches`` proved a Mapping/list at runtime; the deep copy keeps a
        # validated default from aliasing the declaration, the stored schema or
        # another dispatch's filled mapping.
        return cast(JsonValue, deepcopy(value))
    # ``_matches`` proved the scalar type; this cast records the validated fact.
    return cast(JsonValue, value)


def _normalize_choices(raw: object, type_name: str, *, where: str) -> tuple[JsonValue, ...]:
    """Normalize a ``choices`` declaration into a validated tuple."""
    if not _is_sequence(raw):
        raise SceneProviderParamError(
            f"{where}: choices must be a non-empty list of {type_name} values, got {raw!r}"
        )
    values = list(raw)
    if not values:
        raise SceneProviderParamError(f"{where}: choices must be a non-empty list")
    normalized = [
        _validate_value(type_name, f"choices[{index}]", value, where=where)
        for index, value in enumerate(values)
    ]
    for index, value in enumerate(normalized):
        for other in normalized[index + 1 :]:
            if value == other:
                raise SceneProviderParamError(
                    f"{where}: choices must be distinct, got duplicate {value!r}"
                )
    return tuple(normalized)


def _check_parameter_name(name: object, *, where: str) -> str:
    """Validate one parameter name (non-empty string)."""
    if not isinstance(name, str) or not name:
        raise SceneProviderParamError(
            f"{where}: parameter names must be non-empty strings, got {name!r}"
        )
    return name


def _normalize_mapping(raw: Mapping[object, object], *, where: str) -> ProviderParamSpec:
    """Normalize one full mapping declaration (manifest and code forms)."""
    unknown = [key for key in raw if key not in _DECLARATION_KEYS]
    if unknown:
        raise SceneProviderParamError(
            f"{where}: unknown declaration key(s) {sorted(map(str, unknown))} "
            f"(known: {sorted(_DECLARATION_KEYS)})"
        )
    if "type" not in raw:
        raise SceneProviderParamError(f"{where}: declaration requires 'type'")
    type_name = _type_name(raw["type"], where=f"{where}.type")
    choices = (
        _normalize_choices(raw["choices"], type_name, where=where) if "choices" in raw else None
    )
    if "default" not in raw:
        return ProviderParamSpec(type=type_name, required=True, choices=choices)
    default = _validate_value(type_name, "default", raw["default"], where=where)
    if choices is not None and not any(default == choice for choice in choices):
        raise SceneProviderParamError(
            f"{where}: default {default!r} must be one of the choices " f"{list(choices)}"
        )
    return ProviderParamSpec(type=type_name, required=False, default=default, choices=choices)


def _normalize_compact(raw: Sequence[object], *, where: str) -> ProviderParamSpec:
    """Normalize the compact ``type`` / ``(type, default[, choices])`` form."""
    parts = list(raw)
    if not 1 <= len(parts) <= 3:
        raise SceneProviderParamError(
            f"{where}: compact declaration must be (type[, default[, choices]]), "
            f"got {len(parts)} element(s)"
        )
    type_name = _type_name(parts[0], where=f"{where}.type")
    if len(parts) == 1:
        return ProviderParamSpec(type=type_name, required=True)
    default = _validate_value(type_name, "default", parts[1], where=where)
    if len(parts) == 2:
        return ProviderParamSpec(type=type_name, required=False, default=default)
    choices = _normalize_choices(parts[2], type_name, where=where)
    if not any(default == choice for choice in choices):
        raise SceneProviderParamError(
            f"{where}: default {default!r} must be one of the choices " f"{list(choices)}"
        )
    return ProviderParamSpec(type=type_name, required=False, default=default, choices=choices)


def _normalize_declaration(raw: object, *, where: str) -> ProviderParamSpec:
    """Normalize one code-form declaration."""
    if _is_mapping(raw):
        return _normalize_mapping(raw, where=where)
    if _is_sequence(raw):
        return _normalize_compact(raw, where=where)
    return ProviderParamSpec(type=_type_name(raw, where=f"{where}.type"), required=True)


def normalize_manifest_params(
    raw: Mapping[str, Mapping[str, JsonValue]],
) -> dict[str, ProviderParamSpec]:
    """Normalize the manifest ``params:`` declaration mapping.

    Args:
        raw: ``{<parameter>: {type, default?, choices?}}`` as declared in a
            ``kind: scene`` manifest item.

    Returns:
        A fresh ``{<parameter>: ProviderParamSpec}`` mapping.

    Raises:
        SceneProviderParamError: The mapping or one declaration is malformed.
    """
    if not _is_mapping(raw):
        raise SceneProviderParamError(
            f"provider params must be a mapping, got {type(raw).__name__}"
        )
    schema: dict[str, ProviderParamSpec] = {}
    for name, declaration in raw.items():
        parameter = _check_parameter_name(name, where="provider params")
        if not _is_mapping(declaration):
            raise SceneProviderParamError(
                f"provider params[{parameter!r}]: declaration must be a mapping, "
                f"got {type(declaration).__name__}"
            )
        schema[parameter] = _normalize_mapping(
            declaration,
            where=f"provider params[{parameter!r}]",
        )
    return schema


def normalize_code_params(raw: Mapping[str, object]) -> dict[str, ProviderParamSpec]:
    """Normalize the code-registration ``params=`` declaration mapping.

    The compact ``int`` / ``(int, 1)`` / ``(str, "default", ("a", "b"))`` forms
    and the full ``{"type": "int", "default": 1}`` mapping normalize to the
    same :class:`ProviderParamSpec` mapping.

    Args:
        raw: ``{<parameter>: <declaration>}`` passed to
            :func:`register_scene_provider`.

    Returns:
        A fresh ``{<parameter>: ProviderParamSpec}`` mapping.

    Raises:
        SceneProviderParamError: The mapping or one declaration is malformed.
    """
    if not _is_mapping(raw):
        raise SceneProviderParamError(
            f"provider params must be a mapping, got {type(raw).__name__}"
        )
    schema: dict[str, ProviderParamSpec] = {}
    for name, declaration in raw.items():
        parameter = _check_parameter_name(name, where="provider params")
        schema[parameter] = _normalize_declaration(
            declaration, where=f"provider params[{parameter!r}]"
        )
    return schema


def _where(provider: str, preset: str | None) -> str:
    """The dispatch error prefix (provider, and the preset when one is known)."""
    if preset is None:
        return f"scene provider {provider!r}"
    return f"scene provider {provider!r} (preset {preset!r})"


def fill_and_validate(
    schema: Mapping[str, ProviderParamSpec],
    params: Mapping[str, JsonValue],
    *,
    provider: str,
    preset: str | None = None,
) -> dict[str, JsonValue]:
    """Fill defaults and validate supplied values against *schema*.

    Unknown supplied keys are rejected, required parameters must be present,
    and every value is checked for type and (when declared) choices BEFORE the
    provider is called.

    Args:
        schema: The normalized provider parameter schema.
        params: The supplied parameter mapping.
        provider: The provider name named in every error.
        preset: The preset catalog id when the dispatch came from a preset.

    Returns:
        A fresh mapping with defaults filled and every value validated.

    Raises:
        SceneProviderParamError: The supplied mapping is not a mapping, names
            an unknown/required-missing key, or carries an invalid value.
    """
    where = _where(provider, preset)
    if not isinstance(params, Mapping):
        raise SceneProviderParamError(
            f"{where}: params must be a mapping, got {type(params).__name__}"
        )
    unknown = [key for key in params if key not in schema]
    if unknown:
        raise SceneProviderParamError(
            f"{where}: unknown parameter(s) {sorted(map(str, unknown))} "
            f"(declared: {sorted(schema)})"
        )
    missing = [name for name, spec in schema.items() if spec.required and name not in params]
    if missing:
        raise SceneProviderParamError(
            f"{where}: missing required parameter(s) {sorted(missing)} "
            f"(declared: {sorted(schema)})"
        )
    filled: dict[str, JsonValue] = {}
    for name, spec in schema.items():
        # Every dispatch validates its value, including the default branch:
        # validation deep-copies mutable defaults so no two dispatches (or the
        # stored schema) ever share one list/dict instance.
        supplied = params[name] if name in params else spec.default
        value = _validate_value(spec.type, name, supplied, where=where)
        if spec.choices is not None and not any(value == choice for choice in spec.choices):
            raise SceneProviderParamError(
                f"{where}: parameter {name!r} value {value!r} must be one of "
                f"{list(spec.choices)}"
            )
        filled[name] = value
    return filled


def schemas_equal(
    left: Mapping[str, ProviderParamSpec] | None,
    right: Mapping[str, ProviderParamSpec] | None,
) -> bool:
    """Whether two normalized provider schemas (including ``None``) match."""
    if left is None or right is None:
        return left is None and right is None
    return dict(left) == dict(right)
