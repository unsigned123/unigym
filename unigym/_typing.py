"""Shared typing helpers (no runtime behavior).

``JsonValue`` is the recursive JSON tree accepted by config/declaration
boundaries *before* validation.  Use it instead of ``Any`` for raw config and
JSON payloads: it keeps the data shape honest (scalars, lists and string-keyed
maps) while still allowing arbitrary document trees.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeGuard

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def is_json_object(value: object) -> TypeGuard[dict[str, JsonValue]]:
    """Narrow a runtime value to a JSON object.

    Config trees are JSON by construction; this boundary check accepts a
    mapping and trusts the recursive value shape (validating every leaf on
    each access would be redundant).
    """
    return isinstance(value, dict)


def is_json_list(value: object) -> TypeGuard[list[JsonValue]]:
    """Narrow a runtime value to a JSON list (element shape trusted)."""
    return isinstance(value, list)


def is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Narrow a runtime value to a list/tuple of opaque items."""
    return isinstance(value, (list, tuple))


def is_pair(value: object) -> TypeGuard[tuple[object, object]]:
    """Narrow a runtime value to a two-element tuple of opaque items."""
    return isinstance(value, tuple) and len(value) == 2
