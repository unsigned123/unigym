"""Shared typing helpers (no runtime behavior).

``JsonValue`` is the recursive JSON tree accepted by config/declaration
boundaries *before* validation.  Use it instead of ``Any`` for raw config and
JSON payloads: it keeps the data shape honest (scalars, lists and string-keyed
maps) while still allowing arbitrary document trees.
"""
from __future__ import annotations

from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
