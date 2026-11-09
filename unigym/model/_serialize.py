"""Canonical (de)serialization helpers for the model IR.

Floats are written as ``float.hex()`` strings, mapping keys are sorted and
tuples become lists, so a ``to_dict`` capture is byte-stable across
processes and hosts.  The inverse turns the hex spellings back into floats
while leaving ordinary strings untouched.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import Field, fields, is_dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Protocol,
    SupportsInt,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

__all__ = ["JsonValue", "decode", "encode", "from_row", "row_dict"]

# The JSON wire shape: tuples are accepted as list-like input and encode to
# lists, so every raw row value is a member of this recursive type.
JsonValue: TypeAlias = (
    bool
    | int
    | float
    | str
    | None
    | list["JsonValue"]
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
)

_HEX_RE = re.compile(r"^[+-]?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]+)?p[+-]?\d+$")

_T = TypeVar("_T")


class _DataclassInstance(Protocol):
    """Structural view of a dataclass instance (typeshed-internal protocol)."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def _is_mapping(value: object) -> TypeGuard[Mapping[str, JsonValue]]:
    """Narrow *value* to a string-keyed JSON mapping."""
    return isinstance(value, Mapping)


def _is_dict(value: object) -> TypeGuard[dict[str, JsonValue]]:
    """Narrow *value* to a plain dictionary (the encoder's mapping branch)."""
    return isinstance(value, dict)


def _is_dataclass_instance(value: object) -> TypeGuard[_DataclassInstance]:
    """Narrow to dataclass *instances* (never the dataclass type itself)."""
    return is_dataclass(value) and not isinstance(value, type)


def _is_sequence(
    value: object,
) -> TypeGuard[list[JsonValue] | tuple[JsonValue, ...]]:
    """Narrow *value* to a JSON sequence (lists encode; tuples are accepted)."""
    return isinstance(value, (list, tuple))


def encode(value: Any) -> JsonValue:
    """Return a JSON-safe, deterministic representation of *value*.

    *value* is the codec input boundary: it accepts arbitrary runtime trees
    (scalars, Enums, numpy scalars, tuples, dataclasses).  This is the one
    genuine erase boundary -- callers keep their own concrete types.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        if isinstance(value, SupportsInt):
            return encode(int(value))
        raise TypeError(f"encode: cannot convert {type(value).__name__} enum to int")
    if isinstance(value, float):
        return float.hex(value)
    if isinstance(value, int):
        return value
    if _is_dict(value):
        return {
            str(key): encode(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if _is_sequence(value):
        return [encode(item) for item in value]
    if type(value).__module__.split(".")[0] == "numpy":
        return encode(getattr(value, "item")())
    if _is_dataclass_instance(value):
        return encode(row_dict(value))
    return value


def decode(value: JsonValue) -> JsonValue:
    """Reverse :func:`encode` (hex strings back to floats)."""
    if isinstance(value, str):
        if value in ("nan", "inf", "-inf"):
            return float(value)
        if _HEX_RE.match(value):
            return float.fromhex(value)
        return value
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    return value


def row_dict(row: object) -> dict[str, JsonValue]:
    """Return one dataclass/mapping row as a plain attribute dictionary."""
    if _is_mapping(row):
        return {str(name): value for name, value in row.items()}
    if _is_dataclass_instance(row):
        return {field.name: getattr(row, field.name) for field in fields(row)}
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if _is_mapping(result):
            return {str(name): value for name, value in result.items()}
    return {str(name): value for name, value in vars(row).items()}


def from_row(
    cls: Callable[..., _T], row: Mapping[str, JsonValue], tuple_fields: Iterable[str] = ()
) -> _T:
    """Build a frozen dataclass *cls* from one mapping row."""
    dataclass_fields = getattr(cls, "__dataclass_fields__", {})
    kwargs: dict[str, object] = {}
    for name in dataclass_fields:
        if name not in row:
            continue
        value = row[name]
        if name in tuple_fields and value is not None:
            value = tuple(value) if isinstance(value, (list, tuple)) else value
        kwargs[name] = value
    return cls(**kwargs)
