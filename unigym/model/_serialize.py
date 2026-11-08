"""Canonical (de)serialization helpers for the model IR.

Floats are written as ``float.hex()`` strings, mapping keys are sorted and
tuples become lists, so a ``to_dict`` capture is byte-stable across
processes and hosts.  The inverse turns the hex spellings back into floats
while leaving ordinary strings untouched.
"""
from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, cast

__all__ = ["decode", "encode", "from_row", "row_dict"]

_HEX_RE = re.compile(r"^[+-]?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]+)?p[+-]?\d+$")

# JSON codec boundary: the wire form is recursive and heterogeneous
# (str/float/int/list/dict/dataclass/Enum/numpy scalar), so ``Any`` is the
# honest type for the two directions.  Callers keep their own concrete types;
# only this module crosses the erase boundary.
_JSONValue = Any


def encode(value: _JSONValue) -> _JSONValue:
    """Return a JSON-safe, deterministic representation of *value*."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return encode(int(cast(Any, value)))
    if isinstance(value, float):
        return float.hex(value)
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {
            str(key): encode(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return encode(row_dict(value))
    if type(value).__module__.split(".")[0] == "numpy":
        return encode(cast(Any, value).item())
    return value


def decode(value: _JSONValue) -> _JSONValue:
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


def row_dict(row: Any) -> dict[str, Any]:
    """Return one dataclass/mapping row as a plain attribute dictionary."""
    if isinstance(row, dict):
        return dict(row)
    if is_dataclass(row) and not isinstance(row, type):
        return {field.name: getattr(row, field.name) for field in fields(row)}
    return dict(vars(row))


def from_row(cls: type, row: Mapping[str, Any],
             tuple_fields: Iterable[str] = ()) -> Any:
    """Build a frozen dataclass *cls* from one mapping row."""
    kwargs = {}
    for name in cls.__dataclass_fields__:
        if name not in row:
            continue
        value = row[name]
        if name in tuple_fields and value is not None:
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)
