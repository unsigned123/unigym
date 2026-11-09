"""Face-by-face model diffing.

A ``ModelDiff`` answers "what changed between these two models?" without
reaching into a backend: rows are matched by name, every dataclass field
(declared and compiled) is compared, and extension namespaces can be
included as ``ext.<namespace>`` faces.  Comparing a declared IR with a
compiled one is allowed but noisy (compiled-only fields move ``None`` ->
value); ``ignore_compiled=True`` silences exactly those fields.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ._serialize import JsonValue, encode, row_dict

if TYPE_CHECKING:
    from .ir import ModelIR

__all__ = ["ModelDiff"]

_COMPILED_FIELDS: frozenset[str] = frozenset({"mass", "inertia", "ipos", "iquat"})

_PREVIEW = 5

_RowMap = dict[str, JsonValue]
_FaceMap = dict[str, _RowMap]
_FieldChanges = dict[str, tuple[JsonValue, JsonValue]]
_RowChanges = dict[str, _FieldChanges]
_DiffChanges = dict[str, _RowChanges]


def _empty_name_tuple_map() -> dict[str, tuple[str, ...]]:
    """An empty face -> added/removed-name mapping."""
    return {}


def _empty_diff_changes() -> _DiffChanges:
    """An empty face -> row-change mapping."""
    return {}


class _NamedRow(Protocol):
    """A table row carrying its compiled global name."""

    @property
    def name(self) -> str:
        """The row's compiled global name."""
        ...


def _is_compiled_field(name: str) -> bool:
    """Return True for fields a compiled backend fills (ids/addresses)."""
    return name.endswith("_id") or name.endswith("_adr") or name in _COMPILED_FIELDS


def _table_map(table: Iterable[_NamedRow]) -> _FaceMap:
    """Return ``{row name: field dict}`` for one IR table."""
    return {row.name: row_dict(row) for row in table}


def _ext_value(value: JsonValue) -> _FaceMap:
    """Normalize one extension payload value into a name->fields map.

    Entity tables arrive as ``{name: row dict}``; scalar option maps (e.g.
    ``mj.opt``) become ``{key: {"value": scalar}}`` so both diff through
    the same name/field machinery.
    """
    if not isinstance(value, Mapping):
        return {"value": {"value": value}}
    if not value:
        return {}
    if all(isinstance(item, Mapping) for item in value.values()):
        return {
            str(name): dict(item) if isinstance(item, Mapping) else {"value": item}
            for name, item in value.items()
        }
    return {str(name): {"value": item} for name, item in value.items()}


def _faces(ir: ModelIR, include_ext: bool) -> dict[str, _FaceMap]:
    """Return every comparable face as a name->fields map."""
    faces: dict[str, _FaceMap] = {
        "bodies": _table_map(ir.bodies),
        "joints": _table_map(ir.joints),
        "geoms": _table_map(ir.geoms),
        "sites": _table_map(ir.sites),
        "cameras": _table_map(ir.cameras),
        "actuators": _table_map(ir.actuators),
        "materials": _table_map(ir.materials),
    }
    if include_ext:
        for namespace, extension in ir.ext.items():
            for key, value in extension.to_dict().items():
                faces[f"ext.{namespace}.{key}"] = _ext_value(value)
    return faces


def _changed_fields(
    left: _RowMap,
    right: _RowMap,
    ignore_compiled: bool,
) -> _FieldChanges:
    """Return the differing fields between two row maps."""
    changes: _FieldChanges = {}
    for name in sorted(set(left) | set(right)):
        if name == "name":
            continue
        if ignore_compiled and _is_compiled_field(name):
            continue
        before = left.get(name)
        after = right.get(name)
        if before != after:
            changes[name] = (before, after)
    return changes


def _diff_maps(
    left: _FaceMap,
    right: _FaceMap,
    ignore_compiled: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], _RowChanges]:
    """Diff two face maps into added/removed/changed pieces."""
    added = tuple(name for name in right if name not in left)
    removed = tuple(name for name in left if name not in right)
    changed: _RowChanges = {}
    for name in left:
        if name not in right:
            continue
        fields = _changed_fields(left[name], right[name], ignore_compiled)
        if fields:
            changed[name] = fields
    return added, removed, changed


def _count_parts(diff: ModelDiff) -> list[str]:
    """Return the per-face ``+a -r ~c`` count strings for the repr."""
    parts: list[str] = []
    faces = set(diff.added) | set(diff.removed) | set(diff.changed)
    for face in sorted(faces):
        added = len(diff.added.get(face, ()))
        removed = len(diff.removed.get(face, ()))
        changed = len(diff.changed.get(face, {}))
        parts.append(f"{face}: +{added} -{removed} ~{changed}")
    return parts


def _preview_entries(diff: ModelDiff) -> Iterator[str]:
    """Yield a few readable change lines for the repr preview."""
    count = 0
    for face, names in diff.added.items():
        for name in names:
            yield f"  + {face}[{name!r}]"
            count += 1
            if count >= _PREVIEW:
                return
    for face, names in diff.removed.items():
        for name in names:
            yield f"  - {face}[{name!r}]"
            count += 1
            if count >= _PREVIEW:
                return
    for face, entities in diff.changed.items():
        for name, fields in entities.items():
            for field_name, (before, after) in fields.items():
                yield (f"  ~ {face}[{name!r}].{field_name}: " f"{before!r} -> {after!r}")
                count += 1
                if count >= _PREVIEW:
                    return


@dataclass(frozen=True, slots=True)
class ModelDiff:
    """The per-face difference between two model IRs.

    Attributes:
        added: ``{face: (name, ...)}`` rows present only on the right.
        removed: ``{face: (name, ...)}`` rows present only on the left.
        changed: ``{face: {name: {field: (old, new)}}}`` shared rows whose
            compared fields differ.
    """

    added: dict[str, tuple[str, ...]] = field(default_factory=_empty_name_tuple_map)
    removed: dict[str, tuple[str, ...]] = field(default_factory=_empty_name_tuple_map)
    changed: _DiffChanges = field(default_factory=_empty_diff_changes)

    def __bool__(self) -> bool:
        """Return True when any face has an added, removed or changed row."""
        return bool(self.added or self.removed or self.changed)

    def to_dict(self) -> JsonValue:
        """Return a JSON-safe canonical document (floats use ``float.hex``)."""
        return encode(
            {
                "equal": not bool(self),
                "added": {face: list(names) for face, names in self.added.items()},
                "removed": {face: list(names) for face, names in self.removed.items()},
                "changed": self.changed,
            }
        )

    def __repr__(self) -> str:
        """Return counts per face plus the first few concrete changes."""
        if not self:
            return "ModelDiff(equal)"
        lines = ["ModelDiff(" + "; ".join(_count_parts(self)) + ")"]
        lines.extend(_preview_entries(self))
        return "\n".join(lines)


def diff_models(
    left: ModelIR,
    right: ModelIR,
    *,
    include_ext: bool = False,
    ignore_compiled: bool = False,
) -> ModelDiff:
    """Compare two model IRs face by face.

    Args:
        left: The baseline ``ModelIR``.
        right: The ``ModelIR`` to compare against.
        include_ext: Also compare ``ext.<namespace>`` payloads (e.g. ``mj``).
        ignore_compiled: Skip compiled-only fields (``*_id``, ``*_adr``,
            ``mass``, ``inertia``, ``ipos``, ``iquat``).

    Returns:
        A :class:`ModelDiff` (falsy when the models are equal).
    """
    left_faces = _faces(left, include_ext)
    right_faces = _faces(right, include_ext)
    order = list(left_faces) + [face for face in right_faces if face not in left_faces]
    added: dict[str, tuple[str, ...]] = {}
    removed: dict[str, tuple[str, ...]] = {}
    changed: _DiffChanges = {}
    for face in order:
        more, gone, moved = _diff_maps(
            left_faces.get(face, {}), right_faces.get(face, {}), ignore_compiled
        )
        if more:
            added[face] = more
        if gone:
            removed[face] = gone
        if moved:
            changed[face] = moved
    return ModelDiff(added=added, removed=removed, changed=changed)
