"""Face-by-face model diffing (design v098 N5).

A ``ModelDiff`` answers "what changed between these two models?" without
reaching into a backend: rows are matched by name, every dataclass field
(declared and compiled) is compared, and extension namespaces can be
included as ``ext.<namespace>`` faces.  Comparing a declared IR with a
compiled one is allowed but noisy (compiled-only fields move ``None`` ->
value); ``ignore_compiled=True`` silences exactly those fields.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ._serialize import encode, row_dict

__all__ = ["ModelDiff"]

_FACES = ("bodies", "joints", "geoms", "sites", "cameras", "actuators",
          "materials")

_COMPILED_FIELDS = frozenset({"mass", "inertia", "ipos", "iquat"})

_PREVIEW = 5


def _is_compiled_field(name):
    """Return True for fields a compiled backend fills (ids/addresses)."""
    return (name.endswith("_id") or name.endswith("_adr")
            or name in _COMPILED_FIELDS)


def _table_map(table):
    """Return ``{row name: field dict}`` for one IR table."""
    return {row.name: row_dict(row) for row in table}


def _ext_value(value):
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
            str(name): dict(item) if isinstance(item, Mapping)
            else {"value": item}
            for name, item in value.items()
        }
    return {str(name): {"value": item} for name, item in value.items()}


def _faces(ir, include_ext):
    """Return every comparable face as a name->fields map."""
    faces = {name: _table_map(getattr(ir, name)) for name in _FACES}
    if include_ext:
        for namespace, extension in ir.ext.items():
            for key, value in extension.to_dict().items():
                faces[f"ext.{namespace}.{key}"] = _ext_value(value)
    return faces


def _changed_fields(left, right, ignore_compiled):
    """Return the differing fields between two row maps."""
    changes = {}
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


def _diff_maps(left, right, ignore_compiled):
    """Diff two face maps into added/removed/changed pieces."""
    added = tuple(name for name in right if name not in left)
    removed = tuple(name for name in left if name not in right)
    changed = {}
    for name in left:
        if name not in right:
            continue
        fields = _changed_fields(left[name], right[name], ignore_compiled)
        if fields:
            changed[name] = fields
    return added, removed, changed


def _count_parts(diff):
    """Return the per-face ``+a -r ~c`` count strings for the repr."""
    parts = []
    faces = set(diff.added) | set(diff.removed) | set(diff.changed)
    for face in sorted(faces):
        added = len(diff.added.get(face, ()))
        removed = len(diff.removed.get(face, ()))
        changed = len(diff.changed.get(face, {}))
        parts.append(f"{face}: +{added} -{removed} ~{changed}")
    return parts


def _preview_entries(diff):
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
                yield (
                    f"  ~ {face}[{name!r}].{field_name}: "
                    f"{before!r} -> {after!r}")
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

    added: dict = field(default_factory=dict)
    removed: dict = field(default_factory=dict)
    changed: dict = field(default_factory=dict)

    def __bool__(self):
        """Return True when any face has an added, removed or changed row."""
        return bool(self.added or self.removed or self.changed)

    def to_dict(self):
        """Return a JSON-safe canonical document (floats use ``float.hex``)."""
        return encode({
            "equal": not bool(self),
            "added": {face: list(names) for face, names in self.added.items()},
            "removed": {
                face: list(names) for face, names in self.removed.items()},
            "changed": self.changed,
        })

    def __repr__(self):
        """Return counts per face plus the first few concrete changes."""
        if not self:
            return "ModelDiff(equal)"
        lines = ["ModelDiff(" + "; ".join(_count_parts(self)) + ")"]
        lines.extend(_preview_entries(self))
        return "\n".join(lines)


def diff_models(left, right, *, include_ext=False, ignore_compiled=False):
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
    order = list(left_faces) + [
        face for face in right_faces if face not in left_faces]
    added, removed, changed = {}, {}, {}
    for face in order:
        more, gone, moved = _diff_maps(
            left_faces.get(face, {}), right_faces.get(face, {}),
            ignore_compiled)
        if more:
            added[face] = more
        if gone:
            removed[face] = gone
        if moved:
            changed[face] = moved
    return ModelDiff(added=added, removed=removed, changed=changed)
