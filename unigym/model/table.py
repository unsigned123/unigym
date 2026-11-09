"""Named rows with both sequence and mapping access (the introspection face).

``Table`` is deliberately engine-neutral: it stores rows that carry a
``name`` (and optionally a ``qualname``) and adds lookup/filter/display
helpers.  It is a ``Sequence`` (iteration yields rows, integer/slice
indexing works) and a ``Mapping`` (a string key resolves to one row), so
the same object answers both "the third joint" and "the elbow joint".
"""
# The dual protocol (Sequence + Mapping) is a deliberate design conflict:
# integer/slice indexing yields rows while string indexing yields one row, so
# the two ABCs cannot be satisfied simultaneously by static analysis.  The
# runtime contract is correct (covered by tests); scope the two unavoidable
# checker rules to this module instead of scattering per-method ignores.
# pyright: reportGeneralTypeIssues=false, reportIncompatibleMethodOverride=false
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import fields, is_dataclass
from html import escape
from typing import Generic, TypeVar, overload

from ._serialize import JsonValue, _is_mapping

__all__ = ["Table"]

T = TypeVar("T")

_MAX_REPR_ROWS = 20


def _row_dict(row: object) -> dict[str, JsonValue]:
    """Return one row as a plain ``dict`` (no recursive float rewriting)."""
    if _is_mapping(row):
        return {str(name): value for name, value in row.items()}
    if is_dataclass(row) and not isinstance(row, type):
        return {field.name: getattr(row, field.name) for field in fields(row)}
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if _is_mapping(result):
            return {str(name): value for name, value in result.items()}
    return {str(name): value for name, value in vars(row).items()}


def _markdown_cell(value: object) -> str:
    """Escape one cell for a GitHub-flavoured Markdown table."""
    return (str(value).replace("\\", "\\\\").replace("|", "\\|")
            .replace("\n", "<br>"))


class Table(Sequence[T], Mapping[str, T], Generic[T]):
    """An ordered collection of named rows, indexable by name or position.

    Names must be non-empty and unique; a duplicate is a construction
    error (the model IR never silently renames or drops a row).  String
    lookup first tries ``name`` and then ``qualname``.
    """

    __slots__ = ("_rows", "_by_name", "_by_qualname")

    def __init__(self, rows: Iterable[T] = ()) -> None:
        """Build a table from *rows*, rejecting duplicate names.

        Args:
            rows: An iterable of row objects, each carrying a string
                ``name`` attribute (mappings may carry a ``name`` key).

        Raises:
            ValueError: A row has no name or two rows share one.
        """
        ordered: tuple[T, ...] = tuple(rows)
        by_name: dict[str, T] = {}
        by_qualname: dict[str, T] = {}
        for row in ordered:
            name = getattr(row, "name", None)
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"Table rows must carry a non-empty string name, got "
                    f"{name!r} on {row!r}")
            if name in by_name:
                raise ValueError(
                    f"Table row name {name!r} is duplicated: names are the "
                    f"compiled global identity and must be unique")
            by_name[name] = row
            qualname = getattr(row, "qualname", None)
            if isinstance(qualname, str) and qualname:
                if qualname in by_qualname:
                    raise ValueError(
                        f"Table qualname {qualname!r} is duplicated")
                by_qualname[qualname] = row
        self._rows = ordered
        self._by_name = by_name
        self._by_qualname = by_qualname

    @overload
    def __getitem__(self, key: int) -> T:
        ...

    @overload
    def __getitem__(self, key: str) -> T:
        ...

    @overload
    def __getitem__(self, key: slice) -> Table[T]:
        ...

    def __getitem__(self, key: int | str | slice) -> T | Table[T]:
        """Return a row by position/slice, or by name/qualname string."""
        if isinstance(key, slice):
            return Table(self._rows[key])
        if isinstance(key, int):
            return self._rows[key]
        if isinstance(key, str):
            try:
                return self._by_name[key]
            except KeyError:
                try:
                    return self._by_qualname[key]
                except KeyError:
                    raise KeyError(key) from None
        raise TypeError(
            f"Table keys are int, slice or str, not {type(key).__name__}")

    def __iter__(self) -> Iterator[T]:
        """Iterate rows in insertion order (Sequence semantics)."""
        return iter(self._rows)

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._rows)

    def __contains__(self, key: object) -> bool:
        """Membership test for a name, qualname or row object."""
        if isinstance(key, str):
            return key in self._by_name or key in self._by_qualname
        return any(key is row or key == row for row in self._rows)

    def keys(self) -> tuple[str, ...]:
        """Return the row names (name-keyed mapping view)."""
        return tuple(self._by_name)

    def values(self) -> tuple[T, ...]:
        """Return the rows (mapping-values view)."""
        return self._rows

    def items(self) -> tuple[tuple[str, T], ...]:
        """Return ``(name, row)`` pairs in insertion order."""
        return tuple(self._by_name.items())

    def get(self, key: str, default: T | None = None) -> T | None:
        """Return the named row, or *default* when absent."""
        try:
            return self[key]
        except KeyError:
            return default

    def names(self) -> list[str]:
        """Return every row name in insertion order."""
        return list(self._by_name)

    def matching(self, regex: str | re.Pattern[str]) -> Table[T]:
        """Return a new table of rows whose name/qualname matches *regex*."""
        pattern = re.compile(regex) if isinstance(regex, str) else regex
        selected = [
            row for row in self._rows
            if pattern.search(str(getattr(row, "name", "")))
            or pattern.search(str(getattr(row, "qualname", "") or ""))
        ]
        return Table(selected)

    def filter(self, pred: Callable[[T], bool] | None = None,
               **fields: JsonValue) -> Table[T]:
        """Return the rows satisfying *pred* and every equality field.

        Args:
            pred: Optional callable receiving one row.
            **fields: Row attribute names compared by equality.
        """
        selected: list[T] = []
        for row in self._rows:
            if pred is not None and not pred(row):
                continue
            if all(getattr(row, key, None) == value
                   for key, value in fields.items()):
                selected.append(row)
        return Table(selected)

    def to_rows(self) -> list[dict[str, JsonValue]]:
        """Return the rows as plain dictionaries."""
        return [_row_dict(row) for row in self._rows]

    def to_dict(self) -> dict[str, dict[str, JsonValue]]:
        """Return ``{name: row dict}`` in insertion order."""
        return {str(getattr(row, "name")): _row_dict(row)
                for row in self._rows}

    def to_pandas(self) -> object:
        """Return a pandas ``DataFrame`` (pandas is an optional dependency).

        Raises:
            ImportError: pandas is not installed.
        """
        try:
            import pandas
        except ImportError as err:  # pragma: no cover - depends on the host
            raise ImportError(
                "Table.to_pandas() requires pandas; install it or use "
                "to_rows()/to_dict()") from err
        frame = getattr(pandas, "DataFrame")
        return frame(self.to_rows(), index=self.names())

    def _repr_html_(self) -> str:
        """Return an escaped HTML table (first 20 rows, then a count row)."""
        rows = self.to_rows()
        if not rows:
            return ("<table><tbody><tr><td>Table(0 rows)</td></tr>"
                    "</tbody></table>")
        columns = list(rows[0])
        shown = rows[:_MAX_REPR_ROWS]
        lines = ["<table>", "<thead>", "<tr>"]
        lines.extend(f"<th>{escape(str(column))}</th>" for column in columns)
        lines.extend(["</tr>", "</thead>", "<tbody>"])
        for row in shown:
            lines.append("<tr>")
            lines.extend(
                f"<td>{escape(str(row.get(column, '')))}</td>"
                for column in columns)
            lines.append("</tr>")
        remaining = len(rows) - len(shown)
        if remaining:
            lines.append(
                f"<tr><td colspan='{len(columns)}'>... {remaining} more"
                f"</td></tr>")
        lines.extend(["</tbody>", "</table>"])
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Return a GitHub-flavoured Markdown table (all rows, no deps)."""
        rows = self.to_rows()
        if not rows:
            return "Table(0 rows)"
        columns = list(rows[0])
        lines = [
            "| " + " | ".join(_markdown_cell(column) for column in columns)
            + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        lines.extend(
            "| " + " | ".join(
                _markdown_cell(row.get(column, "")) for column in columns)
            + " |"
            for row in rows)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return an aligned text table (first 20 rows, then a count line)."""
        rows = self.to_rows()
        if not rows:
            return "Table(0 rows)"
        columns = list(rows[0])
        shown = rows[:_MAX_REPR_ROWS]
        cells = [[str(row.get(col, "")) for col in columns] for row in shown]
        widths = [
            max(len(columns[i]), *(len(row[i]) for row in cells))
            for i in range(len(columns))
        ]
        lines = [
            "  ".join(columns[i].ljust(widths[i])
                      for i in range(len(columns))),
            "  ".join("-" * widths[i] for i in range(len(columns))),
        ]
        lines.extend(
            "  ".join(cell.ljust(widths[i])
                      for i, cell in enumerate(row))
            for row in cells
        )
        if len(rows) > _MAX_REPR_ROWS:
            lines.append(f"... {len(rows) - _MAX_REPR_ROWS} more")
        return "\n".join(lines)
