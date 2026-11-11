"""Generate a machine-readable catalog from pytest collection.

The tool runs pytest with ``--collect-only`` and records the collected items as
stable JSON.  It never executes test functions, fixtures or backend setup.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import NoReturn, TypedDict

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1


class CatalogRow(TypedDict):
    """One collected test and its derived metadata."""

    nodeid: str
    path: str
    line: int
    layer: str
    domain: str | None
    backend: str | None
    markers: list[str]
    fixtures: list[str]
    docstring: str | None


class CatalogDocument(TypedDict):
    """The generated catalog document."""

    schema_version: int
    test_count: int
    tests: list[CatalogRow]


def _fail(message: str) -> NoReturn:
    """Abort with a teaching error."""
    raise SystemExit(f"test_catalog: {message}")


class _CollectionRecorder:
    """Capture the pytest session after collection finishes."""

    def __init__(self) -> None:
        """Initialize an empty recorder."""
        self.session: pytest.Session | None = None

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Record the session that pytest just finished collecting."""
        self.session = session


def _collect(selectors: list[str]) -> pytest.Session:
    """Run collection only and return the completed pytest session."""
    recorder = _CollectionRecorder()
    arguments = ["--collect-only", "-q", "-p", "no:cacheprovider", *selectors]
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        exit_code = pytest.main(arguments, plugins=[recorder])
    if exit_code != 0:
        details = output.getvalue().strip()
        _fail(f"pytest collection failed with exit code {exit_code}:\n{details}")
    if recorder.session is None:
        _fail("pytest finished without a collection session")
    return recorder.session


def _declared_marker_names(config: pytest.Config) -> set[str]:
    """Return the marker names declared in pytest configuration."""
    names: set[str] = set()
    for declaration in config.getini("markers"):
        name = declaration.split(":", 1)[0].strip()
        if name:
            names.add(name)
    return names


def _marker_field(names: list[str], prefix: str, nodeid: str) -> str | None:
    """Return the single marker with ``prefix``, or fail when ambiguous."""
    matches = [name for name in names if name.startswith(prefix)]
    if len(matches) > 1:
        _fail(f"collected test {nodeid!r} declares multiple {prefix}* markers: {matches}")
    return matches[0] if matches else None


def _first_docstring_line(obj: object | None) -> str | None:
    """Return the first non-empty docstring line, if any."""
    if obj is None:
        return None
    docstring = inspect.getdoc(obj)
    if not docstring:
        return None
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _catalog_row(item: pytest.Item, declared: set[str]) -> CatalogRow:
    """Build one catalog row from a collected item."""
    try:
        relative = item.path.resolve().relative_to(ROOT)
    except ValueError:
        _fail(f"collected test {item.nodeid!r} is outside the repository root")

    location = item.location
    line = location[1]
    if not isinstance(line, int):
        _fail(f"collected test {item.nodeid!r} has no source line")

    markers = sorted(marker.name for marker in item.iter_markers() if marker.name in declared)
    layer = _marker_field(markers, "layer_", item.nodeid)
    if layer is None:
        _fail(f"collected test {item.nodeid!r} has no layer_* marker")
    domain = _marker_field(markers, "domain_", item.nodeid)
    backend = _marker_field(markers, "backend_", item.nodeid)

    fixture_names: list[str] = list(getattr(item, "fixturenames", []))
    fixtures = sorted(set(fixture_names))
    docstring = _first_docstring_line(getattr(item, "obj", None))

    return CatalogRow(
        nodeid=item.nodeid,
        path=relative.as_posix(),
        line=line + 1,
        layer=layer,
        domain=domain,
        backend=backend,
        markers=markers,
        fixtures=fixtures,
        docstring=docstring,
    )


def _build_document(session: pytest.Session) -> CatalogDocument:
    """Build the sorted catalog document from a collected session."""
    declared = _declared_marker_names(session.config)
    rows = sorted(
        (_catalog_row(item, declared) for item in session.items),
        key=lambda row: row["nodeid"],
    )
    if not rows:
        _fail("pytest collected no tests; refusing to write an empty catalog")
    return CatalogDocument(schema_version=SCHEMA_VERSION, test_count=len(rows), tests=rows)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, collect tests and write the catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="PATH",
        help="catalog JSON output path (never written into the repository by default)",
    )
    parser.add_argument(
        "selectors",
        nargs="*",
        metavar="SELECTOR",
        help="pytest selectors; empty uses the project testpaths",
    )
    args = parser.parse_args(argv)

    session = _collect(args.selectors)
    document = _build_document(session)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"test_catalog: wrote {document['test_count']} tests to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
