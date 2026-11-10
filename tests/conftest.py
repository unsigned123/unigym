"""Central test harness: apply path-derived markers and derive the fast lane."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import pytest

from tests._support.markers import (
    BACKEND_MARKER_PREFIX,
    UnknownTestPathError,
    is_fast_lane,
    markers_for_path,
)

_TESTS_ROOT = Path(__file__).resolve().parent
_DATA_ROOT = _TESTS_ROOT / "_support" / "data"


@pytest.fixture
def tiny_mjcf(tmp_path: Path) -> Path:
    """Copy the shared primitives-only MJCF into an isolated tmp directory."""
    source = _DATA_ROOT / "tiny_arm.xml"
    target = tmp_path / "tiny_arm.xml"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _relative_parts(path: Path) -> tuple[str, ...] | None:
    try:
        return path.resolve().relative_to(_TESTS_ROOT).parts
    except ValueError:
        return None


def _marker_names(item: pytest.Item) -> set[str]:
    return {marker.name for marker in item.iter_markers()}


def _add_marker(item: pytest.Item, name: str) -> None:
    if name not in _marker_names(item):
        item.add_marker(name)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Reject test files whose directory the harness cannot map."""
    if not collection_path.is_file():
        return None
    if not any(fnmatch(collection_path.name, pattern) for pattern in config.getini("python_files")):
        return None

    parts = _relative_parts(collection_path)
    if parts is None:
        return None

    try:
        markers_for_path(parts)
    except UnknownTestPathError as exc:
        raise pytest.Collector.CollectError(str(exc)) from exc
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply the central path-to-marker policy to every collected item."""
    for item in items:
        parts = _relative_parts(item.path)
        if parts is None:
            continue

        try:
            markers = markers_for_path(parts)
        except UnknownTestPathError as exc:
            raise pytest.UsageError(str(exc)) from exc

        for name in markers:
            _add_marker(item, name)

        if not any(name.startswith(BACKEND_MARKER_PREFIX) for name in _marker_names(item)):
            _add_marker(item, "backend_agnostic")

        layer = next(name for name in markers if name.startswith("layer_"))
        if is_fast_lane(layer, _marker_names(item)):
            _add_marker(item, "fast")
