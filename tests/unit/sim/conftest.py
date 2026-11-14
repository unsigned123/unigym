"""Keep facade pack activation from leaking into the process-wide robot registry."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from unigym.robots import restore_robot_registry, snapshot_robot_registry


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Restore the robot registry snapshot after each facade test."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)
