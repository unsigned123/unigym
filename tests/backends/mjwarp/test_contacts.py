"""mjwarp serves the shared MuJoCo contact backend through its CPU mirror."""

from __future__ import annotations

from pathlib import Path

import pytest

from unigym.contacts.base import ALL_BODIES

pytestmark = pytest.mark.sim


def test_mjwarp_contact_manager_constructs_and_reads_the_mirror(tiny_env, tiny_mjcf: Path):
    """The mjwarp registry face uses the shared MjModel/MjData contact read."""
    from unigym.contacts.manager import build_contact_manager

    manager = build_contact_manager(
        [
            {
                "name": "robot",
                "sensor_bodies": ["base", "link", "tip"],
                "frame": "world",
            }
        ],
        backend="mjwarp",
    )
    assert manager is not None

    manager.bind_env(tiny_env)

    assert manager.backend.group_axes(manager.specs[0]) == (3, 0)
    value = manager.read()["contact/robot"]
    assert tuple(value.shape) == (1, 3, 3)


def test_mjwarp_all_expands_the_scoped_robot_bodies(tiny_env, tiny_mjcf: Path):
    """contacts: all uses the same articulation-scoped expansion as mujoco."""
    from unigym.contacts.manager import build_contact_manager

    manager = build_contact_manager(
        [{"name": "all", "sensor_bodies": [ALL_BODIES], "frame": "world"}],
        backend="mjwarp",
        contact_assets={"tiny": str(tiny_mjcf)},
    )
    assert manager is not None

    manager.bind_env(tiny_env)

    assert manager.backend.group_axes(manager.specs[0]) == (3, 0)
    value = manager.read()["contact/all"]
    assert tuple(value.shape) == (1, 3, 3)
    assert value.count_nonzero() == 0
