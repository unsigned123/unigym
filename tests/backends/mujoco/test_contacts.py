"""MuJoCo ``contacts: all`` expands scoped robot bodies without failing."""

from __future__ import annotations

from pathlib import Path

from unigym.contacts.base import ALL_BODIES


def _all_manager(contact_assets: dict[str, str]):
    from unigym.contacts.manager import build_contact_manager

    manager = build_contact_manager(
        [{"name": "all", "sensor_bodies": [ALL_BODIES], "frame": "world"}],
        backend="mujoco",
        contact_assets=contact_assets,
    )
    assert manager is not None
    return manager


def test_all_expands_the_scoped_robot_bodies_in_model_order(tiny_env, tiny_mjcf: Path):
    """All resolves base/link/tip (never world or the floor) in model order."""
    manager = _all_manager({"tiny": str(tiny_mjcf)})

    manager.bind_env(tiny_env)

    assert manager.backend.group_axes(manager.specs[0]) == (3, 0)
    value = manager.read()["contact/all"]
    assert tuple(value.shape) == (1, 3, 3)
    assert value.count_nonzero() == 0


def test_all_keeps_collisionless_bodies_as_zero_rows(tiny_mjcf: Path, tmp_path: Path):
    """A body without geoms stays in the B axis with zero force channels."""
    from tests.backends.mujoco.conftest import build_tiny_env

    source = tiny_mjcf.read_text(encoding="utf-8")
    source = source.replace(
        '      <body name="link" pos="0 0 -0.16">',
        '      <body name="ghost" pos="0.1 0 0"/>\n' '      <body name="link" pos="0 0 -0.16">',
        1,
    )
    modified = tmp_path / "tiny_arm_ghost.xml"
    modified.write_text(source, encoding="utf-8")

    env = build_tiny_env(modified)
    manager = _all_manager({"tiny": str(modified)})
    try:
        manager.bind_env(env)
        bodies = [env.mj_model.body(i).name for i in range(env.mj_model.nbody)]
        assert "ghost" in bodies

        b, _ = manager.backend.group_axes(manager.specs[0])
        assert b == 4
        value = manager.read()["contact/all"]
        assert tuple(value.shape) == (1, 4, 3)
        assert value.count_nonzero() == 0
    finally:
        env.close()
