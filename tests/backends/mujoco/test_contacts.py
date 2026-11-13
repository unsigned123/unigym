"""MuJoCo ``contacts: all`` expands scoped robot bodies without failing."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_constructor_binds_an_explicit_owner_group(tiny_env_factory, tiny_mjcf: Path):
    """A constructor-provided manager resolves an explicit owner at bind time."""
    from unigym.contacts.manager import build_contact_manager

    manager = build_contact_manager(
        [
            {
                "name": "owner",
                "articulation": "tiny",
                "sensor_bodies": ["link", "tip"],
                "frame": "world",
            }
        ],
        backend="mujoco",
        contact_assets={"tiny": str(tiny_mjcf)},
        articulation_names=("tiny",),
    )
    assert manager is not None

    tiny_env_factory(mjcf=tiny_mjcf, contact_manager=manager)

    assert manager.backend.group_axes(manager.specs[0]) == (2, 0)
    value = manager.read()["contact/owner"]
    assert tuple(value.shape) == (1, 2, 3)
    assert value.count_nonzero() == 0


def test_constructor_binds_contacts_all(tiny_env_factory, tiny_mjcf: Path):
    """``contacts: all`` needs the post-init articulation tables at bind time."""
    from unigym.contacts.manager import build_contact_manager

    manager = build_contact_manager(
        [{"name": "all", "sensor_bodies": [ALL_BODIES], "frame": "world"}],
        backend="mujoco",
        contact_assets={"tiny": str(tiny_mjcf)},
        articulation_names=("tiny",),
    )
    assert manager is not None

    tiny_env_factory(mjcf=tiny_mjcf, contact_manager=manager)

    assert manager.backend.group_axes(manager.specs[0]) == (3, 0)
    value = manager.read()["contact/all"]
    assert tuple(value.shape) == (1, 3, 3)
    assert value.count_nonzero() == 0


def test_duplicate_resolved_filter_bodies_fail_fast(tiny_env):
    """Overlapping filter expressions are rejected instead of double-counted."""
    from unigym.contacts.manager import build_contact_manager

    manager = build_contact_manager(
        [
            {
                "name": "dup",
                "sensor_bodies": ["tip"],
                "filter_bodies": ["link", "link"],
                "frame": "world",
            }
        ],
        backend="mujoco",
    )
    assert manager is not None

    with pytest.raises(ValueError, match="duplicate bodies"):
        manager.bind_env(tiny_env)


def test_run_sensing_manager_binds_in_the_constructor(tiny_env_factory, tiny_mjcf: Path):
    """The real ``_build_sensing`` manager works through the constructor path."""
    from types import SimpleNamespace

    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.registry import backend_info
    from unigym.run.build import _build_sensing
    from unigym.scenes.articulation import ArticulationSpec

    spec = SimpleNamespace(
        enable_cameras=False,
        direct=False,
        articulations=(ArticulationSpec(name="tiny"),),
    )
    cfg = {
        "contacts": [
            {
                "name": "all",
                "sensor_bodies": {"contact_group": {"articulation": "tiny", "group": "all"}},
                "frame": "world",
            }
        ]
    }
    sensing = _build_sensing(
        spec,
        cfg,
        backend_info("mujoco"),
        "cpu",
        1,
        {},
        {},
        {"tiny": {"all": (ALL_BODIES,)}},
        {"tiny": str(tiny_mjcf)},
        ManualAdapter(),
        None,
    )
    manager = sensing["contact_manager"]
    assert manager is not None

    tiny_env_factory(mjcf=tiny_mjcf, contact_manager=manager)

    assert manager.backend.group_axes(manager.specs[0]) == (3, 0)
    assert tuple(manager.read()["contact/all"].shape) == (1, 3, 3)


def _world_filter_manager(filters):
    from unigym.contacts.manager import build_contact_manager

    manager = build_contact_manager(
        [
            {
                "name": "ground",
                "articulation": "tiny",
                "sensor_bodies": ["tip"],
                "filter_bodies": list(filters),
                "frame": "world",
            }
        ],
        backend="mujoco",
        articulation_names=("tiny",),
    )
    assert manager is not None
    return manager


def _free_prop_mjcf(tiny_mjcf: Path, tmp_path: Path) -> Path:
    """The tiny arm plus a declared prop body parented to the world body."""
    source = tiny_mjcf.read_text(encoding="utf-8")
    injected = source.replace(
        "  </worldbody>",
        '    <body name="Prop_cup" pos="0.3 0 0.1">\n'
        "      <freejoint/>\n"
        '      <geom name="Prop_cup_geom" type="box" '
        'size="0.03 0.03 0.03" mass="0.05"/>\n'
        "    </body>\n"
        "  </worldbody>",
        1,
    )
    modified = tmp_path / "tiny_arm_prop.xml"
    modified.write_text(injected, encoding="utf-8")
    return modified


def test_owned_group_filters_the_world_body_and_reads(tiny_env):
    """An explicit owner can name the world body as a free filter body."""
    manager = _world_filter_manager(["world"])

    manager.bind_env(tiny_env)

    world_id = tiny_env.mj_model.body("world").id
    resolved = manager.backend._resolved["ground"]
    assert resolved["filter_ids"] == [world_id]
    assert world_id not in resolved["sensor_ids"]
    assert manager.backend.group_axes(manager.specs[0]) == (1, 1)
    value = manager.read()["contact/ground"]
    assert tuple(value.shape) == (1, 1, 1, 3)
    assert value.count_nonzero() == 0


def test_owned_group_still_filters_a_free_prop_body(
    tiny_env_factory, tiny_mjcf: Path, tmp_path: Path
):
    """Free prop filtering survives the world-body inclusion in the free set."""
    mjcf = _free_prop_mjcf(tiny_mjcf, tmp_path)
    env = tiny_env_factory(mjcf=mjcf)
    manager = _world_filter_manager(["world", "Prop_cup"])
    manager.bind_env(env)

    model = env.mj_model
    world_id = model.body("world").id
    prop_id = model.body("Prop_cup").id
    assert manager.backend._resolved["ground"]["filter_ids"] == [world_id, prop_id]
    assert manager.backend.group_axes(manager.specs[0]) == (1, 2)
    value = manager.read()["contact/ground"]
    assert tuple(value.shape) == (1, 1, 2, 3)
    assert value.count_nonzero() == 0


def test_owned_sensor_scope_excludes_the_world_body(tiny_env, tiny_mjcf: Path):
    """World stays out of owned sensor bodies (all and native lists)."""
    from unigym.contacts.manager import build_contact_manager

    manager = build_contact_manager(
        [
            {
                "name": "all",
                "articulation": "tiny",
                "sensor_bodies": [ALL_BODIES],
                "frame": "world",
            },
            {
                "name": "ground",
                "articulation": "tiny",
                "sensor_bodies": ["tip"],
                "filter_bodies": ["world"],
                "frame": "world",
            },
        ],
        backend="mujoco",
        contact_assets={"tiny": str(tiny_mjcf)},
        articulation_names=("tiny",),
    )
    assert manager is not None

    manager.bind_env(tiny_env)

    world_id = tiny_env.mj_model.body("world").id
    assert world_id not in manager.backend._resolved["all"]["sensor_ids"]
    assert world_id not in manager.backend._resolved["ground"]["sensor_ids"]
    assert world_id in manager.backend._resolved["ground"]["filter_ids"]
    assert manager.backend.group_axes(manager.specs[0]) == (3, 0)
    assert tuple(manager.read()["contact/all"].shape) == (1, 3, 3)
