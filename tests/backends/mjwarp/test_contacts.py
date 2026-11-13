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


def test_mjwarp_constructor_binds_an_explicit_owner_group(tiny_env_factory, tiny_mjcf: Path):
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
        backend="mjwarp",
        contact_assets={"tiny": str(tiny_mjcf)},
        articulation_names=("tiny",),
    )
    assert manager is not None

    tiny_env_factory(mjcf=tiny_mjcf, contact_manager=manager)

    assert manager.backend.group_axes(manager.specs[0]) == (2, 0)
    value = manager.read()["contact/owner"]
    assert tuple(value.shape) == (1, 2, 3)
    assert value.count_nonzero() == 0


def test_mjwarp_constructor_binds_contacts_all(tiny_env_factory, tiny_mjcf: Path):
    """``contacts: all`` needs the post-init articulation tables at bind time."""
    from unigym.contacts.manager import build_contact_manager

    manager = build_contact_manager(
        [{"name": "all", "sensor_bodies": [ALL_BODIES], "frame": "world"}],
        backend="mjwarp",
        contact_assets={"tiny": str(tiny_mjcf)},
        articulation_names=("tiny",),
    )
    assert manager is not None

    tiny_env_factory(mjcf=tiny_mjcf, contact_manager=manager)

    assert manager.backend.group_axes(manager.specs[0]) == (3, 0)
    value = manager.read()["contact/all"]
    assert tuple(value.shape) == (1, 3, 3)
    assert value.count_nonzero() == 0


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
        backend="mjwarp",
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


def test_mjwarp_owned_group_filters_the_world_body_and_reads(tiny_env):
    """The warp mirror honours the world body as a free filter body."""
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


def test_mjwarp_owned_group_still_filters_a_free_prop_body(
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


def test_mjwarp_owned_sensor_scope_excludes_the_world_body(tiny_env, tiny_mjcf: Path):
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
        backend="mjwarp",
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
