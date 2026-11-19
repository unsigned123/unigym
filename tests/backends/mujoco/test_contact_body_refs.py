"""MuJoCo contact refs: explicit owners, B/F order and force detail facts."""

from __future__ import annotations

from pathlib import Path

import pytest

from unigym.contacts.manager import build_contact_manager
from unigym.runtime.actions import NamedAction


def _mjcf(tmp_path: Path, extra: str = "") -> Path:
    from tests.backends.mujoco.conftest import tiny_mjcf_source

    source = tiny_mjcf_source()
    if extra:
        source = source.replace("  </worldbody>", extra + "\n  </worldbody>", 1)
    path = tmp_path / "cardb.xml"
    path.write_text(source, encoding="utf-8")
    return path


PROP_CRATE = """    <body name="Prop_crate" pos="0.3 0 0.05">
      <freejoint/>
      <geom name="crate_geom" type="box" size="0.05 0.05 0.05" mass="0.5"/>
    </body>"""

SUPPORT_TABLE = """    <body name="table_main_group_main" pos="0 0 -0.1">
      <geom name="table_geom" type="box" size="1 1 0.05" mass="50"/>
    </body>"""

FIXTURE = """    <body name="Prop_microwave_microwave_main" pos="0.5 0 0.5">
      <geom name="mw_geom" type="box" size="0.1 0.1 0.1" mass="5"/>
    </body>"""


def _manager(groups, *, props=None, articulated=None, names=("tiny",)):
    manager = build_contact_manager(
        groups,
        backend="mujoco",
        articulation_names=names,
        prop_names=props,
        articulated_props=articulated,
    )
    assert manager is not None
    return manager


def _step(env, steps=5):
    action = NamedAction.zeros(1, "tiny", ("elbow", "shoulder"), "position")
    for _ in range(steps):
        env.step((action,))


def test_object_sensor_and_environment_filter_force_and_detail(tmp_path, tiny_env_factory):
    mjcf = _mjcf(tmp_path, PROP_CRATE)
    manager = _manager(
        [
            {
                "name": "crate_floor",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [{"owner": "environment", "body": "world"}],
                "frame": "world",
                "format": "contact",
                "threshold": 0.01,
                "aggregate": "any",
                "aggregation": "sum_magnitudes",
            }
        ],
        props=["crate"],
    )
    env = tiny_env_factory(mjcf=mjcf, contact_manager=manager)
    _step(env)

    assert manager.backend.group_axes(manager.specs[0]) == (1, 1)
    value = manager.read()
    assert bool(value["contact/crate_floor"].item())
    assert float(value["contact/crate_floor/force"].item()) > 0.0
    assert int(value["contact/crate_floor/active_bodies"].item()) == 1

    detail = manager.read_detail("crate_floor")
    assert detail
    record = detail[0]
    assert record["sensor_owner"] == "object"
    assert record["sensor_scope"] == "crate"
    assert record["sensor_native"] == "Prop_crate"
    assert record["sensor_key"] == "object/crate/crate"
    assert record["other_owner"] == "environment"
    assert record["other_key"] == "environment/world/world"
    assert record["force_mag"] > 0.0


def test_object_sensor_and_support_filter(tmp_path, tiny_env_factory):
    mjcf = _mjcf(tmp_path, PROP_CRATE + "\n" + SUPPORT_TABLE)
    manager = _manager(
        [
            {
                "name": "crate_table",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [{"owner": "support", "name": "table_main_group_main"}],
                "frame": "world",
            }
        ],
        props=["crate"],
    )
    tiny_env_factory(mjcf=mjcf, contact_manager=manager)

    assert manager.backend.group_axes(manager.specs[0]) == (1, 1)
    assert tuple(manager.read()["contact/crate_table"].shape) == (1, 1, 1, 3)
    assert [row.native for row in manager.backend.inventory.rows if row.owner == "support"] == [
        "table_main_group_main"
    ]


def test_fixture_ref_resolves_an_articulated_prop_link(tmp_path, tiny_env_factory):
    mjcf = _mjcf(tmp_path, FIXTURE)
    manager = _manager(
        [
            {
                "name": "microwave_body",
                "sensor_refs": [
                    {"owner": "fixture", "name": "microwave", "body": "microwave_main"}
                ],
                "frame": "world",
            }
        ],
        props=["microwave"],
        articulated={"microwave"},
    )
    tiny_env_factory(mjcf=mjcf, contact_manager=manager)

    assert manager.backend.group_axes(manager.specs[0]) == (1, 0)
    rows = [row for row in manager.backend.inventory.rows if row.owner == "fixture"]
    assert [(row.scope, row.body, row.native) for row in rows] == [
        ("microwave", "microwave_main", "Prop_microwave_microwave_main")
    ]


def test_arm_and_hand_style_cross_ref_filter_and_declaration_order(tmp_path, tiny_env_factory):
    mjcf = _mjcf(tmp_path, SUPPORT_TABLE)
    manager = _manager(
        [
            {
                "name": "finger_table",
                "sensor_refs": [
                    {"owner": "robot", "articulation": "tiny", "body": "tip"},
                    {"owner": "robot", "articulation": "tiny", "body": "base"},
                ],
                "filter_refs": [
                    {"owner": "support", "name": "table_main_group_main"},
                    {"owner": "environment", "body": "world"},
                ],
                "frame": "world",
            }
        ]
    )
    tiny_env_factory(mjcf=mjcf, contact_manager=manager)

    resolved = manager.backend._resolved["finger_table"]
    assert [row.native for row in resolved["sensor_rows"]] == ["tip", "base"]
    assert [row.native for row in resolved["filter_rows"]] == [
        "table_main_group_main",
        "world",
    ]
    assert manager.backend.group_axes(manager.specs[0]) == (2, 2)
    assert tuple(manager.read()["contact/finger_table"].shape) == (1, 2, 2, 3)


def test_unknown_refs_fail_with_canonical_candidates(tiny_env):
    manager = _manager(
        [
            {
                "name": "ghost",
                "sensor_refs": [{"owner": "object", "name": "ghost"}],
                "frame": "world",
            }
        ]
    )
    with pytest.raises(ValueError, match="unknown object scope") as err:
        manager.bind_env(tiny_env)
    assert "robot/tiny/base" in str(err.value)


def test_duplicate_resolved_bodies_fail_fast(tiny_env):
    manager = _manager(
        [
            {
                "name": "dup",
                "sensor_refs": [
                    {"owner": "robot", "body": "tip"},
                    {"owner": "robot", "body": "t.*"},
                ],
                "frame": "world",
            }
        ]
    )
    with pytest.raises(ValueError, match="duplicate body"):
        manager.bind_env(tiny_env)


def test_robot_wildcard_keeps_model_order_and_collisionless_rows(tiny_env):
    manager = _manager(
        [
            {
                "name": "all",
                "sensor_refs": [{"owner": "robot", "body": "*"}],
                "frame": "world",
            }
        ]
    )
    manager.bind_env(tiny_env)

    assert manager.backend.group_axes(manager.specs[0]) == (3, 0)
    rows = manager.backend._resolved["all"]["sensor_rows"]
    assert [row.native for row in rows] == ["base", "link", "tip"]
    assert tuple(manager.read()["contact/all"].shape) == (1, 3, 3)


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "x", "sensor_bodies": ["tip"], "frame": "world"},
        {
            "name": "x",
            "sensor_refs": [{"owner": "robot", "body": "tip"}],
            "mode": "general",
            "frame": "world",
        },
        {
            "name": "x",
            "sensor_refs": [{"owner": "robot", "body": "tip"}],
            "filter_bodies": ["world"],
            "frame": "world",
        },
    ],
)
def test_legacy_syntax_is_rejected(entry):
    with pytest.raises(ValueError, match="removed keys"):
        build_contact_manager([entry], backend="mujoco")


FREE_ARTICULATION_PROP = """    <body name=\"Prop_laptop\" pos=\"0.6 0 0.2\">
      <freejoint/>
      <geom name=\"laptop_geom\" type=\"box\" size=\"0.12 0.08 0.02\" mass=\"1.2\"/>
    </body>"""


def test_nested_welded_prop_is_object_only_and_default_root_resolves(tmp_path, tiny_env_factory):
    """A prop nested under a robot link is not also a robot body row."""
    from tests.backends.mujoco.conftest import tiny_mjcf_source

    source = tiny_mjcf_source().replace(
        """      </body>
    </body>
  </worldbody>""",
        """        <body name=\"Prop_crate\" pos=\"0 0 -0.05\">
          <geom name=\"crate_geom\" type=\"box\" size=\"0.03 0.03 0.03\" mass=\"0.1\"/>
        </body>
      </body>
    </body>
  </worldbody>""",
        1,
    )
    mjcf = tmp_path / "nested_prop.xml"
    mjcf.write_text(source, encoding="utf-8")
    manager = _manager(
        [
            {
                "name": "crate",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "frame": "world",
            }
        ],
        props=["crate"],
    )
    tiny_env_factory(mjcf=mjcf, contact_manager=manager)

    prop_rows = [row for row in manager.backend.inventory.rows if row.native == "Prop_crate"]
    assert len(prop_rows) == 1
    assert prop_rows[0].owner == "object"
    assert prop_rows[0].root is True
    robot_rows = [row for row in manager.backend.inventory.rows if row.owner == "robot"]
    assert [row.native for row in robot_rows] == ["base", "link", "tip"]
    assert manager.backend.group_axes(manager.specs[0]) == (1, 0)
    assert tuple(manager.read()["contact/crate"].shape) == (1, 1, 3)


def test_free_articulation_props_are_objects_fixtures_stay_explicit(tmp_path, tiny_env_factory):
    """fixture owner is only for declared articulated fixtures; free ones are objects."""
    mjcf = _mjcf(tmp_path, FIXTURE + "\n" + FREE_ARTICULATION_PROP)
    manager = _manager(
        [
            {
                "name": "pairs",
                "sensor_refs": [
                    {"owner": "fixture", "name": "microwave"},
                    {"owner": "object", "name": "laptop"},
                ],
                "filter_refs": [{"owner": "environment", "body": "world"}],
                "frame": "world",
            }
        ],
        props=["microwave", "laptop"],
        articulated={"microwave"},
    )
    tiny_env_factory(mjcf=mjcf, contact_manager=manager)

    rows = {(row.owner, row.scope): row for row in manager.backend.inventory.rows}
    assert ("fixture", "microwave") in rows
    assert ("object", "laptop") in rows
    assert ("object", "microwave") not in rows
    assert ("fixture", "laptop") not in rows


def test_two_articulation_arm_hand_cross_filter(tmp_path, tiny_mjcf):
    """A sensor on the left arm can filter the right arm's bodies."""
    from tests.backends.mujoco.test_env import _two_articulation_env

    manager = _manager(
        [
            {
                "name": "left_right",
                "sensor_refs": [{"owner": "robot", "articulation": "left", "body": "left_tip"}],
                "filter_refs": [{"owner": "robot", "articulation": "right", "body": "right_tip"}],
                "frame": "world",
            }
        ],
        names=("left", "right"),
    )
    env = _two_articulation_env(tmp_path, tiny_mjcf, contact_manager=manager)
    try:
        resolved = manager.backend._resolved["left_right"]
        assert [row.native for row in resolved["sensor_rows"]] == ["left_tip"]
        assert resolved["sensor_rows"][0].scope == "left"
        assert [row.native for row in resolved["filter_rows"]] == ["right_tip"]
        assert resolved["filter_rows"][0].scope == "right"
        assert manager.backend.group_axes(manager.specs[0]) == (1, 1)
        assert tuple(manager.read()["contact/left_right"].shape) == (1, 1, 1, 3)
    finally:
        env.close()
