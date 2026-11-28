"""mjWarp serves typed contact refs through the shared MuJoCo mirror."""

from __future__ import annotations

from unigym.backends.compose import backend_runtime

from pathlib import Path

import pytest

from unigym.contacts.manager import build_contact_manager

pytestmark = pytest.mark.sim

PROP_CRATE = """    <body name="Prop_crate" pos="0.3 0 0.05">
      <freejoint/>
      <geom name="crate_geom" type="box" size="0.05 0.05 0.05" mass="0.5"/>
    </body>"""

SUPPORT_TABLE = """    <body name="table_main_group_main" pos="0 0 -0.1">
      <geom name="table_geom" type="box" size="1 1 0.05" mass="50"/>
    </body>"""


def _mjcf(tmp_path: Path, extra: str = "") -> Path:
    from tests.backends.mujoco.conftest import tiny_mjcf_source

    source = tiny_mjcf_source()
    if extra:
        source = source.replace("  </worldbody>", extra + "\n  </worldbody>", 1)
    path = tmp_path / "cardb_warp.xml"
    path.write_text(source, encoding="utf-8")
    return path


def test_mjwarp_object_sensor_and_environment_filter(tmp_path, tiny_env_factory):
    mjcf = _mjcf(tmp_path, PROP_CRATE)
    manager = build_contact_manager(
        [
            {
                "name": "crate_floor",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [{"owner": "environment", "body": "world"}],
                "frame": "world",
            }
        ],
        backend="mjwarp",
        contact_factory=backend_runtime("mjwarp").bindings.contact_backend,
        articulation_names=("tiny",),
        prop_names=["crate"],
    )
    assert manager is not None
    tiny_env_factory(mjcf=mjcf, contact_manager=manager)

    assert manager.backend.group_axes(manager.specs[0]) == (1, 1)
    assert tuple(manager.read()["contact/crate_floor"].shape) == (1, 1, 1, 3)


def test_mjwarp_declaration_order_across_owner_scopes(tmp_path, tiny_env_factory):
    mjcf = _mjcf(tmp_path, PROP_CRATE + "\n" + SUPPORT_TABLE)
    manager = build_contact_manager(
        [
            {
                "name": "mixed",
                "sensor_refs": [
                    {"owner": "robot", "body": "tip"},
                    {"owner": "object", "name": "crate"},
                ],
                "filter_refs": [
                    {"owner": "support", "name": "table_main_group_main"},
                    {"owner": "environment", "body": "world"},
                ],
                "frame": "world",
            }
        ],
        backend="mjwarp",
        contact_factory=backend_runtime("mjwarp").bindings.contact_backend,
        articulation_names=("tiny",),
        prop_names=["crate"],
    )
    assert manager is not None
    tiny_env_factory(mjcf=mjcf, contact_manager=manager)

    resolved = manager.backend._resolved["mixed"]
    assert [row.native for row in resolved["sensor_rows"]] == ["tip", "Prop_crate"]
    assert [row.native for row in resolved["filter_rows"]] == [
        "table_main_group_main",
        "world",
    ]
    assert manager.backend.group_axes(manager.specs[0]) == (2, 2)


def test_mjwarp_rejects_legacy_syntax():
    with pytest.raises(ValueError, match="removed keys"):
        build_contact_manager(
            [{"name": "x", "sensor_bodies": ["tip"], "frame": "world"}],
            backend="mjwarp",
            contact_factory=backend_runtime("mjwarp").bindings.contact_backend,
        )
