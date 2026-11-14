"""File-backed MJCF asset instances compose, compile, describe and persist."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unigym.assets.asset_ref import AssetRef
from unigym.scenes.artifacts import ASSET_SCENE_DIGEST_VERSION, asset_instance_rows
from unigym.scenes.scene_ir import AssetInstance, Pose, Region, Scene

STATIC_MJCF = """<mujoco model="static_asset">
  <worldbody>
    <body name="root" pos="0 0 0.25">
      <geom name="body" type="box" size="0.1 0.1 0.1" mass="1"/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

DYNAMIC_MJCF = """<mujoco model="dynamic_asset">
  <worldbody>
    <body name="root" pos="0 0 0.5">
      <freejoint/>
      <geom name="body" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _asset_plan():
    """A real empty asset boundary (absolute fixture paths need no pack)."""
    from unigym.assets.plan import AssetPlan
    from unigym.packs.runtime import AssetContext, discover

    return AssetPlan(AssetContext(discover([])))


def _write_mjcf(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / f"{name}.xml"
    path.write_text(text, encoding="utf-8")
    return path


def _scene(
    path: Path,
    *,
    asset_id: str = "crate",
    kind: str = "static_rigid",
    collision: bool = True,
    pose: Pose | None = None,
) -> Scene:
    return Scene(
        name="bench",
        asset_instances=(
            AssetInstance(
                id=asset_id,
                asset=AssetRef(mjcf=str(path)),
                pose=Pose(xyz=(1.0, 2.0, 3.0)) if pose is None else pose,
                source="scene.yaml:1",
                kind=kind,
                collision=collision,
            ),
        ),
    )


def _root_body_id(model, asset_id: str = "crate") -> int:
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"Asset_{asset_id}_root")
    assert body_id >= 0
    return body_id


def test_static_asset_composes_compiles_and_stays_put(tmp_path) -> None:
    """A static asset keeps its authored pose and does not move under gravity."""
    import mujoco

    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene

    path = _write_mjcf(tmp_path, "static", STATIC_MJCF)
    scene = _scene(path)
    composed = compose_scene(scene, _asset_plan())

    assert composed.scene is scene
    assert dict(composed.asset_paths) == {"crate": str(path)}
    assert composed.regions == {}

    model = composed.compile()
    body_id = _root_body_id(model)
    assert model.nq == 0

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = data.xpos[body_id].copy()
    assert start == pytest.approx((1.0, 2.0, 3.25))
    for _ in range(250):
        mujoco.mj_step(model, data)
    assert data.xpos[body_id] == pytest.approx(start)


def test_dynamic_asset_composes_compiles_and_falls(tmp_path) -> None:
    """A dynamic asset owns one freejoint and falls under gravity."""
    import mujoco

    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene

    path = _write_mjcf(tmp_path, "dynamic", DYNAMIC_MJCF)
    scene = _scene(path, kind="dynamic_rigid")
    composed = compose_scene(scene, _asset_plan())

    model = composed.compile()
    body_id = _root_body_id(model)
    assert model.nq == 7

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.xpos[body_id] == pytest.approx((1.0, 2.0, 3.5))
    start_z = float(data.xpos[body_id][2])
    for _ in range(250):
        mujoco.mj_step(model, data)
    assert float(data.xpos[body_id][2]) < start_z - 0.1


@pytest.mark.parametrize(("collision", "expected"), [(False, 0), (True, 1)])
def test_collision_toggle_covers_every_root_geom(tmp_path, collision, expected) -> None:
    """collision=false disables collision on named and unnamed root geoms."""
    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene

    path = _write_mjcf(tmp_path, "static", STATIC_MJCF)
    scene = _scene(path, collision=collision)
    model = compose_scene(scene, _asset_plan()).compile()

    body_id = _root_body_id(model)
    start = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    assert count == 2
    for geom_id in range(start, start + count):
        assert int(model.geom_contype[geom_id]) == expected
        assert int(model.geom_conaffinity[geom_id]) == expected


_REFUSALS = (
    (
        "multiple_roots",
        """<mujoco model="two_roots"><worldbody>
             <body name="a"><geom type="box" size="0.1 0.1 0.1"/></body>
             <body name="b"><geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody></mujoco>""",
        "static_rigid",
        "exactly one root body",
    ),
    (
        "static_joint",
        """<mujoco model="static_joint"><worldbody>
             <body name="root"><joint name="j" type="hinge"/>
               <geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody></mujoco>""",
        "static_rigid",
        "static rigid assets declare no joints",
    ),
    (
        "static_freejoint",
        """<mujoco model="static_free"><worldbody>
             <body name="root"><freejoint/>
               <geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody></mujoco>""",
        "static_rigid",
        "static rigid assets declare no joints",
    ),
    (
        "nested_body",
        """<mujoco model="nested"><worldbody>
             <body name="root"><geom type="box" size="0.1 0.1 0.1"/>
               <body name="child"><geom type="box" size="0.1 0.1 0.1"/></body>
             </body>
           </worldbody></mujoco>""",
        "static_rigid",
        "nested bodies are unsupported",
    ),
    (
        "dynamic_no_freejoint",
        """<mujoco model="no_joint"><worldbody>
             <body name="root"><geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody></mujoco>""",
        "dynamic_rigid",
        "need exactly one root freejoint",
    ),
    (
        "dynamic_hinge",
        """<mujoco model="hinge"><worldbody>
             <body name="root"><joint name="j" type="hinge"/>
               <geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody></mujoco>""",
        "dynamic_rigid",
        "need exactly one root freejoint",
    ),
    (
        "dynamic_ball",
        """<mujoco model="ball"><worldbody>
             <body name="root"><joint name="j" type="ball"/>
               <geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody></mujoco>""",
        "dynamic_rigid",
        "need exactly one root freejoint",
    ),
    (
        "dynamic_multiple_freejoints",
        """<mujoco model="two_free"><worldbody>
             <body name="root"><freejoint/><freejoint/>
               <geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody></mujoco>""",
        "dynamic_rigid",
        "need exactly one root freejoint",
    ),
)


@pytest.mark.parametrize(
    ("name", "text", "kind", "match"),
    _REFUSALS,
    ids=[case[0] for case in _REFUSALS],
)
def test_unsupported_asset_shapes_are_refused(tmp_path, name, text, kind, match) -> None:
    """Every unsupported body/joint shape fails closed with a teaching error."""
    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene

    path = _write_mjcf(tmp_path, name, text)
    scene = _scene(path, kind=kind)

    with pytest.raises(ValueError, match=match):
        compose_scene(scene, _asset_plan())


def test_asset_scene_rejects_regions_instead_of_dropping_them(tmp_path) -> None:
    """Asset instances contribute no regions, so declared regions are refused."""
    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene
    from unigym.scenes.pipeline import SceneBindingError

    path = _write_mjcf(tmp_path, "static", STATIC_MJCF)
    scene = Scene(
        name="bench",
        asset_instances=(
            AssetInstance(
                id="crate",
                asset=AssetRef(mjcf=str(path)),
                pose=Pose(),
                source="scene.yaml:1",
            ),
        ),
        regions=(
            Region(
                name="table",
                owner="crate",
                frame="world",
                extent=(1.0, 1.0),
                source="scene.yaml:3",
            ),
        ),
    )

    with pytest.raises(SceneBindingError, match="regions are not supported"):
        compose_scene(scene, _asset_plan())


def test_describe_asset_scene_slice_is_neutral_and_deterministic(tmp_path, monkeypatch) -> None:
    """describe adds the neutral asset slice without a resolved path."""
    from unigym.backends.shared.mjfamily.scene.scene_engine import MujocoSceneEngine
    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene

    _write_mjcf(tmp_path, "static", STATIC_MJCF)
    _write_mjcf(tmp_path, "dynamic", DYNAMIC_MJCF)
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", str(tmp_path))
    scene = Scene(
        name="bench",
        asset_instances=(
            AssetInstance(
                id="ball",
                asset=AssetRef(mjcf="${UNIGYM_DEPS_ROOT}/dynamic.xml"),
                pose=Pose(xyz=(0.0, 0.0, 1.0)),
                source="scene.yaml:2",
                kind="dynamic_rigid",
            ),
            AssetInstance(
                id="crate",
                asset=AssetRef(mjcf="${UNIGYM_DEPS_ROOT}/static.xml"),
                pose=Pose(xyz=(1.0, 2.0, 3.0)),
                source="scene.yaml:1",
            ),
        ),
    )
    composed = compose_scene(scene, _asset_plan())
    engine = MujocoSceneEngine()

    first = engine.describe(composed)
    second = engine.describe(composed)

    assert first == second
    assert first["digest_version"] == "unigym.modelir/2"
    assert set(first) == {"asset_scene", "base", "digest_version", "ext"}
    asset_scene = first["asset_scene"]
    assert asset_scene["digest_version"] == ASSET_SCENE_DIGEST_VERSION
    expected = [dict(row) for row in asset_instance_rows(scene, {"ball": "mjcf", "crate": "mjcf"})]
    assert asset_scene["instances"] == expected
    for row in asset_scene["instances"]:
        assert "resolved" not in row
        assert set(row) == {
            "id",
            "kind",
            "format",
            "asset",
            "pose",
            "collision",
            "source",
        }
        assert all(not value.startswith("/") for value in row["asset"].values())
        for value in (*row["pose"]["xyz"], *row["pose"]["quat_wxyz"]):
            assert value.startswith(("0x", "-0x"))
    serialized = json.dumps(asset_scene)
    assert '"resolved"' not in serialized
    assert str(tmp_path) not in serialized


def test_describe_empty_asset_scene_slice_is_an_empty_list() -> None:
    """A scene without file assets still carries an empty asset slice."""
    from unigym.backends.shared.mjfamily.scene.scene_engine import MujocoSceneEngine
    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene

    composed = compose_scene(Scene(name="empty"), _asset_plan())
    described = MujocoSceneEngine().describe(composed)

    assert described["asset_scene"] == {
        "digest_version": ASSET_SCENE_DIGEST_VERSION,
        "instances": [],
    }


def test_artifacts_record_resolved_asset_paths(tmp_path) -> None:
    """Artifact rows and payload entries carry the composed resolved path."""
    from unigym.backends.shared.mjfamily.scene.scene_artifacts import (
        build_views,
        write_scene_artifacts,
    )
    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene

    path = _write_mjcf(tmp_path, "static", STATIC_MJCF)
    scene = _scene(path)
    composed = compose_scene(scene, _asset_plan())
    model = composed.compile()

    _draw, resolution, payloads = build_views(scene, "bench", {}, composed.asset_paths)
    rows = resolution.to_mapping()["asset_instances"]
    assert rows[0]["resolved"] == str(path)
    entries = payloads.to_mapping()["entries"]
    assert entries == [
        {
            "id": "crate",
            "kind": "static_rigid",
            "format": "mjcf",
            "asset": {"mjcf": str(path)},
            "source": "scene.yaml:1",
            "resolved": str(path),
        }
    ]

    out_dir = tmp_path / "artifacts"
    write_scene_artifacts(
        out_dir,
        scene,
        "bench",
        model,
        {},
        composed.spec.to_xml(),
        asset_paths=composed.asset_paths,
    )

    written = json.loads((out_dir / "resolution.json").read_text(encoding="utf-8"))
    assert written["asset_instances"][0]["resolved"] == str(path)
    index = json.loads((out_dir / "payload_index.json").read_text(encoding="utf-8"))
    assert index["entries"][0]["resolved"] == str(path)


def test_artifacts_without_asset_paths_add_no_resolved_field(tmp_path) -> None:
    """Omitting the composed path table leaves artifact rows unresolved."""
    from unigym.backends.shared.mjfamily.scene.scene_artifacts import build_views
    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene

    path = _write_mjcf(tmp_path, "static", STATIC_MJCF)
    scene = _scene(path)
    compose_scene(scene, _asset_plan())

    _draw, resolution, payloads = build_views(scene, "bench", {})

    assert "resolved" not in resolution.to_mapping()["asset_instances"][0]
    assert "resolved" not in payloads.to_mapping()["entries"][0]


def test_load_asset_spec_requires_an_absolute_existing_path(tmp_path) -> None:
    """load_asset_spec fails closed on a relative or missing document path."""
    from unigym.backends.shared.mjfamily.scene.mjcf_io import load_asset_spec

    with pytest.raises(ValueError, match="must be an absolute path"):
        load_asset_spec(Path("relative.xml"), _asset_plan())
    with pytest.raises(FileNotFoundError, match="not an existing file"):
        load_asset_spec(tmp_path / "missing.xml", _asset_plan())


def test_load_asset_spec_rejects_relative_leaves_after_resolution(tmp_path) -> None:
    """A rewritten document must use absolute file leaves only."""
    from unigym.backends.shared.mjfamily.scene.mjcf_io import load_asset_spec

    text = """<mujoco model="relative_leaves">
      <asset>
        <texture name="tex" type="2d" file="${UNIGYM_REPO}/docs/tex.png"/>
        <mesh name="mesh" file="mesh.stl"/>
      </asset>
      <worldbody>
        <body name="root"><geom type="mesh" mesh="mesh"/></body>
      </worldbody>
    </mujoco>
    """
    path = _write_mjcf(tmp_path, "relative_leaves", text)

    with pytest.raises(ValueError, match="relative file leaves"):
        load_asset_spec(path, _asset_plan())


@pytest.mark.parametrize(
    "leaf",
    ['file = "mesh.stl"', "file='mesh.stl'", "file = 'mesh.stl'"],
    ids=["double-space", "single", "single-space"],
)
def test_load_asset_spec_rejects_tolerantly_spelled_relative_leaves(tmp_path, leaf) -> None:
    """Whitespace/single-quoted relative leaves are caught after a trigger."""
    from unigym.backends.shared.mjfamily.scene.mjcf_io import load_asset_spec

    text = f"""<mujoco model="relative_leaves">
      <asset>
        <texture name="tex" type="2d" file="${{UNIGYM_REPO}}/docs/tex.png"/>
        <mesh name="mesh" {leaf}/>
      </asset>
      <worldbody>
        <body name="root"><geom type="mesh" mesh="mesh"/></body>
      </worldbody>
    </mujoco>
    """
    path = _write_mjcf(tmp_path, "relative_leaves", text)

    with pytest.raises(ValueError, match="relative file leaves"):
        load_asset_spec(path, _asset_plan())


_META_REFUSALS = (
    (
        "worldbody_geom",
        """<mujoco model="world_geom"><worldbody>
             <geom name="floor" type="plane" size="5 5 0.1"/>
             <body name="root"><geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody></mujoco>""",
        "worldbody-level geometry",
    ),
    (
        "actuator",
        """<mujoco model="actuator"><worldbody>
             <body name="root"><geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody><actuator><general name="g"/></actuator></mujoco>""",
        "section 'actuators'",
    ),
    (
        "sensor",
        """<mujoco model="sensor"><worldbody>
             <body name="root"><geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody><sensor><framepos objtype="body" objname="root"/></sensor></mujoco>""",
        "section 'sensors'",
    ),
    (
        "equality",
        """<mujoco model="equality"><worldbody>
             <body name="root"><geom type="box" size="0.1 0.1 0.1"/></body>
           </worldbody><equality><weld body1="root"/></equality></mujoco>""",
        "section 'equalities'",
    ),
)


@pytest.mark.parametrize(
    ("name", "text", "match"),
    _META_REFUSALS,
    ids=[case[0] for case in _META_REFUSALS],
)
def test_unsupported_asset_sections_are_refused(tmp_path, name, text, match) -> None:
    """World-level geoms and actuator/sensor/equality sections fail closed."""
    from unigym.backends.shared.mjfamily.scene.scene_render import compose_scene

    path = _write_mjcf(tmp_path, name, text)
    scene = _scene(path)

    with pytest.raises(ValueError, match=match):
        compose_scene(scene, _asset_plan())


def test_load_asset_spec_rejects_an_unresolved_pack_marker(tmp_path) -> None:
    """A pack marker outside a file leaf still fails closed after resolution."""
    from unigym.backends.shared.mjfamily.scene.mjcf_io import load_asset_spec

    text = """<mujoco model="pack:missing">
      <worldbody><body name="root"><geom type="box" size="0.1 0.1 0.1"/></body></worldbody>
    </mujoco>
    """
    path = _write_mjcf(tmp_path, "unresolved_pack", text)

    with pytest.raises(ValueError, match="unresolved asset reference"):
        load_asset_spec(path, _asset_plan())


def test_env_artifacts_record_resolved_asset_paths(tmp_path, tiny_env_factory) -> None:
    """The real env composition passes the composed asset paths to artifacts."""
    from unigym.scenes.providers import (
        register_scene_provider,
        restore_scene_providers,
        snapshot_scene_providers,
    )

    path = _write_mjcf(tmp_path, "static", STATIC_MJCF)
    scene = _scene(path)
    snapshot = snapshot_scene_providers()

    def provider(_params, _context):
        """Return the file-backed asset scene for the env composition."""
        return scene

    register_scene_provider("test.asset_scene", provider)
    try:
        out_dir = tmp_path / "artifacts"
        tiny_env_factory(
            params={"scene_provider": "test.asset_scene"},
            artifact_dir=out_dir,
            asset_plan=_asset_plan(),
        )
    finally:
        restore_scene_providers(snapshot)

    resolution = json.loads((out_dir / "resolution.json").read_text(encoding="utf-8"))
    assert resolution["asset_instances"][0]["resolved"] == str(path)
    index = json.loads((out_dir / "payload_index.json").read_text(encoding="utf-8"))
    assert index["entries"][0]["resolved"] == str(path)
