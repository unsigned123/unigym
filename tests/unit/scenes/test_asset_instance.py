"""The neutral AssetInstance IR, its Scene integration and the asset rows."""

from __future__ import annotations

import json

import pytest

from unigym.assets.asset_ref import AssetRef
from unigym.scenes.artifacts import ASSET_SCENE_DIGEST_VERSION, asset_instance_rows
from unigym.scenes.pipeline import Resolution
from unigym.scenes.scene_ir import (
    ASSET_INSTANCE_KINDS,
    AssetInstance,
    Instance,
    Pose,
    Scene,
    validate_scene,
)
from unigym.world import Box, World


def _asset(asset_id: str = "cup") -> AssetInstance:
    """Build one valid file-backed asset instance."""
    return AssetInstance(
        id=asset_id,
        asset=AssetRef(mjcf="pack:acme/assets/cup.mjcf"),
        pose=Pose(xyz=(0.0, 0.0, 1.0)),
        source="scene.yaml:12",
    )


def _instance(instance_id: str = "cup") -> Instance:
    """Build one valid generated instance sharing a candidate id."""
    return Instance(
        id=instance_id,
        generator="box",
        params={},
        pose=Pose(),
        source="scene.yaml:3",
    )


class _UncheckedAssetInstance(AssetInstance):
    """An AssetInstance that skips construction validation on purpose."""

    def __post_init__(self) -> None:
        """Keep the raw field values so the rows boundary can reject them."""


def test_asset_instance_kinds_are_the_closed_role_vocabulary() -> None:
    """The asset instance role vocabulary is exactly the two rigid kinds."""
    assert ASSET_INSTANCE_KINDS == ("static_rigid", "dynamic_rigid")


def test_asset_instance_defaults_are_static_and_colliding() -> None:
    """An asset instance defaults to static_rigid with collision enabled."""
    asset = _asset()

    assert asset.kind == "static_rigid"
    assert asset.collision is True
    assert asset.asset.to_dict() == {"mjcf": "pack:acme/assets/cup.mjcf"}
    assert asset.pose.frame == "world"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"id": "Cup"}, "lowercase snake_case"),
        ({"id": "body"}, "engine element name"),
        ({"asset": {"mjcf": "cup.mjcf"}}, "asset must be an AssetRef"),
        ({"asset": AssetRef()}, "asset declares no format"),
        ({"pose": {"frame": "world"}}, "pose must be a Pose"),
        ({"pose": Pose(frame="slot_a")}, "is not 'world'"),
        ({"source": ""}, "source must be a non-empty"),
        ({"source": "scene.yaml"}, "must name 'file:line'"),
        ({"kind": "articulated"}, "kind 'articulated' must be one of"),
        ({"collision": 1}, "collision must be a bool"),
        ({"collision": 0}, "collision must be a bool"),
        ({"collision": "yes"}, "collision must be a bool"),
    ],
    ids=[
        "bad-id",
        "engine-name",
        "mapping-asset",
        "empty-asset",
        "mapping-pose",
        "non-world-frame",
        "empty-source",
        "source-without-line",
        "unknown-kind",
        "collision-one",
        "collision-zero",
        "collision-truthy",
    ],
)
def test_asset_instance_rejects_a_broken_field_contract(
    overrides: dict[str, object],
    match: str,
) -> None:
    """Every field rule fails at construction with a teaching error."""
    fields: dict[str, object] = {
        "id": "cup",
        "asset": AssetRef(mjcf="pack:acme/assets/cup.mjcf"),
        "pose": Pose(),
        "source": "scene.yaml:12",
    }
    fields.update(overrides)

    with pytest.raises(ValueError, match=match):
        AssetInstance(**fields)


def test_asset_instance_accepts_an_explicit_dynamic_non_colliding_role() -> None:
    """A dynamic, non-colliding asset instance is legal authoring data."""
    asset = AssetInstance(
        id="cup",
        asset=AssetRef(usd="pack:acme/assets/cup.usd"),
        pose=Pose(),
        source="scene.yaml:12",
        kind="dynamic_rigid",
        collision=False,
    )

    assert asset.kind == "dynamic_rigid"
    assert asset.collision is False


@pytest.mark.parametrize("path", ["", "   "], ids=["empty", "whitespace"])
def test_asset_instance_rejects_an_asset_with_no_usable_format(path: str) -> None:
    """A declared path that is empty or whitespace is not a format."""
    with pytest.raises(ValueError, match="asset declares no format"):
        AssetInstance(
            id="cup",
            asset=AssetRef(mjcf=path),
            pose=Pose(),
            source="scene.yaml:12",
        )


def test_scene_rejects_duplicate_asset_instance_ids() -> None:
    """Asset instance ids are unique within one scene."""
    with pytest.raises(ValueError, match="duplicate asset instance id 'cup'"):
        Scene(name="table", asset_instances=(_asset("cup"), _asset("cup")))


def test_scene_rejects_an_instance_asset_instance_id_collision() -> None:
    """Generated instances and asset instances share one id namespace."""
    with pytest.raises(ValueError, match="collides with instance 'cup'.*one namespace"):
        Scene(
            name="table",
            instances=(_instance("cup"),),
            asset_instances=(_asset("cup"),),
        )


def test_scene_positional_construction_still_works() -> None:
    """Appending asset_instances preserves positional construction."""
    scene = Scene("table", (), (), (), (), {}, None)

    assert scene.asset_instances == ()
    assert scene == Scene(name="table")


def test_scene_rejects_a_non_tuple_asset_instance_member() -> None:
    """asset_instances is a typed tuple, never a list."""
    with pytest.raises(ValueError, match="asset_instances must be a tuple"):
        Scene(name="table", asset_instances=[_asset()])


def test_scene_rejects_a_non_asset_instance_entry() -> None:
    """Every asset_instances entry is an AssetInstance."""
    with pytest.raises(ValueError, match="entries must be AssetInstance"):
        Scene(name="table", asset_instances=(object(),))


def test_validate_scene_accepts_a_valid_asset_scene() -> None:
    """The boundary walk accepts a well-formed asset scene."""
    validate_scene(Scene(name="table", asset_instances=(_asset(),)))


def test_asset_instance_rows_are_sorted_and_deterministically_shaped() -> None:
    """Rows are sorted by id, identical across runs, and hex-float only."""
    scene = Scene(
        name="table",
        asset_instances=(_asset("lid"), _asset("cup")),
    )
    formats = {"lid": "mjcf", "cup": "mjcf"}

    rows = asset_instance_rows(scene, formats)

    assert tuple(row["id"] for row in rows) == ("cup", "lid")
    assert rows == asset_instance_rows(scene, dict(reversed(list(formats.items()))))
    assert rows[0] == {
        "id": "cup",
        "kind": "static_rigid",
        "format": "mjcf",
        "asset": {"mjcf": "pack:acme/assets/cup.mjcf"},
        "pose": {
            "xyz": ["0x0.0p+0", "0x0.0p+0", "0x1.0000000000000p+0"],
            "quat_wxyz": [
                "0x1.0000000000000p+0",
                "0x0.0p+0",
                "0x0.0p+0",
                "0x0.0p+0",
            ],
        },
        "collision": True,
        "source": "scene.yaml:12",
    }


def test_asset_instance_rows_carry_no_absolute_paths_or_engine_names() -> None:
    """The neutral row keeps authoring references verbatim, not resolved paths."""
    scene = Scene(name="table", asset_instances=(_asset(),))

    rows = asset_instance_rows(scene, {"cup": "mjcf"})
    serialized = json.dumps(rows)

    assert set(rows[0]) == {
        "id",
        "kind",
        "format",
        "asset",
        "pose",
        "collision",
        "source",
    }
    assert all(not value.startswith("/") for value in rows[0]["asset"].values())
    assert '"path"' not in serialized


@pytest.mark.parametrize(
    ("formats", "match"),
    [
        ({}, "no format was selected"),
        ({"cup": "mjcf", "ghost": "mjcf"}, "undeclared asset instance"),
        ({"cup": "glb"}, "must be one of"),
        ({"cup": "usd"}, "no compatible format"),
    ],
    ids=["missing", "unknown-id", "unknown-format", "undeclared-format"],
)
def test_asset_instance_rows_reject_a_bad_format_selection(
    formats: dict[str, str],
    match: str,
) -> None:
    """Each id needs exactly one selected format the asset declares."""
    scene = Scene(name="table", asset_instances=(_asset(),))

    with pytest.raises(ValueError, match=match):
        asset_instance_rows(scene, formats)


def test_asset_instance_rows_reject_an_empty_declared_path() -> None:
    """A selected format needs a non-empty path, not just a present key."""
    entry = _UncheckedAssetInstance(
        id="cup",
        asset=AssetRef(mjcf=""),
        pose=Pose(),
        source="scene.yaml:12",
    )
    scene = Scene(name="table", asset_instances=(entry,))

    with pytest.raises(ValueError, match="no compatible format"):
        asset_instance_rows(scene, {"cup": "mjcf"})


def test_asset_instance_rows_reject_a_non_scene() -> None:
    """The row writer accepts only a Scene."""
    with pytest.raises(TypeError, match="expects a Scene"):
        asset_instance_rows("not-a-scene", {})


def test_asset_instance_rows_reject_non_mapping_formats() -> None:
    """The selected-format table is a mapping, never a sequence."""
    scene = Scene(name="table", asset_instances=(_asset(),))

    with pytest.raises(TypeError, match="formats must be a mapping"):
        asset_instance_rows(scene, [("cup", "mjcf")])


def test_asset_scene_digest_version_is_the_neutral_tag() -> None:
    """The shared asset-scene digest tag is the stable neutral version."""
    assert ASSET_SCENE_DIGEST_VERSION == "unigym.scene.asset/1"


def test_resolution_to_mapping_includes_asset_instances() -> None:
    """Resolution carries the asset rows in its JSON view."""
    mapping = Resolution(asset_instances=({"id": "cup"},)).to_mapping()

    assert mapping["asset_instances"] == [{"id": "cup"}]
    assert set(mapping) == {
        "instances",
        "regions",
        "slots",
        "materials",
        "asset_instances",
    }
    assert Resolution().to_mapping()["asset_instances"] == []


def test_world_inline_propagates_asset_instances() -> None:
    """An inline world serves the base scene's asset instances unchanged."""
    asset = _asset()
    world = World.inline(Scene(name="asset_scene", asset_instances=(asset,)))

    assert world.to_scene().asset_instances == (asset,)
    assert world.unmapped_static_scene is True


def test_world_inline_without_assets_has_no_unmapped_static_scene() -> None:
    """An empty inline scene is still fully mapped."""
    assert World.inline(Scene(name="empty_scene")).unmapped_static_scene is False


def test_world_object_name_collides_with_an_asset_instance() -> None:
    """Object names and asset instance ids share one uniqueness namespace."""
    world = World.inline(Scene(name="asset_scene", asset_instances=(_asset("cup"),)))

    with pytest.raises(ValueError, match="duplicate object name 'cup'"):
        world.add(Box((0.1, 0.1, 0.1), name="cup"))


def test_world_auto_names_avoid_asset_instance_ids() -> None:
    """Auto-generated primitive names skip declared asset instance ids."""
    world = World.inline(Scene(name="asset_scene", asset_instances=(_asset("ground_1"),)))

    world.flat_ground()
    scene = world.to_scene()

    assert "ground_1" in {entry.id for entry in scene.asset_instances}
    assert "ground_1" not in {entry.id for entry in scene.instances}
    assert "ground_2" in {entry.id for entry in scene.instances}
