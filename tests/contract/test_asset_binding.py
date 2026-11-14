"""The consumer-boundary contract for file-backed asset scenes."""

from __future__ import annotations

import pytest

from unigym.assets.asset_ref import AssetRef
from unigym.scenes.pipeline import (
    SceneBindingError,
    validate_asset_scene,
    validate_binding,
)
from unigym.scenes.scene_ir import (
    AssetInstance,
    Instance,
    Material,
    Pose,
    Region,
    Scene,
    Slot,
)
from unigym.scenes.terrain import Terrain


class _UncheckedAssetInstance(AssetInstance):
    """An AssetInstance that skips construction validation on purpose."""

    def __post_init__(self) -> None:
        """Keep the raw field values so the consumer boundary can reject them."""


def _asset(asset_id: str = "cup") -> AssetInstance:
    """Build one valid file-backed asset instance."""
    return AssetInstance(
        id=asset_id,
        asset=AssetRef(mjcf="pack:acme/assets/cup.mjcf"),
        pose=Pose(xyz=(0.0, 0.0, 1.0)),
        source="scene.yaml:12",
    )


def _unchecked(**overrides: object) -> AssetInstance:
    """Build an unchecked asset instance for boundary-failure tests."""
    fields: dict[str, object] = {
        "id": "cup",
        "asset": AssetRef(mjcf="pack:acme/assets/cup.mjcf"),
        "pose": Pose(),
        "source": "scene.yaml:12",
    }
    fields.update(overrides)
    return _UncheckedAssetInstance(**fields)


def _accepts(_name: str) -> bool:
    """Accept every generator name (the mixed-scene predicate)."""
    return True


def test_validate_asset_scene_accepts_assets_materials_and_primitives() -> None:
    """Materials and generated instances stay legal beside asset instances."""
    scene = Scene(
        name="table",
        instances=(
            Instance(
                id="table_top",
                generator="box",
                params={},
                pose=Pose(),
                source="scene.yaml:3",
            ),
        ),
        materials=(Material("wood"),),
        asset_instances=(_asset(),),
    )

    validate_asset_scene(scene)


@pytest.mark.parametrize(
    "scene",
    [
        Scene(
            name="table",
            terrain=Terrain.heightmap([[0.0, 0.0], [0.0, 0.0]]),
            asset_instances=(_asset(),),
        ),
        Scene(
            name="table",
            engine_hints={"gravity": {"z": -9.81}},
            asset_instances=(_asset(),),
        ),
    ],
    ids=["terrain", "engine-hint"],
)
def test_validate_asset_scene_allows_terrain_and_non_region_engine_hints(
    scene: Scene,
) -> None:
    """Terrain and engine hints other than 'regions' stay legal beside assets."""
    validate_asset_scene(scene)


@pytest.mark.parametrize(
    "scene",
    [
        Scene(
            name="table",
            regions=(Region("top", "table", "table", (1.0, 1.0), "scene.yaml:5"),),
            asset_instances=(_asset(),),
        ),
        Scene(
            name="table",
            slots=(Slot("shelf", "shelf", (1.0, 1.0, 1.0)),),
            asset_instances=(_asset(),),
        ),
        Scene(
            name="table",
            engine_hints={"regions": [{"name": "top"}]},
            asset_instances=(_asset(),),
        ),
    ],
    ids=["regions", "slots", "engine-hint-regions"],
)
def test_validate_asset_scene_rejects_unsupported_scene_semantics(scene: Scene) -> None:
    """Region and slot semantics fail closed beside asset instances."""
    with pytest.raises(SceneBindingError, match="not supported with file-backed"):
        validate_asset_scene(scene)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"asset": {"mjcf": "cup.mjcf"}}, "asset must be an AssetRef"),
        ({"asset": AssetRef()}, "asset declares no format"),
        ({"pose": {"frame": "world"}}, "pose must be a Pose"),
        ({"pose": Pose(frame="slot_a")}, "is not 'world'"),
        ({"source": ""}, "source must be a non-empty"),
        ({"source": "scene.yaml"}, "must name 'file:line'"),
        ({"kind": "floating"}, "kind 'floating' must be one of"),
        ({"collision": 1}, "collision must be a bool"),
        ({"id": "body"}, "engine element name"),
    ],
    ids=[
        "mapping-asset",
        "empty-asset",
        "mapping-pose",
        "non-world-frame",
        "empty-source",
        "source-without-line",
        "unknown-kind",
        "non-bool-collision",
        "engine-name",
    ],
)
def test_validate_asset_scene_reasserts_every_asset_field(
    overrides: dict[str, object],
    match: str,
) -> None:
    """The consumer re-validates fields even when construction was bypassed."""
    scene = Scene(name="table", asset_instances=(_unchecked(**overrides),))

    with pytest.raises(SceneBindingError, match=match):
        validate_asset_scene(scene)


def test_validate_binding_runs_the_asset_scene_checks() -> None:
    """The binding gate is the one caller of the asset-scene validation."""
    scene = Scene(
        name="table",
        regions=(Region("top", "table", "table", (1.0, 1.0), "scene.yaml:5"),),
        asset_instances=(_asset(),),
    )

    with pytest.raises(SceneBindingError, match="regions are not supported"):
        validate_binding(scene, _accepts)


def test_validate_binding_does_not_resolve_asset_generators() -> None:
    """Asset instances never consult the generator registry."""

    def _rejects(_name: str) -> bool:
        return False

    scene = Scene(name="table", asset_instances=(_asset(),))

    validate_binding(scene, _rejects)
