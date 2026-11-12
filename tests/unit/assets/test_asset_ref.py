"""The neutral AssetRef type: formats, mapping normalization and re-export."""

from __future__ import annotations

import dataclasses

import pytest

from unigym.assets.asset_ref import ASSET_FORMATS, AssetRef, AssetRefError


def test_asset_formats_are_the_closed_stable_vocabulary():
    """The asset format vocabulary is exactly mjcf/urdf/usd in stable order."""
    assert ASSET_FORMATS == ("mjcf", "urdf", "usd")


def test_asset_ref_to_dict_keeps_declared_formats_in_stable_order():
    """to_dict drops empty formats and keeps the declaration order."""
    asset = AssetRef(usd="robot.usd", mjcf="robot.xml")

    assert asset.to_dict() == {"mjcf": "robot.xml", "usd": "robot.usd"}


def test_asset_ref_from_mapping_round_trips_and_accepts_none_fields():
    """from_mapping accepts a mapping (all fields optional) and AssetRef self."""
    asset = AssetRef.from_mapping({"mjcf": "robot.xml", "urdf": None})

    assert asset == AssetRef(mjcf="robot.xml")
    assert AssetRef.from_mapping(asset) is asset


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"mjcf": "a", "glb": "b"}, r"unknown keys \['glb'\]"),
        ("robot.urdf", "must be a mapping"),
        ({"mjcf": 3}, "must be a string"),
    ],
)
def test_asset_ref_from_mapping_rejects_malformed_input(value, match):
    """Malformed asset references fail fast with AssetRefError."""
    with pytest.raises(AssetRefError, match=match):
        AssetRef.from_mapping(value)


def test_asset_ref_is_frozen_data():
    """An AssetRef is immutable authoring data."""
    asset = AssetRef(mjcf="robot.xml")

    with pytest.raises(dataclasses.FrozenInstanceError):
        asset.mjcf = "other.xml"


def test_scenes_reexports_the_same_asset_ref_type():
    """The long-lived scenes public path re-exports the moved owner type."""
    from unigym.scenes.articulation import AssetRef as SceneAssetRef

    assert SceneAssetRef is AssetRef
