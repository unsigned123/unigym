"""The neutral AssetRef type: formats, mapping normalization and re-export."""

from __future__ import annotations

import dataclasses

import pytest

from unigym.assets.asset_ref import (
    ASSET_FORMATS,
    AssetRef,
    AssetRefError,
    select_asset_format,
)


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


@pytest.mark.parametrize(
    ("assets", "formats", "expected"),
    [
        (AssetRef(usd="robot.usd", urdf="robot.urdf"), ("usd", "urdf"), "usd"),
        (AssetRef(urdf="robot.urdf", mjcf="robot.xml"), ("usd", "urdf"), "urdf"),
        (AssetRef(mjcf="robot.xml"), ("usd", "urdf", "mjcf"), "mjcf"),
        (AssetRef(usd="robot.usd"), ("mjcf",), None),
    ],
    ids=["usd-first", "urdf-when-no-usd", "mjcf-last", "no-compatible"],
)
def test_select_asset_format_returns_the_first_compatible_declared_format(
    assets, formats, expected
):
    """Selection follows the backend order, never the AssetRef declaration order."""
    if expected is None:
        with pytest.raises(AssetRefError, match="no compatible format"):
            select_asset_format(assets, formats)
        return

    assert select_asset_format(assets, formats) == expected


def test_select_asset_format_treats_an_empty_reference_as_absent():
    """An empty AssetRef field is not a declared format."""
    with pytest.raises(AssetRefError, match="no compatible format"):
        select_asset_format(AssetRef(usd=""), ("usd",))


def test_select_asset_format_rejects_a_missing_asset_reference():
    """No effective asset is a teaching error that names the backend formats."""
    with pytest.raises(AssetRefError, match="no asset reference is declared"):
        select_asset_format(None, ("usd", "urdf"))


@pytest.mark.parametrize(
    ("formats", "match"),
    [
        ((), "must not be empty"),
        (("usd", "glb"), r"unknown format\(s\) \['glb'\]"),
        (("usd", "usd"), "repeats a format"),
    ],
    ids=["empty", "unknown", "duplicate"],
)
def test_select_asset_format_rejects_a_malformed_preference_list(formats, match):
    """The format preference list is validated before selection."""
    with pytest.raises(AssetRefError, match=match):
        select_asset_format(AssetRef(usd="robot.usd"), formats)
