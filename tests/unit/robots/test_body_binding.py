"""Canonical body-id syntax and BodyBinding map-validation invariants."""

from __future__ import annotations

import pytest

from unigym.robots.base import (
    BodyBinding,
    BodyBindingError,
    normalize_canonical_body_id,
    validate_body_bindings,
)
from unigym.assets.asset_ref import AssetRef


@pytest.mark.parametrize(
    "value",
    [
        "",
        " base",
        "base ",
        "left hand",
        "left\thand",
        "left/hand",
        "left\\hand",
        ".base",
        "base.",
        "left..hand",
        "*",
        "base*",
    ],
)
def test_canonical_body_id_rejects_empty_whitespace_paths_and_wildcards(value):
    """A canonical body id is a non-empty semantic name without paths or wildcards."""
    with pytest.raises(BodyBindingError, match="canonical body id"):
        normalize_canonical_body_id(value, "canonical body id")


def test_canonical_body_id_accepts_dotted_semantic_segments():
    """Dotted semantic segments such as left_hand.index_1 are canonical."""
    body_id = normalize_canonical_body_id("left_hand.index_1", "canonical body id")

    assert body_id == "left_hand.index_1"
    assert isinstance(body_id, str)


@pytest.mark.parametrize(
    "names",
    [{}, None, {"mjcf": ""}, {"mjcf": " root"}, {"": "root"}, {" mjcf": "root"}],
)
def test_body_binding_rejects_empty_or_malformed_names(names):
    """A binding needs a non-empty format -> native-name mapping."""
    with pytest.raises(BodyBindingError):
        BodyBinding("base", names)


def test_body_binding_duplicate_native_name_is_rejected_per_format():
    """Two bodies may not share a native name inside one asset format."""
    base = BodyBinding("base", {"mjcf": "root", "usd": "/World/root"})
    foot = BodyBinding("foot", {"mjcf": "root", "usd": "/World/foot"})

    with pytest.raises(
        BodyBindingError,
        match=r"duplicate mjcf native name 'root' for bodies 'base' and 'foot'",
    ):
        validate_body_bindings({base.canonical: base, foot.canonical: foot})


def test_body_binding_same_native_name_across_formats_is_allowed():
    """The duplicate-native check is per format, never global."""
    base = BodyBinding("base", {"mjcf": "shared"})
    foot = BodyBinding("foot", {"usd": "shared"})

    validate_body_bindings({base.canonical: base, foot.canonical: foot})


def test_body_binding_missing_required_id_is_rejected():
    """Every required canonical id must have a binding."""
    base = BodyBinding("base", {"mjcf": "root"})

    with pytest.raises(BodyBindingError, match=r"missing required ids \['foot'\]"):
        validate_body_bindings({"base": base}, required_ids=("foot",))


def test_body_binding_map_key_must_equal_its_record_canonical():
    """A binding-map key is the record's canonical id, never a different id."""
    binding = BodyBinding("base", {"mjcf": "root"})

    with pytest.raises(BodyBindingError, match="does not match its record canonical"):
        validate_body_bindings({"foot": binding})


def test_body_binding_required_format_missing_is_rejected():
    """A required id must carry every explicitly listed format."""
    base = BodyBinding("base", {"mjcf": "root"})

    with pytest.raises(BodyBindingError, match=r"lack required formats.*\['usd'\]"):
        validate_body_bindings(
            {"base": base},
            required_ids=("base",),
            formats=("mjcf", "usd"),
        )


def test_body_binding_unknown_format_key_is_rejected_when_formats_are_declared():
    """A binding may not carry a format outside the explicitly listed set."""
    base = BodyBinding("base", {"mjcf": "root", "usd": "root_usd"})

    with pytest.raises(BodyBindingError, match=r"unknown formats \['usd'\]"):
        validate_body_bindings({"base": base}, formats=("mjcf",))


def test_asset_ref_to_dict_supplies_required_body_binding_formats():
    """AssetRef owns the format vocabulary a BodyBinding validates against."""
    asset = AssetRef(mjcf="g1.xml", usd="g1.usd")
    formats = tuple(asset.to_dict())
    binding = BodyBinding(
        "base",
        {asset_format: f"{asset_format}_base" for asset_format in formats},
    )

    validate_body_bindings({"base": binding}, required_ids=("base",), formats=formats)

    assert formats == ("mjcf", "usd")
