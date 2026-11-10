"""Model IR construction defaults, validation, round-trip and malformed input."""

from __future__ import annotations

import pytest

from unigym.model import (
    Actuator,
    Body,
    Camera,
    Geom,
    Joint,
    Material,
    MjExt,
    ModelIR,
    ModelIRError,
    Site,
    Table,
    Texture,
    UnknownExtensionError,
    UnparsedExt,
)


def _model(**tables) -> ModelIR:
    return ModelIR(name="m", backend=None, compiled=False, **tables)


def test_model_ir_defaults_to_empty_tables_and_mappings():
    """A newly built IR has empty faces and no backend extensions."""
    model = ModelIR(name="m", backend=None, compiled=False)

    for face in ("bodies", "joints", "geoms", "sites", "cameras", "actuators", "materials"):
        table = getattr(model, face)
        assert len(table) == 0, face
        assert table.names() == [], face
    assert model.meta == {}
    assert model.ext == {}
    assert model.mj is None
    assert model.usd is None


@pytest.mark.parametrize(
    ("tables", "match"),
    [
        ({"bodies": Table([Body(name="b", parent="ghost")])}, "missing parent 'ghost'"),
        (
            {
                "bodies": Table([Body(name="world")]),
                "joints": Table([Joint(name="j", type="hinge", body="ghost")]),
            },
            "references missing body 'ghost'",
        ),
        (
            {
                "bodies": Table([Body(name="world")]),
                "joints": Table([Joint(name="j", type="hinge", body="world", range=(1.0, 0.0))]),
            },
            "reversed range",
        ),
        (
            {
                "bodies": Table([Body(name="world")]),
                "geoms": Table([Geom(name="collision", type="box", body="ghost")]),
            },
            "geom 'collision' references missing body 'ghost'",
        ),
        (
            {
                "bodies": Table([Body(name="world")]),
                "sites": Table([Site(name="marker", type="sphere", body="ghost")]),
            },
            "site 'marker' references missing body 'ghost'",
        ),
        (
            {
                "bodies": Table([Body(name="world")]),
                "cameras": Table([Camera(name="cam", body="ghost")]),
            },
            "references missing body 'ghost'",
        ),
        (
            {
                "bodies": Table([Body(name="world")]),
                "actuators": Table([Actuator(name="act", target="ghost", trntype="joint")]),
            },
            "targets missing joint 'ghost'",
        ),
    ],
)
def test_model_ir_validate_rejects_dangling_references_and_reversed_range(tables, match):
    """Validation rejects a dangling parent/body/target or a reversed joint range."""
    with pytest.raises(ModelIRError, match=match):
        _model(**tables).validate()


_MISSING = object()


@pytest.mark.parametrize(
    ("section", "value", "match"),
    [
        ("name", _MISSING, "missing required string key 'name'"),
        ("name", 7, "'name' must be a string, got int"),
        ("bodies", _MISSING, "missing required table 'bodies'"),
        ("joints", [], "table 'joints' must be a mapping, got list"),
        ("geoms", {1: {}}, "table 'geoms' must be a string-keyed mapping"),
        ("bodies", {"world": "nope"}, "table 'bodies' row 'world' must be a mapping"),
        ("bodies", {"world": {}}, "row 'world' must declare a string 'name'"),
        (
            "bodies",
            {"world": {"name": "other"}},
            "row 'world' declares mismatched name 'other'",
        ),
        ("ext", {"mj": "nope"}, "extension 'mj' payload must be a mapping"),
        ("backend", 7, "'backend' must be a string or None"),
        ("compiled", "false", "'compiled' must be a boolean"),
        ("meta", {1: "x"}, "'meta' must be a string-keyed mapping"),
        ("ext", {1: {}}, "'ext' must be a string-keyed mapping"),
        ("meta", [], "'meta' must be a mapping, got list"),
        ("ext", [], "'ext' must be a mapping, got list"),
    ],
)
def test_model_ir_from_dict_requires_canonical_sections(section, value, match):
    """A malformed canonical document fails naming the key and expected type."""
    document = _model(bodies=Table([Body(name="world")])).to_dict()
    if value is _MISSING:
        del document[section]
    else:
        document[section] = value

    with pytest.raises(ModelIRError, match=match):
        ModelIR.from_dict(document)


def test_model_ir_canonical_round_trip_is_stable():
    """``to_dict`` -> ``from_dict`` -> ``to_dict`` preserves the canonical document."""
    model = _model(
        bodies=Table([Body(name="torso", qualname="robot/torso", pos=(0.1, 0.2, 0.3))]),
        joints=Table(
            [
                Joint(
                    name="elbow",
                    type="hinge",
                    body="torso",
                    range=(-1.5, 1.5),
                    limited=True,
                )
            ]
        ),
        materials=Table([Material(name="steel", rgba=(0.1, 0.2, 0.3, 1.0))]),
        meta={"note": "round-trip"},
        ext={"mj": MjExt(textures=Table([Texture(name="tex", file="tex.png")]))},
    )

    document = model.to_dict(include_ext=True, include_meta=True)
    restored = ModelIR.from_dict(document)

    assert restored.validate() is restored
    assert restored.to_dict(include_ext=True, include_meta=True) == document
    assert document["bodies"]["torso"]["pos"] == [
        float.hex(0.1),
        float.hex(0.2),
        float.hex(0.3),
    ]
    assert document["joints"]["elbow"]["range"] == [float.hex(-1.5), float.hex(1.5)]


def test_model_ir_unknown_extension_is_strict_by_default_and_preserved_when_lenient():
    """An unknown extension namespace fails strictly and survives lenient parsing."""
    document = _model(ext={"mj": MjExt()}).to_dict(include_ext=True)
    document["ext"]["custom"] = {"payload": 1}

    with pytest.raises(UnknownExtensionError, match="unknown extension namespace 'custom'"):
        ModelIR.from_dict(document)

    lenient = ModelIR.from_dict(document, strict=False)
    assert isinstance(lenient.ext["custom"], UnparsedExt)
    assert lenient.to_dict(include_ext=True)["ext"]["custom"] == {"payload": 1}


def test_model_ir_to_dict_omits_optional_sections_and_never_serializes_builder():
    """Optional sections are opt-in and the opaque native builder is never written."""
    marker = object()
    model = _model(meta={"builder": marker, "note": "x"}, ext={"mj": MjExt()})

    plain = model.to_dict()
    assert "meta" not in plain
    assert "ext" not in plain

    with_meta = model.to_dict(include_meta=True)
    assert with_meta["meta"] == {"note": "x"}
    assert "ext" not in with_meta

    with_ext = model.to_dict(include_ext=True)
    assert set(with_ext["ext"]) == {"mj"}
    assert "meta" not in with_ext
