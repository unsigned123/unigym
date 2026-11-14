"""Unit contract for the one scene-provider parameter-schema truth."""

from __future__ import annotations

import pytest

from unigym.scenes.provider_params import (
    ProviderParamSpec,
    SceneProviderParamError,
    fill_and_validate,
    normalize_code_params,
    normalize_manifest_params,
    schemas_equal,
)


def test_manifest_normalization_marks_required_and_optional_params() -> None:
    """A declaration without a default is required; with one it is optional."""
    schema = normalize_manifest_params(
        {
            "style": {"type": "int", "default": 1},
            "seed": {"type": "int"},
            "layout": {
                "type": "str",
                "default": "default",
                "choices": ["default", "dense"],
            },
        }
    )

    assert schema["style"] == ProviderParamSpec(type="int", required=False, default=1)
    assert schema["seed"] == ProviderParamSpec(type="int", required=True)
    assert schema["layout"] == ProviderParamSpec(
        type="str",
        required=False,
        default="default",
        choices=("default", "dense"),
    )


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"x": {"default": 1}}, "requires 'type'"),
        ({"x": {"type": "int", "extra": 1}}, "unknown declaration key"),
        ({"x": {"type": "object"}}, "parameter type must be one of"),
        ({"x": {"type": "int", "default": "one"}}, "must be int"),
        ({"x": {"type": "int", "choices": []}}, "non-empty list"),
        ({"x": {"type": "int", "choices": [1, True]}}, "must be int"),
        ({"x": {"type": "int", "choices": [1, 1]}}, "must be distinct"),
        ({"x": {"type": "int", "default": 3, "choices": [1, 2]}}, "must be one of"),
    ],
)
def test_manifest_normalization_rejects_bad_declarations(params, match: str) -> None:
    """Malformed declarations fail closed with the offending rule."""
    with pytest.raises(SceneProviderParamError, match=match):
        normalize_manifest_params(params)


def test_code_normalization_accepts_compact_and_mapping_forms() -> None:
    """Compact tuples and full mappings normalize to the same specs."""
    schema = normalize_code_params(
        {
            "style": (int, 1),
            "seed": int,
            "layout": (str, "default", ("default", "dense")),
            "flag": {"type": "bool", "default": False},
            "scale": (float, 1),
            "tare": (list, [1, 2]),
            "table": (dict, {"a": 1}),
            "nothing": (type(None), None),
        }
    )

    assert (
        schema["style"] == normalize_code_params({"style": {"type": "int", "default": 1}})["style"]
    )
    assert schema["seed"] == ProviderParamSpec(type="int", required=True)
    assert schema["layout"].choices == ("default", "dense")
    assert schema["flag"].default is False
    assert schema["scale"].type == "float"
    assert schema["tare"] == ProviderParamSpec(type="list", required=False, default=[1, 2])
    assert schema["table"] == ProviderParamSpec(type="dict", required=False, default={"a": 1})
    assert schema["nothing"] == ProviderParamSpec(type="null", required=False, default=None)


@pytest.mark.parametrize(
    "declaration",
    [(), (int, 1, ("a",), "extra"), ("object",), (int, 1, ())],
)
def test_code_normalization_rejects_bad_compact_shapes(declaration) -> None:
    """Compact declarations outside the supported shapes fail closed."""
    with pytest.raises(SceneProviderParamError):
        normalize_code_params({"x": declaration})


def test_fill_and_validate_fills_defaults_and_rejects_bad_values() -> None:
    """Defaults fill, unknown keys fail and every value is type-checked."""
    schema = normalize_manifest_params(
        {
            "style": {"type": "int", "default": 1},
            "seed": {"type": "int"},
            "layout": {
                "type": "str",
                "default": "default",
                "choices": ["default", "dense"],
            },
        }
    )

    assert fill_and_validate(schema, {"seed": 3}, provider="acme.table_top") == {
        "style": 1,
        "seed": 3,
        "layout": "default",
    }

    with pytest.raises(SceneProviderParamError, match="unknown parameter"):
        fill_and_validate(schema, {"seed": 3, "extra": 1}, provider="acme.table_top")
    with pytest.raises(SceneProviderParamError, match="missing required parameter"):
        fill_and_validate(schema, {"style": 2}, provider="acme.table_top")
    with pytest.raises(SceneProviderParamError, match="must be int"):
        fill_and_validate(schema, {"seed": True}, provider="acme.table_top")
    with pytest.raises(SceneProviderParamError, match="must be one of"):
        fill_and_validate(schema, {"seed": 3, "layout": "wide"}, provider="acme.table_top")
    with pytest.raises(SceneProviderParamError, match=r"preset 'acme.tabletop'"):
        fill_and_validate(
            schema,
            {"seed": 3, "extra": 1},
            provider="acme.table_top",
            preset="acme.tabletop",
        )


def test_partial_validation_supports_preset_load_before_run_overrides() -> None:
    """Preset load validates declared values but may defer a required parameter."""
    schema = normalize_manifest_params(
        {
            "style": {"type": "int", "default": 1},
            "seed": {"type": "int"},
        }
    )

    assert fill_and_validate(schema, {}, provider="acme.table_top", require_all=False) == {
        "style": 1
    }

    with pytest.raises(SceneProviderParamError, match="missing required parameter"):
        fill_and_validate(schema, {}, provider="acme.table_top")
    with pytest.raises(SceneProviderParamError, match="unknown parameter"):
        fill_and_validate(schema, {"extra": 1}, provider="acme.table_top", require_all=False)
    with pytest.raises(SceneProviderParamError, match="must be int"):
        fill_and_validate(schema, {"seed": True}, provider="acme.table_top", require_all=False)


def test_mutable_defaults_are_fresh_per_dispatch() -> None:
    """Each dispatch deep-copies list/dict defaults instead of sharing them."""
    schema = normalize_code_params({"tare": (list, [1, [2]]), "table": (dict, {"a": {"b": 1}})})

    first = fill_and_validate(schema, {}, provider="acme.mutable")
    first_tare = first["tare"]
    first_table = first["table"]
    assert isinstance(first_tare, list)
    assert isinstance(first_table, dict)
    first_tare.append(3)
    first_table["a"] = 99

    second = fill_and_validate(schema, {}, provider="acme.mutable")
    assert second == {"tare": [1, [2]], "table": {"a": {"b": 1}}}
    assert second["tare"] is not first["tare"]
    assert second["table"] is not first["table"]
    assert schema["tare"].default == [1, [2]]
    assert schema["table"].default == {"a": {"b": 1}}


def test_float_accepts_integers_but_int_rejects_bools() -> None:
    """JSON ints satisfy a float parameter; booleans never satisfy an int."""
    schema = normalize_code_params({"scale": (float, 1.0), "count": int})

    filled = fill_and_validate(schema, {"count": 2}, provider="acme.scaled")
    assert filled["scale"] == 1.0
    assert fill_and_validate(schema, {"scale": 2, "count": 2}, provider="acme.scaled")["scale"] == 2

    with pytest.raises(SceneProviderParamError, match="must be int"):
        fill_and_validate(schema, {"count": False}, provider="acme.scaled")


def test_schemas_equal_compares_normalized_schemas_and_none() -> None:
    """The drift comparator treats equal normalized schemas as equal."""
    code = normalize_code_params({"x": int})
    manifest = normalize_manifest_params({"x": {"type": "int"}})

    assert schemas_equal(code, manifest)
    assert schemas_equal(None, None)
    assert not schemas_equal(code, None)
    assert not schemas_equal(None, code)
    assert not schemas_equal(
        normalize_code_params({"x": (int, 1)}),
        normalize_code_params({"x": (int, 2)}),
    )
