"""Contract for the public ``unigym.pack/1`` schema and pack catalog."""

from __future__ import annotations

import operator
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

import unigym.pack_api as pack_api
from unigym.pack_api import (
    CatalogEntry,
    PackCatalog,
    PackManifestV1,
    ProvidedItem,
    load_manifest_v1,
)
from unigym.packs import PackError
from unigym.packs.catalog import validate_unique


def _document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": "unigym.pack/1",
        "name": "acme",
        "version": "1.2.0",
        "api": 1,
        "trust": "content",
        "provides": {},
    }
    document.update(overrides)
    return document


def _write(root: Path, document: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(document, sort_keys=False)
    (root / "unigym.pack.yaml").write_text(text, encoding="utf-8")


def _touch(root: Path, rel: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("kind: fixture\n", encoding="utf-8")
    return path


def _pack(base: Path, name: str, provides: dict[str, list[dict[str, object]]]) -> PackManifestV1:
    root = base / name
    for items in provides.values():
        for item in items:
            resource = item.get("resource")
            if isinstance(resource, str):
                _touch(root, resource)
    _write(root, _document(name=name, provides=provides))
    return load_manifest_v1(root)


def test_content_manifest_loads(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(
        root,
        _document(
            provides={
                "scene_presets": [
                    {"id": "acme.tabletop", "resource": "scene_presets/tabletop.yaml"}
                ],
            }
        ),
    )
    _touch(root, "scene_presets/tabletop.yaml")

    manifest = load_manifest_v1(root)

    assert isinstance(manifest, PackManifestV1)
    assert manifest.schema == "unigym.pack/1"
    assert manifest.name == "acme"
    assert manifest.version == "1.2.0"
    assert manifest.api == 1
    assert manifest.trust == "content"
    assert manifest.entrypoint is None
    assert dict(manifest.requires) == {}
    assert manifest.provides["scene_presets"] == (
        ProvidedItem(
            id="acme.tabletop",
            resource="scene_presets/tabletop.yaml",
            entrypoint=None,
            kind=None,
        ),
    )
    assert manifest.root == root.resolve()


def test_code_manifest_with_entrypoint_loads(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(root, _document(trust="code", entrypoint="acme_unigym:register"))

    manifest = load_manifest_v1(root)

    assert manifest.trust == "code"
    assert manifest.entrypoint == "acme_unigym:register"


def test_provider_item_with_kind_and_entrypoint_loads(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(
        root,
        _document(
            trust="code",
            provides={
                "providers": [
                    {
                        "id": "acme.dynamic_assets",
                        "kind": "asset_generator",
                        "entrypoint": "acme_unigym.assets:register",
                    }
                ]
            },
        ),
    )

    manifest = load_manifest_v1(root)

    item = manifest.provides["providers"][0]
    assert item.resource is None
    assert item.entrypoint == "acme_unigym.assets:register"
    assert item.kind == "asset_generator"
    assert item.params is None


def test_scene_provider_item_loads_with_an_inline_params_schema(tmp_path: Path) -> None:
    """A providers item with kind: scene accepts and deep-freezes params."""
    root = tmp_path / "acme"
    _write(
        root,
        _document(
            trust="code",
            provides={
                "providers": [
                    {
                        "id": "acme.table_top",
                        "kind": "scene",
                        "entrypoint": "acme_scenes:table_top",
                        "params": {
                            "style": {"type": "int", "default": 1},
                            "seed": {"type": "int"},
                        },
                    }
                ]
            },
        ),
    )

    manifest = load_manifest_v1(root)

    item = manifest.provides["providers"][0]
    assert item.params == {
        "style": {"type": "int", "default": 1},
        "seed": {"type": "int"},
    }
    assert item.params is not None
    with pytest.raises(TypeError):
        item.params["style"]["type"] = "str"  # type: ignore[index]


@pytest.mark.parametrize(
    ("face", "item", "resource", "match"),
    [
        (
            "providers",
            {
                "id": "acme.table_top",
                "kind": "scene",
                "resource": "value.yaml",
            },
            "value.yaml",
            "entrypoint-backed only",
        ),
        (
            "providers",
            {"id": "acme.table_top", "kind": "scene"},
            None,
            "entrypoint-backed",
        ),
        (
            "providers",
            {
                "id": "acme.dynamic_assets",
                "kind": "asset_generator",
                "entrypoint": "acme_assets:register",
                "params": {"style": {"type": "int"}},
            },
            None,
            "unknown provided-item key",
        ),
        (
            "scene_presets",
            {
                "id": "acme.tabletop",
                "resource": "value.yaml",
                "params": {"style": {"type": "int"}},
            },
            "value.yaml",
            "cannot declare",
        ),
        (
            "providers",
            {
                "id": "Acme.TableTop",
                "kind": "scene",
                "entrypoint": "acme_scenes:table_top",
            },
            None,
            "adapter/robot grammar",
        ),
        (
            "providers",
            {
                "id": "acme.table_top",
                "kind": "scene",
                "entrypoint": "acme_scenes:table_top",
                "params": {"style": {"type": "object"}},
            },
            None,
            "params are invalid",
        ),
    ],
)
def test_face_specific_item_rules(
    tmp_path: Path,
    face: str,
    item: dict[str, object],
    resource: str | None,
    match: str,
) -> None:
    """Face, kind, locator, grammar and params rules fail at manifest load."""
    root = tmp_path / "acme"
    if resource is not None:
        _touch(root, resource)
    _write(root, _document(trust="code", provides={face: [item]}))

    with pytest.raises(PackError, match=match):
        load_manifest_v1(root)


def test_catalog_entry_carries_the_provider_params_schema(tmp_path: Path) -> None:
    """The flattened catalog keeps the scene-provider schema."""
    root = tmp_path / "acme"
    _write(
        root,
        _document(
            trust="code",
            provides={
                "providers": [
                    {
                        "id": "acme.table_top",
                        "kind": "scene",
                        "entrypoint": "acme_scenes:table_top",
                        "params": {"style": {"type": "int", "default": 1}},
                    }
                ]
            },
        ),
    )
    manifest = load_manifest_v1(root)

    catalog = PackCatalog((manifest,))

    entry = catalog.entry("providers", "acme.table_top")
    assert entry.params == {"style": {"type": "int", "default": 1}}


def test_scene_provider_params_nested_sequences_are_deep_frozen(tmp_path: Path) -> None:
    """Nested list/tuple declarations freeze to tuples that cannot be mutated."""
    root = tmp_path / "acme"
    _write(
        root,
        _document(
            trust="code",
            provides={
                "providers": [
                    {
                        "id": "acme.table_top",
                        "kind": "scene",
                        "entrypoint": "acme_scenes:table_top",
                        "params": {
                            "layout": {
                                "type": "list",
                                "default": [["a"], "b"],
                                "choices": [[["a"], "b"], ["c"]],
                            }
                        },
                    }
                ]
            },
        ),
    )

    manifest = load_manifest_v1(root)

    item = manifest.provides["providers"][0]
    assert item.params is not None
    declaration = item.params["layout"]
    assert declaration["default"] == (("a",), "b")
    assert isinstance(declaration["default"], tuple)
    assert declaration["choices"] == ((("a",), "b"), ("c",))
    assert isinstance(declaration["choices"], tuple)

    with pytest.raises(TypeError):
        declaration["default"][0][0] = "z"  # type: ignore[index]
    with pytest.raises(AttributeError):
        declaration["choices"].append(("d",))  # type: ignore[union-attr]
    assert item.params["layout"]["default"] == (("a",), "b")


def test_catalog_entry_params_are_deep_frozen(tmp_path: Path) -> None:
    """The catalog re-freezes nested sequences and never exposes a mutable list."""
    root = tmp_path / "acme"
    _write(
        root,
        _document(
            trust="code",
            provides={
                "providers": [
                    {
                        "id": "acme.table_top",
                        "kind": "scene",
                        "entrypoint": "acme_scenes:table_top",
                        "params": {
                            "layout": {
                                "type": "list",
                                "default": ["a", ["b"]],
                                "choices": [["a", ["b"]]],
                            }
                        },
                    }
                ]
            },
        ),
    )
    catalog = PackCatalog((load_manifest_v1(root),))

    entry = catalog.entry("providers", "acme.table_top")

    assert entry.params is not None
    declaration = entry.params["layout"]
    assert declaration["default"] == ("a", ("b",))
    with pytest.raises(TypeError):
        declaration["default"][1][0] = "c"  # type: ignore[index]
    with pytest.raises(AttributeError):
        declaration["choices"].append(())  # type: ignore[union-attr]
    assert declaration["default"] == ("a", ("b",))


def test_requires_is_stored_not_evaluated(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(root, _document(requires={"unigym": ">=99.0,<100.0"}))

    manifest = load_manifest_v1(root)

    assert dict(manifest.requires) == {"unigym": ">=99.0,<100.0"}


@pytest.mark.parametrize("value", [False, True])
def test_shadow_is_rejected_as_an_unknown_key(tmp_path: Path, value: bool) -> None:
    root = tmp_path / "acme"
    _write(root, _document(shadow=value))

    with pytest.raises(PackError, match="unknown pack manifest key"):
        load_manifest_v1(root)


def test_content_pack_rejects_manifest_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(root, _document(trust="content", entrypoint="acme_unigym:register"))

    with pytest.raises(PackError, match="entrypoints require trust: code"):
        load_manifest_v1(root)


def test_malformed_yaml_is_a_pack_error(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    root.mkdir(parents=True)
    (root / "unigym.pack.yaml").write_text("schema: [unterminated\n", encoding="utf-8")

    with pytest.raises(PackError, match="unreadable pack manifest"):
        load_manifest_v1(root)


def test_content_pack_rejects_item_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(
        root,
        _document(
            provides={
                "providers": [
                    {
                        "id": "acme.dynamic_assets",
                        "kind": "asset_generator",
                        "entrypoint": "acme_unigym.assets:register",
                    }
                ]
            }
        ),
    )

    with pytest.raises(PackError, match="trust: code"):
        load_manifest_v1(root)


@pytest.mark.parametrize(
    "item",
    [
        {"id": "acme.item"},
        {
            "id": "acme.item",
            "resource": "item.yaml",
            "entrypoint": "acme_unigym:register",
        },
    ],
)
def test_item_requires_exactly_one_locator(tmp_path: Path, item: dict[str, object]) -> None:
    root = tmp_path / "acme"
    _touch(root, "item.yaml")
    _write(root, _document(trust="code", provides={"assets": [item]}))

    with pytest.raises(PackError, match="exactly one"):
        load_manifest_v1(root)


def test_missing_schema_rejected(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    document = _document()
    del document["schema"]
    _write(root, document)

    with pytest.raises(PackError, match="schema"):
        load_manifest_v1(root)


@pytest.mark.parametrize("schema", ["unigym.pack/2", "pack/1", None])
def test_unknown_schema_rejected(tmp_path: Path, schema: object) -> None:
    root = tmp_path / "acme"
    _write(root, _document(schema=schema))

    with pytest.raises(PackError, match="schema"):
        load_manifest_v1(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Acme"),
        ("name", "9acme"),
        ("name", ""),
        ("version", "1.x"),
        ("version", "1.2.0-beta"),
        ("version", "1..2"),
        ("api", 0),
        ("api", -3),
        ("api", True),
        ("api", "1"),
        ("trust", "trusted"),
    ],
)
def test_invalid_manifest_field_rejected(tmp_path: Path, field: str, value: object) -> None:
    root = tmp_path / "acme"
    _write(root, _document(**{field: value}))

    with pytest.raises(PackError):
        load_manifest_v1(root)


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(root, _document(faces={}))

    with pytest.raises(PackError, match="unknown pack manifest key"):
        load_manifest_v1(root)


def test_unknown_face_rejected(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(root, _document(provides={"widgets": []}))

    with pytest.raises(PackError, match="unknown pack face"):
        load_manifest_v1(root)


def test_removed_scenes_face_names_scene_presets_and_providers(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(root, _document(provides={"scenes": []}))

    with pytest.raises(PackError) as info:
        load_manifest_v1(root)

    message = str(info.value)
    assert "unknown pack face 'scenes'" in message
    assert "scene_presets" in message
    assert "providers" in message
    assert "kind: scene" in message


@pytest.mark.parametrize(
    "item",
    [
        {"id": "acme.tabletop", "entrypoint": "acme_scenes:tabletop"},
        {"id": "acme.tabletop", "resource": "scene_presets/x.yaml", "kind": "scene"},
    ],
    ids=["entrypoint", "kind"],
)
def test_scene_preset_items_are_resource_backed_only(
    tmp_path: Path, item: dict[str, object]
) -> None:
    root = tmp_path / "acme"
    _touch(root, "scene_presets/x.yaml")
    _write(root, _document(trust="code", provides={"scene_presets": [item]}))

    with pytest.raises(PackError, match="resource-backed only"):
        load_manifest_v1(root)


def test_provides_must_be_declared(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    document = _document()
    del document["provides"]
    _write(root, document)

    with pytest.raises(PackError, match="provides"):
        load_manifest_v1(root)


def test_item_requires_non_empty_id(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _write(
        root,
        _document(provides={"scene_presets": [{"id": "  ", "resource": "scene_presets/x.yaml"}]}),
    )

    with pytest.raises(PackError, match="non-empty string id"):
        load_manifest_v1(root)


def test_unknown_item_key_rejected(tmp_path: Path) -> None:
    root = tmp_path / "acme"
    _touch(root, "scene_presets/x.yaml")
    _write(
        root,
        _document(
            provides={
                "scene_presets": [
                    {"id": "acme.x", "resource": "scene_presets/x.yaml", "manifest": "x"}
                ]
            }
        ),
    )

    with pytest.raises(PackError, match="unknown provided-item key"):
        load_manifest_v1(root)


@pytest.mark.parametrize(
    "resource",
    [
        "/tmp/absolute.yaml",
        "../escape.yaml",
        "scene_presets/missing.yaml",
        "",
    ],
)
def test_resource_must_stay_inside_manifest_root(tmp_path: Path, resource: str) -> None:
    root = tmp_path / "acme"
    _write(
        root,
        _document(provides={"scene_presets": [{"id": "acme.scene", "resource": resource}]}),
    )

    with pytest.raises(PackError):
        load_manifest_v1(root)


def test_catalog_preserves_order_and_filters(tmp_path: Path) -> None:
    alpha = _pack(
        tmp_path,
        "alpha",
        {
            "scene_presets": [
                {"id": "alpha.one", "resource": "one.yaml"},
                {"id": "alpha.two", "resource": "two.yaml"},
            ],
            "runs": [{"id": "alpha.demo", "resource": "demo.yaml"}],
        },
    )
    beta = _pack(
        tmp_path,
        "beta",
        {"scene_presets": [{"id": "beta.one", "resource": "one.yaml"}]},
    )

    catalog = PackCatalog((alpha, beta))

    assert catalog.packs() == ("alpha", "beta")
    assert [entry.id for entry in catalog.entries()] == [
        "alpha.one",
        "alpha.two",
        "alpha.demo",
        "beta.one",
    ]
    assert [entry.id for entry in catalog.entries("scene_presets")] == [
        "alpha.one",
        "alpha.two",
        "beta.one",
    ]
    assert catalog.entries("runs") == (
        CatalogEntry(
            pack="alpha",
            face="runs",
            id="alpha.demo",
            resource="demo.yaml",
            entrypoint=None,
            kind=None,
            params=None,
        ),
    )
    assert catalog.get("beta") is beta

    with pytest.raises(PackError, match="unknown pack"):
        catalog.get("missing")
    with pytest.raises(PackError, match="unknown pack face"):
        catalog.entries("widgets")


def test_catalog_rejects_duplicate_pack_names(tmp_path: Path) -> None:
    first = _pack(tmp_path / "left", "acme", {})
    second = _pack(tmp_path / "right", "acme", {})

    with pytest.raises(PackError, match="duplicate pack name"):
        PackCatalog((first, second))


def test_catalog_rejects_duplicate_face_id(tmp_path: Path) -> None:
    first = _pack(
        tmp_path / "left",
        "alpha",
        {"scene_presets": [{"id": "shared.item", "resource": "x.yaml"}]},
    )
    second = _pack(
        tmp_path / "right",
        "beta",
        {"scene_presets": [{"id": "shared.item", "resource": "x.yaml"}]},
    )

    with pytest.raises(PackError, match="duplicate 'scene_presets' id"):
        PackCatalog((first, second))


def test_validate_unique_materializes_one_shot_iterables(tmp_path: Path) -> None:
    """A generator is consumed once, so the second item pass still sees duplicates."""
    first = _pack(
        tmp_path / "left",
        "alpha",
        {"scene_presets": [{"id": "shared.item", "resource": "x.yaml"}]},
    )
    second = _pack(
        tmp_path / "right",
        "beta",
        {"scene_presets": [{"id": "shared.item", "resource": "x.yaml"}]},
    )

    with pytest.raises(PackError, match="duplicate 'scene_presets' id"):
        validate_unique(iter((first, second)))


def test_catalog_and_manifest_are_immutable(tmp_path: Path) -> None:
    manifest = _pack(
        tmp_path,
        "alpha",
        {"scene_presets": [{"id": "alpha.one", "resource": "one.yaml"}]},
    )
    catalog = PackCatalog((manifest,))

    assert isinstance(catalog.manifests, tuple)
    with pytest.raises(FrozenInstanceError):
        setattr(catalog, "manifests", ())
    with pytest.raises(FrozenInstanceError):
        setattr(manifest, "name", "other")
    with pytest.raises(TypeError):
        operator.setitem(manifest.requires, "unigym", ">=0")
    with pytest.raises(TypeError):
        operator.setitem(manifest.provides, "scene_presets", ())


def test_public_pack_api_exports() -> None:
    expected = {
        "ActivationHandle",
        "AssetContext",
        "AssetResourceRef",
        "CatalogEntry",
        "PackCatalog",
        "PackContext",
        "PackInfo",
        "PackManifestV1",
        "PackRuntime",
        "PackSet",
        "ProvidedItem",
        "ProviderInfo",
        "ProviderOrigin",
        "ResourceInfo",
        "RobotRef",
        "ScenePresetRef",
        "SceneProvider",
        "SceneProviderContext",
        "SceneProviderRef",
        "build_scene_from_provider",
        "discover",
        "discover_set",
        "installed",
        "load_asset_manifest",
        "load_manifest_v1",
        "register_adapter",
        "register_prop_template",
        "register_robot",
        "register_scene_content_filler",
        "register_scene_provider",
        "register_termination",
        "registered_scene_providers",
        "require",
    }
    removed = {"asset_path", "code_pack_root", "find", "load_table", "resolve"}

    assert set(pack_api.__all__) == expected
    for name in expected:
        assert getattr(pack_api, name) is not None
    for name in removed:
        assert not hasattr(pack_api, name)


def test_manifest_load_does_not_import_pack_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "acme"
    _write(root, _document(trust="code", entrypoint="acme_sentinel:register"))
    (root / "acme_sentinel.py").write_text(
        "raise AssertionError('pack python imported during manifest load')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop("acme_sentinel", None)

    manifest = load_manifest_v1(root)

    assert manifest.entrypoint == "acme_sentinel:register"
    assert "acme_sentinel" not in sys.modules
