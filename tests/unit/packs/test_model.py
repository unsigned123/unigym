"""Pure stored pack model: Face, locators, PackItem, Pack and PackSet."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from unigym.packs import AssetResourceRef, PackError, PackLoader, PackSet
from unigym.packs.schema import (
    MANIFEST_FACES,
    EntrypointLocator,
    Face,
    ItemStatus,
    Locator,
    Pack,
    PackItem,
    ResourceLocator,
)
from unigym.provenance import ProviderOrigin


def _origin(
    *,
    kind: str = "manifest",
    pack: str = "alpha",
    face: str | None = "assets",
    item: str | None = "alpha.one",
    resource: str | None = "one.yaml",
    entrypoint: str | None = None,
) -> ProviderOrigin:
    return ProviderOrigin(
        kind=kind,
        pack=pack,
        pack_version="1.0.0",
        root=f"/packs/{pack}",
        discovered_by="explicit-root",
        face=face,
        item_id=item,
        resource=resource,
        entrypoint=entrypoint,
    )


def _item(
    *,
    pack: str = "alpha",
    face: Face = Face.ASSETS,
    kind: str | None = None,
    name: str = "alpha.one",
    locator: Locator | None = None,
    params=None,
    status: ItemStatus = ItemStatus.DECLARED,
) -> PackItem:
    return PackItem(
        pack=pack,
        face=face,
        kind=kind,
        name=name,
        locator=locator if locator is not None else ResourceLocator("one.yaml"),
        params=params,
        origin=_origin(pack=pack, face=face.value, item=name),
        status=status,
    )


def _pack(
    *,
    name: str = "alpha",
    items: tuple[PackItem, ...] | None = None,
    origin: ProviderOrigin | None = None,
) -> Pack:
    return Pack(
        schema="unigym.pack/1",
        name=name,
        version="1.0.0",
        api=1,
        trust="content",
        root=Path(f"/packs/{name}"),
        entrypoint=None,
        requires={},
        items=items if items is not None else (_item(pack=name),),
        origin=origin
        if origin is not None
        else ProviderOrigin(
            kind="pack",
            pack=name,
            pack_version="1.0.0",
            root=f"/packs/{name}",
            discovered_by="explicit-root",
        ),
    )


def test_face_enum_is_exactly_the_manifest_vocabulary() -> None:
    """Face has exactly the seven manifest faces and no registration face."""
    assert tuple(Face) == (
        Face.RUNS,
        Face.SCENE_PRESETS,
        Face.ASSETS,
        Face.PROPS,
        Face.TERMINATIONS,
        Face.PROVIDERS,
        Face.SELECTION,
    )
    assert MANIFEST_FACES == tuple(Face)
    assert [face.value for face in Face] == [
        "runs",
        "scene_presets",
        "assets",
        "props",
        "terminations",
        "providers",
        "selection",
    ]


def test_item_status_has_the_diagnostic_failed_member() -> None:
    """ItemStatus exposes the five lifecycle states including diagnostic FAILED."""
    assert [status.value for status in ItemStatus] == [
        "declared",
        "registered",
        "adopted",
        "inert",
        "failed",
    ]


def test_locators_are_distinct_and_structural() -> None:
    """Resource and entrypoint locators carry only their own locator data."""
    resource = ResourceLocator("assets/x.yaml")
    entrypoint = EntrypointLocator(module="acme.mod", attribute="make", raw="acme.mod:make")

    assert resource.path == "assets/x.yaml"
    assert entrypoint.raw == "acme.mod:make"
    assert not isinstance(resource, EntrypointLocator)
    with pytest.raises(AttributeError):
        resource.module  # type: ignore[attr-defined]


def test_pack_item_requires_one_locator() -> None:
    """A stored item cannot be locator-less."""
    with pytest.raises(PackError, match="locator must be a ResourceLocator"):
        PackItem(
            pack="alpha",
            face=Face.ASSETS,
            kind=None,
            name="alpha.one",
            locator=None,  # type: ignore[arg-type]
            params=None,
            origin=_origin(),
        )


def test_provider_item_requires_a_non_empty_kind() -> None:
    """Providers are identified by a non-empty kind."""
    with pytest.raises(PackError, match="requires a non-empty kind"):
        _item(face=Face.PROVIDERS, kind=None)


def test_pack_item_rejects_failed_as_a_stored_status() -> None:
    """FAILED is diagnostic-only and never becomes a stored item state."""
    with pytest.raises(PackError, match="cannot store ItemStatus.FAILED"):
        _item(status=ItemStatus.FAILED)


def test_pack_item_deep_freezes_nested_params() -> None:
    """Stored params never expose a mutable mapping or sequence."""
    item = _item(
        face=Face.PROVIDERS,
        kind="scene",
        name="alpha.table_top",
        locator=EntrypointLocator(module="alpha", attribute="table", raw="alpha:table"),
        params={
            "layout": {
                "type": "list",
                "default": ["a", ["b"]],
                "choices": [["a", ["b"]]],
            }
        },
    )

    assert item.params is not None
    declaration = item.params["layout"]
    assert declaration["default"] == ("a", ("b",))
    assert declaration["choices"] == (("a", ("b",)),)
    with pytest.raises(TypeError):
        declaration["default"][1][0] = "c"  # type: ignore[index]
    with pytest.raises(AttributeError):
        declaration["choices"].append(())  # type: ignore[union-attr]


def test_pack_item_compatibility_projections_read_the_locator() -> None:
    """id/resource/entrypoint are read projections over name and locator."""
    asset = _item()
    provider = _item(
        face=Face.PROVIDERS,
        kind="adapter",
        name="alpha.adapter",
        locator=EntrypointLocator(module="alpha", attribute="Adapter", raw="alpha:Adapter"),
    )

    assert asset.id == asset.name == "alpha.one"
    assert asset.resource == "one.yaml"
    assert asset.entrypoint is None
    assert provider.resource is None
    assert provider.entrypoint == "alpha:Adapter"


def test_pack_items_preserve_declaration_order_and_filter() -> None:
    """Pack keeps one item tuple and filters by face, kind and status."""
    items = (
        _item(name="alpha.run", face=Face.RUNS),
        _item(name="alpha.robot", face=Face.PROVIDERS, kind="robot"),
        _item(
            name="alpha.cup",
            face=Face.ASSETS,
            status=ItemStatus.REGISTERED,
        ),
        _item(name="alpha.scene", face=Face.PROVIDERS, kind="scene", status=ItemStatus.INERT),
    )
    pack = _pack(items=items)

    assert pack.items == items
    assert pack.items_for() == items
    assert pack.items_for(face=Face.PROVIDERS) == (items[1], items[3])
    assert pack.items_for(face=Face.PROVIDERS, kind="scene") == (items[3],)
    assert pack.items_for(status=ItemStatus.REGISTERED) == (items[2],)
    assert pack.item(Face.PROVIDERS, "alpha.robot") is items[1]
    assert pack.provider("scene", "alpha.scene") is items[3]
    assert pack.face_counts == {Face.RUNS: 1, Face.ASSETS: 1, Face.PROVIDERS: 2}


def test_pack_lookup_errors_name_the_pack_and_known_ids() -> None:
    """A missing face/name lookup is a teaching error with the known choices."""
    pack = _pack()

    with pytest.raises(PackError) as missing_item:
        pack.item(Face.ASSETS, "alpha.missing")
    assert "declares no 'assets' id 'alpha.missing'" in str(missing_item.value)
    assert "alpha.one" in str(missing_item.value)

    with pytest.raises(PackError) as missing_provider:
        pack.provider("robot", "alpha.missing")
    assert "has no robot id 'alpha.missing'" in str(missing_provider.value)


def test_pack_equality_includes_items_and_origin() -> None:
    """Two packs that differ only in item data or origin are not equal."""
    base = _pack()
    same = _pack()
    changed_item = _pack(items=(_item(name="alpha.two"),))
    changed_origin = _pack(
        origin=ProviderOrigin(
            kind="pack",
            pack="alpha",
            pack_version="1.0.0",
            root="/packs/alpha",
            discovered_by="env:UNIGYM_PACKS",
        )
    )

    assert base == same
    assert base != changed_item
    assert base != changed_origin


def test_pack_set_is_an_ordered_index_of_the_same_items() -> None:
    """PackSet preserves pack order and returns the exact stored items."""
    alpha = _pack(name="alpha")
    beta = _pack(name="beta", items=(_item(pack="beta", name="beta.one"),))
    pack_set = PackSet((alpha, beta))

    assert pack_set.packs() == (alpha, beta)
    assert pack_set.get("beta") is beta
    assert pack_set.items() == (alpha.items[0], beta.items[0])
    assert pack_set.item(Face.ASSETS, "beta.one") is beta.items[0]
    assert pack_set.item(Face.ASSETS, "alpha.one") is alpha.items[0]


def test_pack_set_filters_items_by_face_kind_and_status() -> None:
    """PackSet is a stable filter over the same immutable items."""
    robot = _item(face=Face.PROVIDERS, kind="robot", name="alpha.robot")
    scene = _item(face=Face.PROVIDERS, kind="scene", name="alpha.scene")
    registered = _item(name="alpha.cup", status=ItemStatus.REGISTERED)
    pack = _pack(items=(robot, scene, registered))
    pack_set = PackSet((pack,))

    assert pack_set.items(face=Face.PROVIDERS) == (robot, scene)
    assert pack_set.items(face=Face.PROVIDERS, kind="scene") == (scene,)
    assert pack_set.items(status=ItemStatus.REGISTERED) == (registered,)
    assert pack_set.providers("robot") == (robot,)
    assert pack_set.resources(Face.ASSETS) == (registered,)
    assert pack_set.origins() == (pack.origin,)


def test_pack_set_duplicate_pack_names_are_rejected() -> None:
    """Two packs with one name are fatal; there is no first match."""
    with pytest.raises(PackError, match="duplicate pack name 'alpha'"):
        PackSet((_pack(name="alpha"), _pack(name="alpha")))


def test_pack_set_duplicate_face_name_across_kinds_is_rejected() -> None:
    """Uniqueness is (face, name): provider names cannot collide across kinds."""
    first = _pack(
        name="alpha",
        items=(_item(pack="alpha", face=Face.PROVIDERS, kind="robot", name="shared"),),
    )
    second = _pack(
        name="beta",
        items=(_item(pack="beta", face=Face.PROVIDERS, kind="scene", name="shared"),),
    )

    with pytest.raises(PackError, match="duplicate 'providers' id 'shared'"):
        PackSet((first, second))


def test_pack_set_construction_touches_no_filesystem() -> None:
    """A pack set indexes hand-built packs with missing roots and resources."""
    pack = _pack(items=(_item(locator=ResourceLocator("missing/never.yaml")),))
    assert not pack.root.exists()

    pack_set = PackSet((pack,))

    assert pack_set.item(Face.ASSETS, "alpha.one") is pack.items[0]


def test_refs_project_stored_items_without_reading_the_filesystem() -> None:
    """A typed ref is minted from a stored item and is never stored itself."""
    pack = _pack(items=(_item(locator=ResourceLocator("missing/never.yaml")),))
    item = pack.items[0]

    ref = AssetResourceRef.from_item(item)

    assert ref.resource == "missing/never.yaml"
    assert ref.origin is item.origin
    assert all(isinstance(stored, PackItem) for stored in pack.items)
    assert ref != item


def test_pack_set_equality_includes_item_data() -> None:
    """Pack-set equality is the ordered pack tuple, item data included."""
    alpha = _pack()
    beta = _pack(name="beta", items=(_item(pack="beta", name="beta.one"),))

    assert PackSet((alpha, beta)) == PackSet((alpha, beta))
    changed_beta = replace(beta, items=(_item(pack="beta", name="beta.two"),))
    assert PackSet((alpha, beta)) != PackSet((alpha, changed_beta))


def test_pack_set_is_pure_and_object_loading_lives_on_the_loader() -> None:
    """Pure construction stays data-only; the loader owns object loading."""
    pack_set = PackSet((_pack(),))
    loader = PackLoader.from_pack_set(pack_set)

    assert loader.pack_set is pack_set
    assert not hasattr(pack_set, "to_runtime")
    assert not hasattr(pack_set, "from_runtime")
    assert not hasattr(pack_set, "resource_path")
    assert not hasattr(pack_set, "robot")
    assert not hasattr(pack_set, "robots")


def test_pack_schema_alias_is_not_part_of_the_public_vocabulary() -> None:
    """The new model owns the names; flattened catalog constructs are gone."""
    from unigym.packs import __all__ as pack_exports

    assert "Pack" in pack_exports
    assert "PackItem" in pack_exports
    assert "PackInfo" not in pack_exports
    assert "ResourceInfo" not in pack_exports
    assert "PackManifestV1" not in pack_exports
    assert "ProvidedItem" not in pack_exports
