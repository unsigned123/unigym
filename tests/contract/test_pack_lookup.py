"""Contract for catalog and runtime lookup by ``(face, id)``.

Lookup is a pure catalog operation plus the existing resource resolution. It
never discovers, activates or imports pack Python, and an entrypoint-only item
fails with a teaching error that points at activation.
"""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from unigym.pack_api import CatalogEntry, PackRuntime, discover
from unigym.packs import PackError

PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
CONTENT_ROOT = PACKS_ROOT / "acme_content"
CODE_ROOT = PACKS_ROOT / "acme_code"


@pytest.fixture(autouse=True)
def _isolate_pack_modules() -> None:
    """Keep fixture import state and sys.path from leaking between tests."""
    path_snapshot = list(sys.path)
    for name in [
        name for name in sys.modules if name == "acme_pack" or name.startswith("acme_pack.")
    ]:
        del sys.modules[name]
    yield
    sys.path[:] = path_snapshot
    for name in [
        name for name in sys.modules if name == "acme_pack" or name.startswith("acme_pack.")
    ]:
        del sys.modules[name]


def _runtime() -> PackRuntime:
    return discover([CONTENT_ROOT, CODE_ROOT])


def test_catalog_entry_returns_the_immutable_declared_entry() -> None:
    catalog = _runtime().catalog

    entry = catalog.entry("scene_presets", "acme_content.tabletop")

    assert entry == CatalogEntry(
        pack="acme_content",
        face="scene_presets",
        id="acme_content.tabletop",
        resource="scene_presets/tabletop.yaml",
        entrypoint=None,
        kind=None,
        params=None,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(entry, "id", "changed")


def test_catalog_entry_unknown_face_names_the_known_faces() -> None:
    catalog = _runtime().catalog

    with pytest.raises(PackError) as info:
        catalog.entry("widgets", "acme_content.tabletop")

    message = str(info.value)
    assert "unknown pack face 'widgets'" in message
    assert "scene_presets" in message


def test_catalog_entry_rejects_none_face_and_resource_for_delegates() -> None:
    runtime = _runtime()

    with pytest.raises(PackError) as info:
        runtime.catalog.entry(None, "acme_content.tabletop")

    assert "unknown pack face None" in str(info.value)
    assert "scene_presets" in str(info.value)

    with pytest.raises(PackError, match="unknown pack face None"):
        runtime.resource_for(None, "acme_content.tabletop")


def test_catalog_entry_unknown_id_names_the_face_id_and_known_ids() -> None:
    catalog = _runtime().catalog

    with pytest.raises(PackError) as info:
        catalog.entry("scene_presets", "acme_content.missing")

    message = str(info.value)
    assert "unknown id 'acme_content.missing'" in message
    assert "pack face 'scene_presets'" in message
    assert "acme_content.tabletop" in message


@pytest.mark.parametrize(
    ("face", "item_id", "rel"),
    [
        ("runs", "acme_content.demo", "runs/demo.yaml"),
        ("scene_presets", "acme_content.tabletop", "scene_presets/tabletop.yaml"),
        ("assets", "acme_content.cup", "assets/cup/asset.yaml"),
    ],
)
def test_resource_for_resolves_resource_backed_items(face: str, item_id: str, rel: str) -> None:
    runtime = _runtime()

    resource = runtime.resource_for(face, item_id)

    assert resource == (CONTENT_ROOT / rel).resolve()
    assert resource.is_absolute()
    assert resource.is_relative_to(CONTENT_ROOT.resolve())
    assert resource.is_file()


def test_resource_for_entrypoint_only_item_points_at_activation() -> None:
    runtime = _runtime()

    with pytest.raises(PackError) as info:
        runtime.resource_for("providers", "acme_code.dynamic_assets")

    message = str(info.value)
    assert "pack 'acme_code'" in message
    assert "face 'providers'" in message
    assert "id 'acme_code.dynamic_assets'" in message
    assert "entrypoint 'acme_pack:register'" in message
    assert "runtime.activate('acme_code')" in message
    assert "acme_pack" not in sys.modules


def test_repeated_lookups_are_deterministic_and_do_not_mutate_the_catalog() -> None:
    catalog = _runtime().catalog
    manifests_before = catalog.manifests
    entries_before = catalog.entries()

    first = catalog.entry("assets", "acme_content.cup")
    second = catalog.entry("assets", "acme_content.cup")

    assert first == second
    assert catalog.manifests is manifests_before
    assert catalog.entries() == entries_before


def test_lookup_does_not_import_pack_python() -> None:
    runtime = _runtime()
    assert "acme_pack" not in sys.modules

    runtime.catalog.entry("providers", "acme_code.dynamic_assets")
    runtime.resource_for("scene_presets", "acme_content.tabletop")
    with pytest.raises(PackError):
        runtime.resource_for("providers", "acme_code.dynamic_assets")

    assert "acme_pack" not in sys.modules
