"""Contract for pack-set lookup by ``(face, id)`` and loader resource resolution.

Lookup is a pure pack-set operation plus the existing loader resource
resolution. It never discovers, activates or imports pack Python, and an
entrypoint-only item fails with a teaching error that points at activation.
"""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from unigym.pack_api import Face, PackLoader, ResourceLocator, discover
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


def _loader() -> PackLoader:
    return discover([CONTENT_ROOT, CODE_ROOT])


def test_pack_set_item_returns_the_immutable_declared_item() -> None:
    loader = _loader()

    entry = loader.item("scene_presets", "acme_content.tabletop")

    assert entry.pack == "acme_content"
    assert entry.face is Face.SCENE_PRESETS
    assert entry.name == "acme_content.tabletop"
    assert entry.locator == ResourceLocator("scene_presets/tabletop.yaml")
    assert entry.entrypoint is None
    assert entry.kind is None
    assert entry.params is None
    with pytest.raises(FrozenInstanceError):
        setattr(entry, "name", "changed")


def test_pack_set_item_unknown_face_names_the_known_faces() -> None:
    loader = _loader()

    with pytest.raises(PackError) as info:
        loader.item("widgets", "acme_content.tabletop")

    message = str(info.value)
    assert "unknown pack face 'widgets'" in message
    assert "scene_presets" in message


def test_pack_set_item_rejects_none_face_and_resolve_delegates() -> None:
    loader = _loader()

    with pytest.raises(PackError) as info:
        loader.item(None, "acme_content.tabletop")  # type: ignore[arg-type]

    assert "unknown pack face None" in str(info.value)
    assert "scene_presets" in str(info.value)

    with pytest.raises(PackError, match="unknown pack face None"):
        loader.resolve(None, "acme_content.tabletop")  # type: ignore[arg-type]


def test_pack_set_item_unknown_id_names_the_face_id_and_known_ids() -> None:
    loader = _loader()

    with pytest.raises(PackError) as info:
        loader.item("scene_presets", "acme_content.missing")

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
def test_resolve_returns_resource_backed_items(face: str, item_id: str, rel: str) -> None:
    loader = _loader()

    resource = loader.resolve(face, item_id)

    assert resource == (CONTENT_ROOT / rel).resolve()
    assert resource.is_absolute()
    assert resource.is_relative_to(CONTENT_ROOT.resolve())
    assert resource.is_file()


def test_resource_for_entrypoint_only_item_points_at_activation() -> None:
    loader = _loader()
    item = loader.item("providers", "acme_code.dynamic_assets")

    with pytest.raises(PackError) as info:
        loader.resource_for(item)

    message = str(info.value)
    assert "pack 'acme_code'" in message
    assert "face 'providers'" in message
    assert "id 'acme_code.dynamic_assets'" in message
    assert "entrypoint 'acme_pack:register'" in message
    assert "loader.activate('acme_code')" in message
    assert "acme_pack" not in sys.modules


def test_repeated_lookups_are_deterministic_and_do_not_mutate_the_pack_set() -> None:
    loader = _loader()
    pack_set = loader.pack_set
    packs_before = pack_set.packs()
    items_before = pack_set.items()

    first = loader.item("assets", "acme_content.cup")
    second = loader.item("assets", "acme_content.cup")

    assert first == second
    assert pack_set.packs() is packs_before
    assert pack_set.items() == items_before


def test_resource_for_rejects_an_item_from_another_loader_scope() -> None:
    """A foreign item cannot resolve through a loader that does not own its pack."""
    code_loader = discover([CODE_ROOT])
    foreign_item = discover([CONTENT_ROOT]).item("assets", "acme_content.cup")

    with pytest.raises(PackError, match="unknown pack 'acme_content'"):
        code_loader.resource_for(foreign_item)


def test_lookup_does_not_import_pack_python() -> None:
    loader = _loader()
    assert "acme_pack" not in sys.modules

    loader.item("providers", "acme_code.dynamic_assets")
    loader.resolve("scene_presets", "acme_content.tabletop")
    with pytest.raises(PackError):
        loader.resource_for(loader.item("providers", "acme_code.dynamic_assets"))

    assert "acme_pack" not in sys.modules
