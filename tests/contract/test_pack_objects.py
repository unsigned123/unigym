"""Contract for pack objects: ``PackSet``, ``PackInfo`` and scene handles.

Handles are read-only values minted from manifest catalogs.  Acquisition is
canonical (``packs.pack(name).scene_preset(id)``), has no attribute sugar and
never activates a pack, imports pack Python or reads a resource path while
minting.  ``resource_path`` resolves a preset resource and refuses an
entrypoint-backed provider with the "activate the pack" teaching error.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from unigym.packs import (
    PackError,
    PackSet,
    ScenePresetRef,
    SceneProviderRef,
    installed,
    require,
)
from unigym.packs.runtime import PackRuntime
from unigym.scene_presets import ScenePresetError, load_scene_preset
from unigym.scenes.provider_params import ProviderParamSpec
from unigym.world import World

PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
CONTENT_ROOT = PACKS_ROOT / "acme_content"
CODE_ROOT = PACKS_ROOT / "acme_code"


def _drop_code_modules() -> None:
    """Remove the fixture pack module so import checks cannot see stale state."""
    for name in [
        name for name in sys.modules if name == "acme_pack" or name.startswith("acme_pack.")
    ]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _isolate_import_state():
    """Drop fixture pack modules and restore ``sys.path`` around every test."""
    path_snapshot = list(sys.path)
    _drop_code_modules()
    yield
    sys.path[:] = path_snapshot
    _drop_code_modules()


@pytest.fixture
def packs(monkeypatch: pytest.MonkeyPatch) -> PackSet:
    """The explicit two-pack dependency set from the configured environment."""
    monkeypatch.setenv("UNIGYM_PACKS", os.pathsep.join([str(CONTENT_ROOT), str(CODE_ROOT)]))
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)
    return require("acme_content", "acme_code")


def _boom(*args: object, **kwargs: object) -> None:
    """Fail if a pure pack-object query activates, imports or resolves a path."""
    raise AssertionError(f"unexpected pack boundary call: args={args!r} kwargs={kwargs!r}")


def test_installed_and_require_are_one_filtered_scope(packs: PackSet) -> None:
    """``installed`` enumerates the scope; ``require`` keeps exactly the names."""
    all_packs = installed()
    names = [pack.name for pack in all_packs.packs()]
    assert "acme_content" in names
    assert "acme_code" in names
    assert names.index("acme_content") < names.index("acme_code")

    filtered = require("acme_code")
    assert [pack.name for pack in filtered.packs()] == ["acme_code"]
    assert [pack.name for pack in require().packs()] == []
    assert "acme_pack" not in sys.modules


def test_require_missing_pack_fails_with_installed_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing explicit dependency fails before any handle exists."""
    monkeypatch.setenv("UNIGYM_PACKS", str(CONTENT_ROOT))
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)

    with pytest.raises(PackError, match="not installed in this scope") as raised:
        require("acme_content", "missing_pack")
    message = str(raised.value)
    assert "'missing_pack'" in message
    assert "acme_content" in message

    with pytest.raises(PackError, match="non-empty strings"):
        require("")


def test_scene_preset_ref_is_a_value_with_manifest_origin(packs: PackSet) -> None:
    """Preset handles are equal by value and carry pack-layer provenance."""
    pack = packs.pack("acme_content")
    assert pack.name == "acme_content"
    assert pack.version == "1.0.0"
    assert pack.origin == "env:UNIGYM_PACKS"

    ref = pack.scene_preset("acme_content.tabletop")
    assert isinstance(ref, ScenePresetRef)
    assert (ref.pack, ref.pack_version, ref.id) == (
        "acme_content",
        "1.0.0",
        "acme_content.tabletop",
    )
    assert ref.resource == "scene_presets/tabletop.yaml"
    assert ref.origin.kind == "manifest"
    assert ref.origin.face == "scene_presets"
    assert ref.origin.item_id == ref.id
    assert ref.origin.locator == ref.resource
    assert ref.origin.root == str(CONTENT_ROOT.resolve())

    assert ref == packs.scene_preset("acme_content.tabletop")
    assert hash(ref) == hash(packs.pack("acme_content").scene_preset("acme_content.tabletop"))
    assert pack.scene_presets() == (ref,)
    assert packs.scene_presets() == (ref,)


def test_scene_provider_ref_carries_normalized_params(packs: PackSet) -> None:
    """Provider handles expose entrypoint, normalized schema and origin."""
    pack = packs.pack("acme_code")
    ref = pack.scene_provider("acme_code.table_top")
    assert isinstance(ref, SceneProviderRef)
    assert (ref.pack, ref.pack_version, ref.id) == (
        "acme_code",
        "1.0.0",
        "acme_code.table_top",
    )
    assert ref.entrypoint == "acme_pack:table_top"
    assert not hasattr(ref, "resource")
    assert ref.origin.face == "providers"
    assert ref.origin.locator == ref.entrypoint

    assert ref.params_schema is not None
    assert ref.params_schema["style"] == ProviderParamSpec(type="int", required=False, default=1)
    assert ref.params_schema["seed"] == ProviderParamSpec(type="int", required=True)
    assert pack.scene_providers() == (ref,)
    assert packs.scene_providers() == (ref,)
    assert ref == packs.scene_provider("acme_code.table_top")
    assert hash(ref) == hash(packs.scene_provider("acme_code.table_top"))

    with pytest.raises(PackError, match="not 'scene'"):
        pack.scene_provider("acme_code.dynamic_assets")


def test_provider_without_params_has_no_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unparameterized scene provider mints ``params_schema=None``."""
    root = tmp_path / "acme_bare"
    root.mkdir()
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "unigym.pack/1",
                "name": "acme_bare",
                "version": "2.0.0",
                "api": 1,
                "trust": "code",
                "provides": {
                    "providers": [
                        {
                            "id": "acme_bare.plain",
                            "kind": "scene",
                            "entrypoint": "acme_bare:plain",
                        }
                    ]
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("UNIGYM_PACKS", str(root))
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)

    ref = require("acme_bare").pack("acme_bare").scene_provider("acme_bare.plain")
    assert ref.params_schema is None
    assert ref.origin.root == str(root.resolve())
    assert "acme_bare" not in sys.modules


def test_pack_scoped_and_cross_pack_lookup(packs: PackSet) -> None:
    """Pack-scoped lookup owns only its pack; PackSet is the edge lookup."""
    with pytest.raises(PackError, match="declares no 'providers' id"):
        packs.pack("acme_content").scene_provider("acme_code.table_top")
    assert packs.scene_provider("acme_code.table_top") == packs.pack("acme_code").scene_provider(
        "acme_code.table_top"
    )

    with pytest.raises(PackError, match="unknown pack"):
        packs.pack("missing_pack")
    with pytest.raises(PackError, match="unknown scene preset id"):
        packs.scene_preset("missing_preset")
    with pytest.raises(PackError, match="unknown scene provider id"):
        packs.scene_provider("missing_provider")


def test_no_attribute_sugar(packs: PackSet) -> None:
    """Pack names never resolve through attribute access."""
    assert not hasattr(packs, "acme_content")
    assert not hasattr(packs, "acme_code")
    assert packs.pack("acme_content").name == "acme_content"


def test_refs_never_activate_import_or_resolve_a_path(
    packs: PackSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Minting, comparing and reading provenance stay catalog-only."""
    monkeypatch.setattr(PackRuntime, "activate", _boom)
    monkeypatch.setattr(PackRuntime, "resource", _boom)
    monkeypatch.setattr(PackRuntime, "resource_for", _boom)

    preset = packs.pack("acme_content").scene_preset("acme_content.tabletop")
    provider = packs.pack("acme_code").scene_provider("acme_code.table_top")
    assert preset == packs.scene_preset("acme_content.tabletop")
    assert provider == packs.scene_provider("acme_code.table_top")
    assert preset.origin.pack == "acme_content"
    assert provider.params_schema is not None

    runtime = packs.to_runtime()
    assert runtime.catalog.packs() == ("acme_content", "acme_code")
    assert "acme_pack" not in sys.modules


def test_resource_path_preset_and_provider(packs: PackSet) -> None:
    """Only preset handles resolve a path; providers direct to activation."""
    preset = packs.pack("acme_content").scene_preset("acme_content.tabletop")
    expected = (CONTENT_ROOT / "scene_presets" / "tabletop.yaml").resolve()
    assert packs.resource_path(preset) == expected

    provider = packs.pack("acme_code").scene_provider("acme_code.table_top")
    with pytest.raises(PackError, match="activate the pack") as raised:
        packs.resource_path(provider)
    assert "acme_code.table_top" in str(raised.value)

    foreign = replace(preset, pack="acme_code")
    with pytest.raises(PackError, match="declared by pack 'acme_content', not 'acme_code'"):
        packs.resource_path(foreign)

    with pytest.raises(PackError, match="requires a ScenePresetRef or SceneProviderRef"):
        packs.resource_path("acme_content.tabletop")  # type: ignore[arg-type]


def test_world_and_loader_accept_refs(packs: PackSet) -> None:
    """``World`` and ``load_scene_preset`` accept the pack-object handles."""
    runtime = packs.to_runtime()
    preset = packs.pack("acme_content").scene_preset("acme_content.tabletop")
    provider = packs.pack("acme_code").scene_provider("acme_code.table_top")

    assert World.from_preset(preset, runtime=runtime).fragment() == {
        "scene_preset": "acme_content.tabletop"
    }
    assert World.from_provider(provider, params={"style": 2}).fragment() == {
        "scene_preset": {"provider": "acme_code.table_top", "params": {"style": 2}}
    }
    assert load_scene_preset(preset, runtime).catalog_id == "acme_content.tabletop"

    foreign = replace(preset, pack="acme_code")
    with pytest.raises(ScenePresetError, match="declared by pack 'acme_content', not 'acme_code'"):
        load_scene_preset(foreign, runtime)
