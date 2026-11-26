"""Contract for pack objects: ``PackSet``, ``Pack`` and scene handles.

Handles are read-only values minted from one stored item.  Acquisition is
canonical (``packs.scene_preset(id)`` / ``packs.scene_provider(id)``), has no
attribute sugar and never activates a pack, imports pack Python or reads a
resource path while minting.  ``resource_path`` resolves a preset resource and
refuses an entrypoint-backed provider with the "activate the pack" teaching
error.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from unigym.packs import (
    AssetResourceRef,
    Face,
    PackError,
    PackItem,
    PackSet,
    ProviderOrigin,
    ScenePresetRef,
    SceneProviderRef,
    installed,
    require,
)
from unigym.packs.runtime import PackRuntime
from unigym.robots.base import RobotProfile
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
    assert pack.origin.kind == "pack"
    assert pack.origin.discovered_by == "env:UNIGYM_PACKS"
    assert pack.origin.pack == "acme_content"
    assert pack.origin.pack_version == "1.0.0"
    assert pack.root == CONTENT_ROOT.resolve()

    ref = packs.scene_preset("acme_content.tabletop")
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
    assert ref.origin.resource == ref.resource
    assert ref.origin.entrypoint is None
    assert ref.origin.root == str(CONTENT_ROOT.resolve())

    assert ref == packs.scene_preset("acme_content.tabletop")
    assert hash(ref) == hash(packs.scene_preset("acme_content.tabletop"))
    assert packs.scene_presets() == (ref,)


def test_scene_provider_ref_carries_normalized_params(packs: PackSet) -> None:
    """Provider handles expose entrypoint, normalized schema and origin."""
    ref = packs.scene_provider("acme_code.table_top")
    assert isinstance(ref, SceneProviderRef)
    assert (ref.pack, ref.pack_version, ref.id) == (
        "acme_code",
        "1.0.0",
        "acme_code.table_top",
    )
    assert ref.entrypoint == "acme_pack:table_top"
    assert not hasattr(ref, "resource")
    assert ref.origin.face == "providers"
    assert ref.origin.entrypoint == ref.entrypoint
    assert ref.origin.resource is None

    assert ref.params_schema is not None
    assert ref.params_schema["style"] == ProviderParamSpec(type="int", required=False, default=1)
    assert ref.params_schema["seed"] == ProviderParamSpec(type="int", required=True)
    assert packs.scene_providers() == (ref,)
    assert ref == packs.scene_provider("acme_code.table_top")
    assert hash(ref) == hash(packs.scene_provider("acme_code.table_top"))

    with pytest.raises(PackError, match="not 'scene'"):
        packs.scene_provider("acme_code.dynamic_assets")


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

    ref = require("acme_bare").scene_provider("acme_bare.plain")
    assert ref.params_schema is None
    assert ref.origin.root == str(root.resolve())
    assert "acme_bare" not in sys.modules


def test_pack_scoped_and_cross_pack_lookup(packs: PackSet) -> None:
    """Pack-scoped lookup owns only its pack; PackSet is the edge lookup."""
    with pytest.raises(PackError, match="declares no 'providers' id"):
        packs.pack("acme_content").item(Face.PROVIDERS, "acme_code.table_top")
    provider = packs.scene_provider("acme_code.table_top")
    assert provider.pack == "acme_code"

    with pytest.raises(PackError, match="unknown pack"):
        packs.pack("missing_pack")
    with pytest.raises(PackError, match="unknown id .* on pack face 'scene_presets'"):
        packs.scene_preset("missing_preset")
    with pytest.raises(PackError, match="unknown id .* on pack face 'providers'"):
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

    preset = packs.scene_preset("acme_content.tabletop")
    provider = packs.scene_provider("acme_code.table_top")
    assert preset == packs.scene_preset("acme_content.tabletop")
    assert provider == packs.scene_provider("acme_code.table_top")
    assert preset.origin.pack == "acme_content"
    assert provider.params_schema is not None

    runtime = packs.to_runtime()
    assert runtime.catalog.packs() == ("acme_content", "acme_code")
    assert "acme_pack" not in sys.modules


def test_resource_path_preset_and_provider(packs: PackSet) -> None:
    """Only preset handles resolve a path; providers direct to activation."""
    preset = packs.scene_preset("acme_content.tabletop")
    expected = (CONTENT_ROOT / "scene_presets" / "tabletop.yaml").resolve()
    assert packs.resource_path(preset) == expected

    provider = packs.scene_provider("acme_code.table_top")
    with pytest.raises(PackError, match="activate the pack") as raised:
        packs.resource_path(provider)
    assert "acme_code.table_top" in str(raised.value)

    foreign = replace(preset, pack="acme_code")
    with pytest.raises(PackError, match="declared by pack 'acme_content', not 'acme_code'"):
        packs.resource_path(foreign)

    with pytest.raises(PackError, match="requires a ScenePresetRef, SceneProviderRef"):
        packs.resource_path("acme_content.tabletop")  # type: ignore[arg-type]


def test_world_and_loader_accept_refs(packs: PackSet) -> None:
    """``World`` and ``load_scene_preset`` accept the pack-object handles."""
    runtime = packs.to_runtime()
    preset = packs.scene_preset("acme_content.tabletop")
    provider = packs.scene_provider("acme_code.table_top")

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


def test_pack_and_set_provider_resource_queries(packs: PackSet) -> None:
    """Providers, resources and face counts expose the declared catalog items."""
    content = packs.pack("acme_content")
    assert content.face_counts == {"runs": 1, "scene_presets": 1, "assets": 1}
    assert content.items_for(face=Face.PROVIDERS) == ()
    asset_items = content.items_for(face=Face.ASSETS)
    assert len(asset_items) == 1
    assert isinstance(asset_items[0], PackItem)
    assert asset_items[0].name == "acme_content.cup"
    assert asset_items[0].resource == "assets/cup/asset.yaml"
    assert asset_items[0].origin.face == "assets"
    assert asset_items[0].origin.item_id == "acme_content.cup"

    code = packs.pack("acme_code")
    provider_ids = [item.name for item in code.items_for(face=Face.PROVIDERS)]
    assert provider_ids == ["acme_code.dynamic_assets", "acme_code.table_top"]
    assert all(isinstance(item, PackItem) for item in code.items_for(face=Face.PROVIDERS))
    assert [item.name for item in code.items_for(face=Face.PROVIDERS, kind="scene")] == [
        "acme_code.table_top"
    ]
    assert [item.name for item in packs.providers(kind="scene")] == ["acme_code.table_top"]

    assert [item.name for item in packs.resources(face=Face.SCENE_PRESETS)] == [
        "acme_content.tabletop"
    ]
    with pytest.raises(PackError, match="scene presets are declared under 'scene_presets'"):
        packs.resources(face="scenes")


def test_robot_and_asset_refs_are_minted_from_the_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resource robot and an asset mint equal refs with resolved paths."""
    root = tmp_path / "acme_robot"
    (root / "robots").mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "robots" / "acme_arm.yaml").write_text(
        "schema: unigym.robot/1\n"
        "name: acme.arm\n"
        "joint_sets:\n"
        "  body: [shoulder]\n"
        "default_joint_pos:\n"
        "  shoulder: 0.0\n",
        encoding="utf-8",
    )
    (root / "assets" / "cup.yaml").write_text("kind: fixture\n", encoding="utf-8")
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "unigym.pack/1",
                "name": "acme_robot",
                "version": "2.1.0",
                "api": 1,
                "trust": "content",
                "provides": {
                    "providers": [
                        {"id": "acme.arm", "kind": "robot", "resource": "robots/acme_arm.yaml"}
                    ],
                    "assets": [{"id": "acme.cup", "resource": "assets/cup.yaml"}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("UNIGYM_PACKS", str(root))
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)

    from unigym.robots import ROBOT_REGISTRY, snapshot_robots

    registry_before = snapshot_robots()
    packs = require("acme_robot")
    pack = packs.pack("acme_robot")
    robot = packs.robot("acme.arm")
    assert snapshot_robots() == registry_before
    assert "acme.arm" not in ROBOT_REGISTRY
    assert isinstance(robot, RobotProfile)
    assert robot.name == "acme.arm"
    assert robot.joint_sets == {"body": ("shoulder",)}
    assert packs.robot("acme.arm") == robot
    assert packs.robots() == (robot,)
    provider = pack.items_for(face=Face.PROVIDERS, kind="robot")[0]
    assert provider.name == "acme.arm"
    assert provider.origin.kind == "manifest"
    assert provider.origin.item_id == "acme.arm"
    assert provider.origin.resource == "robots/acme_arm.yaml"
    assert provider.resource == "robots/acme_arm.yaml"

    asset = packs.asset("acme.cup")
    assert isinstance(asset, AssetResourceRef)
    assert asset.origin.face == "assets"
    assert packs.asset("acme.cup") == asset
    assert packs.assets() == (asset,)
    assert packs.resource_path(asset) == (root / "assets" / "cup.yaml").resolve()
    assert pack.face_counts == {"providers": 1, "assets": 1}

    with pytest.raises(PackError, match="has no robot id 'acme.missing'"):
        pack.provider("robot", "acme.missing")
    with pytest.raises(PackError, match="unknown robot id 'acme.missing'"):
        packs.robot("acme.missing")
    with pytest.raises(PackError, match="unknown id 'acme.missing'"):
        packs.asset("acme.missing")


def test_pack_set_equality_origins_and_require_strings(
    packs: PackSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pack sets compare by ordered packs; origins are pack-level values."""
    whole = installed()
    assert whole == packs
    assert whole != require("acme_content")
    assert [origin.pack for origin in packs.origins()] == ["acme_content", "acme_code"]
    assert packs.origins(pack="acme_code") == (packs.pack("acme_code").origin,)
    assert all(isinstance(origin, ProviderOrigin) for origin in packs.origins())
    assert packs.pack("acme_content").origin.kind == "pack"

    with pytest.raises(PackError, match="non-empty strings"):
        require(1)  # type: ignore[arg-type]
    with pytest.raises(PackError, match="unknown pack"):
        packs.origins(pack="missing_pack")


def test_duplicate_catalog_ids_fail_at_pack_set_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two packs declaring the same ``(face, id)`` fail instead of first-match."""
    roots = []
    for name in ("left", "right"):
        root = tmp_path / name
        (root / "scene_presets").mkdir(parents=True)
        (root / "scene_presets" / "shared.yaml").write_text("name: shared\n", encoding="utf-8")
        (root / "unigym.pack.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": "unigym.pack/1",
                    "name": name,
                    "version": "1.0.0",
                    "api": 1,
                    "trust": "content",
                    "provides": {
                        "scene_presets": [
                            {"id": "shared.item", "resource": "scene_presets/shared.yaml"}
                        ]
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        roots.append(str(root))
    monkeypatch.setenv("UNIGYM_PACKS", os.pathsep.join(roots))
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)

    with pytest.raises(PackError, match="duplicate 'scene_presets' id 'shared.item'"):
        installed()


def test_pack_robot_entrypoint_calls_the_factory_without_registry_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entrypoint robot provider returns a RobotProfile and registers nothing."""
    root = tmp_path / "entry_robot"
    (root / "python").mkdir(parents=True)
    (root / "python" / "acme_entry_robot.py").write_text(
        "from unigym.robots.base import RobotProfile\n"
        "\n"
        "\n"
        "def make_profile():\n"
        "    return RobotProfile(\n"
        "        name='acme.entry',\n"
        "        joint_sets={'body': ('j',)},\n"
        "        default_joint_pos={'j': 0.0},\n"
        "        assets=None,\n"
        "        extra_action_sets={},\n"
        "        base_body=None,\n"
        "        foot_bodies=(),\n"
        "        body_sets={},\n"
        "        body_bindings={},\n"
        "    )\n",
        encoding="utf-8",
    )
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "unigym.pack/1",
                "name": "entry_robot",
                "version": "1.0.0",
                "api": 1,
                "trust": "code",
                "provides": {
                    "providers": [
                        {
                            "id": "acme.entry",
                            "kind": "robot",
                            "entrypoint": "acme_entry_robot:make_profile",
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

    from unigym.robots import ROBOT_REGISTRY, snapshot_robots

    before = snapshot_robots()
    robot = require("entry_robot").robot("acme.entry")
    assert isinstance(robot, RobotProfile)
    assert robot.name == "acme.entry"
    assert snapshot_robots() == before
    assert "acme.entry" not in ROBOT_REGISTRY
    assert "acme_entry_robot" in sys.modules


def test_require_can_extend_scope_with_installed_entry_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``require(..., entry_points=True)`` resolves the opt-in entry-point scope."""
    root = tmp_path / "entry_pack"
    root.mkdir()
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "unigym.pack/1",
                "name": "entry_pack",
                "version": "1.0.0",
                "api": 1,
                "trust": "content",
                "provides": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("UNIGYM_PACKS", raising=False)
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)
    monkeypatch.setattr(
        "unigym.packs.runtime._entry_point_roots",
        lambda: [(root, "entry_pack")],
    )

    assert require("entry_pack", entry_points=True).pack("entry_pack").name == "entry_pack"
    assert require("entry_pack").pack("entry_pack").name == "entry_pack"
    with pytest.raises(PackError, match="not installed"):
        require("entry_pack", entry_points=False)
