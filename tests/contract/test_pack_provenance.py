"""Contract for installed-vs-activated packs, provenance and pack handles.

Constructing a :class:`~unigym.packs.PackSet` is a catalog operation: it must
not activate a pack, import pack Python or mutate any registry.  Provenance is
read-only data minted from the manifests; it is never serialized into a run
config.  Ambiguity is fatal and names both packs, both roots and both
discovery labels.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

from unigym.adapters import (
    snapshot_adapter_registry,
    restore_adapter_registry,
)
from unigym.pack_api import discover
from unigym.packs import (
    PackError,
    PackSet,
    ProviderOrigin,
    ScenePresetRef,
    SceneProviderRef,
    discover_set,
    installed,
    require,
)
from unigym.packs.loader import PackLoader
from unigym.props.templates import (
    restore_prop_templates,
    snapshot_prop_templates,
)
from unigym.robots import snapshot_robot_registry, restore_robot_registry
from unigym.scenes.pipeline import (
    restore_scene_content_fillers,
    snapshot_scene_content_fillers,
)
from unigym.scenes.providers import (
    restore_scene_providers,
    snapshot_scene_providers,
)
from unigym.terminations import (
    restore_termination_registry,
    snapshot_termination_registry,
)

PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
CONTENT_ROOT = PACKS_ROOT / "acme_content"
CODE_ROOT = PACKS_ROOT / "acme_code"


def _snapshot_registries():
    """Capture every pack-writable registry as one comparable tuple."""
    return (
        snapshot_robot_registry(),
        snapshot_adapter_registry(),
        snapshot_scene_providers(),
        snapshot_scene_content_fillers(),
        snapshot_prop_templates(),
        snapshot_termination_registry(),
    )


def _restore_registries(snapshots) -> None:
    """Restore the registries captured by :func:`_snapshot_registries`."""
    (
        robots,
        adapters,
        scene_providers,
        content_fillers,
        templates,
        terminations,
    ) = snapshots
    restore_robot_registry(robots)
    restore_adapter_registry(adapters)
    restore_scene_providers(scene_providers)
    restore_scene_content_fillers(content_fillers)
    restore_prop_templates(templates)
    restore_termination_registry(terminations)


@pytest.fixture(autouse=True)
def _isolate_process_state():
    """Restore registries, ``sys.path`` and imported modules around every test."""
    path_snapshot = list(sys.path)
    modules_snapshot = set(sys.modules)
    snapshots = _snapshot_registries()
    yield
    _restore_registries(snapshots)
    sys.path[:] = path_snapshot
    for name in set(sys.modules) - modules_snapshot:
        del sys.modules[name]


def _write_manifest(root: Path, document: dict[str, object]) -> None:
    """Write one pack manifest, creating the root."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def _providers_pack(root: Path, name: str, items: list[dict[str, object]]) -> None:
    """Write a code-trust pack declaring exactly *items* on ``providers``."""
    _write_manifest(
        root,
        {
            "schema": "unigym.pack/1",
            "name": name,
            "version": "1.0.0",
            "api": 1,
            "trust": "code",
            "provides": {"providers": items},
        },
    )


def _trap_pack(root: Path) -> Path:
    """A code pack whose module writes a marker and raises at import time."""
    package = root / "python" / "trap_pack"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '"""Trap module: importing it is a test failure."""\n'
        "\n"
        "from pathlib import Path\n"
        "\n"
        'Path(__file__).with_name("IMPORTED").write_text("imported", encoding="utf-8")\n'
        'raise RuntimeError("trap pack module must not be imported")\n',
        encoding="utf-8",
    )
    _write_manifest(
        root,
        {
            "schema": "unigym.pack/1",
            "name": "trap_pack",
            "version": "1.0.0",
            "api": 1,
            "trust": "code",
            "entrypoint": "trap_pack:register",
            "provides": {
                "providers": [
                    {
                        "id": "trap_pack.thing",
                        "kind": "asset_generator",
                        "entrypoint": "trap_pack:register",
                    }
                ]
            },
        },
    )
    return root


def test_installed_never_activates_a_trap_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed catalog construction imports nothing and mutates no registry."""
    root = _trap_pack(tmp_path / "trap_pack")
    monkeypatch.setenv("UNIGYM_PACKS", str(root))
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", "")

    before = _snapshot_registries()

    views = (
        installed(),
        require("trap_pack"),
        discover_set([root]),
        discover([root]).pack_set,
    )

    after = _snapshot_registries()
    assert after == before
    assert not (root / "python" / "trap_pack" / "IMPORTED").exists()
    assert "trap_pack" not in sys.modules
    for view in views:
        assert [pack.name for pack in view.packs()] == ["trap_pack"]
        provider = view.pack("trap_pack").items_for(face="providers")[0]
        assert provider.name == "trap_pack.thing"
        assert provider.origin.entrypoint == "trap_pack:register"


def _same_named_packs(tmp_path: Path) -> tuple[Path, Path]:
    """Two roots that declare the same pack name."""
    first = tmp_path / "left" / "shared"
    second = tmp_path / "right" / "shared"
    for root in (first, second):
        _write_manifest(
            root,
            {
                "schema": "unigym.pack/1",
                "name": "shared_name",
                "version": "1.0.0",
                "api": 1,
                "trust": "content",
                "provides": {},
            },
        )
    return first, second


def _same_item_packs(tmp_path: Path) -> tuple[Path, Path]:
    """Two differently named roots that declare the same ``(face, id)``."""
    first = tmp_path / "left" / "alpha"
    second = tmp_path / "right" / "beta"
    for root, name in ((first, "alpha_pack"), (second, "beta_pack")):
        (root / "scene_presets").mkdir(parents=True)
        (root / "scene_presets" / "shared.yaml").write_text("name: shared\n", encoding="utf-8")
        _write_manifest(
            root,
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
        )
    return first, second


def test_duplicate_pack_name_is_fatal_and_names_roots_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate pack names fail at construction with every ambiguity fact."""
    first, second = _same_named_packs(tmp_path)
    monkeypatch.setenv("UNIGYM_PACKS", str(first))
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)
    overlay = tmp_path / "unigym.local.yaml"
    overlay.write_text(yaml.safe_dump({"packs": [str(second)]}), encoding="utf-8")

    with pytest.raises(PackError) as raised:
        discover(config_path=overlay)

    message = str(raised.value)
    assert "duplicate pack name 'shared_name'" in message
    assert str(first.resolve()) in message
    assert str(second.resolve()) in message
    assert "'env:UNIGYM_PACKS'" in message
    assert "'local-yaml'" in message


def test_duplicate_item_is_fatal_and_names_packs_roots_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate ``(face, id)`` fails at construction with every ambiguity fact."""
    first, second = _same_item_packs(tmp_path)
    monkeypatch.setenv("UNIGYM_PACKS", str(first))
    monkeypatch.delenv("UNIGYM_DEPS_ROOT", raising=False)
    overlay = tmp_path / "unigym.local.yaml"
    overlay.write_text(yaml.safe_dump({"packs": [str(second)]}), encoding="utf-8")

    with pytest.raises(PackError) as raised:
        discover(config_path=overlay)

    message = str(raised.value)
    assert "duplicate 'scene_presets' id 'shared.item'" in message
    assert "'alpha_pack'" in message
    assert "'beta_pack'" in message
    assert str(first.resolve()) in message
    assert str(second.resolve()) in message
    assert "'env:UNIGYM_PACKS'" in message
    assert "'local-yaml'" in message


def test_resource_only_robot_ref_resolves_without_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A robot may be a declared YAML document; the ref carries its provenance."""
    root = tmp_path / "acme_robots"
    (root / "robots").mkdir(parents=True)
    (root / "robots" / "acme_arm.yaml").write_text(
        "schema: unigym.robot/1\n"
        "name: acme.arm\n"
        "joint_sets:\n"
        "  body: [shoulder]\n"
        "default_joint_pos:\n"
        "  shoulder: 0.0\n",
        encoding="utf-8",
    )
    _write_manifest(
        root,
        {
            "schema": "unigym.pack/1",
            "name": "acme_robots",
            "version": "2.1.0",
            "api": 1,
            "trust": "content",
            "provides": {
                "providers": [
                    {"id": "acme.arm", "kind": "robot", "resource": "robots/acme_arm.yaml"}
                ]
            },
        },
    )
    monkeypatch.setenv("UNIGYM_PACKS", str(root))
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", "")
    assert not (root / "python").exists()

    packs = require("acme_robots")
    loader = PackLoader.from_pack_set(packs)
    robot = loader.robot_profile("acme_robots", "acme.arm")
    assert robot.name == "acme.arm"
    assert loader.robot_profile("acme_robots", "acme.arm") == robot
    provider = packs.pack("acme_robots").items_for(face="providers", kind="robot")[0]
    assert provider.entrypoint is None
    assert provider.origin.pack == "acme_robots"
    assert provider.origin.pack_version == "2.1.0"
    assert provider.origin.root == str(root.resolve())
    assert provider.origin.resource == "robots/acme_arm.yaml"
    assert "acme_pack" not in sys.modules


def _fixture_view(monkeypatch: pytest.MonkeyPatch) -> PackSet:
    """The two fixture packs over the configured environment."""
    monkeypatch.setenv("UNIGYM_PACKS", os.pathsep.join([str(CONTENT_ROOT), str(CODE_ROOT)]))
    monkeypatch.setenv("UNIGYM_DEPS_ROOT", "")
    return installed()


def test_string_edge_and_canonical_handles_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strings resolve at the edge; packs expose methods, never attributes."""
    packs = _fixture_view(monkeypatch)

    preset = packs.scene_preset("acme_content.tabletop")
    assert isinstance(preset, ScenePresetRef)
    assert preset == packs.scene_preset("acme_content.tabletop")

    provider = packs.scene_provider("acme_code.table_top")
    assert isinstance(provider, SceneProviderRef)
    assert provider == packs.scene_provider("acme_code.table_top")

    assert getattr(PackSet, "__getattr__", None) is None
    assert not hasattr(packs, "acme_content")
    with pytest.raises(AttributeError):
        getattr(packs, "acme_content")

    with pytest.raises(PackError, match="unknown pack 'missing_pack'"):
        packs.pack("missing_pack")


def test_from_pack_set_is_pure_and_keeps_loader_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_pack_set`` binds a loader without rediscovery."""
    packs = _fixture_view(monkeypatch)
    loader = PackLoader.from_pack_set(packs)

    wrapped = PackLoader.from_pack_set(loader.pack_set)
    assert wrapped.pack_set is loader.pack_set

    explicit = PackLoader.from_pack_set(packs, origins={"acme_content": "wheel"}).pack_set
    assert explicit.pack("acme_content").origin.discovered_by == "wheel"
    assert explicit.pack("acme_content").origin.kind == "pack"
    assert explicit.pack("acme_code").origin.discovered_by == "env:UNIGYM_PACKS"

    origin = ProviderOrigin(
        kind="pack",
        pack="acme_code",
        pack_version="1.0.0",
        root=str(CODE_ROOT.resolve()),
        discovered_by="deployed",
    )
    overridden = PackLoader.from_pack_set(packs, origins={"acme_code": origin}).pack_set
    assert overridden.pack("acme_code").origin == origin

    with pytest.raises(PackError, match="origins name unknown packs"):
        PackLoader.from_pack_set(packs, origins={"missing": "wheel"})
    with pytest.raises(PackError, match="must be a pack-level ProviderOrigin"):
        PackLoader.from_pack_set(
            packs,
            origins={
                "acme_code": ProviderOrigin(
                    kind="manifest", pack="acme_code", item_id="acme_code.table_top"
                )
            },
        )

    for field in ("pack", "pack_version", "root"):
        origin = ProviderOrigin(
            kind="pack",
            pack="other_pack" if field == "pack" else "acme_code",
            pack_version="9.9.9" if field == "pack_version" else "1.0.0",
            root=str(CODE_ROOT.parent) if field == "root" else str(CODE_ROOT.resolve()),
        )
        with pytest.raises(PackError, match=f"has {field} "):
            PackLoader.from_pack_set(packs, origins={"acme_code": origin})


def test_loader_reports_unknown_when_a_pack_origin_has_no_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pack without a discovery label honestly reports ``unknown``."""
    from unigym.packs.schema import pack_origin, with_pack_origin

    packs = _fixture_view(monkeypatch)
    bare_packs = PackSet(
        tuple(with_pack_origin(pack, pack_origin(pack, "unknown")) for pack in packs.packs())
    )
    loader = PackLoader.from_pack_set(bare_packs)

    for pack in loader.pack_set.packs():
        assert pack.origin.kind == "pack"
        assert pack.origin.discovered_by == "unknown"
        assert pack.origin.location_imported is False
        assert pack.origin.pack == pack.name
        assert pack.origin.root == str(pack.root)
    assert set(loader.origins.values()) == {"unknown"}


def test_from_pack_set_does_not_call_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_pack_set`` never re-walks roots or re-reads manifests."""
    packs = _fixture_view(monkeypatch)
    loader = PackLoader.from_pack_set(packs)

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("from_pack_set must not rediscover")

    monkeypatch.setattr("unigym.packs.loader.discover", boom)
    assert PackLoader.from_pack_set(loader.pack_set).pack_set.packs() == packs.packs()
